#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production ALPR Service for Jetson Nano - PanoramaCity Integration

Features:
- Watches FTP inbox for new images (from Hikvision cameras)
- Runs GPU-accelerated ALPR detection and OCR
- Checks plate authorization against remote database
- Opens gate barrier if authorized
- Uploads annotated images to cloud API
- Protects SD card using /dev/shm (RAM filesystem)

Modes:
- API mode (default): Uploads to PanoramaCity API, controls gates
- FTP mode (legacy): Uploads to FTP server

Author: Ionut/Claude
"""
from __future__ import print_function
import os
import sys
import time
import signal
import logging
import ftplib
import threading
from datetime import datetime

# Python 2/3 compatibility for HTTP server
try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
except ImportError:
    from BaseHTTPServer import HTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler

import numpy as np
import cv2

# Import ALPR engine
from jetson_alpr import JetsonALPR, EnhancedALPR
from concurrent.futures import ThreadPoolExecutor

# Import API client (optional, graceful fallback)
try:
    from plate_api import PlateAPIClient, APIConfig, parse_gate_from_filename
    HAS_API = True
except ImportError:
    HAS_API = False


# =============================================================================
# Configuration
# =============================================================================
class Config:
    """Service configuration from environment variables."""

    # Watch directory (use /dev/shm for RAM-based storage)
    WATCH_DIR = os.environ.get('ALPR_WATCH_DIR', '/dev/shm/alpr_inbox')
    OUTPUT_DIR = os.environ.get('ALPR_OUTPUT_DIR', '/dev/shm/alpr_outbox')

    # Mode: 'api' (PanoramaCity) or 'ftp' (legacy)
    MODE = os.environ.get('ALPR_MODE', 'api').lower()

    # FTP settings (legacy mode)
    FTP_HOST = os.environ.get('ALPR_FTP_HOST', '192.168.100.238')
    FTP_USER = os.environ.get('ALPR_FTP_USER', 'photo')
    FTP_PASS = os.environ.get('ALPR_FTP_PASS', 'photo')
    FTP_DIR = os.environ.get('ALPR_FTP_DIR', '/')

    # Detection settings
    CONFIDENCE_THRESHOLD = float(os.environ.get('ALPR_CONF_THRESH', '0.4'))
    DELETE_AFTER_PROCESS = os.environ.get('ALPR_DELETE_PROCESSED', 'true').lower() == 'true'

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Polling interval (seconds) - fallback if inotify unavailable
    POLL_INTERVAL = float(os.environ.get('ALPR_POLL_INTERVAL', '0.5'))

    # Enhanced detection (for distant plates) - requires vehicle model (yolov8n.onnx)
    # Disabled by default as vehicle model is not included
    ENHANCED_DETECTION = os.environ.get('ALPR_ENHANCED_DETECTION', 'false').lower() == 'true'
    MIN_PLATE_SIZE = int(os.environ.get('ALPR_MIN_PLATE_SIZE', '30'))
    VEHICLE_CONF_THRESH = float(os.environ.get('ALPR_VEHICLE_CONF_THRESH', '0.3'))

    # Crop margins for enhanced detection (car usually centered in frame)
    # Values are percentages (0.0-0.5). E.g., 0.2 = crop 20% from each side
    ENHANCED_CROP_LEFT = float(os.environ.get('ALPR_ENHANCED_CROP_LEFT', '0.0'))
    ENHANCED_CROP_RIGHT = float(os.environ.get('ALPR_ENHANCED_CROP_RIGHT', '0.0'))
    ENHANCED_CROP_TOP = float(os.environ.get('ALPR_ENHANCED_CROP_TOP', '0.0'))
    ENHANCED_CROP_BOTTOM = float(os.environ.get('ALPR_ENHANCED_CROP_BOTTOM', '0.0'))

    # Test mode - only print detections, no API calls
    TEST_MODE = os.environ.get('ALPR_TEST_MODE', 'false').lower() == 'true'

    # Live preview HTTP server port (test mode only)
    PREVIEW_PORT = int(os.environ.get('ALPR_PREVIEW_PORT', '8080'))

    # Annotation style
    BOX_COLOR = (0, 255, 0)  # Green BGR
    BOX_THICKNESS = 3
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.2
    FONT_THICKNESS = 3
    TEXT_COLOR = (0, 255, 0)
    TEXT_BG_COLOR = (0, 0, 0)

    # Logging
    LOG_LEVEL = os.environ.get('ALPR_LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('ALPR_LOG_FILE', '')


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging():
    """Configure logging."""
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if Config.LOG_FILE:
        handlers.append(logging.FileHandler(Config.LOG_FILE))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger('alpr_service')


# =============================================================================
# Live Preview HTTP Server
# =============================================================================
class LivePreviewHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves latest.jpg with auto-refresh HTML page."""

    def __init__(self, *args, output_dir=None, **kwargs):
        self.output_dir = output_dir or Config.OUTPUT_DIR
        # Python 2 compatibility - call parent init
        SimpleHTTPRequestHandler.__init__(self, *args, **kwargs)

    def log_message(self, format, *args):
        """Suppress HTTP logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/latest.jpg':
            self._serve_latest_image()
        elif self.path == '/status.json':
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_html(self):
        """Serve auto-refreshing HTML page - simple debug view."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>ALPR Debug</title>
    <style>
        body { margin: 0; padding: 10px; background: #1a1a1a; font-family: monospace; color: #0f0; }
        #status { font-size: 12px; margin-bottom: 5px; }
        img { max-width: 100%; max-height: 90vh; border: 1px solid #333; }
    </style>
</head>
<body>
    <div id="status">Waiting...</div>
    <img id="preview" src="/latest.jpg">
    <script>
        setInterval(function() {
            document.getElementById('preview').src = '/latest.jpg?t=' + Date.now();
            fetch('/status.json?t=' + Date.now())
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    document.getElementById('status').innerText =
                        d.plate + ' | ' + d.confidence + ' | ' + (d.enhanced ? 'ENHANCED' : 'direct') + ' | ' + d.timestamp;
                })
                .catch(function() {});
        }, 5000);
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(html))
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_latest_image(self):
        """Serve the latest.jpg file."""
        img_path = os.path.join(self.output_dir, 'latest.jpg')
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(data))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def _serve_status(self):
        """Serve status JSON."""
        import json
        status_path = os.path.join(self.output_dir, 'status.json')
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(data))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(data.encode('utf-8'))
        else:
            data = '{"plate": null, "confidence": null, "timestamp": null}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data.encode('utf-8'))


class LivePreviewServer:
    """Background HTTP server for live preview in test mode."""

    def __init__(self, port, output_dir, logger):
        self.port = port
        self.output_dir = output_dir
        self.logger = logger
        self.server = None
        self.thread = None

    def start(self):
        """Start the HTTP server in a background thread."""
        def handler_factory(*args, **kwargs):
            return LivePreviewHandler(*args, output_dir=self.output_dir, **kwargs)

        try:
            self.server = HTTPServer(('0.0.0.0', self.port), handler_factory)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.logger.info("Live preview server started at http://0.0.0.0:%d", self.port)
        except Exception as e:
            self.logger.error("Failed to start preview server: %s", e)

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.logger.info("Live preview server stopped")


# =============================================================================
# Image Annotation
# =============================================================================
def annotate_image(image, results):
    """Draw bounding boxes and plate text on image."""
    annotated = image.copy()

    for result in results:
        x1, y1, x2, y2 = result['bbox']
        plate_text = result['text']
        confidence = result['ocr_confidence'] * 100

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                     Config.BOX_COLOR, Config.BOX_THICKNESS)

        # Prepare label
        label = "{} {:.1f}%".format(plate_text, confidence)

        # Calculate text size
        (text_w, text_h), _ = cv2.getTextSize(
            label, Config.FONT, Config.FONT_SCALE, Config.FONT_THICKNESS)

        # Position text above box
        text_x = x1
        text_y = y1 - 10
        if text_y - text_h < 0:
            text_y = y2 + text_h + 10

        # Draw background and text
        cv2.rectangle(annotated,
                     (text_x - 2, text_y - text_h - 5),
                     (text_x + text_w + 2, text_y + 5),
                     Config.TEXT_BG_COLOR, -1)
        cv2.putText(annotated, label, (text_x, text_y),
                   Config.FONT, Config.FONT_SCALE,
                   Config.TEXT_COLOR, Config.FONT_THICKNESS)

    return annotated


# =============================================================================
# FTP Upload (Legacy)
# =============================================================================
def upload_to_ftp(local_path, remote_filename, logger):
    """Upload file to FTP server with retry logic."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with ftplib.FTP(Config.FTP_HOST, timeout=30) as ftp:
                ftp.login(Config.FTP_USER, Config.FTP_PASS)
                if Config.FTP_DIR and Config.FTP_DIR != '/':
                    try:
                        ftp.cwd(Config.FTP_DIR)
                    except ftplib.error_perm:
                        ftp.mkd(Config.FTP_DIR)
                        ftp.cwd(Config.FTP_DIR)

                with open(local_path, 'rb') as f:
                    ftp.storbinary('STOR {}'.format(remote_filename), f)

                logger.info("FTP uploaded: %s", remote_filename)
                return True

        except Exception as e:
            logger.warning("FTP attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2)

    logger.error("FTP upload failed: %s", remote_filename)
    return False


# =============================================================================
# File Watcher
# =============================================================================
class FileWatcher:
    """Watch directory for new image files using inotify or polling."""

    def __init__(self, watch_dir, callback, logger):
        self.watch_dir = watch_dir
        self.callback = callback
        self.logger = logger
        self.running = False
        self.processed_files = set()
        self._use_inotify = False

        try:
            import inotify.adapters
            self._inotify = inotify.adapters.Inotify()
            self._use_inotify = True
            self.logger.info("Using inotify for file watching")
        except ImportError:
            self.logger.info("Using polling for file watching")

    def start(self):
        """Start watching."""
        self.running = True
        os.makedirs(self.watch_dir, exist_ok=True)

        if self._use_inotify:
            self._watch_inotify()
        else:
            self._watch_polling()

    def stop(self):
        """Stop watching."""
        self.running = False

    def _is_image(self, filename):
        """Check if file is a supported image."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in Config.IMAGE_EXTENSIONS

    def _watch_inotify(self):
        """Watch using inotify (efficient, event-driven)."""
        import inotify.adapters

        self._inotify.add_watch(self.watch_dir)

        for event in self._inotify.event_gen(yield_nones=True):
            if not self.running:
                break
            if event is None:
                continue

            (_, type_names, path, filename) = event

            if 'IN_CLOSE_WRITE' in type_names or 'IN_MOVED_TO' in type_names:
                if self._is_image(filename):
                    filepath = os.path.join(path, filename)
                    if filepath not in self.processed_files:
                        self.processed_files.add(filepath)
                        time.sleep(0.1)  # Ensure file fully written
                        self.callback(filepath)

    def _watch_polling(self):
        """Watch using polling (fallback)."""
        while self.running:
            try:
                for filename in os.listdir(self.watch_dir):
                    if not self._is_image(filename):
                        continue

                    filepath = os.path.join(self.watch_dir, filename)
                    if filepath in self.processed_files:
                        continue

                    # Check file is complete
                    try:
                        size1 = os.path.getsize(filepath)
                        time.sleep(0.2)
                        size2 = os.path.getsize(filepath)
                        if size1 == size2 and size1 > 0:
                            self.processed_files.add(filepath)
                            self.callback(filepath)
                    except OSError:
                        continue

            except Exception as e:
                self.logger.error("Polling error: %s", e)

            time.sleep(Config.POLL_INTERVAL)


# =============================================================================
# ALPR Service
# =============================================================================
class ALPRService:
    """Main ALPR processing service."""

    def __init__(self):
        self.logger = setup_logging()
        self.alpr = None
        self.enhanced_alpr = None
        self.api_client = None
        self.watcher = None
        self.running = False
        self._enhanced_executor = None
        self.preview_server = None
        self.stats = {
            'processed': 0,
            'plates_found': 0,
            'gates_opened': 0,
            'enhanced_detections': 0,
            'errors': 0,
            'start_time': None
        }

    def initialize(self):
        """Initialize ALPR models and API client."""
        self.logger.info("=" * 60)
        self.logger.info("ALPR Service Starting")
        self.logger.info("=" * 60)
        self.logger.info("Mode: %s%s", Config.MODE.upper(), " [TEST MODE]" if Config.TEST_MODE else "")
        self.logger.info("Watch: %s", Config.WATCH_DIR)
        self.logger.info("Output: %s", Config.OUTPUT_DIR)
        if Config.TEST_MODE:
            self.logger.info("*** TEST MODE - No API calls, detection only ***")
            self.logger.info("Live preview: http://0.0.0.0:%d", Config.PREVIEW_PORT)

        # Create directories
        os.makedirs(Config.WATCH_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Initialize API client (if API mode and not test mode)
        if Config.MODE == 'api' and HAS_API and not Config.TEST_MODE:
            self.logger.info("Initializing API client...")
            self.api_client = PlateAPIClient()
            if not self.api_client.initialize():
                self.logger.warning("API init failed, will retry on first detection")
        elif Config.MODE == 'ftp':
            self.logger.info("FTP destination: %s@%s:%s",
                           Config.FTP_USER, Config.FTP_HOST, Config.FTP_DIR)

        # Initialize ALPR
        self.logger.info("Loading ALPR models...")
        self.alpr = JetsonALPR(use_gpu=True, conf_thresh=Config.CONFIDENCE_THRESHOLD)
        self.logger.info("ALPR models loaded")

        # Initialize Enhanced ALPR for distant plates (with vehicle detection)
        if Config.ENHANCED_DETECTION:
            self.logger.info("Initializing enhanced detection (vehicle detection)...")
            self.enhanced_alpr = EnhancedALPR(
                alpr=self.alpr,
                use_gpu=True,
                vehicle_conf_thresh=Config.VEHICLE_CONF_THRESH,
                min_plate_size=Config.MIN_PLATE_SIZE
            )
            if self.enhanced_alpr.has_vehicle_detector:
                self.logger.info("  Vehicle detector loaded")
                # Background executor for enhanced detection (1 worker to not overload GPU)
                self._enhanced_executor = ThreadPoolExecutor(max_workers=1)
            else:
                self.logger.warning("  Vehicle model not found, enhanced detection disabled")
                self.enhanced_alpr = None

        # Warmup GPU
        self.logger.info("Warming up GPU...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.alpr.predict(dummy)
        self.logger.info("GPU ready")

        # Start live preview server in test mode
        if Config.TEST_MODE:
            self.preview_server = LivePreviewServer(
                Config.PREVIEW_PORT, Config.OUTPUT_DIR, self.logger)
            self.preview_server.start()

    def process_image(self, filepath):
        """Process a single image."""
        filename = os.path.basename(filepath)
        self.logger.info("Processing: %s", filename)

        try:
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                self.logger.warning("Could not read: %s", filepath)
                self.stats['errors'] += 1
                return

            # Run ALPR
            start_time = time.time()
            results = self.alpr.predict(image)
            inference_ms = (time.time() - start_time) * 1000

            self.stats['processed'] += 1

            if results:
                self.stats['plates_found'] += len(results)

                # Get best detection
                best = max(results, key=lambda r: r['ocr_confidence'])
                plate_text = best['text']
                confidence = best['ocr_confidence']
                bbox = best['bbox']

                self.logger.info("  Plate: %s (%.1f%%) [%dms]",
                               plate_text, confidence * 100, inference_ms)

                # Annotate image
                annotated = annotate_image(image, results)

                # Process based on mode
                if Config.TEST_MODE:
                    # Test mode - just save annotated image, no API calls
                    self._process_test_mode(plate_text, annotated, filename, confidence, bbox)
                elif Config.MODE == 'api' and self.api_client:
                    self._process_api_mode(
                        plate_text, annotated, filename, confidence, bbox)
                else:
                    self._process_ftp_mode(
                        plate_text, annotated, filename, confidence)

                # Check if plate is too small - queue enhanced detection in background
                if self._should_run_enhanced(results) and self.enhanced_alpr:
                    x1, y1, x2, y2 = best['bbox']
                    plate_w, plate_h = x2 - x1, y2 - y1
                    self.logger.info("  [ENHANCED] Plate too small (%dx%d < %d), sending to vehicle detection",
                                   plate_w, plate_h, Config.MIN_PLATE_SIZE)
                    self._queue_enhanced_detection(image.copy(), filename)
            else:
                self.logger.info("  No plate detected [%dms]", inference_ms)

                # No results - queue enhanced detection in background
                if self.enhanced_alpr:
                    self.logger.info("  [ENHANCED] No direct detection, sending to vehicle detection")
                    self._queue_enhanced_detection(image.copy(), filename)

            # Cleanup input file
            if Config.DELETE_AFTER_PROCESS:
                self._safe_remove(filepath)

        except Exception as e:
            self.logger.error("Error processing %s: %s", filepath, e)
            self.stats['errors'] += 1

    def _process_api_mode(self, plate, image, filename, confidence, bbox):
        """Process detection in API mode."""
        gate_id = parse_gate_from_filename(filename) if HAS_API else 1

        gate_opened, reason = self.api_client.process_detection(
            plate=plate,
            image=image,
            filename=filename,
            gate_id=gate_id,
            confidence=confidence,
            bbox=bbox
        )

        if gate_opened:
            self.stats['gates_opened'] += 1

        self.logger.info("  Access: %s | Gate: %s",
                        reason, "OPENED" if gate_opened else "closed")

    def _process_ftp_mode(self, plate, image, filename, confidence):
        """Process detection in FTP mode (legacy)."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = os.path.splitext(filename)[0]
        output_name = "{}_{}_{}.jpg".format(timestamp, plate, base)
        output_path = os.path.join(Config.OUTPUT_DIR, output_name)

        # Save annotated image
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Upload to FTP
        upload_to_ftp(output_path, output_name, self.logger)

        # Cleanup
        if Config.DELETE_AFTER_PROCESS:
            self._safe_remove(output_path)

    def _process_test_mode(self, plate, image, filename, confidence, bbox, enhanced=False):
        """Process detection in test mode - overwrite latest.jpg for live preview."""
        import json

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        x1, y1, x2, y2 = bbox
        plate_w, plate_h = x2 - x1, y2 - y1

        # Always overwrite latest.jpg (single file, no disk filling)
        output_path = os.path.join(Config.OUTPUT_DIR, 'latest.jpg')
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Write status JSON for live preview
        status = {
            'plate': plate,
            'confidence': '{:.1f}%'.format(confidence * 100),
            'timestamp': timestamp,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'size': {'w': plate_w, 'h': plate_h},
            'enhanced': enhanced,
            'filename': filename
        }
        status_path = os.path.join(Config.OUTPUT_DIR, 'status.json')
        with open(status_path, 'w') as f:
            json.dump(status, f)

        # Log info
        prefix = "[ENHANCED] " if enhanced else ""
        self.logger.info("  %s[TEST] Plate: %s (%.1f%%) -> latest.jpg",
                        prefix, plate, confidence * 100)
        self.logger.info("  %s[TEST] BBox: (%d,%d)-(%d,%d) Size: %dx%d",
                        prefix, x1, y1, x2, y2, plate_w, plate_h)

    def _safe_remove(self, filepath):
        """Safely remove a file."""
        try:
            os.remove(filepath)
        except OSError:
            pass

    def _should_run_enhanced(self, results):
        """Check if we should run enhanced detection (plates too small)."""
        if not results:
            return True

        for r in results:
            x1, y1, x2, y2 = r['bbox']
            plate_w = x2 - x1
            plate_h = y2 - y1
            # If any plate is large enough, don't run enhanced
            if plate_w >= Config.MIN_PLATE_SIZE and plate_h >= Config.MIN_PLATE_SIZE // 2:
                return False
        return True

    def _queue_enhanced_detection(self, image, filename):
        """Queue enhanced detection to run in background thread."""
        if self._enhanced_executor:
            # Apply crop margins if configured (car usually centered)
            h, w = image.shape[:2]
            crop_left = int(w * Config.ENHANCED_CROP_LEFT)
            crop_right = int(w * Config.ENHANCED_CROP_RIGHT)
            crop_top = int(h * Config.ENHANCED_CROP_TOP)
            crop_bottom = int(h * Config.ENHANCED_CROP_BOTTOM)

            if crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0:
                self.logger.debug("  [ENHANCED] Applying crop margins: L=%d R=%d T=%d B=%d",
                                crop_left, crop_right, crop_top, crop_bottom)
                image = image[crop_top:h-crop_bottom, crop_left:w-crop_right]

            self._enhanced_executor.submit(
                self._process_enhanced_detection, image, filename
            )

    def _process_enhanced_detection(self, image, filename):
        """
        Run enhanced detection in background thread.
        Detects vehicles, crops them, and re-runs plate detection.
        """
        try:
            start_time = time.time()
            results = self.enhanced_alpr.predict_enhanced(image, force_vehicle_detection=True)
            inference_ms = (time.time() - start_time) * 1000

            if results:
                self.stats['enhanced_detections'] += 1

                # Get best detection
                best = max(results, key=lambda r: r['ocr_confidence'])
                plate_text = best['text']
                confidence = best['ocr_confidence']
                bbox = best['bbox']

                source = best.get('from_vehicle', 'direct')
                self.logger.info("[ENHANCED] Plate: %s (%.1f%%) from %s [%dms]",
                               plate_text, confidence * 100, source, inference_ms)

                # Annotate image
                annotated = annotate_image(image, results)

                # Process based on mode
                if Config.TEST_MODE:
                    self._process_test_mode(plate_text, annotated, filename, confidence, bbox, enhanced=True)
                elif Config.MODE == 'api' and self.api_client:
                    self._process_api_mode(
                        plate_text, annotated, filename, confidence, bbox)
                else:
                    self._process_ftp_mode(
                        plate_text, annotated, filename, confidence)
            else:
                self.logger.info("[ENHANCED] No plates found [%dms]", inference_ms)

        except Exception as e:
            self.logger.error("[ENHANCED] Error: %s", e)

    def run(self):
        """Start the service."""
        self.running = True
        self.stats['start_time'] = time.time()

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.initialize()

        self.watcher = FileWatcher(Config.WATCH_DIR, self.process_image, self.logger)
        self.logger.info("-" * 60)
        self.logger.info("Watching for images...")
        self.logger.info("-" * 60)

        try:
            self.watcher.start()
        except KeyboardInterrupt:
            pass

        self._shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received signal %d, shutting down...", signum)
        self.running = False
        if self.watcher:
            self.watcher.stop()

    def _shutdown(self):
        """Clean shutdown."""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0

        self.logger.info("-" * 60)
        self.logger.info("Service Statistics:")
        self.logger.info("  Uptime: %.1f seconds", uptime)
        self.logger.info("  Images processed: %d", self.stats['processed'])
        self.logger.info("  Plates found: %d", self.stats['plates_found'])
        self.logger.info("  Enhanced detections: %d", self.stats['enhanced_detections'])
        self.logger.info("  Gates opened: %d", self.stats['gates_opened'])
        self.logger.info("  Errors: %d", self.stats['errors'])
        self.logger.info("-" * 60)

        # Shutdown enhanced detection executor
        if self._enhanced_executor:
            self.logger.info("Waiting for background tasks...")
            self._enhanced_executor.shutdown(wait=True)

        if self.api_client:
            self.api_client.shutdown()

        # Stop preview server
        if self.preview_server:
            self.preview_server.stop()

        self.logger.info("Service stopped")


# =============================================================================
# Main
# =============================================================================
def main():
    service = ALPRService()
    service.run()


if __name__ == '__main__':
    main()
