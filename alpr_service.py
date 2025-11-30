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
from datetime import datetime

import numpy as np
import cv2

# Import ALPR engine
from jetson_alpr import JetsonALPR

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
        self.api_client = None
        self.watcher = None
        self.running = False
        self.stats = {
            'processed': 0,
            'plates_found': 0,
            'gates_opened': 0,
            'errors': 0,
            'start_time': None
        }

    def initialize(self):
        """Initialize ALPR models and API client."""
        self.logger.info("=" * 60)
        self.logger.info("ALPR Service Starting")
        self.logger.info("=" * 60)
        self.logger.info("Mode: %s", Config.MODE.upper())
        self.logger.info("Watch: %s", Config.WATCH_DIR)
        self.logger.info("Output: %s", Config.OUTPUT_DIR)

        # Create directories
        os.makedirs(Config.WATCH_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Initialize API client (if API mode)
        if Config.MODE == 'api' and HAS_API:
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

        # Warmup GPU
        self.logger.info("Warming up GPU...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.alpr.predict(dummy)
        self.logger.info("GPU ready")

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
                if Config.MODE == 'api' and self.api_client:
                    self._process_api_mode(
                        plate_text, annotated, filename, confidence, bbox)
                else:
                    self._process_ftp_mode(
                        plate_text, annotated, filename, confidence)
            else:
                self.logger.info("  No plate detected [%dms]", inference_ms)

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

    def _safe_remove(self, filepath):
        """Safely remove a file."""
        try:
            os.remove(filepath)
        except OSError:
            pass

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
        self.logger.info("  Gates opened: %d", self.stats['gates_opened'])
        self.logger.info("  Errors: %d", self.stats['errors'])
        self.logger.info("-" * 60)

        if self.api_client:
            self.api_client.shutdown()

        self.logger.info("Service stopped")


# =============================================================================
# Main
# =============================================================================
def main():
    service = ALPRService()
    service.run()


if __name__ == '__main__':
    main()
