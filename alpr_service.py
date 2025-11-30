#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production ALPR Service for Jetson Nano
- Watches FTP inbox directory for new images (from Hikvision camera)
- Runs ALPR detection and OCR
- Annotates image with bounding box and plate text
- Uploads annotated image to destination FTP server
- Cleans up processed files

Designed for /dev/shm (tmpfs) to protect SD card from excessive writes.
"""
from __future__ import print_function
import os
import sys
import time
import signal
import logging
import threading
import ftplib
from datetime import datetime
from collections import deque

import numpy as np
import cv2

# Import ALPR components from jetson_alpr
from jetson_alpr import JetsonALPR

# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Watch directory (use /dev/shm for RAM-based storage)
    WATCH_DIR = os.environ.get('ALPR_WATCH_DIR', '/dev/shm/ftp_inbox')

    # Processed/output directory
    OUTPUT_DIR = os.environ.get('ALPR_OUTPUT_DIR', '/dev/shm/ftp_outbox')

    # Destination FTP server
    FTP_HOST = os.environ.get('ALPR_FTP_HOST', '192.168.100.238')
    FTP_USER = os.environ.get('ALPR_FTP_USER', 'photo')
    FTP_PASS = os.environ.get('ALPR_FTP_PASS', 'photo')
    FTP_DIR = os.environ.get('ALPR_FTP_DIR', '/')  # Remote directory

    # Processing settings
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
    TEXT_COLOR = (0, 255, 0)  # Green
    TEXT_BG_COLOR = (0, 0, 0)  # Black background

    # Logging
    LOG_LEVEL = os.environ.get('ALPR_LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('ALPR_LOG_FILE', '')  # Empty = stdout only


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging():
    """Configure logging"""
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
    """
    Draw bounding boxes and plate text on image.
    Style matches the example: green box with "PLATE 97.6%" above.
    """
    annotated = image.copy()

    for result in results:
        x1, y1, x2, y2 = result['bbox']
        plate_text = result['text']
        confidence = result['ocr_confidence'] * 100

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                     Config.BOX_COLOR, Config.BOX_THICKNESS)

        # Prepare label text
        label = "{} {:.2f}%".format(plate_text, confidence)

        # Calculate text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, Config.FONT, Config.FONT_SCALE, Config.FONT_THICKNESS
        )

        # Position text above the box
        text_x = x1
        text_y = y1 - 10

        # Ensure text stays within image bounds
        if text_y - text_h < 0:
            text_y = y2 + text_h + 10  # Put below if no room above

        # Draw text background
        cv2.rectangle(annotated,
                     (text_x - 2, text_y - text_h - 5),
                     (text_x + text_w + 2, text_y + 5),
                     Config.TEXT_BG_COLOR, -1)

        # Draw text
        cv2.putText(annotated, label, (text_x, text_y),
                   Config.FONT, Config.FONT_SCALE,
                   Config.TEXT_COLOR, Config.FONT_THICKNESS)

    return annotated


# =============================================================================
# FTP Upload
# =============================================================================
def upload_to_ftp(local_path, remote_filename, logger):
    """Upload file to FTP server with retry logic"""
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            with ftplib.FTP(Config.FTP_HOST, timeout=30) as ftp:
                ftp.login(Config.FTP_USER, Config.FTP_PASS)

                # Change to target directory if specified
                if Config.FTP_DIR and Config.FTP_DIR != '/':
                    try:
                        ftp.cwd(Config.FTP_DIR)
                    except ftplib.error_perm:
                        # Try to create directory
                        ftp.mkd(Config.FTP_DIR)
                        ftp.cwd(Config.FTP_DIR)

                # Upload file
                with open(local_path, 'rb') as f:
                    ftp.storbinary('STOR {}'.format(remote_filename), f)

                logger.info("Uploaded to FTP: {}".format(remote_filename))
                return True

        except Exception as e:
            logger.warning("FTP upload attempt {}/{} failed: {}".format(
                attempt + 1, max_retries, e))
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    logger.error("FTP upload failed after {} attempts: {}".format(
        max_retries, remote_filename))
    return False


# =============================================================================
# File Watcher
# =============================================================================
class FileWatcher:
    """Watch directory for new image files"""

    def __init__(self, watch_dir, callback, logger):
        self.watch_dir = watch_dir
        self.callback = callback
        self.logger = logger
        self.running = False
        self.processed_files = set()
        self._use_inotify = False

        # Try to use inotify for efficiency
        try:
            import inotify.adapters
            self._inotify = inotify.adapters.Inotify()
            self._use_inotify = True
            self.logger.info("Using inotify for file watching")
        except ImportError:
            self.logger.info("inotify not available, using polling")

    def start(self):
        """Start watching"""
        self.running = True

        # Ensure watch directory exists
        os.makedirs(self.watch_dir, exist_ok=True)

        if self._use_inotify:
            self._watch_inotify()
        else:
            self._watch_polling()

    def stop(self):
        """Stop watching"""
        self.running = False

    def _is_image(self, filename):
        """Check if file is a supported image"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in Config.IMAGE_EXTENSIONS

    def _watch_inotify(self):
        """Watch using inotify (Linux kernel events)"""
        import inotify.adapters
        import inotify.constants

        self._inotify.add_watch(self.watch_dir)

        for event in self._inotify.event_gen(yield_nones=True):
            if not self.running:
                break

            if event is None:
                continue

            (_, type_names, path, filename) = event

            # Look for file close events (file fully written)
            if 'IN_CLOSE_WRITE' in type_names or 'IN_MOVED_TO' in type_names:
                if self._is_image(filename):
                    filepath = os.path.join(path, filename)
                    if filepath not in self.processed_files:
                        self.processed_files.add(filepath)
                        # Small delay to ensure file is fully written
                        time.sleep(0.1)
                        self.callback(filepath)

    def _watch_polling(self):
        """Watch using polling (fallback)"""
        while self.running:
            try:
                for filename in os.listdir(self.watch_dir):
                    if not self._is_image(filename):
                        continue

                    filepath = os.path.join(self.watch_dir, filename)

                    if filepath in self.processed_files:
                        continue

                    # Check if file is complete (not being written)
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
                self.logger.error("Polling error: {}".format(e))

            time.sleep(Config.POLL_INTERVAL)


# =============================================================================
# ALPR Service
# =============================================================================
class ALPRService:
    """Main ALPR processing service"""

    def __init__(self):
        self.logger = setup_logging()
        self.alpr = None
        self.watcher = None
        self.running = False
        self.stats = {
            'processed': 0,
            'plates_found': 0,
            'errors': 0,
            'start_time': None
        }

    def initialize(self):
        """Initialize ALPR models"""
        self.logger.info("=" * 60)
        self.logger.info("ALPR Service Starting")
        self.logger.info("=" * 60)
        self.logger.info("Watch directory: {}".format(Config.WATCH_DIR))
        self.logger.info("Output directory: {}".format(Config.OUTPUT_DIR))
        self.logger.info("FTP destination: {}@{}:{}".format(
            Config.FTP_USER, Config.FTP_HOST, Config.FTP_DIR))

        # Create directories
        os.makedirs(Config.WATCH_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Initialize ALPR
        self.logger.info("Loading ALPR models...")
        self.alpr = JetsonALPR(use_gpu=True, conf_thresh=Config.CONFIDENCE_THRESHOLD)
        self.logger.info("ALPR models loaded successfully")

        # Warmup
        self.logger.info("Warming up GPU...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.alpr.predict(dummy)
        self.logger.info("GPU warmup complete")

    def process_image(self, filepath):
        """Process a single image"""
        filename = os.path.basename(filepath)
        self.logger.info("Processing: {}".format(filename))

        try:
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                self.logger.warning("Could not read image: {}".format(filepath))
                self.stats['errors'] += 1
                return

            # Run ALPR
            start_time = time.time()
            results = self.alpr.predict(image)
            inference_time = (time.time() - start_time) * 1000

            self.stats['processed'] += 1

            if results:
                self.stats['plates_found'] += len(results)

                # Log detections
                for r in results:
                    self.logger.info("  Plate: {} (conf: {:.1f}%)".format(
                        r['text'], r['ocr_confidence'] * 100))

                # Annotate image
                annotated = annotate_image(image, results)

                # Generate output filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plate_text = results[0]['text'] if results else 'noplate'
                base, ext = os.path.splitext(filename)
                output_filename = "{}_{}_{}.jpg".format(timestamp, plate_text, base)

                # Save annotated image
                output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # Upload to FTP
                upload_to_ftp(output_path, output_filename, self.logger)

                # Clean up output file
                if Config.DELETE_AFTER_PROCESS:
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
            else:
                self.logger.info("  No plates detected")

            self.logger.debug("  Inference time: {:.1f}ms".format(inference_time))

            # Clean up input file
            if Config.DELETE_AFTER_PROCESS:
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error("Error processing {}: {}".format(filepath, e))
            self.stats['errors'] += 1

    def run(self):
        """Start the service"""
        self.running = True
        self.stats['start_time'] = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize
        self.initialize()

        # Start file watcher
        self.watcher = FileWatcher(Config.WATCH_DIR, self.process_image, self.logger)

        self.logger.info("Service ready - watching for images...")
        self.logger.info("-" * 60)

        try:
            self.watcher.start()
        except KeyboardInterrupt:
            pass

        self._shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received signal {}, shutting down...".format(signum))
        self.running = False
        if self.watcher:
            self.watcher.stop()

    def _shutdown(self):
        """Clean shutdown"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0

        self.logger.info("-" * 60)
        self.logger.info("Service Statistics:")
        self.logger.info("  Uptime: {:.1f} seconds".format(uptime))
        self.logger.info("  Images processed: {}".format(self.stats['processed']))
        self.logger.info("  Plates found: {}".format(self.stats['plates_found']))
        self.logger.info("  Errors: {}".format(self.stats['errors']))
        self.logger.info("Service stopped")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    service = ALPRService()
    service.run()


if __name__ == '__main__':
    main()
