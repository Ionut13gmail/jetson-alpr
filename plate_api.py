#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plate API Module - Authorization and Upload for PanoramaCity ALPR System

Handles:
- Plate authorization checking against remote database
- Async image upload with detection results
- Gate control integration
"""
from __future__ import print_function
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import cv2
import requests

logger = logging.getLogger('plate_api')


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class APIConfig:
    """API configuration with environment variable overrides."""
    API_URL: str = field(default_factory=lambda: os.environ.get(
        'ALPR_API_URL', 'https://acces.panoramacity.ro/utilitypagesAPI/_x_ApI_nano.php'))
    API_KEY: str = field(default_factory=lambda: os.environ.get(
        'ALPR_API_KEY', 'A954C6A5-5207-4DF8-91BC-E9E3B73664BF'))
    CACHE_TTL: int = field(default_factory=lambda: int(os.environ.get(
        'ALPR_CACHE_TTL', '3600')))  # Cache refresh interval (seconds)
    REQUEST_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get(
        'ALPR_REQUEST_TIMEOUT', '10')))
    JPEG_QUALITY: int = field(default_factory=lambda: int(os.environ.get(
        'ALPR_JPEG_QUALITY', '70')))
    # Gate IPs: index 0 unused, 1=gate1, 2=gate2
    GATE_IPS: List[str] = field(default_factory=lambda: [
        '',  # Index 0 unused
        os.environ.get('ALPR_GATE1_IP', '192.168.1.230'),
        os.environ.get('ALPR_GATE2_IP', '192.168.1.234'),
    ])
    GATE_ENABLED: bool = field(default_factory=lambda: os.environ.get(
        'ALPR_GATE_ENABLED', 'false').lower() == 'true')


# =============================================================================
# Data Classes
# =============================================================================
@dataclass(frozen=True)
class PlateInfo:
    """Plate authorization status."""
    plate: str
    blacklisted: bool = False
    deleted: bool = False

    @property
    def is_authorized(self) -> bool:
        """Check if plate is authorized (not blacklisted or deleted)."""
        return not self.blacklisted and not self.deleted


@dataclass
class DetectionData:
    """Data to upload for a detection."""
    plate: str
    gate_id: int  # 1 or 2
    direction: str  # 'IN' or 'OUT'
    image: 'np.ndarray'
    filename: str
    gate_opened: bool = False
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


# =============================================================================
# Plate Database with Caching
# =============================================================================
class PlateDatabase:
    """
    Thread-safe plate authorization database with local caching.

    Fetches authorized plates from API and caches them locally for fast lookups.
    Cache auto-refreshes based on TTL.
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._cache: Dict[str, PlateInfo] = {}
        self._cache_lock = threading.RLock()
        self._last_update: float = 0
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def _should_refresh(self) -> bool:
        """Check if cache needs refresh based on TTL."""
        return (time.time() - self._last_update) >= self.config.CACHE_TTL

    def refresh(self, force: bool = False) -> bool:
        """
        Refresh plate cache from API.

        Args:
            force: Force refresh even if TTL not expired

        Returns:
            True if refresh successful
        """
        if not force and not self._should_refresh():
            return True

        try:
            response = self._session.post(
                self.config.API_URL,
                data={
                    'apiK': self.config.API_KEY,
                    'read': 'plates',
                    'readAuthorized': '1'
                },
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'success':
                logger.error("API error: %s", data.get('message', 'Unknown'))
                return False

            # Update cache atomically
            new_cache = {}
            for plate_data in data.get('plates', []):
                plate = plate_data.get('plate', '').upper().strip()
                if plate:
                    new_cache[plate] = PlateInfo(
                        plate=plate,
                        blacklisted=int(plate_data.get('blacklisted', 0)) == 1,
                        deleted=int(plate_data.get('sters', 0)) == 1
                    )

            with self._cache_lock:
                self._cache = new_cache
                self._last_update = time.time()

            logger.info("Refreshed plate cache: %d plates loaded", len(new_cache))
            return True

        except requests.RequestException as e:
            logger.error("Failed to refresh plate cache: %s", e)
            return False

    def check_access(self, plate: str, filename: str = "") -> Tuple[bool, str]:
        """
        Check if plate has access.

        Args:
            plate: License plate number
            filename: Original filename (used to detect exit vs entry)

        Returns:
            Tuple of (has_access, reason)
        """
        # Ensure cache is fresh
        self.refresh()

        plate = plate.upper().strip()
        is_exit = filename.upper().startswith("CPI_")

        # Exit direction: always allow (let them leave)
        if is_exit:
            return True, "exit_allowed"

        # Check authorization
        with self._cache_lock:
            plate_info = self._cache.get(plate)

        if plate_info is None:
            return False, "unknown_plate"

        if plate_info.blacklisted:
            return False, "blacklisted"

        if plate_info.deleted:
            return False, "deleted"

        return True, "authorized"

    def get_plate_count(self) -> int:
        """Get number of plates in cache."""
        with self._cache_lock:
            return len(self._cache)


# =============================================================================
# Gate Controller
# =============================================================================
class GateController:
    """Controls physical gate barriers."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._session = requests.Session()

    def open_gate(self, gate_id: int) -> bool:
        """
        Open specified gate.

        Args:
            gate_id: Gate number (1 or 2)

        Returns:
            True if gate opened successfully
        """
        if not self.config.GATE_ENABLED:
            logger.debug("Gate control disabled, skipping")
            return False

        if gate_id < 1 or gate_id >= len(self.config.GATE_IPS):
            logger.error("Invalid gate ID: %d", gate_id)
            return False

        gate_ip = self.config.GATE_IPS[gate_id]
        if not gate_ip:
            logger.error("No IP configured for gate %d", gate_id)
            return False

        try:
            url = "http://{}/deschide1".format(gate_ip)
            response = self._session.get(url, timeout=5)

            if response.status_code == 200:
                logger.info("Gate %d opened (%s)", gate_id, gate_ip)
                return True
            else:
                logger.error("Gate %d error: HTTP %d", gate_id, response.status_code)
                return False

        except requests.RequestException as e:
            logger.error("Gate %d connection error: %s", gate_id, e)
            return False


# =============================================================================
# API Uploader
# =============================================================================
class APIUploader:
    """Handles async image uploads to the API."""

    def __init__(self, config: Optional[APIConfig] = None, max_workers: int = 2):
        self.config = config or APIConfig()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def upload(self, data: DetectionData) -> Optional[str]:
        """
        Upload detection data synchronously.

        Args:
            data: Detection data including image and plate info

        Returns:
            Remote filename if successful, None otherwise
        """
        try:
            # Encode image to JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.JPEG_QUALITY]
            _, img_encoded = cv2.imencode('.jpg', data.image, encode_params)

            payload = {
                'apiK': self.config.API_KEY,
                'LPR': data.plate,
                'autoLPR': str(data.gate_id),
                'directie': data.direction,
                'gateOpened': '1' if data.gate_opened else '0'
            }

            files = {
                'file': (data.filename, img_encoded.tobytes(), 'image/jpeg')
            }

            response = self._session.post(
                self.config.API_URL,
                data=payload,
                files=files,
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()

            remote_name = result.get('picName', '')
            if remote_name:
                logger.info("Uploaded %s -> %s", data.plate, remote_name)
                return remote_name
            else:
                logger.warning("Upload response missing picName: %s", result)
                return None

        except requests.RequestException as e:
            logger.error("Upload failed for %s: %s", data.plate, e)
            return None

    def upload_async(self, data: DetectionData) -> None:
        """
        Upload detection data asynchronously (fire and forget).

        Args:
            data: Detection data including image and plate info
        """
        self._executor.submit(self._upload_with_retry, data)

    def _upload_with_retry(self, data: DetectionData, max_retries: int = 2) -> Optional[str]:
        """Upload with retry logic."""
        for attempt in range(max_retries + 1):
            result = self.upload(data)
            if result:
                return result
            if attempt < max_retries:
                time.sleep(1)
                logger.debug("Retrying upload for %s (attempt %d)", data.plate, attempt + 2)
        return None

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


# =============================================================================
# Combined API Client
# =============================================================================
class PlateAPIClient:
    """
    Combined client for all plate-related API operations.

    Provides unified interface for:
    - Plate authorization checking
    - Detection result uploading
    - Gate control
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.database = PlateDatabase(self.config)
        self.uploader = APIUploader(self.config)
        self.gate = GateController(self.config)

    def initialize(self) -> bool:
        """Initialize the client and load plate database."""
        logger.info("Initializing Plate API client...")
        logger.info("  API URL: %s", self.config.API_URL)
        logger.info("  Cache TTL: %ds", self.config.CACHE_TTL)
        logger.info("  Gate control: %s", "enabled" if self.config.GATE_ENABLED else "disabled")

        success = self.database.refresh(force=True)
        if success:
            logger.info("  Plates loaded: %d", self.database.get_plate_count())
        return success

    def process_detection(
        self,
        plate: str,
        image,
        filename: str,
        gate_id: int = 1,
        confidence: float = 0.0,
        bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Tuple[bool, str]:
        """
        Process a plate detection: check access, open gate if authorized, upload result.

        Args:
            plate: Detected license plate text
            image: OpenCV image (BGR)
            filename: Original filename
            gate_id: Gate number (1 or 2)
            confidence: OCR confidence
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (gate_opened, reason)
        """
        # Determine direction from filename
        is_exit = filename.upper().startswith("CPI_")
        direction = "OUT" if is_exit else "IN"

        # Check authorization
        has_access, reason = self.database.check_access(plate, filename)

        # Open gate if authorized
        gate_opened = False
        if has_access:
            gate_opened = self.gate.open_gate(gate_id)

        # Upload result asynchronously
        detection_data = DetectionData(
            plate=plate,
            gate_id=gate_id,
            direction=direction,
            image=image,
            filename=filename,
            gate_opened=gate_opened,
            confidence=confidence,
            bbox=bbox
        )
        self.uploader.upload_async(detection_data)

        logger.info("Plate %s: %s (gate: %s)",
                   plate, reason, "opened" if gate_opened else "closed")

        return gate_opened, reason

    def shutdown(self):
        """Clean shutdown."""
        self.uploader.shutdown()


# =============================================================================
# Utility Functions
# =============================================================================
def parse_gate_from_filename(filename: str) -> int:
    """
    Extract gate ID from filename.

    Expected formats:
    - CPI_1_xxx.jpg -> gate 1 (exit)
    - CPO_2_xxx.jpg -> gate 2 (entry)
    - 1_xxx.jpg -> gate 1
    - Default: gate 1
    """
    name = filename.upper()

    # Check for CPI/CPO prefix with gate number
    if name.startswith("CPI_") or name.startswith("CPO_"):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])

    # Check if starts with gate number
    if name[0].isdigit():
        return int(name[0])

    return 1  # Default gate


def parse_direction_from_filename(filename: str) -> str:
    """
    Determine direction from filename.

    CPI_ prefix = Camera Parcaj Interior = Exit (OUT)
    Otherwise = Entry (IN)
    """
    return "OUT" if filename.upper().startswith("CPI_") else "IN"
