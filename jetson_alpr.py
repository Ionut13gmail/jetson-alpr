#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastALPR for Jetson Nano - Python 3.6 Compatible
Direct ONNX Runtime inference without fast-plate-ocr dependencies
"""
from __future__ import print_function
import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

# Model paths - use local bundled models (opset 15 for ONNX Runtime 1.10 compatibility)
# Models are bundled in /app/models/ in the container
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# Fallback URLs if local models not found
DETECTOR_URL = "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx"
OCR_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/global_mobile_vit_v2_ocr.onnx"

# Model cache directory (for downloaded models)
CACHE_DIR = os.path.expanduser("~/.cache/jetson-alpr")

# OCR vocabulary (global plates)
OCR_VOCABULARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
MAX_PLATE_SLOTS = 7

def download_file(url, dest_path):
    """Download file using urllib (Python 3.6 compatible)"""
    import urllib.request
    import ssl

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print("Using cached:", dest_path)
        return dest_path

    print("Downloading:", url)
    # Create SSL context that doesn't verify (for GitHub)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        # Try with wget as fallback
        print("urllib failed, trying wget...")
        os.system("wget -q -O '{}' '{}'".format(dest_path, url))

    print("Downloaded to:", dest_path)
    return dest_path


def letterbox_resize(image, target_size):
    """Resize image with letterbox padding to maintain aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    # Calculate padding offsets
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Place resized image
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return padded, scale, pad_x, pad_y


class JetsonALPR(object):
    """License Plate Recognition for Jetson Nano with GPU support"""

    def __init__(self, use_gpu=True, conf_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.detector_size = (384, 384)
        self.ocr_size = (140, 70)  # width, height

        # Setup providers
        if use_gpu:
            self.providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        else:
            self.providers = ['CPUExecutionProvider']

        # Download and load models
        self._load_models()

    def _load_models(self):
        """Load ONNX models - prefer local bundled models (opset 15)"""
        # Check for local bundled models first (opset 15 compatible)
        local_detector = os.path.join(MODELS_DIR, "detector_opset15.onnx")
        local_ocr = os.path.join(MODELS_DIR, "ocr_opset15.onnx")

        if os.path.exists(local_detector) and os.path.exists(local_ocr):
            print("Using bundled models (opset 15)")
            detector_path = local_detector
            ocr_path = local_ocr
        else:
            # Fallback to downloading (may not work with old ONNX Runtime)
            print("Warning: Bundled models not found, downloading (may have opset issues)")
            detector_path = download_file(
                DETECTOR_URL,
                os.path.join(CACHE_DIR, "yolo-v9-t-384-detector.onnx")
            )
            ocr_path = download_file(
                OCR_URL,
                os.path.join(CACHE_DIR, "global-ocr.onnx")
            )

        # Session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load detector
        print("Loading detector model...")
        self.detector = ort.InferenceSession(
            detector_path,
            sess_options=sess_opts,
            providers=self.providers
        )
        det_provider = self.detector.get_providers()[0]
        print("Detector using:", det_provider)

        # Get detector input/output names
        self.det_input_name = self.detector.get_inputs()[0].name
        self.det_output_names = [o.name for o in self.detector.get_outputs()]

        # Load OCR
        print("Loading OCR model...")
        self.ocr = ort.InferenceSession(
            ocr_path,
            sess_options=sess_opts,
            providers=self.providers
        )
        ocr_provider = self.ocr.get_providers()[0]
        print("OCR using:", ocr_provider)

        # Get OCR input/output names
        self.ocr_input_name = self.ocr.get_inputs()[0].name
        self.ocr_output_names = [o.name for o in self.ocr.get_outputs()]

        print("Models loaded successfully!")

    def _preprocess_detector(self, image):
        """Preprocess image for YOLO detector"""
        # Letterbox resize
        padded, scale, pad_x, pad_y = letterbox_resize(image, self.detector_size)

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, scale, pad_x, pad_y

    def _postprocess_detector(self, outputs, orig_shape, scale, pad_x, pad_y):
        """Postprocess YOLO detector outputs"""
        detections = []

        # Debug: print output shapes (only first time)
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("Detector outputs:")
            for i, out in enumerate(outputs):
                print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
                if out.size < 50:
                    print(f"    Values: {out.flatten()[:20]}")

        # YOLO end2end output format: typically multiple outputs
        # [num_dets], [boxes], [scores], [labels] or combined format
        if len(outputs) >= 4:
            # Format: num_dets, boxes, scores, labels
            num_dets = int(outputs[0][0]) if outputs[0].size > 0 else 0
            boxes = outputs[1][0] if len(outputs[1].shape) > 1 else outputs[1]
            scores = outputs[2][0] if len(outputs[2].shape) > 1 else outputs[2]

            for i in range(min(num_dets, len(boxes))):
                conf = float(scores[i])
                if conf >= self.conf_thresh:
                    x1, y1, x2, y2 = boxes[i][:4]

                    x1 = (float(x1) - pad_x) / scale
                    y1 = (float(y1) - pad_y) / scale
                    x2 = (float(x2) - pad_x) / scale
                    y2 = (float(y2) - pad_y) / scale

                    h, w = orig_shape[:2]
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf
                        })

        elif len(outputs) >= 1:
            output = outputs[0]

            # Handle different output shapes
            if len(output.shape) == 2:
                # Shape: [num_dets, 7] where 7 = [batch_idx, x1, y1, x2, y2, class_id, conf]
                for det in output:
                    if len(det) >= 7:
                        # Format: [batch_idx, x1, y1, x2, y2, class_id, confidence]
                        conf = float(det[6])
                        if conf >= self.conf_thresh:
                            x1, y1, x2, y2 = det[1:5]

                            x1 = (float(x1) - pad_x) / scale
                            y1 = (float(y1) - pad_y) / scale
                            x2 = (float(x2) - pad_x) / scale
                            y2 = (float(y2) - pad_y) / scale

                            h, w = orig_shape[:2]
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))

                            if x2 > x1 and y2 > y1:
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': conf
                                })
                    elif len(det) >= 6:
                        # Alternate format: [x1, y1, x2, y2, conf, class]
                        conf = float(det[4])
                        if conf >= self.conf_thresh:
                            x1, y1, x2, y2 = det[0:4]

                            x1 = (float(x1) - pad_x) / scale
                            y1 = (float(y1) - pad_y) / scale
                            x2 = (float(x2) - pad_x) / scale
                            y2 = (float(y2) - pad_y) / scale

                            h, w = orig_shape[:2]
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))

                            if x2 > x1 and y2 > y1:
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': conf
                                })
            elif len(output.shape) == 3:
                # Shape: [batch, num_dets, 7]
                for det in output[0]:
                    if len(det) >= 7:
                        conf = float(det[6])
                        if conf >= self.conf_thresh:
                            x1, y1, x2, y2 = det[1:5]

                            x1 = (float(x1) - pad_x) / scale
                            y1 = (float(y1) - pad_y) / scale
                            x2 = (float(x2) - pad_x) / scale
                            y2 = (float(y2) - pad_y) / scale

                            h, w = orig_shape[:2]
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))

                            if x2 > x1 and y2 > y1:
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': conf
                                })

        return detections

    def _preprocess_ocr(self, plate_crop):
        """Preprocess cropped plate for OCR"""
        # Convert to grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop

        # Resize to OCR input size
        resized = cv2.resize(gray, self.ocr_size, interpolation=cv2.INTER_LINEAR)

        # Check what input type the model expects
        input_info = self.ocr.get_inputs()[0]
        input_type = input_info.type

        if 'uint8' in input_type:
            # Model expects uint8 (0-255)
            input_data = resized.astype(np.uint8)
        else:
            # Model expects float (0-1)
            input_data = resized.astype(np.float32) / 255.0

        # Add batch and channel dimensions: BHWC format
        input_tensor = np.expand_dims(input_data, axis=(0, -1))

        return input_tensor

    def _decode_ocr(self, outputs):
        """Decode OCR model outputs to text"""
        chars = []
        confidences = []

        vocab_size = len(OCR_VOCABULARY)  # 37

        # Debug: print OCR output shapes (only first time)
        if not hasattr(self, '_ocr_debug_printed'):
            self._ocr_debug_printed = True
            print("OCR outputs:")
            for i, out in enumerate(outputs):
                print("  Output {}: shape={}, dtype={}".format(i, out.shape, out.dtype))

        # Check if single combined output (e.g., shape (1, 333) = 9 slots x 37 chars)
        if len(outputs) == 1:
            output = outputs[0].flatten()
            total_size = len(output)

            # Check if this is a combined format (multiple of vocab_size)
            if total_size > vocab_size and total_size % vocab_size == 0:
                num_slots = total_size // vocab_size
                if not hasattr(self, '_ocr_debug_printed2'):
                    self._ocr_debug_printed2 = True
                    print("OCR: Combined format detected - {} slots x {} vocab".format(num_slots, vocab_size))

                # Reshape to (num_slots, vocab_size)
                probs_matrix = output.reshape(num_slots, vocab_size)

                for slot_idx in range(num_slots):
                    slot_probs = probs_matrix[slot_idx]

                    # Softmax if not already probabilities
                    if np.max(slot_probs) > 1.0:
                        exp_probs = np.exp(slot_probs - np.max(slot_probs))
                        slot_probs = exp_probs / np.sum(exp_probs)

                    best_idx = int(np.argmax(slot_probs))
                    best_prob = float(slot_probs[best_idx])

                    if best_idx < vocab_size:
                        char = OCR_VOCABULARY[best_idx]
                        if char != '_':  # Skip padding character
                            chars.append(char)
                            confidences.append(best_prob)
            else:
                # Single slot output
                probs = output[:vocab_size] if len(output) >= vocab_size else output
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])

                if best_idx < vocab_size:
                    char = OCR_VOCABULARY[best_idx]
                    if char != '_':
                        chars.append(char)
                        confidences.append(best_prob)
        else:
            # Multiple outputs format (one per character slot)
            for i, output in enumerate(outputs):
                if i >= MAX_PLATE_SLOTS:
                    break

                probs = output.flatten()
                if len(probs) == 0:
                    continue

                if len(probs) >= vocab_size:
                    probs = probs[:vocab_size]

                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])

                if best_idx < vocab_size:
                    char = OCR_VOCABULARY[best_idx]
                    if char != '_':
                        chars.append(char)
                        confidences.append(best_prob)

        text = ''.join(chars)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return text, avg_conf

    def detect_plates(self, image):
        """Detect license plates in image"""
        # Preprocess
        input_tensor, scale, pad_x, pad_y = self._preprocess_detector(image)

        # Run inference
        outputs = self.detector.run(self.det_output_names, {self.det_input_name: input_tensor})

        # Postprocess
        detections = self._postprocess_detector(outputs, image.shape, scale, pad_x, pad_y)

        return detections

    def read_plate(self, plate_crop):
        """Read text from cropped plate image"""
        # Preprocess
        input_tensor = self._preprocess_ocr(plate_crop)

        # Run inference
        outputs = self.ocr.run(self.ocr_output_names, {self.ocr_input_name: input_tensor})

        # Decode
        text, confidence = self._decode_ocr(outputs)

        return text, confidence

    def predict(self, image):
        """Full ALPR pipeline: detect plates and read text"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Could not read image: " + str(image))

        results = []

        # Detect plates
        detections = self.detect_plates(image)

        # Read each plate
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size > 0:
                text, ocr_conf = self.read_plate(plate_crop)

                results.append({
                    'bbox': det['bbox'],
                    'det_confidence': det['confidence'],
                    'text': text,
                    'ocr_confidence': ocr_conf
                })

        return results


def benchmark(alpr, image_path, num_runs=10):
    """Benchmark ALPR performance"""
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image:", image_path)
        return

    # Warmup
    print("Warming up...")
    for _ in range(3):
        alpr.predict(image)

    # Benchmark
    print("Benchmarking ({} runs)...".format(num_runs))
    times = []
    for _ in range(num_runs):
        start = time.time()
        results = alpr.predict(image)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1.0 / avg_time

    print("")
    print("=" * 50)
    print("Benchmark Results")
    print("=" * 50)
    print("Average: {:.1f}ms ({:.1f} FPS)".format(avg_time * 1000, fps))
    print("Min: {:.1f}ms, Max: {:.1f}ms".format(min_time * 1000, max_time * 1000))
    print("")

    return results


def main():
    print("=" * 60)
    print("Jetson ALPR - GPU Accelerated License Plate Recognition")
    print("=" * 60)
    print("")
    print("Python version:", sys.version)
    print("ONNX Runtime version:", ort.__version__)
    print("OpenCV version:", cv2.__version__)
    print("Available providers:", ort.get_available_providers())
    print("")

    # Check for GPU
    providers = ort.get_available_providers()
    use_gpu = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers

    if use_gpu:
        print("GPU: ENABLED")
    else:
        print("GPU: NOT AVAILABLE (using CPU)")
    print("")

    # Initialize ALPR
    print("Initializing ALPR...")
    alpr = JetsonALPR(use_gpu=use_gpu, conf_thresh=0.4)
    print("")

    # Test with sample image
    test_images = [
        "/app/assets/test_image.png",
        "assets/test_image.png",
        "/app/data/test.jpg",
    ]

    test_image = None
    for path in test_images:
        if os.path.exists(path):
            test_image = path
            break

    if test_image:
        print("Testing with:", test_image)
        print("")

        # Run benchmark
        results = benchmark(alpr, test_image, num_runs=10)

        # Show results
        if results:
            print("Detected {} plate(s):".format(len(results)))
            for i, r in enumerate(results):
                print("  Plate {}: {} (det: {:.1f}%, ocr: {:.1f}%)".format(
                    i + 1,
                    r['text'],
                    r['det_confidence'] * 100,
                    r['ocr_confidence'] * 100
                ))
        else:
            print("No plates detected")
    else:
        print("No test image found. Usage:")
        print("  python3 jetson_alpr.py [image_path]")
        print("")
        print("ALPR system ready!")

    print("")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process command line image
        image_path = sys.argv[1]

        providers = ort.get_available_providers()
        use_gpu = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers

        alpr = JetsonALPR(use_gpu=use_gpu)
        results = alpr.predict(image_path)

        print("Results for:", image_path)
        for r in results:
            print("  Plate: {} (confidence: {:.1f}%)".format(
                r['text'], r['ocr_confidence'] * 100
            ))
    else:
        main()
