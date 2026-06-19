import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from utils.image_utils import (
    ImageType,
    preprocess_for_inference,
    compute_iou,
)
from schemas.models import Region, RegionType, BBox
from config import Config


class LayoutDetector:
    def __init__(self, model_path: str = None, use_gpu: bool = False,
                 num_workers: int = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.use_gpu = use_gpu
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = [rt.value for rt in RegionType]
        self._lock = threading.Lock()

        self.num_workers = num_workers or min(4, os.cpu_count() or 2)
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self._load_model()

    def _load_model(self) -> None:
        if ort is None:
            print("ONNX Runtime not available, using heuristic detector")
            self.session = None
            return

        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}, using heuristic detector")
            self.session = None
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}, using heuristic detector")
            self.session = None

    def _heuristic_detect(self, img: ImageType) -> List[Region]:
        h, w = img.shape[:2]
        regions = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=2)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=2)

        v_density = np.sum(vertical_lines > 0) / (h * w)
        h_density = np.sum(horizontal_lines > 0) / (h * w)
        has_table_structure = v_density > 0.01 and h_density > 0.01

        text_blocks = self._detect_text_blocks(binary)
        image_regions = self._detect_image_regions(binary, gray)

        header_region = self._detect_header_footer(binary, is_header=True)
        footer_region = self._detect_header_footer(binary, is_header=False)

        if header_region:
            x, y, bw, bh, conf = header_region
            regions.append(Region(
                id=str(uuid.uuid4()),
                type=RegionType.HEADER,
                bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                confidence=min(0.95, conf + 0.5),
            ))

        if footer_region:
            x, y, bw, bh, conf = footer_region
            regions.append(Region(
                id=str(uuid.uuid4()),
                type=RegionType.FOOTER,
                bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                confidence=min(0.95, conf + 0.5),
            ))

        title_blocks = []
        body_blocks = []

        for block in text_blocks:
            x, y, bw, bh, conf = block

            if y < h * 0.15 and bw > w * 0.5:
                if bh > h * 0.04:
                    title_blocks.append((x, y, bw, bh, conf, 1))
                else:
                    title_blocks.append((x, y, bw, bh, conf, 2))
            elif y > h * 0.85:
                continue
            else:
                body_blocks.append(block)

        title_blocks.sort(key=lambda b: b[1])
        for i, (x, y, bw, bh, conf, level) in enumerate(title_blocks[:3]):
            if i == 0:
                rtype = RegionType.TITLE_H1
            elif i == 1:
                rtype = RegionType.TITLE_H2
            else:
                rtype = RegionType.TITLE_H3

            regions.append(Region(
                id=str(uuid.uuid4()),
                type=rtype,
                bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                confidence=min(0.98, conf + 0.3),
            ))

        body_blocks.sort(key=lambda b: (b[1], b[0]))
        for x, y, bw, bh, conf in body_blocks:
            regions.append(Region(
                id=str(uuid.uuid4()),
                type=RegionType.TEXT,
                bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                confidence=conf,
            ))

        for x, y, bw, bh, conf in image_regions:
            regions.append(Region(
                id=str(uuid.uuid4()),
                type=RegionType.FIGURE,
                bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                confidence=conf,
            ))

            cap_y = y + bh + 5
            cap_h = int(h * 0.04)
            if cap_y + cap_h < h * 0.95:
                regions.append(Region(
                    id=str(uuid.uuid4()),
                    type=RegionType.CAPTION,
                    bbox=BBox(x=x / w, y=cap_y / h, width=bw / w, height=cap_h / h),
                    confidence=min(0.85, conf - 0.1),
                ))

        if has_table_structure:
            table_region = self._find_table_region(binary, vertical_lines, horizontal_lines)
            if table_region:
                x, y, bw, bh = table_region
                overlap = False
                for r in regions:
                    if compute_iou(
                        (x / w, y / h, bw / w, bh / h),
                        (r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height)
                    ) > 0.3:
                        overlap = True
                        break

                if not overlap:
                    regions.append(Region(
                        id=str(uuid.uuid4()),
                        type=RegionType.TABLE,
                        bbox=BBox(x=x / w, y=y / h, width=bw / w, height=bh / h),
                        confidence=0.82,
                    ))

        regions = [r for r in regions if r.bbox.width > 0.01 and r.bbox.height > 0.01]

        return self._nms(regions, iou_threshold=0.3)

    def _detect_text_blocks(self, binary: np.ndarray) -> List[tuple]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        blocks = []
        h, w = binary.shape[:2]

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if bw < 30 or bh < 20:
                continue
            if bw > w * 0.95 and bh > h * 0.5:
                continue

            fill_ratio = area / (bw * bh) if bw * bh > 0 else 0
            if fill_ratio < 0.05:
                continue

            aspect = bw / bh if bh > 0 else 0
            if aspect > 20:
                continue

            density = fill_ratio
            confidence = min(0.95, 0.3 + density * 3 + bh / 100)
            confidence = max(0.5, min(0.98, confidence))

            blocks.append((x, y, bw, bh, confidence))

        blocks.sort(key=lambda b: b[1])
        return blocks

    def _detect_image_regions(self, binary: np.ndarray, gray: np.ndarray) -> List[tuple]:
        h, w = binary.shape[:2]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        regions = []

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if bw < w * 0.1 or bh < h * 0.1:
                continue
            if bw > w * 0.95 and bh > h * 0.9:
                continue
            if area < 5000:
                continue

            fill_ratio = area / (bw * bh) if bw * bh > 0 else 0
            if fill_ratio < 0.1:
                continue

            roi = gray[y:y + bh, x:x + bw]
            if roi.size > 0:
                std = np.std(roi)
                texture_score = min(1.0, std / 50)
            else:
                texture_score = 0.5

            aspect = min(bw / bh, bh / bw) if bh > 0 and bw > 0 else 0
            size_score = min(1.0, (bw * bh) / (w * h * 0.1))

            confidence = 0.4 + texture_score * 0.3 + size_score * 0.3
            confidence = max(0.5, min(0.95, confidence))

            regions.append((x, y, bw, bh, confidence))

        regions.sort(key=lambda r: -r[4])
        return regions[:3]

    def _detect_header_footer(self, binary: np.ndarray, is_header: bool = True) -> Optional[tuple]:
        h, w = binary.shape[:2]
        margin = int(h * 0.1)

        if is_header:
            roi = binary[:margin, :]
            y_offset = 0
        else:
            roi = binary[h - margin:, :]
            y_offset = h - margin

        horizontal_proj = np.sum(roi, axis=1) / 255

        has_content = horizontal_proj > w * 0.05

        if not np.any(has_content):
            return None

        content_rows = np.where(has_content)[0]
        if len(content_rows) < 3:
            return None

        top = content_rows[0]
        bottom = content_rows[-1]

        if bottom - top < 10:
            return None

        left = int(w * 0.05)
        right = int(w * 0.95)

        bw = right - left
        bh = bottom - top

        density = np.sum(roi[top:bottom, left:right] > 0) / (bw * bh) if bw * bh > 0 else 0
        confidence = 0.5 + density * 2
        confidence = max(0.5, min(0.9, confidence))

        return (left, y_offset + top, bw, bh, confidence)

    def _find_table_region(self, binary: np.ndarray,
                            v_lines: np.ndarray, h_lines: np.ndarray) -> Optional[tuple]:
        h, w = binary.shape[:2]

        h_proj_v = np.sum(v_lines > 0, axis=0)
        h_proj_h = np.sum(h_lines > 0, axis=1)

        v_threshold = h * 0.1
        h_threshold = w * 0.1

        v_peaks = np.where(h_proj_v > v_threshold)[0]
        h_peaks = np.where(h_proj_h > h_threshold)[0]

        if len(v_peaks) < 2 or len(h_peaks) < 2:
            return None

        x1, x2 = v_peaks[0], v_peaks[-1]
        y1, y2 = h_peaks[0], h_peaks[-1]

        bw = x2 - x1
        bh = y2 - y1

        if bw < w * 0.15 or bh < h * 0.1:
            return None
        if bw > w * 0.95 or bh > h * 0.8:
            return None

        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2 - x1, y2 - y1)

    def _nms(self, regions: List[Region], iou_threshold: float = 0.5) -> List[Region]:
        if not regions:
            return []

        regions_sorted = sorted(regions, key=lambda r: r.confidence, reverse=True)
        keep = []

        while regions_sorted:
            current = regions_sorted.pop(0)
            keep.append(current)

            regions_sorted = [
                r for r in regions_sorted
                if compute_iou(
                    (current.bbox.x, current.bbox.y, current.bbox.width, current.bbox.height),
                    (r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height)
                ) < iou_threshold
            ]

        return keep

    def detect(self, img: ImageType) -> List[Region]:
        if self.session is None:
            return self._heuristic_detect(img)

        try:
            h, w = img.shape[:2]
            input_tensor = preprocess_for_inference(img)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            with self._lock:
                outputs = self.session.run(
                    self.output_names,
                    {self.input_name: input_tensor}
                )

            boxes = outputs[0] if len(outputs) > 0 else np.array([])
            scores = outputs[1] if len(outputs) > 1 else np.array([])
            labels = outputs[2] if len(outputs) > 2 else np.array([])

            regions = []
            for i in range(len(boxes)):
                if scores[i] < 0.5:
                    continue

                x1, y1, x2, y2 = boxes[i]
                x1, x2 = max(0, min(1, x1 / w)), max(0, min(1, x2 / w))
                y1, y2 = max(0, min(1, y1 / h)), max(0, min(1, y2 / h))
                width, height = x2 - x1, y2 - y1

                if width <= 0.01 or height <= 0.01:
                    continue

                class_idx = int(labels[i])
                if class_idx >= len(self.class_names):
                    continue

                region_type = RegionType(self.class_names[class_idx])
                region = Region(
                    id=str(uuid.uuid4()),
                    type=region_type,
                    bbox=BBox(x=x1, y=y1, width=width, height=height),
                    confidence=float(scores[i]),
                )
                regions.append(region)

            return self._nms(regions)

        except Exception as e:
            print(f"Detection error: {e}")
            return self._heuristic_detect(img)

    def detect_batch(self, images: List[ImageType]) -> List[List[Region]]:
        if not images:
            return []

        if self.session is not None:
            batch_size = Config.BATCH_INFERENCE_SIZE
            all_results = []

            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                try:
                    batch_tensors = np.stack([preprocess_for_inference(img) for img in batch])

                    with self._lock:
                        outputs = self.session.run(
                            self.output_names,
                            {self.input_name: batch_tensors}
                        )

                    batch_results = []
                    for idx, img in enumerate(batch):
                        h, w = img.shape[:2]
                        boxes = outputs[0][idx] if len(outputs) > 0 else np.array([])
                        scores = outputs[1][idx] if len(outputs) > 1 else np.array([])
                        labels = outputs[2][idx] if len(outputs) > 2 else np.array([])

                        regions = []
                        for j in range(len(boxes)):
                            if scores[j] < 0.5:
                                continue

                            x1, y1, x2, y2 = boxes[j]
                            x1, x2 = max(0, min(1, x1 / w)), max(0, min(1, x2 / w))
                            y1, y2 = max(0, min(1, y1 / h)), max(0, min(1, y2 / h))
                            width, height = x2 - x1, y2 - y1

                            if width <= 0.01 or height <= 0.01:
                                continue

                            class_idx = int(labels[j])
                            if class_idx >= len(self.class_names):
                                continue

                            region_type = RegionType(self.class_names[class_idx])
                            region = Region(
                                id=str(uuid.uuid4()),
                                type=region_type,
                                bbox=BBox(x=x1, y=y1, width=width, height=height),
                                confidence=float(scores[j]),
                            )
                            regions.append(region)

                        batch_results.append(self._nms(regions))

                    all_results.extend(batch_results)

                except Exception as e:
                    print(f"Batch inference error: {e}, falling back to sequential")
                    for img in batch:
                        all_results.append(self.detect(img))

            return all_results

        else:
            results = [None] * len(images)
            futures = {}

            for idx, img in enumerate(images):
                future = self._executor.submit(self._heuristic_detect, img)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Heuristic detect error for image {idx}: {e}")
                    results[idx] = []

            return results

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
