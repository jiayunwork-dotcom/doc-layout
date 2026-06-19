import os
import numpy as np
from typing import List, Tuple, Optional
import uuid

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from utils.image_utils import (
    ImageType,
    preprocess_for_inference,
    compute_iou,
)
from schemas.models import Region, RegionType, BBox, Page
from config import Config


class LayoutDetector:
    def __init__(self, model_path: str = None, use_gpu: bool = False):
        self.model_path = model_path or Config.MODEL_PATH
        self.use_gpu = use_gpu
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = [rt.value for rt in RegionType]
        self._load_model()

    def _load_model(self) -> None:
        if ort is None:
            print("ONNX Runtime not available, using mock detector")
            self.session = None
            return

        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}, using mock detector")
            self.session = None
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}, using mock detector")
            self.session = None

    def _mock_detect(self, img: ImageType) -> List[Region]:
        h, w = img.shape[:2]
        regions = []

        mock_regions = [
            (RegionType.HEADER, 0.05, 0.02, 0.9, 0.08, 0.92),
            (RegionType.TITLE_H1, 0.05, 0.12, 0.9, 0.06, 0.95),
            (RegionType.TEXT, 0.05, 0.2, 0.55, 0.15, 0.88),
            (RegionType.TEXT, 0.05, 0.36, 0.55, 0.15, 0.85),
            (RegionType.FIGURE, 0.65, 0.2, 0.3, 0.3, 0.90),
            (RegionType.CAPTION, 0.65, 0.51, 0.3, 0.04, 0.80),
            (RegionType.TITLE_H2, 0.05, 0.52, 0.9, 0.04, 0.93),
            (RegionType.TABLE, 0.05, 0.58, 0.9, 0.25, 0.87),
            (RegionType.TEXT, 0.05, 0.85, 0.9, 0.08, 0.82),
            (RegionType.FOOTER, 0.05, 0.94, 0.9, 0.05, 0.90),
        ]

        for idx, (region_type, x, y, width, height, conf) in enumerate(mock_regions):
            if y + height <= 1.0 and x + width <= 1.0:
                region = Region(
                    id=str(uuid.uuid4()),
                    type=region_type,
                    bbox=BBox(x=x, y=y, width=width, height=height),
                    confidence=conf,
                )
                regions.append(region)

        return regions

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
            return self._mock_detect(img)

        try:
            h, w = img.shape[:2]
            input_tensor = preprocess_for_inference(img)
            input_tensor = np.expand_dims(input_tensor, axis=0)

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
            return self._mock_detect(img)

    def detect_batch(self, images: List[ImageType]) -> List[List[Region]]:
        if not images:
            return []

        batch_size = Config.BATCH_INFERENCE_SIZE
        all_results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            if self.session is None:
                batch_results = [self._mock_detect(img) for img in batch]
            else:
                batch_results = []
                for img in batch:
                    batch_results.append(self.detect(img))

            all_results.extend(batch_results)

        return all_results
