from typing import List, Dict, Tuple, Optional
import numpy as np
from schemas.models import (
    Page,
    Region,
    RegionType,
    EvaluationMetrics,
    BBox,
)
from utils.image_utils import compute_iou


class EvaluationMetricsCalculator:
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def compute_metrics(self, detected_pages: List[Page],
                        ground_truth: List[Dict]) -> EvaluationMetrics:
        if not ground_truth:
            return EvaluationMetrics()

        all_detections = []
        all_ground_truth = []

        for page in detected_pages:
            for region in page.regions:
                all_detections.append({
                    "page": page.page_number,
                    "type": region.type.value,
                    "bbox": (region.bbox.x, region.bbox.y, region.bbox.width, region.bbox.height),
                    "confidence": region.confidence,
                })

        for gt_item in ground_truth:
            for gt_region in gt_item.get("regions", []):
                bbox = gt_region.get("bbox", {})
                all_ground_truth.append({
                    "page": gt_item.get("page_number", 1),
                    "type": gt_region.get("type"),
                    "bbox": (
                        bbox.get("x", 0),
                        bbox.get("y", 0),
                        bbox.get("width", 0),
                        bbox.get("height", 0),
                    ),
                })

        per_class_iou = self._compute_per_class_iou(all_detections, all_ground_truth)
        per_class_ap = self._compute_per_class_ap(all_detections, all_ground_truth)

        mean_iou = np.mean(list(per_class_iou.values())) if per_class_iou else 0.0
        mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0

        return EvaluationMetrics(
            mAP=float(mAP),
            mean_iou=float(mean_iou),
            per_class_iou={k: float(v) for k, v in per_class_iou.items()},
            per_class_ap={k: float(v) for k, v in per_class_ap.items()},
        )

    def _compute_per_class_iou(self, detections: List[Dict],
                               ground_truth: List[Dict]) -> Dict[str, float]:
        class_ious = {}

        classes = set()
        for d in detections:
            classes.add(d["type"])
        for g in ground_truth:
            classes.add(g["type"])

        for cls in classes:
            cls_detections = [d for d in detections if d["type"] == cls]
            cls_gt = [g for g in ground_truth if g["type"] == cls]

            if not cls_detections or not cls_gt:
                class_ious[cls] = 0.0
                continue

            matched = set()
            ious = []

            for det in cls_detections:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(cls_gt):
                    if (det["page"], gt_idx) in matched:
                        continue

                    if det["page"] != gt["page"]:
                        continue

                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    matched.add((det["page"], best_gt_idx))
                    ious.append(best_iou)

            class_ious[cls] = np.mean(ious) if ious else 0.0

        return class_ious

    def _compute_per_class_ap(self, detections: List[Dict],
                              ground_truth: List[Dict]) -> Dict[str, float]:
        class_aps = {}

        classes = set()
        for d in detections:
            classes.add(d["type"])
        for g in ground_truth:
            classes.add(g["type"])

        for cls in classes:
            cls_detections = [d for d in detections if d["type"] == cls]
            cls_gt = [g for g in ground_truth if g["type"] == cls]

            if not cls_gt:
                class_aps[cls] = 0.0
                continue

            cls_detections.sort(key=lambda x: x["confidence"], reverse=True)

            aps = []
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_ap_at_iou(cls_detections, cls_gt, iou_thresh)
                aps.append(ap)

            class_aps[cls] = np.mean(aps) if aps else 0.0

        return class_aps

    def _compute_ap_at_iou(self, detections: List[Dict], ground_truth: List[Dict],
                            iou_threshold: float) -> float:
        if not detections or not ground_truth:
            return 0.0

        n_gt = len(ground_truth)
        matched_gt = set()

        tp = []
        fp = []
        scores = []

        for det in detections:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if (det["page"], gt_idx) in matched_gt:
                    continue
                if det["page"] != gt["page"]:
                    continue

                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            scores.append(det["confidence"])
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gt.add((det["page"], best_gt_idx))
            else:
                tp.append(0)
                fp.append(1)

        if n_gt == 0:
            return 0.0

        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)

        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        return ap

    @staticmethod
    def compute_region_confidence_stats(pages: List[Page]) -> Dict[str, Dict]:
        stats = {}

        for page in pages:
            for region in page.regions:
                rtype = region.type.value
                if rtype not in stats:
                    stats[rtype] = {"count": 0, "confidences": []}
                stats[rtype]["count"] += 1
                stats[rtype]["confidences"].append(region.confidence)

        result = {}
        for rtype, data in stats.items():
            confidences = np.array(data["confidences"])
            result[rtype] = {
                "count": data["count"],
                "mean_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "std_confidence": float(np.std(confidences)),
            }

        return result

    @staticmethod
    def compute_page_density(page: Page) -> float:
        total_area = 0.0
        for region in page.regions:
            total_area += region.bbox.width * region.bbox.height
        return min(1.0, total_area)
