from typing import List, Dict, Tuple
from schemas.models import Region, RegionType
from processing.xy_cut import XYCut
from utils.image_utils import compute_iou


class ReadingOrderInference:
    def __init__(self):
        self.xy_cut = XYCut()

    def infer_reading_order(self, regions: List[Region]) -> List[Region]:
        if not regions:
            return []

        content_regions = [
            r for r in regions
            if r.type not in {RegionType.HEADER, RegionType.FOOTER}
        ]
        header_footer = [
            r for r in regions
            if r.type in {RegionType.HEADER, RegionType.FOOTER}
        ]

        ordered_content = self.xy_cut.get_reading_order(content_regions)

        self._associate_captions(ordered_content)

        self._associate_formulas(ordered_content)

        headers = sorted(
            [r for r in header_footer if r.type == RegionType.HEADER],
            key=lambda r: r.bbox.y
        )
        footers = sorted(
            [r for r in header_footer if r.type == RegionType.FOOTER],
            key=lambda r: r.bbox.y
        )

        all_ordered = headers + ordered_content + footers

        for i, region in enumerate(all_ordered):
            region.reading_order = i + 1

        return all_ordered

    def _associate_captions(self, regions: List[Region]) -> None:
        captions = [r for r in regions if r.type == RegionType.CAPTION]
        figures = [r for r in regions if r.type == RegionType.FIGURE]
        tables = [r for r in regions if r.type == RegionType.TABLE]

        for caption in captions:
            caption_center_y = caption.bbox.y + caption.bbox.height / 2

            best_target = None
            best_distance = float("inf")

            for fig in figures:
                fig_bottom = fig.bbox.y + fig.bbox.height
                if fig_bottom <= caption.bbox.y + 0.02:
                    distance = caption_center_y - fig_bottom
                    if distance < best_distance:
                        best_distance = distance
                        best_target = fig

            for tbl in tables:
                tbl_bottom = tbl.bbox.y + tbl.bbox.height
                if tbl_bottom <= caption.bbox.y + 0.02:
                    distance = caption_center_y - tbl_bottom
                    if distance < best_distance:
                        best_distance = distance
                        best_target = tbl

            if best_target:
                caption.parent_id = best_target.id
                if caption.id not in best_target.children:
                    best_target.children.append(caption.id)

    def _associate_formulas(self, regions: List[Region]) -> None:
        formulas = [r for r in regions if r.type in {RegionType.FORMULA, RegionType.FORMULA_INLINE}]
        texts = [r for r in regions if r.type == RegionType.TEXT]

        for formula in formulas:
            formula_center_y = formula.bbox.y + formula.bbox.height / 2
            formula_center_x = formula.bbox.x + formula.bbox.width / 2

            best_text = None
            best_score = float("inf")

            for text in texts:
                iou = compute_iou(
                    (formula.bbox.x, formula.bbox.y, formula.bbox.width, formula.bbox.height),
                    (text.bbox.x, text.bbox.y, text.bbox.width, text.bbox.height)
                )

                if iou > 0.3:
                    best_text = text
                    break

                text_center_y = text.bbox.y + text.bbox.height / 2
                text_center_x = text.bbox.x + text.bbox.width / 2

                vertical_overlap = max(
                    0,
                    min(formula.bbox.y + formula.bbox.height, text.bbox.y + text.bbox.height) -
                    max(formula.bbox.y, text.bbox.y)
                )

                if vertical_overlap > formula.bbox.height * 0.5:
                    horizontal_dist = abs(formula_center_x - text_center_x)
                    if horizontal_dist < best_score:
                        best_score = horizontal_dist
                        best_text = text

            if best_text:
                formula.parent_id = best_text.id
                if formula.id not in best_text.children:
                    best_text.children.append(formula.id)

    def detect_columns(self, regions: List[Region]) -> int:
        if not regions:
            return 1

        content_regions = [
            r for r in regions
            if r.type not in {RegionType.HEADER, RegionType.FOOTER, RegionType.SIDEBAR}
        ]

        x_centers = [r.bbox.x + r.bbox.width / 2 for r in content_regions]

        if len(x_centers) < 3:
            return 1

        try:
            from sklearn.cluster import KMeans
            import numpy as np

            X = np.array(x_centers).reshape(-1, 1)

            inertias = []
            for k in range(1, 4):
                if len(X) < k:
                    break
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            if len(inertias) < 2:
                return 1

            diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]

            if len(diffs) >= 1 and diffs[0] < inertias[0] * 0.1:
                return 1
            elif len(diffs) >= 2 and diffs[1] < diffs[0] * 0.3:
                return 2
            else:
                return min(len(inertias), 3)
        except ImportError:
            x_sorted = sorted(x_centers)
            gaps = [x_sorted[i + 1] - x_sorted[i] for i in range(len(x_sorted) - 1)]
            large_gaps = [g for g in gaps if g > 0.1]
            return min(len(large_gaps) + 1, 3)
