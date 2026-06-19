import numpy as np
from typing import List, Tuple, Optional
from schemas.models import Region, BBox


class XYCut:
    def __init__(self, gap_threshold: float = 0.02, min_region_size: float = 0.01):
        self.gap_threshold = gap_threshold
        self.min_region_size = min_region_size

    def _get_projection(self, regions: List[Region], axis: str) -> np.ndarray:
        projection = np.zeros(1000)
        for region in regions:
            if axis == 'y':
                start = int(region.bbox.y * 1000)
                end = int((region.bbox.y + region.bbox.height) * 1000)
            else:
                start = int(region.bbox.x * 1000)
                end = int((region.bbox.x + region.bbox.width) * 1000)
            start = max(0, start)
            end = min(999, end)
            projection[start:end] += 1
        return projection

    def _find_split_points(self, projection: np.ndarray, threshold: int = 0) -> List[int]:
        gaps = []
        in_gap = False
        gap_start = 0

        for i, val in enumerate(projection):
            if val <= threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif val > threshold and in_gap:
                in_gap = False
                gap_end = i - 1
                gap_size = gap_end - gap_start
                if gap_size >= self.gap_threshold * 1000:
                    gaps.append((gap_start + gap_end) // 2)

        return gaps

    def _split_regions(self, regions: List[Region], split_point: float, axis: str) -> Tuple[List[Region], List[Region]]:
        group1, group2 = [], []

        for region in regions:
            if axis == 'y':
                center = region.bbox.y + region.bbox.height / 2
            else:
                center = region.bbox.x + region.bbox.width / 2

            if center < split_point:
                group1.append(region)
            else:
                group2.append(region)

        return group1, group2

    def _is_single_block(self, regions: List[Region]) -> bool:
        if len(regions) <= 1:
            return True

        y_proj = self._get_projection(regions, 'y')
        x_proj = self._get_projection(regions, 'x')

        y_gaps = self._find_split_points(y_proj)
        x_gaps = self._find_split_points(x_proj)

        return len(y_gaps) == 0 and len(x_gaps) == 0

    def _get_spanning_regions(self, regions: List[Region], split_point: float, axis: str) -> List[Region]:
        spanning = []
        for region in regions:
            if axis == 'y':
                r_start = region.bbox.y
                r_end = region.bbox.y + region.bbox.height
            else:
                r_start = region.bbox.x
                r_end = region.bbox.x + region.bbox.width

            if r_start < split_point < r_end:
                spanning.append(region)
        return spanning

    def xy_cut(self, regions: List[Region], depth: int = 0, max_depth: int = 10) -> List[List[Region]]:
        if depth >= max_depth or not regions or self._is_single_block(regions):
            return [regions]

        y_proj = self._get_projection(regions, 'y')
        y_gaps = self._find_split_points(y_proj)

        if y_gaps:
            split_y = y_gaps[len(y_gaps) // 2] / 1000.0

            spanning = self._get_spanning_regions(regions, split_y, 'y')
            non_spanning = [r for r in regions if r not in spanning]

            upper, lower = self._split_regions(non_spanning, split_y, 'y')

            result = []
            if spanning:
                result.extend([[r] for r in sorted(spanning, key=lambda r: r.bbox.y)])
            if upper:
                result.extend(self.xy_cut(upper, depth + 1, max_depth))
            if lower:
                result.extend(self.xy_cut(lower, depth + 1, max_depth))
            return result

        x_proj = self._get_projection(regions, 'x')
        x_gaps = self._find_split_points(x_proj)

        if x_gaps:
            split_x = x_gaps[len(x_gaps) // 2] / 1000.0

            spanning = self._get_spanning_regions(regions, split_x, 'x')
            non_spanning = [r for r in regions if r not in spanning]

            left, right = self._split_regions(non_spanning, split_x, 'x')

            result = []
            if spanning:
                result.append(sorted(spanning, key=lambda r: r.bbox.y))
            if left:
                result.extend(self.xy_cut(left, depth + 1, max_depth))
            if right:
                result.extend(self.xy_cut(right, depth + 1, max_depth))
            return result

        return [regions]

    def get_reading_order(self, regions: List[Region]) -> List[Region]:
        if not regions:
            return []

        try:
            ordered_blocks = self.xy_cut(regions)

            ordered_regions = []
            for block in ordered_blocks:
                if len(block) == 1:
                    ordered_regions.append(block[0])
                else:
                    ordered_regions.extend(
                        sorted(block, key=lambda r: (r.bbox.y, r.bbox.x))
                    )

            for i, region in enumerate(ordered_regions):
                region.reading_order = i + 1

            return ordered_regions

        except Exception as e:
            print(f"XY-Cut failed, falling back to topological sort: {e}")
            return self._topological_sort(regions)

    def _topological_sort(self, regions: List[Region]) -> List[Region]:
        sorted_regions = sorted(
            regions,
            key=lambda r: (
                round(r.bbox.y, 2),
                r.bbox.x,
            )
        )

        for i, region in enumerate(sorted_regions):
            region.reading_order = i + 1

        return sorted_regions
