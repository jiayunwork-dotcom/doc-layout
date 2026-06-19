import uuid
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment

from schemas.models import (
    Region,
    BBox,
    DiffRecord,
    DiffType,
    PageDiff,
    ComparisonStats,
    DisplacementVector,
)
from utils.image_utils import compute_iou


IOU_THRESHOLD_SAME = 0.6
IOU_THRESHOLD_MOVED = 0.2


def generate_diff_id() -> str:
    return str(uuid.uuid4())


def calculate_iou_matrix(
    source_regions: List[Region],
    target_regions: List[Region],
) -> np.ndarray:
    n_source = len(source_regions)
    n_target = len(target_regions)
    iou_matrix = np.zeros((n_source, n_target), dtype=np.float64)

    for i, src in enumerate(source_regions):
        for j, tgt in enumerate(target_regions):
            if src.type != tgt.type:
                continue
            src_box = (src.bbox.x, src.bbox.y, src.bbox.width, src.bbox.height)
            tgt_box = (tgt.bbox.x, tgt.bbox.y, tgt.bbox.width, tgt.bbox.height)
            iou_matrix[i, j] = compute_iou(src_box, tgt_box)

    return iou_matrix


def optimal_matching(
    source_regions: List[Region],
    target_regions: List[Region],
) -> List[Tuple[int, int, float]]:
    if not source_regions or not target_regions:
        return []

    iou_matrix = calculate_iou_matrix(source_regions, target_regions)

    cost_matrix = -iou_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if i < len(source_regions) and j < len(target_regions):
            iou = iou_matrix[i, j]
            if iou > 0:
                matches.append((i, j, iou))

    return matches


def check_content_modified(
    source_region: Region,
    target_region: Region,
) -> Tuple[bool, Optional[str]]:
    src_text = source_region.text
    tgt_text = target_region.text

    if src_text is None and tgt_text is None:
        return False, None

    if src_text is None or tgt_text is None:
        summary = "文本内容从无到有" if tgt_text else "文本内容丢失"
        return True, summary

    if src_text == tgt_text:
        return False, None

    src_lines = src_text.strip().split('\n')
    tgt_lines = tgt_text.strip().split('\n')

    common = set(src_lines) & set(tgt_lines)
    src_only = set(src_lines) - common
    tgt_only = set(tgt_lines) - common

    summary_parts = []
    if src_only:
        preview = '; '.join(list(src_only)[:2])
        summary_parts.append(f"移除 {len(src_only)} 行: {preview}...")
    if tgt_only:
        preview = '; '.join(list(tgt_only)[:2])
        summary_parts.append(f"新增 {len(tgt_only)} 行: {preview}...")

    return True, "; ".join(summary_parts)


def calculate_displacement(
    source_region: Region,
    target_region: Region,
) -> DisplacementVector:
    src_center_x = source_region.bbox.x + source_region.bbox.width / 2
    src_center_y = source_region.bbox.y + source_region.bbox.height / 2
    tgt_center_x = target_region.bbox.x + target_region.bbox.width / 2
    tgt_center_y = target_region.bbox.y + target_region.bbox.height / 2

    return DisplacementVector(
        dx=tgt_center_x - src_center_x,
        dy=tgt_center_y - src_center_y,
    )


def match_regions_by_type(
    source_regions: List[Region],
    target_regions: List[Region],
    page_number: int,
) -> List[DiffRecord]:
    from collections import defaultdict

    source_by_type: Dict[str, List[Tuple[int, Region]]] = defaultdict(list)
    target_by_type: Dict[str, List[Tuple[int, Region]]] = defaultdict(list)

    for idx, r in enumerate(source_regions):
        source_by_type[r.type.value].append((idx, r))
    for idx, r in enumerate(target_regions):
        target_by_type[r.type.value].append((idx, r))

    all_types = set(source_by_type.keys()) | set(target_by_type.keys())

    diffs: List[DiffRecord] = []

    matched_source_indices: set = set()
    matched_target_indices: set = set()

    for region_type in all_types:
        src_list = source_by_type.get(region_type, [])
        tgt_list = target_by_type.get(region_type, [])

        src_regions_only = [r for _, r in src_list]
        tgt_regions_only = [r for _, r in tgt_list]

        matches = optimal_matching(src_regions_only, tgt_regions_only)

        for local_i, local_j, iou in matches:
            src_orig_idx, src_region = src_list[local_i]
            tgt_orig_idx, tgt_region = tgt_list[local_j]

            matched_source_indices.add(src_orig_idx)
            matched_target_indices.add(tgt_orig_idx)

            if iou >= IOU_THRESHOLD_SAME:
                modified, summary = check_content_modified(src_region, tgt_region)
                if modified:
                    diffs.append(DiffRecord(
                        id=generate_diff_id(),
                        type=DiffType.MODIFIED,
                        page_number=page_number,
                        source_region_id=src_region.id,
                        target_region_id=tgt_region.id,
                        source_region=src_region,
                        target_region=tgt_region,
                        content_summary=summary,
                        iou=iou,
                    ))
                else:
                    diffs.append(DiffRecord(
                        id=generate_diff_id(),
                        type=DiffType.UNCHANGED,
                        page_number=page_number,
                        source_region_id=src_region.id,
                        target_region_id=tgt_region.id,
                        source_region=src_region,
                        target_region=tgt_region,
                        iou=iou,
                    ))
            elif iou >= IOU_THRESHOLD_MOVED:
                displacement = calculate_displacement(src_region, tgt_region)
                diffs.append(DiffRecord(
                    id=generate_diff_id(),
                    type=DiffType.MOVED,
                    page_number=page_number,
                    source_region_id=src_region.id,
                    target_region_id=tgt_region.id,
                    source_region=src_region,
                    target_region=tgt_region,
                    displacement=displacement,
                    iou=iou,
                ))

    for idx, region in enumerate(source_regions):
        if idx not in matched_source_indices:
            diffs.append(DiffRecord(
                id=generate_diff_id(),
                type=DiffType.REMOVED,
                page_number=page_number,
                source_region_id=region.id,
                source_region=region,
            ))

    for idx, region in enumerate(target_regions):
        if idx not in matched_target_indices:
            diffs.append(DiffRecord(
                id=generate_diff_id(),
                type=DiffType.ADDED,
                page_number=page_number,
                target_region_id=region.id,
                target_region=region,
            ))

    return diffs


def compare_pages(
    source_page: dict,
    target_page: dict,
    page_number: int,
) -> PageDiff:
    from schemas.models import Region, BBox

    def dict_to_region(region_dict: dict) -> Region:
        return Region(
            id=region_dict["id"],
            type=region_dict["type"],
            bbox=BBox(
                x=region_dict["bbox"]["x"],
                y=region_dict["bbox"]["y"],
                width=region_dict["bbox"]["width"],
                height=region_dict["bbox"]["height"],
            ),
            confidence=region_dict["confidence"],
            text=region_dict.get("text"),
            reading_order=region_dict.get("reading_order"),
        )

    source_regions = [dict_to_region(r) for r in source_page.get("regions", [])]
    target_regions = [dict_to_region(r) for r in target_page.get("regions", [])]

    diffs = match_regions_by_type(source_regions, target_regions, page_number)

    return PageDiff(
        page_number=page_number,
        source_width=source_page.get("width", 0),
        source_height=source_page.get("height", 0),
        target_width=target_page.get("width", 0),
        target_height=target_page.get("height", 0),
        diffs=diffs,
    )


def calculate_stats(page_diffs: List[PageDiff]) -> ComparisonStats:
    stats = ComparisonStats()
    for page_diff in page_diffs:
        for diff in page_diff.diffs:
            stats.total += 1
            if diff.type == DiffType.ADDED:
                stats.added += 1
            elif diff.type == DiffType.REMOVED:
                stats.removed += 1
            elif diff.type == DiffType.MOVED:
                stats.moved += 1
            elif diff.type == DiffType.MODIFIED:
                stats.modified += 1
            elif diff.type == DiffType.UNCHANGED:
                stats.unchanged += 1
    return stats
