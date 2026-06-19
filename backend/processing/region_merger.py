import uuid
from typing import List
from schemas.models import Region, RegionType, BBox


def merge_adjacent_regions(regions: List[Region]) -> List[Region]:
    if not regions:
        return regions

    type_groups: dict[RegionType, List[Region]] = {}
    for r in regions:
        type_groups.setdefault(r.type, []).append(r)

    merged_regions = []
    for rtype, group in type_groups.items():
        merged_group = _merge_group(group)
        merged_regions.extend(merged_group)

    return merged_regions


def _merge_group(regions: List[Region]) -> List[Region]:
    if len(regions) <= 1:
        return list(regions)

    sorted_regions = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x))

    changed = True
    while changed:
        changed = False
        new_regions = []
        used = [False] * len(sorted_regions)

        for i in range(len(sorted_regions)):
            if used[i]:
                continue

            current = sorted_regions[i]

            for j in range(i + 1, len(sorted_regions)):
                if used[j]:
                    continue

                candidate = sorted_regions[j]

                if _should_merge(current, candidate):
                    current = _merge_two(current, candidate)
                    used[j] = True
                    changed = True

            new_regions.append(current)

        sorted_regions = new_regions

    return sorted_regions


def _should_merge(r1: Region, r2: Region) -> bool:
    if r1.type != r2.type:
        return False

    r1_bottom = r1.bbox.y + r1.bbox.height
    r2_bottom = r2.bbox.y + r2.bbox.height

    if r1.bbox.y <= r2.bbox.y:
        vertical_gap = r2.bbox.y - r1_bottom
    else:
        vertical_gap = r1.bbox.y - r2_bottom

    min_height = min(r1.bbox.height, r2.bbox.height)
    if min_height <= 0:
        return False

    if vertical_gap >= 0.3 * min_height:
        return False

    r1_left = r1.bbox.x
    r1_right = r1.bbox.x + r1.bbox.width
    r2_left = r2.bbox.x
    r2_right = r2.bbox.x + r2.bbox.width

    overlap_left = max(r1_left, r2_left)
    overlap_right = min(r1_right, r2_right)
    overlap_width = max(0, overlap_right - overlap_left)

    min_width = min(r1.bbox.width, r2.bbox.width)
    if min_width <= 0:
        return False

    if overlap_width / min_width <= 0.7:
        return False

    return True


def _merge_two(r1: Region, r2: Region) -> Region:
    x = min(r1.bbox.x, r2.bbox.x)
    y = min(r1.bbox.y, r2.bbox.y)
    x1 = max(r1.bbox.x + r1.bbox.width, r2.bbox.x + r2.bbox.width)
    y1 = max(r1.bbox.y + r1.bbox.height, r2.bbox.y + r2.bbox.height)

    confidence = max(r1.confidence, r2.confidence)

    texts = []
    if r1.text:
        texts.append(r1.text)
    if r2.text:
        texts.append(r2.text)
    merged_text = "\n".join(texts) if texts else None

    return Region(
        id=str(uuid.uuid4()),
        type=r1.type,
        bbox=BBox(x=x, y=y, width=x1 - x, height=y1 - y),
        confidence=confidence,
        text=merged_text,
        reading_order=None,
    )
