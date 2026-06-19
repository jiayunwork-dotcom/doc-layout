import os
from typing import List, Tuple, Optional, Any
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

GRID_SIZE = 32


def _get_bbox_dict(bbox: Any) -> dict:
    if hasattr(bbox, "model_dump"):
        return bbox.model_dump()
    elif hasattr(bbox, "dict"):
        return bbox.dict()
    elif isinstance(bbox, dict):
        return bbox
    else:
        return {
            "x": float(bbox.x),
            "y": float(bbox.y),
            "width": float(bbox.width),
            "height": float(bbox.height),
        }


def _get_diff_type_str(diff_type: Any) -> str:
    if hasattr(diff_type, "value"):
        return diff_type.value
    return str(diff_type)


def _normalize_bbox(bbox: Any, page_width: int, page_height: int) -> Tuple[float, float, float, float]:
    bbox_dict = _get_bbox_dict(bbox)
    x = bbox_dict["x"] * page_width
    y = bbox_dict["y"] * page_height
    w = bbox_dict["width"] * page_width
    h = bbox_dict["height"] * page_height
    return x, y, x + w, y + h


def _compute_intersection_area(
    bbox: Tuple[float, float, float, float],
    cell_x0: float, cell_y0: float, cell_x1: float, cell_y1: float
) -> float:
    x0, y0, x1, y1 = bbox
    ix0 = max(x0, cell_x0)
    iy0 = max(y0, cell_y0)
    ix1 = min(x1, cell_x1)
    iy1 = min(y1, cell_y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _heatmap_color(value: float, max_value: float) -> Tuple[int, int, int, int]:
    if max_value <= 0 or value <= 0:
        return (0, 0, 0, 0)

    ratio = min(1.0, value / max_value)

    if ratio < 0.25:
        t = ratio / 0.25
        r = 255
        g = 255
        b = 0
        a = int(255 * 0.3 * t)
    elif ratio < 0.5:
        t = (ratio - 0.25) / 0.25
        r = 255
        g = int(255 - 90 * t)
        b = 0
        a = int(255 * 0.3 + 255 * 0.2 * t)
    elif ratio < 0.75:
        t = (ratio - 0.5) / 0.25
        r = 255
        g = int(165 - 100 * t)
        b = 0
        a = int(255 * 0.5 + 255 * 0.2 * t)
    else:
        t = (ratio - 0.75) / 0.25
        r = 255
        g = int(65 * (1 - t))
        b = 0
        a = int(255 * 0.7 + 255 * 0.3 * t)

    return (r, g, b, a)


def generate_heatmap_image(
    page_diff: Any,
    base_image: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    page_width = getattr(page_diff, "source_width", 0)
    page_height = getattr(page_diff, "source_height", 0)

    if page_width <= 0 or page_height <= 0:
        page_width, page_height = base_image.size

    cell_w = page_width / GRID_SIZE
    cell_h = page_height / GRID_SIZE

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    diffs = getattr(page_diff, "diffs", [])
    for diff in diffs:
        diff_type = _get_diff_type_str(getattr(diff, "type", "unchanged"))
        if diff_type == "unchanged":
            continue

        bboxes_to_process = []

        source_region = getattr(diff, "source_region", None)
        target_region = getattr(diff, "target_region", None)

        if diff_type == "moved":
            if source_region:
                src_bbox = _normalize_bbox(
                    getattr(source_region, "bbox", None),
                    page_width, page_height
                )
                bboxes_to_process.append(src_bbox)
            if target_region:
                tgt_bbox = _normalize_bbox(
                    getattr(target_region, "bbox", None),
                    page_width, page_height
                )
                bboxes_to_process.append(tgt_bbox)
        else:
            region = source_region or target_region
            if region:
                bbox = _normalize_bbox(
                    getattr(region, "bbox", None),
                    page_width, page_height
                )
                bboxes_to_process.append(bbox)

        for bbox in bboxes_to_process:
            x0, y0, x1, y1 = bbox
            bbox_area = (x1 - x0) * (y1 - y0)
            if bbox_area <= 0:
                continue

            grid_x0 = max(0, int(np.floor(x0 / cell_w)))
            grid_y0 = max(0, int(np.floor(y0 / cell_h)))
            grid_x1 = min(GRID_SIZE - 1, int(np.floor((x1 - 1e-9) / cell_w)))
            grid_y1 = min(GRID_SIZE - 1, int(np.floor((y1 - 1e-9) / cell_h)))

            for gy in range(grid_y0, grid_y1 + 1):
                for gx in range(grid_x0, grid_x1 + 1):
                    cell_x0 = gx * cell_w
                    cell_y0 = gy * cell_h
                    cell_x1 = (gx + 1) * cell_w
                    cell_y1 = (gy + 1) * cell_h

                    inter_area = _compute_intersection_area(bbox, cell_x0, cell_y0, cell_x1, cell_y1)
                    cell_area = cell_w * cell_h
                    if cell_area > 0:
                        weight = inter_area / cell_area
                        grid[gy, gx] += weight

    max_value = grid.max()

    resized_base = base_image.convert("RGBA").resize((page_width, page_height))

    heatmap_overlay = Image.new("RGBA", (page_width, page_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap_overlay)

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            value = grid[gy, gx]
            if value <= 0:
                continue

            color = _heatmap_color(value, max_value)
            r, g, b, a = color
            a = int(a * alpha)

            cell_x0 = int(gx * cell_w)
            cell_y0 = int(gy * cell_h)
            cell_x1 = int((gx + 1) * cell_w)
            cell_y1 = int((gy + 1) * cell_h)

            draw.rectangle(
                [cell_x0, cell_y0, cell_x1, cell_y1],
                fill=(r, g, b, a)
            )

    result = Image.alpha_composite(resized_base, heatmap_overlay)
    return result.convert("RGB")


def heatmap_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
