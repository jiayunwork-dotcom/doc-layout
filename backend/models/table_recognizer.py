import cv2
import numpy as np
from typing import List, Tuple, Dict
import uuid
from utils.image_utils import (
    ImageType,
    detect_edges,
    detect_lines,
    crop_region,
)
from schemas.models import Region, TableStructure, TableCell, BBox
from processing.ocr_engine import OCREngine


class TableRecognizer:
    def __init__(self, ocr_engine: OCREngine = None):
        self.ocr_engine = ocr_engine
        self.min_line_length = 30
        self.max_line_gap = 10

    def recognize(self, img: ImageType, table_region: Region) -> TableStructure:
        bbox = table_region.bbox.to_absolute(img.shape[1], img.shape[0])
        table_img = crop_region(img, bbox, padding=10)

        try:
            structure = self._recognize_ruled_table(table_img, img.shape[1], img.shape[0], bbox)
        except Exception as e:
            structure = self._recognize_unruled_table(table_img, img.shape[1], img.shape[0], bbox)

        if self.ocr_engine and self.ocr_engine.is_available():
            for cell in structure.cells:
                cell_img = crop_region(img, cell.bbox.to_absolute(img.shape[1], img.shape[0]), padding=2)
                cell.text = self.ocr_engine.extract_text(cell_img)

        return structure

    def _recognize_ruled_table(self, table_img: ImageType, page_w: int, page_h: int,
                                offset_bbox: Tuple[int, int, int, int]) -> TableStructure:
        ox1, oy1, _, _ = offset_bbox
        h, w = table_img.shape[:2]

        edges = detect_edges(table_img, low_threshold=50, high_threshold=150)
        horizontal_lines = self._detect_lines_by_orientation(edges, orientation="horizontal")
        vertical_lines = self._detect_lines_by_orientation(edges, orientation="vertical")

        if not horizontal_lines or not vertical_lines:
            return self._recognize_unruled_table(table_img, page_w, page_h, offset_bbox)

        h_lines = self._merge_lines(horizontal_lines, orientation="horizontal")
        v_lines = self._merge_lines(vertical_lines, orientation="vertical")

        h_lines = sorted(h_lines, key=lambda l: l[1])
        v_lines = sorted(v_lines, key=lambda l: l[0])

        row_positions = [l[1] for l in h_lines]
        col_positions = [l[0] for l in v_lines]

        row_positions = sorted(set(row_positions))
        col_positions = sorted(set(col_positions))

        if len(row_positions) < 2:
            row_positions = [0, h]
        if len(col_positions) < 2:
            col_positions = [0, w]

        cells = []
        for i in range(len(row_positions) - 1):
            for j in range(len(col_positions) - 1):
                x1, y1 = col_positions[j], row_positions[i]
                x2, y2 = col_positions[j + 1], row_positions[i + 1]

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                norm_x = (ox1 + x1) / page_w
                norm_y = (oy1 + y1) / page_h
                norm_w = (x2 - x1) / page_w
                norm_h = (y2 - y1) / page_h

                cell = TableCell(
                    row_index=i,
                    col_index=j,
                    row_span=1,
                    col_span=1,
                    bbox=BBox(x=norm_x, y=norm_y, width=norm_w, height=norm_h),
                )
                cells.append(cell)

        cells = self._detect_merged_cells(cells, len(row_positions) - 1, len(col_positions) - 1)

        return TableStructure(
            rows=len(row_positions) - 1,
            cols=len(col_positions) - 1,
            cells=cells,
        )

    def _detect_lines_by_orientation(self, edges: ImageType, orientation: str = "horizontal") -> List:
        h, w = edges.shape[:2]
        lines = detect_lines(edges, threshold=50, min_line_length=self.min_line_length,
                             max_line_gap=self.max_line_gap)

        filtered = []
        for x1, y1, x2, y2 in lines:
            if orientation == "horizontal":
                if abs(y2 - y1) < 10 and abs(x2 - x1) > w * 0.1:
                    y = (y1 + y2) // 2
                    filtered.append((min(x1, x2), y, max(x1, x2), y))
            else:
                if abs(x2 - x1) < 10 and abs(y2 - y1) > h * 0.1:
                    x = (x1 + x2) // 2
                    filtered.append((x, min(y1, y2), x, max(y1, y2)))

        return filtered

    def _merge_lines(self, lines: List, orientation: str) -> List:
        if not lines:
            return []

        threshold = 5 if orientation == "horizontal" else 5
        merged = []
        lines = sorted(lines, key=lambda l: l[1] if orientation == "horizontal" else l[0])

        current_group = [lines[0]]

        for line in lines[1:]:
            pos = line[1] if orientation == "horizontal" else line[0]
            last_pos = current_group[-1][1] if orientation == "horizontal" else current_group[-1][0]

            if abs(pos - last_pos) <= threshold:
                current_group.append(line)
            else:
                avg_line = self._average_lines(current_group, orientation)
                merged.append(avg_line)
                current_group = [line]

        if current_group:
            avg_line = self._average_lines(current_group, orientation)
            merged.append(avg_line)

        return merged

    def _average_lines(self, lines: List, orientation: str) -> Tuple:
        if orientation == "horizontal":
            avg_y = int(np.mean([l[1] for l in lines]))
            min_x = min(l[0] for l in lines)
            max_x = max(l[2] for l in lines)
            return (min_x, avg_y, max_x, avg_y)
        else:
            avg_x = int(np.mean([l[0] for l in lines]))
            min_y = min(l[1] for l in lines)
            max_y = max(l[3] for l in lines)
            return (avg_x, min_y, avg_x, max_y)

    def _recognize_unruled_table(self, table_img: ImageType, page_w: int, page_h: int,
                                  offset_bbox: Tuple[int, int, int, int]) -> TableStructure:
        ox1, oy1, _, _ = offset_bbox
        h, w = table_img.shape[:2]

        if len(table_img.shape) == 3:
            gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_img

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        horizontal_projection = np.sum(binary, axis=1)
        vertical_projection = np.sum(binary, axis=0)

        row_positions = self._find_separators(horizontal_projection, h)
        col_positions = self._find_separators(vertical_projection, w)

        if len(row_positions) < 2:
            row_positions = [0, h]
        if len(col_positions) < 2:
            col_positions = [0, w]

        cells = []
        for i in range(len(row_positions) - 1):
            for j in range(len(col_positions) - 1):
                x1, y1 = col_positions[j], row_positions[i]
                x2, y2 = col_positions[j + 1], row_positions[i + 1]

                if x2 - x1 < 15 or y2 - y1 < 15:
                    continue

                norm_x = (ox1 + x1) / page_w
                norm_y = (oy1 + y1) / page_h
                norm_w = (x2 - x1) / page_w
                norm_h = (y2 - y1) / page_h

                cell = TableCell(
                    row_index=i,
                    col_index=j,
                    row_span=1,
                    col_span=1,
                    bbox=BBox(x=norm_x, y=norm_y, width=norm_w, height=norm_h),
                )
                cells.append(cell)

        return TableStructure(
            rows=len(row_positions) - 1,
            cols=len(col_positions) - 1,
            cells=cells,
        )

    def _find_separators(self, projection: np.ndarray, length: int) -> List[int]:
        threshold = np.mean(projection) * 0.1
        separators = [0]

        in_gap = False
        gap_start = 0

        for i, val in enumerate(projection):
            if val <= threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif val > threshold and in_gap:
                in_gap = False
                gap_end = i - 1
                if gap_end - gap_start > 5:
                    separators.append((gap_start + gap_end) // 2)

        separators.append(length)
        return sorted(set(separators))

    def _detect_merged_cells(self, cells: List[TableCell], rows: int, cols: int) -> List[TableCell]:
        if not cells:
            return cells

        cell_grid = {}
        for cell in cells:
            cell_grid[(cell.row_index, cell.col_index)] = cell

        merged = []
        visited = set()

        for i in range(rows):
            for j in range(cols):
                if (i, j) in visited or (i, j) not in cell_grid:
                    continue

                current = cell_grid[(i, j)]
                visited.add((i, j))

                row_span = 1
                col_span = 1

                while j + col_span < cols and (i, j + col_span) in cell_grid:
                    next_cell = cell_grid[(i, j + col_span)]
                    if self._cells_merged(current, next_cell, "horizontal"):
                        col_span += 1
                        visited.add((i, j + col_span - 1))
                    else:
                        break

                while i + row_span < rows:
                    can_merge = True
                    for k in range(col_span):
                        if (i + row_span, j + k) not in cell_grid:
                            can_merge = False
                            break
                        next_cell = cell_grid[(i + row_span, j + k)]
                        above_cell = cell_grid[(i + row_span - 1, j + k)]
                        if not self._cells_merged(above_cell, next_cell, "vertical"):
                            can_merge = False
                            break

                    if can_merge:
                        for k in range(col_span):
                            visited.add((i + row_span, j + k))
                        row_span += 1
                    else:
                        break

                if row_span > 1 or col_span > 1:
                    last_row = i + row_span - 1
                    last_col = j + col_span - 1
                    bottom_right = cell_grid.get((last_row, last_col), current)

                    x = current.bbox.x
                    y = current.bbox.y
                    w = bottom_right.bbox.x + bottom_right.bbox.width - x
                    h = bottom_right.bbox.y + bottom_right.bbox.height - y

                    current.bbox = BBox(x=x, y=y, width=w, height=h)
                    current.row_span = row_span
                    current.col_span = col_span

                merged.append(current)

        return merged

    def _cells_merged(self, cell1: TableCell, cell2: TableCell, direction: str) -> bool:
        if direction == "horizontal":
            gap = cell2.bbox.x - (cell1.bbox.x + cell1.bbox.width)
            return gap < 0.001
        else:
            gap = cell2.bbox.y - (cell1.bbox.y + cell1.bbox.height)
            return gap < 0.001

    def get_table_as_2d_array(self, structure: TableStructure) -> List[List[str]]:
        grid = [["" for _ in range(structure.cols)] for _ in range(structure.rows)]

        for cell in structure.cells:
            text = cell.text or ""
            for i in range(cell.row_span):
                for j in range(cell.col_span):
                    ri = cell.row_index + i
                    ci = cell.col_index + j
                    if 0 <= ri < structure.rows and 0 <= ci < structure.cols:
                        grid[ri][ci] = text

        return grid
