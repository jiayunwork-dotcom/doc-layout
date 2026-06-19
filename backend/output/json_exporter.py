import json
from typing import Dict, Any
from collections import Counter
from schemas.models import AnalysisResult


class JSONExporter:
    @staticmethod
    def export(result: AnalysisResult) -> str:
        return json.dumps(JSONExporter.to_dict(result), indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_page_statistics(page) -> Dict[str, Any]:
        regions = page.regions
        if not regions:
            return {
                "total_regions": 0,
                "type_counts": {},
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
            }

        type_counts = Counter(r.type.value for r in regions)
        confidences = [r.confidence for r in regions]

        return {
            "total_regions": len(regions),
            "type_counts": dict(type_counts),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "min_confidence": round(min(confidences), 4),
        }

    @staticmethod
    def to_dict(result: AnalysisResult) -> Dict[str, Any]:
        return {
            "task_id": result.task_id,
            "status": result.status,
            "metadata": result.metadata.model_dump() if result.metadata else None,
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "dpi": page.dpi,
                    "image_path": page.image_path,
                    "preprocessing_applied": page.preprocessing_applied,
                    "statistics": JSONExporter._compute_page_statistics(page),
                    "regions": [
                        {
                            "id": region.id,
                            "type": region.type.value,
                            "bbox": {
                                "x": region.bbox.x,
                                "y": region.bbox.y,
                                "width": region.bbox.width,
                                "height": region.bbox.height,
                            },
                            "confidence": region.confidence,
                            "text": region.text,
                            "reading_order": region.reading_order,
                            "parent_id": region.parent_id,
                            "children": region.children,
                            "table_structure": (
                                {
                                    "rows": region.table_structure.rows,
                                    "cols": region.table_structure.cols,
                                    "cells": [
                                        {
                                            "row_index": cell.row_index,
                                            "col_index": cell.col_index,
                                            "row_span": cell.row_span,
                                            "col_span": cell.col_span,
                                            "text": cell.text,
                                            "bbox": {
                                                "x": cell.bbox.x,
                                                "y": cell.bbox.y,
                                                "width": cell.bbox.width,
                                                "height": cell.bbox.height,
                                            },
                                        }
                                        for cell in region.table_structure.cells
                                    ],
                                }
                                if region.table_structure else None
                            ),
                        }
                        for region in page.regions
                    ],
                }
                for page in result.pages
            ],
            "hierarchy": (
                JSONExporter._hierarchy_to_dict(result.hierarchy)
                if result.hierarchy else None
            ),
            "evaluation": (
                result.evaluation.model_dump() if result.evaluation else None
            ),
            "error": result.error,
        }

    @staticmethod
    def _hierarchy_to_dict(node) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "title": node.title,
            "level": node.level,
            "region_ids": node.region_ids,
            "children": [
                JSONExporter._hierarchy_to_dict(child)
                for child in node.children
            ],
        }

    @staticmethod
    def save(result: AnalysisResult, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(JSONExporter.export(result))
