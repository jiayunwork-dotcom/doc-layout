from typing import List, Optional, Dict
import os
import json
from datetime import datetime

from schemas.models import (
    AnalysisResult,
    Page,
    RegionType,
    Metadata,
    EvaluationMetrics,
)
from processing.document_loader import DocumentLoader
from processing.preprocessor import DocumentPreprocessor
from processing.ocr_engine import OCREngine
from processing.reading_order import ReadingOrderInference
from processing.hierarchy import HierarchyExtractor
from models.layout_detector import LayoutDetector
from models.table_recognizer import TableRecognizer
from output.json_exporter import JSONExporter
from output.hocr_exporter import HOCRExporter
from output.alto_exporter import ALTOExporter
from core.metrics import EvaluationMetricsCalculator
from utils.file_utils import (
    get_file_size,
    get_upload_path,
    get_analysis_result_path,
    format_timestamp,
)
from config import Config
from utils.image_utils import load_image


class AnalysisPipeline:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.preprocessor = DocumentPreprocessor(
            enable_deskew=True,
            enable_denoise=False,
            enable_binarize=False,
        )
        self.layout_detector = LayoutDetector()
        self.table_recognizer = TableRecognizer()
        self.ocr_engine = OCREngine(enabled=Config.OCR_ENABLED)
        self.reading_order = ReadingOrderInference()
        self.hierarchy_extractor = HierarchyExtractor()
        self.metrics_calculator = EvaluationMetricsCalculator()

    def analyze(self, file_path: str, task_id: str,
                ocr_enabled: bool = True,
                output_format: str = "json",
                ground_truth: List[Dict] = None,
                progress_callback=None) -> AnalysisResult:
        try:
            if progress_callback:
                progress_callback(0.0, "Loading document...")

            filename = os.path.basename(file_path)
            file_size = get_file_size(file_path)

            pages = self.document_loader.load_document(file_path, task_id)
            page_count = len(pages)

            if progress_callback:
                progress_callback(0.1, f"Loaded {page_count} pages")

            file_type = "pdf" if filename.lower().endswith(".pdf") else "image"
            metadata = Metadata(
                filename=filename,
                file_size=file_size,
                page_count=page_count,
                file_type=file_type,
                ocr_enabled=ocr_enabled,
                output_format=output_format,
            )

            if progress_callback:
                progress_callback(0.2, "Preprocessing pages...")

            pages = self.preprocessor.preprocess_pages(pages, task_id)

            if progress_callback:
                progress_callback(0.4, "Detecting layout regions...")

            images = []
            for page in pages:
                if page.image_path and os.path.exists(page.image_path):
                    img = load_image(page.image_path)
                    images.append(img)
                else:
                    images.append(None)

            all_regions = self.layout_detector.detect_batch(images)

            for page, regions in zip(pages, all_regions):
                page.regions = regions

            if progress_callback:
                progress_callback(0.6, "Analyzing table structures...")

            for page_idx, page in enumerate(pages):
                img = images[page_idx]
                if img is None:
                    continue

                for region in page.regions:
                    if region.type == RegionType.TABLE:
                        try:
                            region.table_structure = self.table_recognizer.recognize(img, region)
                        except Exception as e:
                            print(f"Table recognition error: {e}")

            if ocr_enabled and self.ocr_engine.is_available():
                if progress_callback:
                    progress_callback(0.7, "Extracting text with OCR...")

                for page_idx, page in enumerate(pages):
                    img = images[page_idx]
                    if img is None:
                        continue

                    for region in page.regions:
                        if region.type in {
                            RegionType.TEXT,
                            RegionType.TITLE_H1,
                            RegionType.TITLE_H2,
                            RegionType.TITLE_H3,
                            RegionType.CAPTION,
                            RegionType.LIST,
                        }:
                            region.text = self.ocr_engine.extract_text(img, region)

            if progress_callback:
                progress_callback(0.8, "Inferring reading order...")

            for page in pages:
                page.regions = self.reading_order.infer_reading_order(page.regions)

            if progress_callback:
                progress_callback(0.9, "Extracting document hierarchy...")

            hierarchy = self.hierarchy_extractor.extract_hierarchy(pages)

            evaluation = None
            if ground_truth:
                if progress_callback:
                    progress_callback(0.95, "Computing evaluation metrics...")
                evaluation = self.metrics_calculator.compute_metrics(pages, ground_truth)

            if progress_callback:
                progress_callback(1.0, "Analysis complete")

            result = AnalysisResult(
                task_id=task_id,
                status="completed",
                metadata=metadata,
                pages=pages,
                hierarchy=hierarchy,
                evaluation=evaluation,
            )

            result_path = get_analysis_result_path(task_id)
            JSONExporter.save(result, result_path)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                task_id=task_id,
                status="failed",
                metadata=Metadata(
                    filename=os.path.basename(file_path),
                    file_size=0,
                    page_count=0,
                    file_type="unknown",
                ),
                pages=[],
                error=str(e),
            )

    def export_result(self, result: AnalysisResult, output_format: str) -> str:
        output_format = output_format.lower()
        if output_format == "json":
            return JSONExporter.export(result)
        elif output_format == "hocr":
            return HOCRExporter.export(result)
        elif output_format == "alto":
            return ALTOExporter.export(result)
        else:
            return JSONExporter.export(result)
