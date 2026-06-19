import threading
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

from schemas.models import TaskInfo, TaskStatus, AnalysisResult
from core.pipeline import AnalysisPipeline
from utils.file_utils import (
    generate_task_id,
    save_uploaded_file,
    get_upload_path,
    get_analysis_result_path,
    delete_task_data,
    format_timestamp,
)
from config import Config


class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.tasks: Dict[str, TaskInfo] = {}
        self.results: Dict[str, AnalysisResult] = {}
        self.progress: Dict[str, float] = {}
        self.messages: Dict[str, str] = {}

        self.task_queue: deque = deque()
        self.active_tasks: set = set()
        self.max_concurrent = 2

        self.pipeline = AnalysisPipeline()

        self._start_worker()

    def _start_worker(self):
        worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        worker_thread.start()

    def _process_queue(self):
        while True:
            if self.task_queue and len(self.active_tasks) < self.max_concurrent:
                task_id = self.task_queue.popleft()
                self.active_tasks.add(task_id)

                process_thread = threading.Thread(
                    target=self._process_task,
                    args=(task_id,),
                    daemon=True
                )
                process_thread.start()

            import time
            time.sleep(0.1)

    def _process_task(self, task_id: str):
        try:
            self._update_status(task_id, TaskStatus.PROCESSING, 0.0, "Starting analysis...")

            upload_dir = os.path.join(Config.UPLOAD_FOLDER, task_id)
            files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
            if not files:
                raise FileNotFoundError(f"No files found for task {task_id}")

            file_path = os.path.join(upload_dir, files[0])

            task_info = self.tasks.get(task_id)
            ocr_enabled = True
            output_format = "json"
            ground_truth = None

            if task_info and task_info.metadata:
                ocr_enabled = task_info.metadata.ocr_enabled
                output_format = task_info.metadata.output_format

            def progress_callback(progress: float, message: str):
                self._update_status(task_id, TaskStatus.PROCESSING, progress, message)

            result = self.pipeline.analyze(
                file_path=file_path,
                task_id=task_id,
                ocr_enabled=ocr_enabled,
                output_format=output_format,
                ground_truth=ground_truth,
                progress_callback=progress_callback,
            )

            self.results[task_id] = result

            if result.status == "completed":
                self._update_status(task_id, TaskStatus.COMPLETED, 1.0, "Analysis complete")
            else:
                self._update_status(task_id, TaskStatus.FAILED, 0.0, result.error or "Analysis failed")

        except Exception as e:
            self._update_status(task_id, TaskStatus.FAILED, 0.0, f"Error: {str(e)}")
        finally:
            self.active_tasks.discard(task_id)

    def _update_status(self, task_id: str, status: TaskStatus, progress: float, message: str = None):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            task.progress = progress
            task.message = message
            task.updated_at = format_timestamp()
            self.tasks[task_id] = task

            self.progress[task_id] = progress
            if message:
                self.messages[task_id] = message

    def create_task(self, files, ocr_enabled: bool = True, output_format: str = "json") -> List[str]:
        task_ids = []

        if not isinstance(files, list):
            files = [files]

        if len(files) > Config.MAX_BATCH_FILES:
            raise ValueError(f"Maximum {Config.MAX_BATCH_FILES} files allowed per batch")

        for file_storage in files:
            if not file_storage or not file_storage.filename:
                continue

            if not Config.allowed_file(file_storage.filename):
                raise ValueError(f"Invalid file type: {file_storage.filename}")

            task_id = generate_task_id()

            save_uploaded_file(file_storage, task_id)

            from schemas.models import Metadata
            task_info = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=format_timestamp(),
                updated_at=format_timestamp(),
                progress=0.0,
                message="Task queued",
                metadata=Metadata(
                    filename=file_storage.filename,
                    file_size=0,
                    page_count=0,
                    file_type="unknown",
                    ocr_enabled=ocr_enabled,
                    output_format=output_format,
                ),
            )

            self.tasks[task_id] = task_info
            self.task_queue.append(task_id)

            task_ids.append(task_id)

        return task_ids

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[AnalysisResult]:
        if task_id not in self.results:
            result_path = get_analysis_result_path(task_id)
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    from schemas.models import AnalysisResult
                    self.results[task_id] = AnalysisResult(**data)
                except Exception:
                    pass
        return self.results.get(task_id)

    def get_page_result(self, task_id: str, page_number: int) -> Optional[dict]:
        result = self.get_result(task_id)
        if not result:
            return None

        page = next((p for p in result.pages if p.page_number == page_number), None)
        if not page:
            return None

        return {
            "task_id": task_id,
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "dpi": page.dpi,
            "image_path": page.image_path,
            "preprocessing_applied": page.preprocessing_applied,
            "regions": [
                {
                    "id": r.id,
                    "type": r.type.value,
                    "bbox": {
                        "x": r.bbox.x,
                        "y": r.bbox.y,
                        "width": r.bbox.width,
                        "height": r.bbox.height,
                    },
                    "confidence": r.confidence,
                    "text": r.text,
                    "reading_order": r.reading_order,
                    "table_structure": (
                        {
                            "rows": r.table_structure.rows,
                            "cols": r.table_structure.cols,
                            "cells": [
                                {
                                    "row_index": c.row_index,
                                    "col_index": c.col_index,
                                    "row_span": c.row_span,
                                    "col_span": c.col_span,
                                    "text": c.text,
                                }
                                for c in r.table_structure.cells
                            ],
                            "grid": self.pipeline.table_recognizer.get_table_as_2d_array(r.table_structure),
                        }
                        if r.table_structure else None
                    ),
                }
                for r in page.regions
            ],
        }

    def list_tasks(self) -> List[TaskInfo]:
        return list(self.tasks.values())

    def delete_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            del self.tasks[task_id]
        if task_id in self.results:
            del self.results[task_id]
        if task_id in self.progress:
            del self.progress[task_id]
        if task_id in self.messages:
            del self.messages[task_id]

        delete_task_data(task_id)
        return True

    def get_all_tasks(self) -> List[TaskInfo]:
        return list(self.tasks.values())
