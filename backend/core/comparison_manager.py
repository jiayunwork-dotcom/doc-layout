import threading
import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

from schemas.models import (
    ComparisonInfo,
    ComparisonStatus,
    ComparisonResult,
    TaskBasicInfo,
    TaskStatus,
)
from core.comparison import compare_pages, calculate_stats
from utils.file_utils import (
    generate_task_id,
    format_timestamp,
    get_result_path,
)
from config import Config


class ComparisonManager:
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

        self.comparisons: Dict[str, ComparisonInfo] = {}
        self.results: Dict[str, ComparisonResult] = {}
        self.progress: Dict[str, float] = {}
        self.messages: Dict[str, str] = {}

        self.comparison_queue: deque = deque()
        self.active_comparisons: set = set()
        self.max_concurrent = 2

        self._task_manager = None

        self._start_worker()

    def set_task_manager(self, task_manager):
        self._task_manager = task_manager

    def _start_worker(self):
        worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        worker_thread.start()

    def _process_queue(self):
        while True:
            if self.comparison_queue and len(self.active_comparisons) < self.max_concurrent:
                comparison_id = self.comparison_queue.popleft()
                self.active_comparisons.add(comparison_id)

                process_thread = threading.Thread(
                    target=self._process_comparison,
                    args=(comparison_id,),
                    daemon=True
                )
                process_thread.start()

            import time
            time.sleep(0.1)

    def _process_comparison(self, comparison_id: str):
        try:
            self._update_status(comparison_id, ComparisonStatus.PROCESSING, 0.0, "Starting comparison...")

            comparison_info = self.comparisons.get(comparison_id)
            if not comparison_info:
                raise ValueError(f"Comparison {comparison_id} not found")

            source_id = comparison_info.source_id
            target_id = comparison_info.target_id
            page_number = comparison_info.page_number

            if not self._task_manager:
                raise ValueError("TaskManager not set")

            source_task = self._task_manager.get_task(source_id)
            target_task = self._task_manager.get_task(target_id)

            if not source_task:
                raise ValueError(f"Source task {source_id} not found")
            if not target_task:
                raise ValueError(f"Target task {target_id} not found")

            if source_task.status != TaskStatus.COMPLETED:
                raise ValueError(f"Source task {source_id} is not completed")
            if target_task.status != TaskStatus.COMPLETED:
                raise ValueError(f"Target task {target_id} is not completed")

            source_result = self._task_manager.get_result(source_id)
            target_result = self._task_manager.get_result(target_id)

            if not source_result or not target_result:
                raise ValueError("Could not load analysis results")

            source_info = TaskBasicInfo(
                task_id=source_id,
                filename=source_result.metadata.filename,
                page_count=source_result.metadata.page_count,
            )
            target_info = TaskBasicInfo(
                task_id=target_id,
                filename=target_result.metadata.filename,
                page_count=target_result.metadata.page_count,
            )

            source_pages = source_result.pages
            target_pages = target_result.pages

            if page_number is not None:
                source_page = next((p for p in source_pages if p.page_number == page_number), None)
                target_page = next((p for p in target_pages if p.page_number == page_number), None)
                if not source_page or not target_page:
                    raise ValueError(f"Page {page_number} not found in one of the documents")
                pages_to_compare = [(page_number, source_page, target_page)]
            else:
                min_pages = min(len(source_pages), len(target_pages))
                pages_to_compare = []
                for i in range(min_pages):
                    src_page = source_pages[i]
                    tgt_page = target_pages[i]
                    pages_to_compare.append((src_page.page_number, src_page, tgt_page))

            total_pages = len(pages_to_compare)
            page_diffs = []

            for idx, (pn, src_page, tgt_page) in enumerate(pages_to_compare):
                progress = (idx + 1) / total_pages * 0.9
                self._update_status(
                    comparison_id,
                    ComparisonStatus.PROCESSING,
                    progress,
                    f"Comparing page {pn}/{total_pages}..."
                )

                src_page_dict = self._task_manager.get_page_result(source_id, pn)
                tgt_page_dict = self._task_manager.get_page_result(target_id, pn)

                if not src_page_dict or not tgt_page_dict:
                    continue

                page_diff = compare_pages(src_page_dict, tgt_page_dict, pn)
                page_diffs.append(page_diff)

            self._update_status(
                comparison_id,
                ComparisonStatus.PROCESSING,
                0.95,
                "Calculating statistics..."
            )

            stats = calculate_stats(page_diffs)

            result = ComparisonResult(
                comparison_id=comparison_id,
                status=ComparisonStatus.COMPLETED,
                source_info=source_info,
                target_info=target_info,
                page_diffs=page_diffs,
                stats=stats,
            )

            self.results[comparison_id] = result
            self._save_result(comparison_id, result)

            self._update_status(comparison_id, ComparisonStatus.COMPLETED, 1.0, "Comparison complete")

        except Exception as e:
            self._update_status(comparison_id, ComparisonStatus.FAILED, 0.0, f"Error: {str(e)}")
        finally:
            self.active_comparisons.discard(comparison_id)

    def _update_status(self, comparison_id: str, status: ComparisonStatus, progress: float, message: str = None):
        if comparison_id in self.comparisons:
            comp = self.comparisons[comparison_id]
            comp.status = status
            comp.progress = progress
            comp.message = message
            comp.updated_at = format_timestamp()
            self.comparisons[comparison_id] = comp

            self.progress[comparison_id] = progress
            if message:
                self.messages[comparison_id] = message

    def _save_result(self, comparison_id: str, result: ComparisonResult):
        try:
            result_path = get_result_path(comparison_id, "comparison_result.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def create_comparison(self, source_id: str, target_id: str, page_number: Optional[int] = None) -> str:
        if not self._task_manager:
            raise ValueError("TaskManager not set")

        source_task = self._task_manager.get_task(source_id)
        target_task = self._task_manager.get_task(target_id)

        if not source_task:
            raise ValueError(f"Source task {source_id} not found")
        if not target_task:
            raise ValueError(f"Target task {target_id} not found")

        comparison_id = str(uuid.uuid4())

        comparison_info = ComparisonInfo(
            comparison_id=comparison_id,
            status=ComparisonStatus.PENDING,
            source_id=source_id,
            target_id=target_id,
            page_number=page_number,
            created_at=format_timestamp(),
            updated_at=format_timestamp(),
            progress=0.0,
            message="Comparison queued",
        )

        self.comparisons[comparison_id] = comparison_info
        self.comparison_queue.append(comparison_id)

        return comparison_id

    def get_comparison(self, comparison_id: str) -> Optional[ComparisonInfo]:
        return self.comparisons.get(comparison_id)

    def get_result(self, comparison_id: str) -> Optional[ComparisonResult]:
        if comparison_id not in self.results:
            result_path = get_result_path(comparison_id, "comparison_result.json")
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    from schemas.models import ComparisonResult
                    self.results[comparison_id] = ComparisonResult(**data)
                except Exception:
                    pass
        return self.results.get(comparison_id)

    def list_comparisons(self) -> List[ComparisonInfo]:
        return list(self.comparisons.values())
