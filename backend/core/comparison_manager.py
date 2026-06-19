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
    GlobalComparisonStats,
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
        self.comparison_start_times: Dict[str, float] = {}
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
        import time
        start_time = time.time()
        try:
            self.comparison_start_times[comparison_id] = start_time
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

            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            self._update_status(comparison_id, ComparisonStatus.COMPLETED, 1.0, "Comparison complete", duration_ms=duration_ms)

        except Exception as e:
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            self._update_status(comparison_id, ComparisonStatus.FAILED, 0.0, f"Error: {str(e)}", duration_ms=duration_ms)
        finally:
            self.active_comparisons.discard(comparison_id)

    def _update_status(self, comparison_id: str, status: ComparisonStatus, progress: float, message: str = None, duration_ms: int = None):
        if comparison_id in self.comparisons:
            comp = self.comparisons[comparison_id]
            comp.status = status
            comp.progress = progress
            comp.message = message
            comp.updated_at = format_timestamp()
            if duration_ms is not None:
                comp.duration_ms = duration_ms
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

    def create_comparison(self, source_id: str, target_id: str, page_number: Optional[int] = None, label: Optional[str] = None) -> str:
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
            label=label,
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

    def list_comparisons(
        self,
        page: int = 1,
        page_size: int = 20,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> tuple[List[ComparisonInfo], int]:
        all_comps = list(self.comparisons.values())

        all_comps.sort(key=lambda c: c.created_at, reverse=True)

        filtered = []
        for comp in all_comps:
            if source_id and comp.source_id != source_id:
                continue
            if target_id and comp.target_id != target_id:
                continue
            if label and comp.label:
                if label.lower() not in comp.label.lower():
                    continue
            if label and not comp.label:
                continue
            filtered.append(comp)

        total = len(filtered)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged = filtered[start_idx:end_idx]

        return paged, total

    def get_global_stats(self) -> GlobalComparisonStats:
        stats = GlobalComparisonStats()

        completed_comps = [c for c in self.comparisons.values() if c.status == ComparisonStatus.COMPLETED]
        stats.total_comparisons = len(completed_comps)

        if not completed_comps:
            return stats

        total_diffs = 0
        type_counts = {"added": 0, "removed": 0, "moved": 0, "modified": 0, "unchanged": 0}
        total_durations = 0
        count_with_duration = 0

        for comp in completed_comps:
            result = self.get_result(comp.comparison_id)
            if result:
                total_diffs += result.stats.total
                type_counts["added"] += result.stats.added
                type_counts["removed"] += result.stats.removed
                type_counts["moved"] += result.stats.moved
                type_counts["modified"] += result.stats.modified
                type_counts["unchanged"] += result.stats.unchanged

            if comp.duration_ms is not None:
                total_durations += comp.duration_ms
                count_with_duration += 1

        stats.avg_diff_count = round(total_diffs / len(completed_comps), 2) if completed_comps else 0.0
        stats.avg_duration_ms = round(total_durations / count_with_duration, 2) if count_with_duration > 0 else 0.0

        total_type_counts = sum(type_counts.values())
        if total_type_counts > 0:
            stats.type_distribution = {
                k: round((v / total_type_counts) * 100, 2)
                for k, v in type_counts.items()
            }
        else:
            stats.type_distribution = {k: 0.0 for k in type_counts}

        return stats
