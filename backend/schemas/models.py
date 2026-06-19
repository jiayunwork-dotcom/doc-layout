from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class RegionType(str, Enum):
    TEXT = "text"
    TITLE_H1 = "title_h1"
    TITLE_H2 = "title_h2"
    TITLE_H3 = "title_h3"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    FORMULA = "formula"
    FORMULA_INLINE = "formula_inline"
    LIST = "list"


class BBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0, description="Normalized x coordinate of top-left corner")
    y: float = Field(ge=0.0, le=1.0, description="Normalized y coordinate of top-left corner")
    width: float = Field(ge=0.0, le=1.0, description="Normalized width")
    height: float = Field(ge=0.0, le=1.0, description="Normalized height")

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    def to_absolute(self, page_width: int, page_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x * page_width),
            int(self.y * page_height),
            int(self.x1 * page_width),
            int(self.y1 * page_height),
        )


class TableCell(BaseModel):
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    text: Optional[str] = None
    bbox: BBox


class TableStructure(BaseModel):
    rows: int
    cols: int
    cells: List[TableCell]


class Region(BaseModel):
    id: str
    type: RegionType
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    text: Optional[str] = None
    reading_order: Optional[int] = None
    table_structure: Optional[TableStructure] = None
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)


class PageStatistics(BaseModel):
    total_regions: int
    type_counts: Dict[str, int]
    avg_confidence: float
    max_confidence: float
    min_confidence: float


class Page(BaseModel):
    page_number: int
    width: int
    height: int
    dpi: int = 72
    image_path: Optional[str] = None
    regions: List[Region] = Field(default_factory=list)
    preprocessing_applied: List[str] = Field(default_factory=list)


class HierarchyNode(BaseModel):
    node_id: str
    title: Optional[str] = None
    level: int
    region_ids: List[str] = Field(default_factory=list)
    children: List["HierarchyNode"] = Field(default_factory=list)


HierarchyNode.model_rebuild()


class Metadata(BaseModel):
    filename: str
    file_size: int
    page_count: int
    file_type: str
    ocr_enabled: bool = False
    output_format: str = "json"


class EvaluationMetrics(BaseModel):
    mAP: Optional[float] = None
    mean_iou: Optional[float] = None
    per_class_iou: Optional[Dict[str, float]] = None
    per_class_ap: Optional[Dict[str, float]] = None


class AnalysisResult(BaseModel):
    task_id: str
    status: str = "completed"
    metadata: Metadata
    pages: List[Page]
    hierarchy: Optional[HierarchyNode] = None
    evaluation: Optional[EvaluationMetrics] = None
    error: Optional[str] = None


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Metadata] = None


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    total: int


class DiffType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class DisplacementVector(BaseModel):
    dx: float
    dy: float


class DiffRecord(BaseModel):
    id: str
    type: DiffType
    page_number: int
    source_region_id: Optional[str] = None
    target_region_id: Optional[str] = None
    source_region: Optional[Region] = None
    target_region: Optional[Region] = None
    displacement: Optional[DisplacementVector] = None
    content_summary: Optional[str] = None
    iou: Optional[float] = None


class PageDiff(BaseModel):
    page_number: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    diffs: List[DiffRecord] = Field(default_factory=list)


class ComparisonStats(BaseModel):
    added: int = 0
    removed: int = 0
    moved: int = 0
    modified: int = 0
    unchanged: int = 0
    total: int = 0


class TaskBasicInfo(BaseModel):
    task_id: str
    filename: str
    page_count: int


class ComparisonStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonInfo(BaseModel):
    comparison_id: str
    status: ComparisonStatus
    source_id: str
    target_id: str
    page_number: Optional[int] = None
    label: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: str
    updated_at: str
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None


class GlobalComparisonStats(BaseModel):
    total_comparisons: int = 0
    avg_diff_count: float = 0.0
    type_distribution: Dict[str, float] = Field(default_factory=dict)
    avg_duration_ms: float = 0.0


class ComparisonResult(BaseModel):
    comparison_id: str
    status: ComparisonStatus
    source_info: TaskBasicInfo
    target_info: TaskBasicInfo
    page_diffs: List[PageDiff] = Field(default_factory=list)
    stats: ComparisonStats = Field(default_factory=ComparisonStats)
    error: Optional[str] = None


class ComparisonExport(BaseModel):
    comparison_id: str
    source_info: TaskBasicInfo
    target_info: TaskBasicInfo
    page_diffs: List[PageDiff]
    stats: ComparisonStats
    created_at: str
