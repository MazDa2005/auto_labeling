"""Pydantic модели для API."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SourceType(str, Enum):
    VIDEO = "video"
    IMAGES = "images"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""

class ExtractFramesRequest(BaseModel):
    project: str
    fps: int = Field(default=5, ge=1, le=30)


class PipelineRequest(BaseModel):
    project: str
    classes: Optional[list[dict] | list[str]] = None


class ConvertRequest(BaseModel):
    project: str


class MergeRequest(BaseModel):
    projects: list[str] = Field(..., min_length=2)
    output_name: str


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(accept|reject|skip)$")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""


class ProjectInfo(BaseModel):
    name: str
    source_type: SourceType
    description: str = ""
    num_images: int = 0
    status: str = "created"
    created_at: str = ""


class DetectionInfo(BaseModel):
    class_name: str
    bbox: list[float]
    confidence: Optional[float] = None
    qc_bucket: Optional[str] = None
    qc_reason: Optional[str] = None
    mask_path: Optional[str] = None


class FrameInfo(BaseModel):
    stem: str
    total: int
    accepted: int
    needs_review: int
    rejected: int


class FrameDetails(BaseModel):
    stem: str
    image: str
    annotated: str
    detections: list[dict]