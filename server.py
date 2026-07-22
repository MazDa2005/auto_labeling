"""
FastAPI сервер для фоновых задач пайплайна.
Запуск: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Auto-Labeling Backend")

# CORS для Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище задач (в памяти)
TASKS = {}

PROJECTS_DIR = Path("projects")


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, done, failed
    stage: Optional[str] = None
    progress: float = 0.0
    message: str = ""


class ExtractFramesRequest(BaseModel):
    project: str
    fps: int = 5


class PipelineRequest(BaseModel):
    project: str
    classes: Optional[list[str]] = None


class ConvertRequest(BaseModel):
    project: str


def run_extract_frames(task_id: str, project: str, fps: int):
    """Фоновая задача: извлечение кадров."""
    project_dir = PROJECTS_DIR / project
    video_path = project_dir / "video.mp4"

    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "extract_frames"
    TASKS[task_id].message = "Извлечение кадров..."

    cmd = [
        sys.executable, "frame_out.py",
        "--video", str(video_path),
        "--output", str(project_dir / "frames"),
        "--fps", str(fps)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Кадры извлечены"
    except subprocess.TimeoutExpired:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = "Превышено время ожидания (timeout)"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


def run_pipeline(task_id: str, project: str, classes: list[str]):
    """Фоновая задача: запуск пайплайна."""
    project_dir = PROJECTS_DIR / project

    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "batch_label"
    TASKS[task_id].message = "Запуск детекции..."

    cmd = [
        sys.executable, "batch_label.py",
        "--images-dir", str(project_dir / "frames"),
        "--classes", ",".join(classes),
        "--out-dir", str(project_dir / "ann"),
        "--config", "pipeline_config.yaml"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Пайплайн завершен"
    except subprocess.TimeoutExpired:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = "Превышено время ожидания (timeout)"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


def run_convert(task_id: str, project: str):
    """Фоновая задача: конвертация в YOLO."""
    project_dir = PROJECTS_DIR / project

    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "convert"
    TASKS[task_id].message = "Конвертация в YOLO..."

    cmd = [
        sys.executable, "convert_to_yolo_seg.py",
        "--annotations-dir", str(project_dir / "ann" / "clean"),
        "--classes-file", "classes.json",
        "--output-dir", str(project_dir / "dataset_yolo")
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Конвертация завершена"
    except subprocess.TimeoutExpired:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = "Превышено время ожидания (timeout)"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


@app.post("/extract-frames", response_model=TaskStatus)
async def extract_frames(req: ExtractFramesRequest, background_tasks: BackgroundTasks):
    """Извлечь кадры из видео."""
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")

    background_tasks.add_task(run_extract_frames, task_id, req.project, req.fps)

    return TASKS[task_id]


@app.post("/run-pipeline", response_model=TaskStatus)
async def run_pipeline_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks):
    """Запустить пайплайн детекции."""
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")

    # Загрузить классы по умолчанию
    if req.classes is None:
        import json
        with open("classes.json", encoding="utf-8") as f:
            data = json.load(f)
        classes = [c["name"] for c in data["classes"]]
    else:
        classes = req.classes

    background_tasks.add_task(run_pipeline, task_id, req.project, classes)

    return TASKS[task_id]


@app.post("/convert-to-yolo", response_model=TaskStatus)
async def convert_to_yolo(req: ConvertRequest, background_tasks: BackgroundTasks):
    """Конвертировать в YOLO формат."""
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")

    background_tasks.add_task(run_convert, task_id, req.project)

    return TASKS[task_id]


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Получить статус задачи."""
    if task_id not in TASKS:
        raise HTTPException(404, "Задача не найдена")
    return TASKS[task_id]


@app.get("/tasks")
async def list_tasks():
    """Список всех задач."""
    return {"tasks": list(TASKS.values())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)