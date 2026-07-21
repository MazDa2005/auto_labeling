"""Запуск этапов пайплайна."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.schemas.models import (
    PipelineRequest, 
    ExtractFramesRequest,
    ConvertRequest, 
    TaskStatusResponse
)
from api.services.task_manager import task_manager
from api.services.pipeline_runner import (
    run_extract_frames_sync,
    run_pipeline_sync, 
    run_convert_sync
)
from pathlib import Path

router = APIRouter()


@router.post("/extract-frames", response_model=TaskStatusResponse)
async def extract_frames(req: ExtractFramesRequest, background_tasks: BackgroundTasks):
    """Извлечь кадры из видео (отдельный этап, не часть пайплайна)."""
    project_dir = Path("projects") / req.project
    if not project_dir.exists():
        raise HTTPException(404, f"Проект '{req.project}' не найден")
    
    video_path = project_dir / "video.mp4"
    if not video_path.exists():
        raise HTTPException(400, "Видео не найдено в проекте")
    
    task_id = task_manager.create_task()
    
    background_tasks.add_task(
        run_extract_frames_sync,
        task_id,
        req.project,
        req.fps
    )
    
    return task_manager.get_task(task_id)


@router.post("/run-full", response_model=TaskStatusResponse)
async def run_full_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Запустить весь пайплайн (detect → refine → qc_filter) БЕЗ извлечения кадров.
    Кадры должны быть уже извлечены через /extract-frames.
    """
    project_dir = Path("projects") / req.project
    if not project_dir.exists():
        raise HTTPException(404, f"Проект '{req.project}' не найден")
    
    # Проверяем, что кадры уже есть
    frames_dir = project_dir / "frames"
    if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
        raise HTTPException(400, "Кадры не найдены. Сначала запустите /extract-frames")
    
    task_id = task_manager.create_task()
    
    # Загружаем классы по умолчанию или извлекаем имена из объектов
    if req.classes is None:
        import json
        with open("classes.json") as f:
            data = json.load(f)
        classes = [c["name"] for c in data["classes"]]
    elif req.classes and isinstance(req.classes[0], dict):
        classes = [c["name"] for c in req.classes]
    else:
        classes = req.classes
    
    background_tasks.add_task(
        run_pipeline_sync,
        task_id,
        req.project,
        classes
    )
    
    return task_manager.get_task(task_id)


@router.post("/convert-to-yolo", response_model=TaskStatusResponse)
async def convert_to_yolo(req: ConvertRequest, background_tasks: BackgroundTasks):
    """Конвертировать аннотации в YOLO формат."""
    project_dir = Path("projects") / req.project
    if not project_dir.exists():
        raise HTTPException(404, f"Проект '{req.project}' не найден")
    
    clean_dir = project_dir / "ann" / "clean"
    if not clean_dir.exists():
        raise HTTPException(400, "Папка clean/ не найдена. Сначала запустите пайплайн.")
    
    task_id = task_manager.create_task()
    
    background_tasks.add_task(
        run_convert_sync,
        task_id,
        req.project
    )
    
    return task_manager.get_task(task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Получить статус задачи."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    return task


@router.get("/tasks")
async def list_tasks():
    """Список всех задач."""
    return {"tasks": task_manager.list_tasks()}