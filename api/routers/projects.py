"""Управление проектами."""
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from api.schemas.models import ProjectInfo, SourceType

router = APIRouter()

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)


@router.post("/from-video")
async def create_project_from_video(
    name: str = Form(...),
    description: str = Form(""),
    video: UploadFile = File(...)
):
    """Создать проект из видео."""
    project_dir = PROJECTS_DIR / name
    if project_dir.exists():
        raise HTTPException(400, f"Проект '{name}' уже существует")
    
    project_dir.mkdir(parents=True)
    
    # Сохраняем видео
    video_path = project_dir / "video.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Метаданные
    project_info = {
        "name": name,
        "description": description,
        "source_type": "video",
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "num_images": 0
    }
    
    with open(project_dir / "project.json", "w") as f:
        json.dump(project_info, f, indent=2, ensure_ascii=False)
    
    return {"name": name, "status": "created", "message": "Проект создан"}


@router.post("/from-images")
async def create_project_from_images(
    name: str = Form(...),
    description: str = Form(""),
    files: list[UploadFile] = File(...)
):
    """Создать проект из загруженных картинок."""
    project_dir = PROJECTS_DIR / name
    if project_dir.exists():
        raise HTTPException(400, f"Проект '{name}' уже существует")
    
    project_dir.mkdir(parents=True)
    images_dir = project_dir / "images"
    images_dir.mkdir()
    
    total_files = 0
    
    for file in files:
        if file.filename.endswith(".zip"):
            # Распаковываем ZIP
            zip_path = project_dir / file.filename
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    if member.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        zip_ref.extract(member, images_dir)
                        total_files += 1
            
            zip_path.unlink()
        else:
            # Сохраняем отдельный файл
            if file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_path = images_dir / file.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                total_files += 1
    
    # Метаданные
    project_info = {
        "name": name,
        "description": description,
        "source_type": "images",
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "num_images": total_files
    }
    
    with open(project_dir / "project.json", "w") as f:
        json.dump(project_info, f, indent=2, ensure_ascii=False)
    
    return {
        "name": name,
        "status": "created",
        "num_images": total_files,
        "message": f"Загружено {total_files} картинок"
    }


@router.get("/")
async def list_projects():
    """Список всех проектов."""
    projects = []
    
    for project_dir in PROJECTS_DIR.iterdir():
        if project_dir.is_dir():
            project_json = project_dir / "project.json"
            if project_json.exists():
                with open(project_json) as f:
                    info = json.load(f)
                projects.append(info)
    
    return {"projects": projects, "total": len(projects)}


@router.get("/{project}")
async def get_project(project: str):
    """Детали проекта."""
    project_dir = PROJECTS_DIR / project
    if not project_dir.exists():
        raise HTTPException(404, f"Проект '{project}' не найден")
    
    project_json = project_dir / "project.json"
    if not project_json.exists():
        raise HTTPException(404, "project.json не найден")
    
    with open(project_json) as f:
        info = json.load(f)
    
    # Добавляем информацию о кадрах
    frames_dir = project_dir / "frames"
    if frames_dir.exists():
        info["num_frames"] = len(list(frames_dir.glob("frame_*.jpg")))
    
    # Информация о аннотациях
    ann_dir = project_dir / "ann"
    if ann_dir.exists():
        clean_dir = ann_dir / "clean"
        review_dir = ann_dir / "review"
        info["num_clean"] = len(list(clean_dir.glob("*.json"))) if clean_dir.exists() else 0
        info["num_review"] = len(list(review_dir.glob("*.json"))) if review_dir.exists() else 0
    
    return info