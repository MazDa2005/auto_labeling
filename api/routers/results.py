"""Скачивание результатов."""
import zipfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/{project}/annotated/{frame}")
async def get_annotated_image(project: str, frame: str):
    """Скачать картинку с масками."""
    # Ищем в clean/ или review/
    for bucket in ["clean", "review"]:
        img_path = Path("projects") / project / "ann" / bucket / f"{frame}_annotated.jpg"
        if img_path.exists():
            return FileResponse(img_path)
    
    raise HTTPException(404, "Annotated image not found")


@router.get("/{project}/dataset")
async def download_dataset(project: str):
    """Скачать YOLO-датасет в виде ZIP."""
    dataset_dir = Path("projects") / project / "dataset_yolo"
    if not dataset_dir.exists():
        raise HTTPException(404, "Dataset not found. Run convert-to-yolo first.")
    
    # Создаём ZIP архив
    zip_path = Path("projects") / project / f"{project}_dataset.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(dataset_dir)
                zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{project}_dataset.zip"
    )


@router.get("/{project}/status")
async def get_project_status(project: str):
    """Текущий статус проекта."""
    project_dir = Path("projects") / project
    if not project_dir.exists():
        raise HTTPException(404, "Project not found")
    
    status = {
        "project": project,
        "has_frames": (project_dir / "frames").exists(),
        "has_annotations": (project_dir / "ann").exists(),
        "has_dataset": (project_dir / "dataset_yolo").exists(),
    }
    
    # Считаем количество файлов
    if status["has_frames"]:
        status["num_frames"] = len(list((project_dir / "frames").glob("*.jpg")))
    
    if status["has_annotations"]:
        clean_dir = project_dir / "ann" / "clean"
        review_dir = project_dir / "ann" / "review"
        status["num_clean"] = len(list(clean_dir.glob("*.json"))) if clean_dir.exists() else 0
        status["num_review"] = len(list(review_dir.glob("*.json"))) if review_dir.exists() else 0
    
    return status