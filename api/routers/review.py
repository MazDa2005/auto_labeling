"""Ручная проверка детекций."""
import json
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.schemas.models import DecisionRequest

router = APIRouter()


@router.get("/{project}")
async def list_review(project: str):
    """Список картинок в review/."""
    review_dir = Path("projects") / project / "ann" / "review"
    if not review_dir.exists():
        return {"frames": [], "total": 0}
    
    frames = []
    for json_file in sorted(review_dir.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        
        with open(json_file) as f:
            data = json.load(f)
        
        detections = data.get("detections", [])
        frames.append({
            "stem": json_file.stem,
            "total": len(detections),
            "accepted": sum(1 for d in detections if d.get("qc_bucket") == "accepted"),
            "needs_review": sum(1 for d in detections if d.get("qc_bucket") == "needs_review"),
            "rejected": sum(1 for d in detections if d.get("qc_bucket") == "rejected"),
        })
    
    return {"frames": frames, "total": len(frames)}


@router.get("/{project}/{frame}")
async def get_frame(project: str, frame: str):
    """Детекции + путь к annotated-картинке."""
    json_path = Path("projects") / project / "ann" / "review" / f"{frame}.json"
    if not json_path.exists():
        raise HTTPException(404, "Frame not found")
    
    with open(json_path) as f:
        data = json.load(f)
    
    return {
        "stem": frame,
        "image": data.get("image"),
        "annotated": f"/static/projects/{project}/ann/review/{frame}_annotated.jpg",
        "detections": data.get("detections", [])
    }


@router.post("/{project}/{frame}/{idx}/decide")
async def decide(project: str, frame: str, idx: int, req: DecisionRequest):
    """Принять/отклонить детекцию."""
    json_path = Path("projects") / project / "ann" / "review" / f"{frame}.json"
    if not json_path.exists():
        raise HTTPException(404, "Frame not found")
    
    with open(json_path) as f:
        data = json.load(f)
    
    detections = data.get("detections", [])
    if idx < 0 or idx >= len(detections):
        raise HTTPException(400, "Invalid detection index")
    
    det = detections[idx]
    
    if req.decision == "accept":
        det["qc_bucket"] = "accepted"
        det["qc_reason"] = "accepted via API"
    elif req.decision == "reject":
        det["qc_bucket"] = "rejected"
        det["qc_reason"] = "rejected via API"
        # Удаляем маску
        mask_path = det.get("mask_path")
        if mask_path and Path(mask_path).exists():
            Path(mask_path).unlink()
    # "skip" — ничего не делаем
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return {"status": "ok", "detection": det}


@router.post("/{project}/{frame}/move-to-clean")
async def move_to_clean(project: str, frame: str):
    """Перенести картинку в clean/, если все детекции обработаны."""
    ann_dir = Path("projects") / project / "ann"
    json_path = ann_dir / "review" / f"{frame}.json"
    
    if not json_path.exists():
        raise HTTPException(404, "Frame not found")
    
    with open(json_path) as f:
        data = json.load(f)
    
    has_review = any(d.get("qc_bucket") == "needs_review" for d in data["detections"])
    if has_review:
        raise HTTPException(400, "Still has needs_review detections")
    
    # Переносим файлы
    shutil.move(str(json_path), str(ann_dir / "clean" / json_path.name))
    
    ann_jpg = ann_dir / "review" / f"{frame}_annotated.jpg"
    if ann_jpg.exists():
        shutil.move(str(ann_jpg), str(ann_dir / "clean" / ann_jpg.name))
    
    masks_src = ann_dir / "review" / "masks"
    masks_dst = ann_dir / "clean" / "masks"
    masks_dst.mkdir(parents=True, exist_ok=True)
    
    for mask in masks_src.glob(f"{frame}_*"):
        shutil.move(str(mask), str(masks_dst / mask.name))
    
    return {"status": "moved", "destination": "clean"}