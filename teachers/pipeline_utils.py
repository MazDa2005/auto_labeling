"""Общие утилиты для teacher-скриптов и оркестраторов — чтобы не дублировать код."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir: str) -> list[Path]:
    return sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def merge_new_detections(existing: list[dict], new: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    merged = list(existing)
    for d in new:
        is_dup = any(
            e["class"] == d["class"] and iou(e["bbox"], d["bbox"]) > iou_threshold
            for e in existing
        )
        if not is_dup:
            merged.append(d)
    return merged


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def save_mask_png(mask, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(out_path)


# Работа с classes.json — единый источник промптов для всех моделей 

def load_classes_with_prompts(classes_file: str = "classes.json") -> list[dict]:
    """Читает classes.json целиком, возвращает список классов (отсортирован по index)."""
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data["classes"], key=lambda c: c["index"])


def get_class_names(classes_file: str = "classes.json") -> list[str]:
    """Просто список имён классов, в порядке index."""
    return [c["name"] for c in load_classes_with_prompts(classes_file)]


def get_primary_prompts(classes_file: str = "classes.json") -> dict[str, str]:
    """name -> первый (основной, самый короткий) промпт."""
    return {c["name"]: c["prompts"][0] for c in load_classes_with_prompts(classes_file)}


def get_all_prompts(classes_file: str = "classes.json") -> dict[str, list[str]]:
    """name -> ВЕСЬ список промптов. Для мест с мульти-промпт агрегацией."""
    return {c["name"]: c["prompts"] for c in load_classes_with_prompts(classes_file)}


def build_reverse_prompt_lookup(classes_file: str = "classes.json") -> dict[str, str]:
    """Нужно, чтобы сопоставить текстовый ответ модели обратно с каноническим именем из classes.json"""
    lookup = {}
    for c in load_classes_with_prompts(classes_file):
        primary = c["prompts"][0].lower().strip()
        lookup[primary] = c["name"]
    return lookup