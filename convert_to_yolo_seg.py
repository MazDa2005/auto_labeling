"""
Конвертация нашего внутреннего формата (annotations/*.json + masks/*.png) в YOLO segment.

YOLO segment label — по одной строке на объект в файле <image_stem>.txt:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn
координаты полигона нормализованы в диапазон [0, 1] от ширины/высоты картинки.

Если у детекции есть маска (mask_path) — полигон строится из контура маски (cv2.findContours).
Если маски нет (например, refine-этап был выключен/не сработал) — используется bbox
как прямоугольный полигон (4 точки) — это fallback, а не полноценная сегментация.

Результат — готовая структура датасета:
    <output-dir>/
    ├── images/train/*.jpg
    ├── labels/train/*.txt
    └── data.yaml

Пример:
    python convert_to_yolo_seg.py --annotations-dir ann/clean/ --classes-file classes.json --output-dir dataset_yolo/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_class_mapping(classes_file: str) -> dict[str, int]:
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["name"]: c["index"] for c in data["classes"]}


def mask_to_polygon(mask_path: str, epsilon_frac: float = 0.005) -> list[tuple[float, float]] | None:
    """Извлекает контур маски и упрощает его (уменьшает число точек полигона)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:  # слишком маленький контур — не стоит внимания
        return None

    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) < 3:
        return None

    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def bbox_to_polygon(bbox: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def find_file(file_path: str, annotations_dir: str) -> Path | None:
    """
    Ищет файл по нескольким стратегиям:
    1. Как есть (абсолютный путь)
    2. Относительно текущей директории
    3. Относительно корня проекта (на 2 уровня выше annotations_dir)
    """
    # 1. Пробуем как есть
    p = Path(file_path)
    if p.exists():
        return p
    
    # 2. Пробуем относительно cwd
    p = Path.cwd() / file_path
    if p.exists():
        return p
    
    # 3. Пробуем относительно корня проекта (ann/clean/ -> ann/ -> project_root/)
    # Структура: project_root/ann/clean/ или project_root/ann/review/
    project_root = Path(annotations_dir).parent.parent
    p = project_root / file_path
    if p.exists():
        return p
    
    return None


def convert_detection(det: dict, img_w: int, img_h: int, class_mapping: dict[str, int], 
                     annotations_dir: str) -> str | None:
    # Если детекция прошла через qc_filter.py — конвертируем ТОЛЬКО accepted.
    # Если поля qc_bucket нет вообще (QC не запускался) — конвертируем как раньше, без фильтрации.
    qc_bucket = det.get("qc_bucket")
    if qc_bucket is not None and qc_bucket != "accepted":
        return None

    cls_name = det["class"]
    if cls_name not in class_mapping:
        print(f"[WARN] класс '{cls_name}' отсутствует в classes.json — пропускаю детекцию", file=sys.stderr)
        return None
    class_id = class_mapping[cls_name]

    polygon = None
    mask_path = det.get("mask_path")
    
    # Ищем маску с fallback
    if mask_path:
        mask_file = find_file(mask_path, annotations_dir)
        if mask_file:
            polygon = mask_to_polygon(str(mask_file))

    if polygon is None:
        polygon = bbox_to_polygon(det["bbox"])  # fallback — прямоугольник вместо маски

    normalized = []
    for x, y in polygon:
        normalized.append(min(1.0, max(0.0, x / img_w)))
        normalized.append(min(1.0, max(0.0, y / img_h)))

    coords_str = " ".join(f"{v:.6f}" for v in normalized)
    return f"{class_id} {coords_str}"


def convert(annotations_dir: str, classes_file: str, output_dir: str, split: str = "train"):
    class_mapping = load_class_mapping(classes_file)

    images_out = Path(output_dir) / "images" / split
    labels_out = Path(output_dir) / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    json_files = [
        p for p in Path(annotations_dir).glob("*.json")
        if not p.name.startswith("_") 
    ]

    if not json_files:
        print(f"Не найдено JSON-аннотаций в {annotations_dir}", file=sys.stderr)
        sys.exit(1)

    converted, skipped = 0, 0
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("image")
        width, height = data.get("width"), data.get("height")
        detections = data.get("detections", [])

        if not image_path:
            print(f"[WARN] {json_path.name}: нет пути к картинке, пропуск", file=sys.stderr)
            skipped += 1
            continue

        # Ищем картинку с fallback
        image_file = find_file(image_path, annotations_dir)
        if not image_file:
            print(f"[WARN] {json_path.name}: картинка не найдена ({image_path}), пропуск", file=sys.stderr)
            skipped += 1
            continue

        if not width or not height:
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"[WARN] {json_path.name}: не удалось прочитать картинку, пропуск", file=sys.stderr)
                skipped += 1
                continue
            height, width = img.shape[:2]

        lines = []
        for det in detections:
            line = convert_detection(det, width, height, class_mapping, annotations_dir)
            if line:
                lines.append(line)

        stem = Path(image_path).stem  
        label_path = labels_out / f"{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        image_ext = Path(image_path).suffix
        shutil.copy(str(image_file), images_out / f"{stem}{image_ext}")  

        converted += 1
        print(f"{stem}: {len(lines)} объектов -> {label_path.name}")

    # data.yaml для обучения YOLO
    with open(classes_file, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    names = [c["name"] for c in sorted(classes_data["classes"], key=lambda c: c["index"])]

    data_yaml = {
        "path": str(Path(output_dir).resolve()),
        "train": f"images/{split}",
        "val": f"images/{split}",  # пока один и тот же split — поправить, когда появится реальное разделение
        "nc": len(names),
        "names": names,
    }
    with open(Path(output_dir) / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nГотово: {converted} картинок сконвертировано, {skipped} пропущено.")
    print(f"Датасет: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--classes-file", default="classes.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", help="train/val/test — имя подпапки")
    args = parser.parse_args()

    convert(args.annotations_dir, args.classes_file, args.output_dir, args.split)