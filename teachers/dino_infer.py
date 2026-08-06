"""
Grounding DINO в роли DETECT.

Формат текстового запроса Grounding DINO: промпты из classes.json через точку с пробелом,
например "a person. a safety helmet. work gloves." — модель сама возвращает, какой класс
к какому боксу относится.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from pipeline_utils import list_images, get_primary_prompts, build_reverse_prompt_lookup,iou

def dedup_overlapping(detections: list[dict], iou_threshold: float = 0.6) -> list[dict]:
    """Убирает дубликаты боксов одного класса"""
    detections = sorted(detections, key=lambda d: d.get("confidence") or 0, reverse=True)
    kept = []
    for d in detections:
        if not any(d["class"] == k["class"] and iou(d["bbox"], k["bbox"]) > iou_threshold for k in kept):
            kept.append(d)
    return kept

def load_prompts_with_fallback(classes: list[str], classes_file: str) -> dict[str, str]:
    """
    Загружает промпты из classes.json. Если у класса нет поля 'prompts' —
    использует fallback: имя класса с заменой '_' на пробелы.
    """
    try:
        all_prompts = get_primary_prompts(classes_file)
        if all_prompts:
            has_real_prompts = any(v for v in all_prompts.values())
            if has_real_prompts:
                for cls in classes:
                    if cls not in all_prompts or not all_prompts[cls]:
                        all_prompts[cls] = cls.replace("_", " ")
                return all_prompts
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] не удалось прочитать промпты из {classes_file}: {e}. "
              f"Использую fallback.", file=sys.stderr)

    return {cls: cls.replace("_", " ") for cls in classes}


def build_query(classes: list[str], classes_file: str) -> str:
    """
    Строит текстовый запрос для Grounding DINO из промптов в classes.json.
    """
    prompts = load_prompts_with_fallback(classes, classes_file)
    selected = [prompts[c] for c in classes if c in prompts]
    return ". ".join(selected) + "."


def build_label_lookup(classes: list[str], classes_file: str) -> dict[str, str]:
    """
    Обратное сопоставление: промпт (lowercase) -> каноническое имя класса.
    """
    try:
        lookup = build_reverse_prompt_lookup(classes_file)
        if lookup:
            return lookup
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        pass

    return {c.lower().strip().replace("_", " "): c for c in classes}


def process_image(model, processor, device, image_path: Path, classes: list[str],
                   classes_file: str,
                   box_threshold: float = 0.45, text_threshold: float = 0.35) -> list[dict]:
    """Обрабатывает изображения"""
    img = Image.open(image_path).convert("RGB")
    query = build_query(classes, classes_file)
    label_lookup = build_label_lookup(classes, classes_file)

    inputs = processor(images=img, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        raw_label = label.strip().lower()
        canonical = label_lookup.get(raw_label)
        if canonical is None:
            canonical = next((c for phrase, c in label_lookup.items()
                              if phrase in raw_label or raw_label in phrase), raw_label)
        detections.append({
            "class": canonical,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(score),
            "source": "grounding_dino",
        })
    detections = dedup_overlapping(detections)
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes", required=True, help="comma-separated, e.g. person,helmet,gloves")
    parser.add_argument("--classes-file", default="classes.json",
                        help="Файл с классами и промптами (единый источник правды)")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--text-threshold", type=float, default=0.35)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    images = list_images(args.images_dir)

    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grounding_dino] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"[grounding_dino] загрузка модели ...")

    processor = AutoProcessor.from_pretrained(args.model,use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    print("[grounding_dino] модель загружена, начинаю обработку")

    for i, image_path in enumerate(images, 1):
        out_path = Path(args.out_dir) / f"{image_path.stem}.json"
        try:
            detections = process_image(
                model, processor, device, image_path, classes, args.classes_file,
                args.box_threshold, args.text_threshold,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"detections": detections}, f, indent=2, ensure_ascii=False)
            print(f"[{i}/{len(images)}] {image_path.name}: {len(detections)} детекций")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "detections": []}, f, indent=2)
            print(f"[{i}/{len(images)}] {image_path.name}: ОШИБКА {e}", file=sys.stderr)

    print(f"[grounding_dino] готово, результаты в {args.out_dir}")
