"""
LocateAnything в роли DETECT.

Запускается в conda-окружении `nvidia`.
ПАКЕТНЫЙ режим: модель грузится ОДИН РАЗ, затем обрабатывает все картинки в --images-dir.
Для каждой картинки пишет отдельный JSON в --out-dir (по имени файла картинки).

Промпты берутся из classes.json (единый источник правды). Если у класса нет поля 'prompts' —
используется fallback: имя класса с заменой '_' на пробелы.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"

import torch
from PIL import Image

from locate_anything_worker import LocateAnythingWorker
from pipeline_utils import list_images, get_primary_prompts, build_reverse_prompt_lookup


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


def process_image(worker, image_path: Path, classes: list[str], classes_file: str) -> list[dict]:
    """Обрабатывает изображения"""
    img = Image.open(image_path).convert("RGB")

    prompts = load_prompts_with_fallback(classes, classes_file)
    prompt_classes = [prompts.get(c, c) for c in classes]

    result = worker.detect(img, prompt_classes)

    boxes = LocateAnythingWorker.parse_boxes_with_refs(result["answer"], *img.size)

    label_lookup = build_label_lookup(classes, classes_file)

    detections = []
    for b in boxes:
        raw_label = b["class"].lower().strip()
        canonical = label_lookup.get(raw_label)

        if canonical is None:
            canonical = next(
                (c for phrase, c in label_lookup.items()
                 if phrase in raw_label or raw_label in phrase),
                b["class"]
            )

        detections.append({
            "class": canonical,
            "bbox": [b["x1"], b["y1"], b["x2"], b["y2"]],
            "confidence": None,
            "source": "locate_anything",
        })

    print(f"Итоговых детекций: {len(detections)}")

    torch.cuda.empty_cache()
    gc.collect()
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes", required=True, help="comma-separated, e.g. person,helmet,gloves")
    parser.add_argument("--classes-file", default="classes.json",
                        help="Файл с классами и промптами (единый источник правды)")
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    images = list_images(args.images_dir)

    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[locate_anything] загрузка модели...")
    if not torch.cuda.is_available():
        print("[WARN] LocateAnything работает на CPU — будет очень медленно", file=sys.stderr)

    worker = LocateAnythingWorker(args.model)
    print("[locate_anything] модель загружена, начинаю обработку")

    for i, image_path in enumerate(images, 1):
        out_path = Path(args.out_dir) / f"{image_path.stem}.json"
        try:
            detections = process_image(worker, image_path, classes, args.classes_file)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"detections": detections}, f, indent=2, ensure_ascii=False)
            print(f"[{i}/{len(images)}] {image_path.name}: {len(detections)} детекций")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "detections": []}, f, indent=2)
            print(f"[{i}/{len(images)}] {image_path.name}: ОШИБКА {e}", file=sys.stderr)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"[locate_anything] готово, результаты в {args.out_dir}")
