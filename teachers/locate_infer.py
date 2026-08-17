"""
LocateAnything в роли DETECT (с chunking классов для экономии VRAM).
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
from pipeline_utils import list_images, get_primary_prompts, build_reverse_prompt_lookup, merge_new_detections


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Разбивает список на чанки заданного размера."""
    if chunk_size <= 0:
        return [lst]
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_prompts_with_fallback(classes: list[str], classes_file: str) -> dict[str, str]:
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
        print(f"[WARN] не удалось прочитать промпты: {e}. Использую fallback.", file=sys.stderr)

    return {cls: cls.replace("_", " ") for cls in classes}


def build_label_lookup(classes: list[str], classes_file: str) -> dict[str, str]:
    try:
        lookup = build_reverse_prompt_lookup(classes_file)
        if lookup:
            return lookup
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        pass
    return {c.lower().strip().replace("_", " "): c for c in classes}


def process_image_chunk(
    worker, 
    image: Image.Image, 
    classes_chunk: list[str], 
    classes_file: str,
    img_width: int,
    img_height: int,
) -> list[dict]:
    """Обрабатывает изображение для одного чанка классов."""
    prompts = load_prompts_with_fallback(classes_chunk, classes_file)
    prompt_classes = [prompts.get(c, c) for c in classes_chunk]

    result = worker.detect(image, prompt_classes)
    boxes = LocateAnythingWorker.parse_boxes_with_refs(result["answer"], img_width, img_height)

    label_lookup = build_label_lookup(classes_chunk, classes_file)

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

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes", required=True, help="comma-separated")
    parser.add_argument("--classes-file", default="classes.json")
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=15,  # 🛡️ НОВОЕ
                        help="Максимум классов за один промпт (экономит VRAM)")
    parser.add_argument("--max-image-size", type=int, default=1024,  # 🛡️ НОВОЕ
                        help="Максимальный размер изображения (px)")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    images = list_images(args.images_dir)

    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 🛡️ Разбиваем классы на чанки
    class_chunks = chunk_list(classes, args.chunk_size)
    if len(class_chunks) > 1:
        print(f"🔀 Разбиваю {len(classes)} классов на {len(class_chunks)} чанков по {args.chunk_size}")

    print(f"[locate_anything] загрузка модели (max_image_size={args.max_image_size})...")
    if not torch.cuda.is_available():
        print("[WARN] LocateAnything работает на CPU — будет очень медленно", file=sys.stderr)

    # 🛡️ Передаём ограничение размера изображения в worker
    worker = LocateAnythingWorker(
        args.model,
        max_image_size=args.max_image_size,
    )
    print("[locate_anything] модель загружена, начинаю обработку")

    for i, image_path in enumerate(images, 1):
        out_path = Path(args.out_dir) / f"{image_path.stem}.json"
        try:
            img = Image.open(image_path).convert("RGB")
            img_width, img_height = img.size
            
            all_detections = []
            
            # 🛡️ Обрабатываем каждый чанк классов отдельно
            for chunk_idx, class_chunk in enumerate(class_chunks):
                if len(class_chunks) > 1:
                    print(f"  Чанк {chunk_idx + 1}/{len(class_chunks)}: {len(class_chunk)} классов")
                
                chunk_dets = process_image_chunk(
                    worker, img, class_chunk, args.classes_file,
                    img_width, img_height,
                )
                all_detections = merge_new_detections(all_detections, chunk_dets)
                
                # 🛡️ Очистка памяти после каждого чанка
                torch.cuda.empty_cache()
                gc.collect()
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"detections": all_detections}, f, indent=2, ensure_ascii=False)
            
            print(f"[{i}/{len(images)}] {image_path.name}: {len(all_detections)} детекций")
            
            img.close()
            
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "detections": []}, f, indent=2)
            print(f"[{i}/{len(images)}] {image_path.name}: ОШИБКА {e}", file=sys.stderr)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"[locate_anything] готово, результаты в {args.out_dir}")
