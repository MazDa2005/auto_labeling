"""
CLIP-верификация через open_clip.

CLIP сравнивает кроп СО ВСЕМИ классами сразу и смотрит, совпадает ли его "лучшая догадка"
с тем, что предсказал детектор. Если не совпадает — это содержательный сигнал несогласия.

ВАЖНО — эта версия ОБОГАЩАЕТ каждую детекцию полями:
    clip_agrees          — bool, совпадает ли лучшая догадка CLIP с классом от детектора
    clip_predicted_class — какой класс CLIP считает наиболее вероятным
    clip_predicted_score — score этого класса (после softmax по классам)
    clip_own_class_score — score именно того класса, что предсказал детектор
Решение "отбросить / на проверку / принять" принимает qc_filter.py.
"""

import argparse
import json
import sys
from pathlib import Path
import os
from transformers import AutoTokenizer
import torch
from PIL import Image
import open_clip

from pipeline_utils import list_images, get_all_prompts

# Fallback-промпты — используются, если в classes.json нет поля 'prompts'
FALLBACK_PROMPTS = {
    "person": ["a photo of a person", "a worker", "a human figure", "someone standing"],
    "helmet": ["a safety helmet", "a hard hat", "a protective helmet"],
    "gloves": ["work gloves", "safety gloves", "gloves on hands"],
    "welding_gloves": ["welding gloves", "thick leather welding gloves"],
    "welding": ["a welding area", "a welding zone", "welding equipment", "welding sparks"],
    "orange_vest": ["an orange safety vest", "a high-visibility vest"],
    "gas_mask": ["a gas mask", "a respirator mask"],
    "welding_mask": ["a welding mask", "a welding helmet with face shield"],
    "mask": ["a face mask", "a protective mask"],
    "glasses": ["safety glasses", "protective goggles"],
    "protective_headphones": ["protective headphones", "ear defenders"],
    "railcar": ["a railway freight car", "a railroad car on tracks", "a metal freight wagon"],
}


def load_prompts(classes_file: str) -> dict[str, list[str]]:
    """Загружает промпты из classes.json с fallback."""
    try:
        prompts = get_all_prompts(classes_file)
        if prompts:
            has_real_prompts = any(len(v) > 0 for v in prompts.values())
            if has_real_prompts:
                for cls in list(prompts.keys()):
                    if not prompts[cls]:
                        prompts[cls] = FALLBACK_PROMPTS.get(
                            cls, [f"a photo of {cls.replace('_', ' ')}"]
                        )
                return prompts
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] не удалось прочитать промпты из {classes_file}: {e}. "
              f"Использую fallback.", file=sys.stderr)

    return dict(FALLBACK_PROMPTS)


def build_prompt_index(all_prompts: dict[str, list[str]]) -> tuple[list[str], dict[int, str], list[str]]:
    """Собирает плоский список текстов и обратное сопоставление индекс->класс."""
    texts = []
    idx_to_class = {}
    for cls, prompts in all_prompts.items():
        for p in prompts:
            idx_to_class[len(texts)] = cls
            texts.append(p)
    all_classes = sorted(set(all_prompts.keys()))
    return texts, idx_to_class, all_classes

@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Кодирует все тексты один раз."""
    # HuggingFace tokenizer возвращает BatchEncoding, нам нужен именно тензор input_ids
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    # Передаём в open_clip только тензор с ID токенов
    features = model.encode_text(tokens["input_ids"])
    features = features / features.norm(dim=-1, keepdim=True)
    return features

@torch.no_grad()
def verify_batch(
    model,
    text_features: torch.Tensor,
    idx_to_class: dict[int, str],
    all_classes: list[str],
    device: str,
    crops: list[Image.Image],
    preprocess_fn,
    predicted_classes: list[str],
    batch_size: int = 32,
) -> list[dict]:
    """Обрабатывает батч кропов. Softmax ПОСЛЕ агрегации по классам."""
    if not crops:
        return []

    results = []

    for start in range(0, len(crops), batch_size):
        end = min(start + batch_size, len(crops))
        batch_crops = crops[start:end]
        batch_preds = predicted_classes[start:end]

        batch_tensor = torch.stack([preprocess_fn(c) for c in batch_crops]).to(device)

        image_features = model.encode_image(batch_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

        for k in range(similarity.shape[0]):
            sim_k = similarity[k]

            class_scores: dict[str, float] = {}
            for t_idx, score in enumerate(sim_k.tolist()):
                cls = idx_to_class[t_idx]
                class_scores[cls] = max(class_scores.get(cls, -float("inf")), score)

            scores_tensor = torch.tensor(
                [class_scores[cls] for cls in all_classes], dtype=torch.float32
            )
            probs = torch.softmax(scores_tensor, dim=0).tolist()
            class_probs = dict(zip(all_classes, probs))

            best_class = max(class_probs, key=class_probs.get)
            best_score = class_probs[best_class]
            own_score = class_probs.get(batch_preds[k], 0.0)

            results.append({
                "clip_agrees": best_class == batch_preds[k],
                "clip_predicted_class": best_class,
                "clip_predicted_score": best_score,
                "clip_own_class_score": own_score,
            })

    return results


def process_image(
    model,
    preprocess_fn,
    text_features: torch.Tensor,
    idx_to_class: dict[int, str],
    all_classes: list[str],
    device: str,
    image_path: Path,
    input_detections: list[dict],
    batch_size: int = 32,
    context_scale: float = 2.0,
) -> list[dict]:
    """Обрабатывает одну картинку, вырезая кропы с расширенным контекстом."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    valid_crops = []
    valid_indices = []
    enriched = [dict(det) for det in input_detections]

    for i, det in enumerate(input_detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        w, h = x2 - x1, y2 - y1

        if w < 2 or h < 2:
            continue

        # РАСШИРЕНИЕ ОКНА (CONTEXT SCALE) 
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        new_w = w * context_scale
        new_h = h * context_scale
        
        new_x1 = max(0, int(cx - new_w / 2.0))
        new_y1 = max(0, int(cy - new_h / 2.0))
        new_x2 = min(img_w, int(cx + new_w / 2.0))
        new_y2 = min(img_h, int(cy + new_h / 2.0))
        
        crop = img.crop((new_x1, new_y1, new_x2, new_y2))

        valid_crops.append(crop)
        valid_indices.append(i)

    if not valid_crops:
        return enriched

    predicted_classes = [input_detections[i]["class"] for i in valid_indices]
    clip_results = verify_batch(
        model, text_features, idx_to_class, all_classes, device,
        valid_crops, preprocess_fn, predicted_classes, batch_size=batch_size,
    )

    for idx, clip_res in zip(valid_indices, clip_results):
        enriched[idx].update(clip_res)

    return enriched

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--boxes-dir", required=True)
    parser.add_argument("--model", default="ViT-SO400M-14-SigLIP")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--classes-file", default="classes.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Не используется (решение принимает qc_filter.py)")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[clip] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"[clip] загрузка модели {args.model}/{args.pretrained}...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()

    tokenizer_dir = os.path.dirname(args.pretrained) if os.path.isfile(args.pretrained) else args.pretrained
    print(f"[clip] загрузка токенизатора из локальной папки: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    all_prompts = load_prompts(args.classes_file)

    texts, idx_to_class, all_classes = build_prompt_index(all_prompts)
    text_features = encode_texts(model, tokenizer, texts, device)

    for i, image_path in enumerate(images, 1):
        boxes_path = Path(args.boxes_dir) / f"{image_path.stem}.json"
        out_path = Path(args.out_dir) / f"{image_path.stem}.json"

        if not boxes_path.exists():
            print(f"[{i}/{len(images)}] {image_path.name}: нет входных боксов, пропуск",
                  file=sys.stderr)
            continue

        with open(boxes_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Читаем весь словарь целиком
            input_detections = data.get("detections", [])

        try:
            enriched = process_image(
                model, preprocess, text_features, idx_to_class, all_classes, device,
                image_path, input_detections, batch_size=args.batch_size,
            )
            data['detections'] = enriched
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            n_disagree = sum(1 for d in enriched if d.get("clip_agrees") is False)
            print(f"[{i}/{len(images)}] {image_path.name}: {len(enriched)} детекций, "
                  f"{n_disagree} несогласий CLIP")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "detections": []}, f, indent=2)
            print(f"[{i}/{len(images)}] {image_path.name}: ОШИБКА {e}", file=sys.stderr)

    print(f"[clip] готово, результаты в {args.out_dir}")
