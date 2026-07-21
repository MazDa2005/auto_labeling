"""
QC filter — intermediate step between batch_label.py and convert_to_yolo_seg.py.

Splits IMAGES into two groups:
  - clean/   -> all detections accepted/rejected
  - review/  -> at least one needs_review

Each group has its own masks/ subfolder with COPIES of the masks.

Box colors:
  green  = accepted
  orange  = needs_review
  red  = rejected
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from teachers.pipeline_utils import iou

# ─────────────────────────────────────────────────────────────────────

MIN_CONFIDENCE = 0.5
MIN_BBOX_AREA_REL = 0.000025
LOW_REFINE_IOU = 0.6
CONFLICT_IOU_THRESHOLD = 0.7
DUPLICATE_IOU = 0.8

PERSON_CONTEXT_CONF_THRESHOLD = 0.8

COMPATIBLE_WITH_PERSON = {
    "helmet", "orange_vest", "gloves", "welding_gloves", "gas_mask",
    "mask", "welding_mask", "glasses", "protective_headphones",
}

MUTUALLY_EXCLUSIVE_GROUPS = [
    {"gloves", "welding_gloves"},
    {"mask", "gas_mask", "welding_mask"},
]

_BUCKET_PRIORITY = {"rejected": 2, "needs_review": 1, "accepted": 0}

CLASS_COLORS = {
    "person":              (200, 80, 80),
    "helmet":              (0, 180, 100),
    "orange_vest":         (255, 140, 0),
    "gas_mask":            (180, 150, 30),
    "mask":                (220, 80, 220),
    "welding_mask":        (170, 120, 20),
    "gloves":              (120, 30, 60),
    "welding_gloves":      (30, 100, 10),
    "welding":             (180, 50, 100),
    "glasses":             (80, 220, 200),
    "protective_headphones": (100, 200, 140),
    "railcar":             (100, 100, 100),
}

BUCKET_OUTLINE = {
    "accepted":     (0, 200, 0),
    "needs_review": (255, 140, 0),
    "rejected":     (200, 0, 0),
}
# ───────────────────────────────────────────────────────────────────────────────

def is_expected_overlap(class_a: str, class_b: str) -> bool:
    if class_a == class_b: return True
    if class_a == "person" and class_b in COMPATIBLE_WITH_PERSON: return True
    if class_b == "person" and class_a in COMPATIBLE_WITH_PERSON: return True
    return False

def is_exclusive_conflict(class_a: str, class_b: str) -> bool:
    for group in MUTUALLY_EXCLUSIVE_GROUPS:
        if class_a in group and class_b in group and class_a != class_b:
            return True
    return False

def bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def check_detection_level(det: dict, img_w: int, img_h: int) -> tuple[str, str] | None:
    conf = det.get("confidence")
    if conf is not None and conf < MIN_CONFIDENCE:
        return ("rejected", f"low confidence ({conf:.2f} < {MIN_CONFIDENCE})")

    img_area = img_w * img_h
    rel_area = bbox_area(det["bbox"]) / img_area if img_area > 0 else 0
    if rel_area < MIN_BBOX_AREA_REL:
        return ("rejected", f"degenerate bbox (rel.area={rel_area*100:.4f}%)")

    refine_iou = det.get("refine_iou")
    if refine_iou is not None and refine_iou < LOW_REFINE_IOU:
        return ("needs_review", f"detect/refine diverged (IoU={refine_iou:.2f})")

    # CLIP-проверка (если есть)
    if det.get("clip_agrees") is False:
        clip_class = det.get("clip_predicted_class")
        clip_score = det.get("clip_predicted_score") or 0.0
        own_score = det.get("clip_own_class_score") or 0.0
        if clip_score >= 0.25 and (clip_score - own_score) > 0.15:
            return ("needs_review",
                    f"CLIP disagrees: '{clip_class}' ({clip_score:.2f}) vs '{det['class']}' ({own_score:.2f})")

    return None

def check_pairwise_conflicts(detections: list[dict]) -> dict[int, tuple[str, str]]:
    flagged: dict[int, tuple[str, str]] = {}
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            d1, d2 = detections[i], detections[j]
            box_iou = iou(d1["bbox"], d2["bbox"])
            if box_iou < CONFLICT_IOU_THRESHOLD: continue

            if d1["class"] == d2["class"] and box_iou > DUPLICATE_IOU:
                c1, c2 = d1.get("confidence") or 0.0, d2.get("confidence") or 0.0
                loser = i if c1 <= c2 else j
                reason = f"duplicate '{d1['class']}' (IoU={box_iou:.2f})"
                prev = flagged.get(loser)
                if prev is None or _BUCKET_PRIORITY["rejected"] > _BUCKET_PRIORITY[prev[0]]:
                    flagged[loser] = ("rejected", reason)
                continue

            if is_expected_overlap(d1["class"], d2["class"]): continue

            if is_exclusive_conflict(d1["class"], d2["class"]):
                reason = f"conflict '{d1['class']}' vs '{d2['class']}' (IoU={box_iou:.2f})"
            else:
                reason = f"overlap '{d1['class']}'/'{d2['class']}' (IoU={box_iou:.2f})"

            for idx in (i, j):
                prev = flagged.get(idx)
                if prev is None or _BUCKET_PRIORITY["needs_review"] > _BUCKET_PRIORITY[prev[0]]:
                    flagged[idx] = ("needs_review", reason)
    return flagged

def check_person_context(detections: list[dict]) -> dict[int, str]:
    flagged = {}
    all_classes = {d["class"] for d in detections}
    has_ppe = bool(all_classes & COMPATIBLE_WITH_PERSON)
    if has_ppe: return flagged

    for idx, det in enumerate(detections):
        if det["class"] == "person" and (det.get("confidence") or 0) >= PERSON_CONTEXT_CONF_THRESHOLD:
            flagged[idx] = f"person without PPE on image (conf={det['confidence']:.2f})"
    return flagged

def classify_detections(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    pairwise_flags = check_pairwise_conflicts(detections)
    context_flags = check_person_context(detections)
    result = []

    for idx, det in enumerate(detections):
        det = dict(det)
        single = check_detection_level(det, img_w, img_h)
        pair = pairwise_flags.get(idx)
        ctx = context_flags.get(idx)

        candidates = [c for c in (single, pair, (ctx and ("needs_review", ctx))) if c is not None]
        if not candidates:
            det["qc_bucket"] = "accepted"
            det["qc_reason"] = None
        else:
            bucket, reason = max(candidates, key=lambda c: _BUCKET_PRIORITY[c[0]])
            det["qc_bucket"] = bucket
            det["qc_reason"] = reason

        result.append(det)
    return result


def _copy_masks_for_image(detections: list[dict], image_stem: str, target_masks_dir: Path) -> list[dict]:
    result = []
    for det in detections:
        det = dict(det)
        old_mask_path = det.get("mask_path")
        if old_mask_path and Path(old_mask_path).exists():
            old_name = Path(old_mask_path).name
            new_name = f"{image_stem}_{old_name}"
            new_path = target_masks_dir / new_name
            shutil.copy(old_mask_path, new_path)
            det["mask_path"] = str(new_path.resolve())
        result.append(det)
    return result


def load_class_colors(classes_file: str) -> dict[str, tuple[int, int, int]]:
    """Загружает цвета классов из classes.json и конвертирует #RRGGBB -> (R, G, B)."""
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    colors = {}
    for c in data["classes"]:
        hex_color = c.get("color", "#808080").lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            colors[c["name"]] = (r, g, b)
    return colors


def draw_qc_annotated(image_path: str, detections: list[dict], out_path: str,
                      class_colors: dict[str, tuple[int, int, int]] = None):
    """
    Рисует маски (цвет из classes.json) + боксы (цвет по статусу QC) + номера детекций.
    """
    if class_colors is None:
        class_colors = {}
    default_color = (150, 150, 150)

    img = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # ── 1. Полупрозрачные МАСКИ (цвет из classes.json) ──
    for det in detections:
        mask_path = det.get("mask_path")
        if mask_path and Path(mask_path).exists():
            cls = det["class"]
            rgb = class_colors.get(cls, default_color)
            mask_img = Image.open(mask_path).convert("L").resize(img.size)
            mask_array = np.array(mask_img) > 127
            colored = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
            colored[mask_array] = (*rgb, 70)  # альфа 70 — полупрозрачность
            overlay = Image.alpha_composite(overlay, Image.fromarray(colored, mode="RGBA"))

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── 2. Боксы, номера и подписи ──
    for idx, det in enumerate(detections):
        bucket = det["qc_bucket"]
        cls = det["class"]
        x1, y1, x2, y2 = det["bbox"]

        # Цвет рамки — по статусу QC
        outline_color = BUCKET_OUTLINE[bucket]
        # Цвет подписи/фона — по классу (из classes.json)
        class_color = class_colors.get(cls, default_color)

        # Рамка
        width = 4 if bucket == "needs_review" else (2 if bucket == "accepted" else 1)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)

        # Подпись: [номер] класс conf | причина
        label = f"[{idx}] {cls}"
        if det.get("confidence") is not None:
            label += f" {det['confidence']:.2f}"
        if bucket != "accepted" and det.get("qc_reason"):
            reason = det["qc_reason"]
            if len(reason) > 40:
                reason = reason[:37] + "..."
            label += f" | {reason}"

        # Размер текста
        try:
            bbox = draw.textbbox((0, 0), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(label) * 7, 12

        # Фон подписи — затемнённый цвет класса
        bg_color = tuple(max(0, c - 60) for c in class_color)
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=bg_color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255))

        # Номер в кружке
        num_label = str(idx)
        try:
            nb = draw.textbbox((0, 0), num_label)
            nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        except AttributeError:
            nw, nh = 8, 12
        draw.ellipse(
            [x1 + 2, y1 + 2, x1 + nw + 8, y1 + nh + 6],
            fill=outline_color,
        )
        draw.text((x1 + 5, y1 + 3), num_label, fill=(255, 255, 255))

    img.save(out_path)


def get_image_size(data: dict) -> tuple[int, int]:
    w, h = data.get("width"), data.get("height")
    if w and h: return int(w), int(h)
    image_path = data.get("image")
    if image_path and Path(image_path).exists():
        with Image.open(image_path) as im: return im.size
    raise ValueError(f"Cannot determine image size: {image_path}")


def process_annotations(annotations_dir: str, output_dir: str, 
                        classes_file: str = "classes.json",
                        masks_root_to_delete: str = None):
    # Загружаем цвета классов из classes.json
    class_colors = load_class_colors(classes_file)
    print(f"[qc] загружено {len(class_colors)} цветов классов из {classes_file}")
    
    out = Path(output_dir)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)

    json_files = [p for p in Path(annotations_dir).glob("*.json") if not p.name.startswith("_")]
    if not json_files:
        print(f"No JSON files found in {annotations_dir}", file=sys.stderr)
        sys.exit(1)

    totals = {"clean": 0, "review": 0, "skipped": 0}
    bucket_totals = {"accepted": 0, "needs_review": 0, "rejected": 0}
    masks_copied = 0
    summary = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            image_path = data.get("image")
            if not image_path:
                print(f"[WARN] {json_path.name}: нет пути к картинке")
                totals["skipped"] += 1
                continue

            image_file = Path(image_path)
            if not image_file.exists():
                alt_path = Path(annotations_dir) / image_path
                if alt_path.exists():
                    image_file = alt_path
                else:
                    alt_path = Path.cwd() / image_path
                    if alt_path.exists():
                        image_file = alt_path
                    else:
                        print(f"[WARN] {json_path.name}: image not found ({image_path})")
                        totals["skipped"] += 1
                        continue

        try:
            img_w, img_h = get_image_size(data)
        except ValueError as e:
            print(f"[WARN] {json_path.name}: {e}")
            totals["skipped"] += 1
            continue

        detections = classify_detections(data.get("detections", []) or [], img_w, img_h)
        for det in detections:
            bucket_totals[det["qc_bucket"]] += 1

        has_review = any(d["qc_bucket"] in ("needs_review", "rejected") for d in detections)
        target = out / ("review" if has_review else "clean")
        totals["review" if has_review else "clean"] += 1

        target_masks_dir = target / "masks"
        target_masks_dir.mkdir(parents=True, exist_ok=True)
        detections_with_masks = _copy_masks_for_image(detections, json_path.stem, target_masks_dir)
        masks_copied += sum(1 for d in detections_with_masks if d.get("mask_path"))

        out_data = dict(data)
        out_data["detections"] = detections_with_masks
        with open(target / json_path.name, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        draw_qc_annotated(str(image_file), detections_with_masks, str(target / f"{json_path.stem}_annotated.jpg"),class_colors=class_colors)
        s = {
            "image": json_path.stem,
            "status": "review" if has_review else "clean",
            "accepted": sum(1 for d in detections if d["qc_bucket"] == "accepted"),
            "needs_review": sum(1 for d in detections if d["qc_bucket"] == "needs_review"),
            "rejected": sum(1 for d in detections if d["qc_bucket"] == "rejected"),
        }
        summary.append(s)
        print(f"{json_path.stem}: {'REVIEW' if has_review else 'clean'} "
              f"(a={s['accepted']}, r={s['needs_review']}, x={s['rejected']})")

    with open(out / "_qc_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "images": totals,
            "detections": bucket_totals,
            "masks_copied": masks_copied,
            "per_image": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nTotal: clean={totals['clean']}, review={totals['review']}, skipped={totals['skipped']}")
    print(f"Masks copied: {masks_copied}")

    masks_root = Path(annotations_dir).parent / "masks"
    if masks_root.exists() and masks_root.is_dir():
        shutil.rmtree(masks_root)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--classes-file", default="classes.json", 
                   help="Путь к classes.json для загрузки цветов")
    p.add_argument("--masks-root", default=None,
                   help="Общая папка масок для удаления после фильтрации")
    args = p.parse_args()
    process_annotations(args.annotations_dir, args.output_dir, 
                        args.classes_file, args.masks_root)
