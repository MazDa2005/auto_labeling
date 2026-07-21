"""
SAM3 в режиме REFINE.
Для каждой картинки читает detections из --boxes-dir (по имени файла), уточняет их
до сегментационной маски, bbox пересчитывается из маски. Маска сохраняется как PNG.

Запускается в conda-окружении `sam3`.

"""
import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from pipeline_utils import list_images, mask_to_bbox, save_mask_png, iou


def process_image(predictor, image_path: Path, input_detections: list[dict], masks_dir: str) -> list[dict]:
    predictor.set_image(str(image_path))

    refined = []
    for i, det in enumerate(input_detections):
        bbox = det["bbox"]
        results = predictor(bboxes=[bbox])

        r = results[0] if results else None
        new_bbox = bbox
        mask_area = None
        mask_path = None
        confidence = det.get("confidence")

        if r is not None and r.masks is not None and len(r.masks) > 0:
            mask = r.masks.data[0].cpu().numpy()
            candidate = mask_to_bbox(mask)
            if candidate is not None:
                new_bbox = candidate
                mask_area = float((mask > 0.5).sum())

            mask_path = str(Path(masks_dir) / f"det_{i:03d}_{det['class']}.png")
            save_mask_png(mask, mask_path)

        # проверка, что sam3 сегментировал нужный объект
        refine_iou = iou(bbox, new_bbox)

        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            try:
                confidence = float(r.boxes.conf[0])
            except (IndexError, AttributeError, TypeError):
                pass

        refined.append({
            "class": det["class"],
            "bbox": new_bbox,
            "confidence": confidence,
            "source": f"{det.get('source', 'unknown')}+sam3_refine",
            "mask_area": mask_area,
            "mask_path": mask_path,
            "refine_iou": refine_iou,
        })

    return refined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--boxes-dir", required=True, help="Папка с JSON detections от предыдущего этапа")
    parser.add_argument("--model", default="sam3.pt")
    parser.add_argument("--masks-root", default="masks")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[sam3] загрузка модели (один раз на весь батч из {len(images)} картинок)...")
    print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("[WARN] SAM3 работает на CPU — будет очень медленно", file=sys.stderr)

    overrides = dict(
        conf=0.4, task="segment", mode="predict", model=args.model,
        half=torch.cuda.is_available(), save=False, verbose=False, device=device,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("[sam3] модель загружена, начинаю обработку")

    for i, image_path in enumerate(images, 1):
        boxes_path = Path(args.boxes_dir) / f"{image_path.stem}.json"
        out_path = Path(args.out_dir) / f"{image_path.stem}.json"
        masks_dir = str(Path(args.masks_root) / image_path.stem)

        if not boxes_path.exists():
            print(f"[{i}/{len(images)}] {image_path.name}: нет входных боксов, пропуск", file=sys.stderr)
            continue

        with open(boxes_path, "r", encoding="utf-8") as f:
            data = json.load(f)  
            input_detections = data.get("detections", [])
    
        try:
            refined = process_image(predictor, image_path, input_detections, masks_dir)
            data["detections"] = refined  
        
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[{i}/{len(images)}] {image_path.name}: {len(refined)} детекций уточнено")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "detections": []}, f, indent=2)
            print(f"[{i}/{len(images)}] {image_path.name}: ОШИБКА {e}", file=sys.stderr)

    print(f"[sam3] готово, результаты в {args.out_dir}")
