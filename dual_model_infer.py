"""
dual_model_infer.py — инференс двумя YOLO-seg моделями сразу (старая + новые классы)
с мёржем результатов в одну разметку.

Имена классов читаются напрямую из весов каждой модели (model.names) —
merged_classes.json / classes.json НЕ нужен.

Пример:
    python dual_model_infer.py \
        --weights-old projects/my/runs/run_1/weights/best.pt \
        --weights-new runs_new_head/welding_v1/weights/best.pt \
        --input test.jpg \
        --output out.jpg \
        --conf 0.25
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_merged_model import build_class_colors


def iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _detections_from_result(result, class_offset: int,
                              class_names: list[str], source: str) -> list[dict]:
    dets = []
    if result.boxes is None:
        return dets
    has_masks = result.masks is not None
    for i, box in enumerate(result.boxes):
        cls_id_local = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det = {
            "class": class_names[cls_id_local] if cls_id_local < len(class_names) else str(cls_id_local),
            "class_id_global": cls_id_local + class_offset,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "source": source,
        }
        if has_masks and i < len(result.masks.data):
            det["_mask_np"] = result.masks.data[i].cpu().numpy()
        dets.append(det)
    return dets


def merge_detections(dets_old: list[dict], dets_new: list[dict],
                      cross_model_dedup: bool,
                      dedup_iou_threshold: float = 0.8) -> list[dict]:
    """
    По умолчанию просто конкатенирует: классы двух моделей не пересекаются,
    настоящих дубликатов нет. cross_model_dedup включается только как страховка.
    """
    if not cross_model_dedup:
        return dets_old + dets_new
    merged = list(dets_old)
    for d in dets_new:
        is_dup = any(
            e["class"] == d["class"] and iou(e["bbox"], d["bbox"]) > dedup_iou_threshold
            for e in dets_old
        )
        if not is_dup:
            merged.append(d)
    return merged


def _draw_detection(img, det: dict, class_colors: dict):
    x1, y1, x2, y2 = det["bbox"]
    color = class_colors.get(det["class"], (150, 150, 150))

    mask_np = det.get("_mask_np")
    if mask_np is not None:
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
        mask_bool = mask_resized > 0.5
        overlay = img.copy()
        overlay[mask_bool] = color
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)

    label = f"{det['class']} {det['confidence']:.2f} [{det['source']}]"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + 4), color, -1)
    cv2.putText(img, label, (x1 + 3, ty + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_on_image(model_old: YOLO, model_new: YOLO,
                  names_old: list[str], names_new: list[str],
                  class_colors: dict,
                  image_path: str, out_path: str,
                  conf: float, cross_model_dedup: bool) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {image_path}")

    t0 = time.time()
    result_old = model_old.predict(img, conf=conf, verbose=False)[0]
    result_new = model_new.predict(img, conf=conf, verbose=False)[0]
    inference_ms = (time.time() - t0) * 1000

    dets_old = _detections_from_result(result_old, 0, names_old, "old_model")
    dets_new = _detections_from_result(result_new, len(names_old), names_new, "new_model")
    detections = merge_detections(dets_old, dets_new, cross_model_dedup)

    for det in detections:
        _draw_detection(img, det, class_colors)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)

    clean_detections = [{k: v for k, v in d.items() if k != "_mask_np"} for d in detections]
    return {
        "detections": clean_detections,
        "inference_ms": round(inference_ms, 1),
        "n_old": len(dets_old),
        "n_new": len(dets_new),
    }


def run_on_video(model_old: YOLO, model_new: YOLO,
                  names_old: list[str], names_new: list[str],
                  class_colors: dict,
                  video_path: str, out_path: str,
                  conf: float, cross_model_dedup: bool,
                  max_frames: int | None = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w += w % 2
    h += h % 2

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_times = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        t0 = time.time()
        result_old = model_old.predict(frame, conf=conf, verbose=False)[0]
        result_new = model_new.predict(frame, conf=conf, verbose=False)[0]
        frame_times.append(time.time() - t0)

        dets_old = _detections_from_result(result_old, 0, names_old, "old_model")
        dets_new = _detections_from_result(result_new, len(names_old), names_new, "new_model")
        for det in merge_detections(dets_old, dets_new, cross_model_dedup):
            _draw_detection(frame, det, class_colors)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Обработано кадров: {frame_idx}")

    cap.release()
    writer.release()

    avg_ms = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0
    return {
        "frames_processed": frame_idx,
        "avg_inference_ms": round(avg_ms, 1),
        "avg_fps": round(1000 / avg_ms, 1) if avg_ms else 0,
        "output_path": out_path,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-old", required=True)
    p.add_argument("--weights-new", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--result-json", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--cross-model-dedup", action="store_true",
                   help="IoU-дедуп между моделями (нужен только при рассинхроне классов)")
    args = p.parse_args()

    model_old = YOLO(args.weights_old)
    model_new = YOLO(args.weights_new)

    # Имена классов берём прямо из весов — никакого JSON
    names_old = [model_old.names[i] for i in range(len(model_old.names))]
    names_new = [model_new.names[i] for i in range(len(model_new.names))]
    all_names = names_old + names_new
    class_colors = build_class_colors(all_names)

    print(f"[dual] старая модель: {names_old}")
    print(f"[dual] новая модель:  {names_new}")

    suffix = Path(args.input).suffix.lower()
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        result = run_on_video(model_old, model_new, names_old, names_new, class_colors,
                               args.input, args.output, args.conf,
                               args.cross_model_dedup, args.max_frames)
    else:
        result = run_on_image(model_old, model_new, names_old, names_new, class_colors,
                               args.input, args.output, args.conf, args.cross_model_dedup)

    print(json.dumps(result, ensure_ascii=False))
    if args.result_json:
        Path(args.result_json).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()