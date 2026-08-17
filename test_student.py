"""
test_student.py — инференс обученной student-модели на картинке или видео.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import colorsys
import yaml

def _draw_mask(img, mask_array, color, alpha=70):
    """Полупрозрачная маска поверх картинки (аналогично qc_filter.py)."""
    colored = np.zeros_like(img)
    colored[mask_array] = color
    blended = cv2.addWeighted(colored, alpha / 255, img, 1 - alpha / 255, 0)
    img[mask_array] = blended[mask_array]

def _parse_color(value):
    """
    Поддерживает:
      "#RRGGBB"
      "RRGGBB"
      [R, G, B]
      (R, G, B)

    Возвращает BGR для OpenCV.
    """
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip().lstrip("#")

        if len(s) == 6:
            try:
                r = int(s[0:2], 16)
                g = int(s[2:4], 16)
                b = int(s[4:6], 16)
                return (b, g, r)
            except ValueError:
                return None

        return None

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            r = int(value[0])
            g = int(value[1])
            b = int(value[2])
            return (b, g, r)
        except (TypeError, ValueError):
            return None

    return None


def _default_bgr(class_id: int) -> tuple[int, int, int]:
    """
    Генерирует детерминированный цвет для класса,
    если цвета нет в data.yaml.
    """
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)

    return int(b * 255), int(g * 255), int(r * 255)


def load_class_colors(data_yaml: str, class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """
    Читает цвета из data.yaml.

    Ожидаемый формат:
      colors:
        person: "#c06060"
        helmet: "#008060"

    или:
      colors:
        - "#c06060"
        - "#008060"

    Если цветов нет — создаёт автоматические цвета.
    """
    colors_cfg = None
    yaml_path = Path(data_yaml)

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if isinstance(data, dict):
            colors_cfg = data.get("colors")

            # Запасной вариант, если вдруг data.yaml хранит classes как список словарей:
            # classes:
            #   - name: person
            #     color: "#c06060"
            if colors_cfg is None and isinstance(data.get("classes"), list):
                colors_cfg = {}
                for item in data.get("classes", []):
                    if isinstance(item, dict) and "name" in item and "color" in item:
                        colors_cfg[str(item["name"]).strip()] = item["color"]
    else:
        print(f"[WARN] data.yaml не найден: {data_yaml}, цвета будут автоматическими")

    colors = {}

    for i, name in enumerate(class_names):
        raw = None

        if isinstance(colors_cfg, dict):
            raw = colors_cfg.get(name)
        elif isinstance(colors_cfg, list) and i < len(colors_cfg):
            raw = colors_cfg[i]

        parsed = _parse_color(raw)
        colors[name] = parsed if parsed is not None else _default_bgr(i)

    return colors



def _draw_detection(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + 4), color, -1)
    cv2.putText(img, label, (x1 + 3, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_on_image(model: YOLO, image_path: str, out_path: str, class_colors: dict, conf: float = 0.25) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать картинку: {image_path}")

    t0 = time.time()
    results = model.predict(img, conf=conf, verbose=False)[0]
    inference_ms = (time.time() - t0) * 1000

    masks_data = results.masks.data.cpu().numpy() if results.masks is not None else None

    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = class_colors.get(cls_name, (150, 150, 150))

        if masks_data is not None and i < len(masks_data):
            mask_resized = cv2.resize(masks_data[i], (img.shape[1], img.shape[0]))
            _draw_mask(img, mask_resized > 0.5, color)

        _draw_detection(img, x1, y1, x2, y2, f"{cls_name} {conf_score:.2f}", color)
        detections.append({"class": cls_name, "confidence": conf_score, "bbox": [x1, y1, x2, y2]})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(out_path, img):
        raise ValueError(f"Не удалось сохранить результат: {out_path}")
    return {
        "frames_processed": 1,
        "avg_inference_ms": round(inference_ms, 1),
        "avg_fps": round(1000 / inference_ms, 1) if inference_ms else 0,
        "output_path": out_path,
        "detections": detections,
    }
def run_on_video(model: YOLO, video_path: str, out_path: str, class_colors: dict, conf: float = 0.25, max_frames: int | None = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback, если FPS не определен
        print(f"[WARN] FPS не определен, использую {fps}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    
    print(f"Видео: {w}x{h}, FPS={fps}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        print(f"[WARN] mp4v не работает, пробую XVID...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path_xvid = str(Path(out_path).with_suffix(".avi"))
        writer = cv2.VideoWriter(out_path_xvid, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось инициализировать VideoWriter для {out_path}")
        out_path = out_path_xvid
        print(f"Использую XVID, выход: {out_path}")

    frame_times = []
    frame_idx = 0
    skipped_frames = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            skipped_frames += 1

        t0 = time.time()
        results = model.predict(frame, conf=conf, verbose=False)[0]
        frame_times.append(time.time() - t0)

        masks_data = results.masks.data.cpu().numpy() if results.masks is not None else None

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = class_colors.get(cls_name, (150, 150, 150))

            if masks_data is not None and i < len(masks_data):
                mask_resized = cv2.resize(masks_data[i], (frame.shape[1], frame.shape[0]))
                _draw_mask(frame, mask_resized > 0.5, color)

            _draw_detection(frame, x1, y1, x2, y2, f"{cls_name} {conf_score:.2f}", color)
        writer.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Обработано кадров: {frame_idx}")

    cap.release()
    writer.release()

    if skipped_frames > 0:
        print(f"[WARN] Пропущено/ресайзнуто кадров: {skipped_frames}")

    avg_ms = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0
    return {
        "frames_processed": frame_idx,
        "avg_inference_ms": round(avg_ms, 1),
        "avg_fps": round(1000 / avg_ms, 1) if avg_ms else 0,
        "output_path": out_path
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-yaml", default="data.yaml", help="data.yaml с names и опционально colors")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--result-json", default=None)
    p.add_argument("--max-frames", type=int, default=None, help="Ограничить количество кадров для теста")
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Файл весов не найден: {args.weights}")

    model = YOLO(args.weights)

    if isinstance(model.names, dict):
        class_names = [
            str(model.names[k]).strip()
            for k in sorted(model.names, key=lambda x: int(x))
        ]
    else:
        class_names = [str(x).strip() for x in model.names]

    class_colors = load_class_colors(args.data_yaml, class_names)

    suffix = Path(args.input).suffix.lower()
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        result = run_on_video(model, args.input, args.output, class_colors, args.conf, args.max_frames)
    else:
        result = run_on_image(model, args.input, args.output, class_colors, args.conf)

    print(json.dumps(result, ensure_ascii=False))
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
