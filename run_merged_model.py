"""
run_merged_model.py — инференс уже собранной merged_model.pt.
Импортирует ядро из merge_model.py.

Пример:
    python run_merged_model.py \\
        --model projects/detection/merged_models/merged_model.pt \\
        --input test.jpg --output out.jpg --conf 0.25
"""
import argparse
import json
import sys
from pathlib import Path

from merge_model import load_merged_model, run_on_image, run_on_video


def main():
    p = argparse.ArgumentParser(description="Инференс merged YOLO-seg модели")
    p.add_argument("--model", required=True, help="Путь к merged_model.pt")
    p.add_argument("--input", required=True, help="Картинка или видео")
    p.add_argument("--output", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--result-json", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Файл модели не найден: {args.model}")
        sys.exit(1)

    device = (
        f"cuda:{args.device}"
        if args.device != "cpu" and __import__("torch").cuda.is_available()
        else "cpu"
    )

    print(f"[load] Загружаю {args.model}...")
    model, names, class_colors = load_merged_model(args.model, device)

    input_path = Path(args.input)
    if input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        result = run_on_video(
            model,
            args.input,
            args.output,
            names,
            class_colors,
            args.conf,
            args.iou,
            args.imgsz,
            args.max_frames,
        )
    else:
        result = run_on_image(
            model,
            args.input,
            args.output,
            names,
            class_colors,
            args.conf,
            args.iou,
            args.imgsz,
        )

    n_det = result.get("n_detections", result.get("total_detections", 0))
    ms = result.get("inference_ms", result.get("avg_inference_ms", 0))
    print(f"[infer] {n_det} объектов, {ms} ms → {args.output}")
    print(json.dumps(result, ensure_ascii=False))

    if args.result_json:
        Path(args.result_json).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()