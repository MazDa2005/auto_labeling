"""
build_merged_model.py — собирает ОДНУ модель из старого чекпоинта + новой головы
и опционально гоняет инференс.

Имена классов читаются из весов моделей (model.names) — classes.json НЕ нужен.
Опционально можно передать --data-yaml для переопределения имён и цветов.

Пример:
    python build_merged_model.py \\
        --old-weights projects/my/runs/run_1/weights/best.pt \\
        --new-weights runs_new_head/welding_v1/weights/best.pt \\
        --save-merged merged_model.pt \\
        --input test.jpg --output out.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from merge_model import (
    MergedSegModel,
    build_class_colors,
    export_to_onnx,
    run_on_image,
    run_on_video,
)


def main():
    p = argparse.ArgumentParser(description="Сборка + инференс merged YOLO-seg модели")
    p.add_argument("--old-weights", required=True)
    p.add_argument("--new-weights", required=True)
    p.add_argument(
        "--data-yaml",
        default=None,
        help="data.yaml для переопределения имён классов и цветов",
    )
    p.add_argument(
        "--classes-file",
        default=None,
        help="classes.json для цветов (fallback, если нет data.yaml)",
    )
    p.add_argument("--input", default=None, help="Картинка или видео для теста")
    p.add_argument("--output", default=None)
    p.add_argument("--save-merged", default=None, help="Путь для сохранения merged_model.pt")
    p.add_argument("--export-onnx", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--result-json", default=None)
    p.add_argument("--max-frames", type=int, default=None, help="Ограничить кадры для видео")
    p.add_argument("--verbose", action="store_true", help="Подробный вывод")
    args = p.parse_args()

    device = (
        f"cuda:{args.device}"
        if args.device != "cpu" and torch.cuda.is_available()
        else "cpu"
    )

    print("[build] Загружаю модели...")
    try:
        model = MergedSegModel(args.old_weights, args.new_weights, verbose=args.verbose)
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модели: {e}")
        sys.exit(1)

    model = model.to(device).eval()

    names = list(model.names)
    if args.data_yaml:
        with open(args.data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("names", cfg.get("classes"))
        if isinstance(raw, dict):
            names = [
                str(raw[k]).strip()
                for k in sorted(raw, key=lambda x: int(x))
            ]
        elif isinstance(raw, list):
            names = [str(x).strip() for x in raw]
        print(f"[build] Имена переопределены из data.yaml: {names}")
    else:
        print(f"[build] Имена из весов: {model.names_old} + {model.names_new}")

    class_colors = build_class_colors(names, args.data_yaml, args.classes_file)

    if args.save_merged:
        metadata = {
            "old_weights": args.old_weights,
            "new_weights": args.new_weights,
            "data_yaml": args.data_yaml,
        }
        model.save_merged(args.save_merged, metadata=metadata)

    if args.export_onnx:
        export_to_onnx(model, args.export_onnx, args.imgsz)

    if args.input:
        input_path = Path(args.input)
        if args.output is None:
            args.output = str(input_path.with_stem(input_path.stem + "_merged"))

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