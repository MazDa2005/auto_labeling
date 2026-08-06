"""
benchmark_student.py — оценка обученной student-модели по двум осям:

1. КАЧЕСТВО: precision/recall/mAP50/mAP50-95, общие и по каждому классу
   отдельно (через ultralytics model.val() на val-сплите датасета).

2. СКОРОСТЬ: FPS на одном потоке и при N параллельных потоках одновременно —
   симуляция нескольких камер. Каждый поток — ОТДЕЛЬНЫЙ ПРОЦЕСС со своим
   экземпляром модели (multiprocessing, не threading): так безопаснее
   (не шарим один объект YOLO между потоками) и реалистичнее отражает
   реальный деплой, где каждая камера — независимый воркер.

Пример запуска:
    python benchmark_student.py \
        --weights projects/my_project/runs/run_1/weights/best.pt \
        --data-yaml projects/my_project/dataset_yolo/data.yaml \
        --sample-images-dir projects/my_project/dataset_yolo/images/val \
        --streams 1,2,4,8 \
        --duration 15 \
        --output-json benchmark_result.json
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from ultralytics import YOLO

def benchmark_quality(weights: str, data_yaml: str, imgsz: int, device: str) -> dict:
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False, plots=False)
    names = model.names

    try:
        class_indices = metrics.box.ap_class_index
    except Exception:
        class_indices = list(range(len(names)))

    per_class = {}
    for i, cls_id in enumerate(class_indices):
        cls_name = names[int(cls_id)]
        per_class[cls_name] = {
            "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else None,
            "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else None,
            "mAP50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else None,
            "mAP50-95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else None,
        }

    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    return {"overall": overall, "per_class": per_class}


def _stream_worker(weights: str, sample_images: list[str], imgsz: int, device: str,
                    duration: float, conf: float, result_queue: mp.Queue) -> None:
    """Один 'поток камеры' — отдельный процесс со своим экземпляром модели."""
    import cv2

    model = YOLO(weights)
    imgs = [cv2.imread(p) for p in sample_images]
    imgs = [im for im in imgs if im is not None]

    if not imgs:
        result_queue.put({"frames": 0, "elapsed": 0.0})
        return

    frames = 0
    idx = 0
    start = time.time()
    while time.time() - start < duration:
        img = imgs[idx % len(imgs)]
        model.predict(img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        frames += 1
        idx += 1
    elapsed = time.time() - start

    result_queue.put({"frames": frames, "elapsed": elapsed})


def benchmark_speed(weights: str, sample_images_dir: str, imgsz: int, device: str,
                     stream_counts: list[int], duration: float, conf: float,
                     max_sample_images: int = 50) -> dict:
    sample_images = sorted(
        str(p) for p in Path(sample_images_dir).glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:max_sample_images]

    if not sample_images:
        raise ValueError(f"Нет картинок для бенчмарка в {sample_images_dir}")

    results = {}
    for n_streams in stream_counts:
        result_queue: mp.Queue = mp.Queue()
        procs = [
            mp.Process(
                target=_stream_worker,
                args=(weights, sample_images, imgsz, device, duration, conf, result_queue),
            )
            for _ in range(n_streams)
        ]

        for p in procs:
            p.start()
        for p in procs:
            p.join()

        total_frames = 0
        max_elapsed = 0.0
        while not result_queue.empty():
            r = result_queue.get()
            total_frames += r["frames"]
            max_elapsed = max(max_elapsed, r["elapsed"])

        aggregate_fps = total_frames / max_elapsed if max_elapsed else 0.0
        per_stream_fps = aggregate_fps / n_streams if n_streams else 0.0

        results[str(n_streams)] = {
            "total_frames": total_frames,
            "aggregate_fps": round(aggregate_fps, 2),
            "per_stream_fps": round(per_stream_fps, 2),
        }
        print(f"[bench] {n_streams} поток(ов): суммарно {aggregate_fps:.1f} FPS "
              f"({per_stream_fps:.1f} FPS на поток)")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Путь к best.pt")
    p.add_argument("--data-yaml", required=True, help="data.yaml датасета (для метрик качества)")
    p.add_argument("--sample-images-dir", required=True,
                    help="Папка с картинками для бенчмарка скорости (например, images/val)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="'0' для GPU, 'cpu' для CPU")
    p.add_argument("--streams", default="1,2,4,8",
                    help="Через запятую: сколько параллельных потоков тестировать")
    p.add_argument("--duration", type=float, default=15.0,
                    help="Секунд на каждую конфигурацию потоков")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--skip-quality", action="store_true", help="Пропустить метрики качества")
    p.add_argument("--skip-speed", action="store_true", help="Пропустить бенчмарк скорости")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    result = {"weights": args.weights, "imgsz": args.imgsz, "device": args.device}

    if not args.skip_quality:
        print("[bench] === Метрики качества ===")
        result["quality"] = benchmark_quality(args.weights, args.data_yaml, args.imgsz, args.device)
        print(f"[bench] Общий mAP50={result['quality']['overall']['mAP50']:.3f}, "
              f"mAP50-95={result['quality']['overall']['mAP50-95']:.3f}")
        print("[bench] По классам:")
        for cls_name, m in result["quality"]["per_class"].items():
            print(f"  {cls_name:24s} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50-95']:.3f}")

    if not args.skip_speed:
        print("\n[bench] === Скорость (FPS) ===")
        stream_counts = [int(s.strip()) for s in args.streams.split(",") if s.strip()]
        result["speed"] = benchmark_speed(
            args.weights, args.sample_images_dir, args.imgsz, args.device,
            stream_counts, args.duration, args.conf,
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[bench] Результат сохранён: {args.output_json}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
