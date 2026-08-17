"""
benchmark_dual.py — сравнение FPS: старая модель одна vs дуал (старая+новая) при N
параллельных "камерах". Адаптация вашего benchmark_student.py.

Отвечает на главный вопрос варианта B: насколько 2x forward pass реально просаживает
FPS на целевом железе (RTX 5060 Ti) при вашей цели 15-30 FPS на несколько потоков.

Пример:
    python benchmark_dual.py \
        --weights-old projects/my_project/runs/run_1/weights/best.pt \
        --weights-new runs_new_head/welding_v1/weights/best.pt \
        --sample-images-dir projects/my_project/dataset_yolo/images/val \
        --streams 1,2,4,8 \
        --duration 15 \
        --output-json bench_dual_result.json
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from ultralytics import YOLO


def _worker_single(weights: str, sample_images: list[str], imgsz: int, device: str,
                    duration: float, conf: float, result_queue: mp.Queue) -> None:
    import cv2
    model = YOLO(weights)
    imgs = [im for im in (cv2.imread(p) for p in sample_images) if im is not None]
    if not imgs:
        result_queue.put({"frames": 0, "elapsed": 0.0})
        return

    frames, idx = 0, 0
    start = time.time()
    while time.time() - start < duration:
        model.predict(imgs[idx % len(imgs)], imgsz=imgsz, device=device, conf=conf, verbose=False)
        frames += 1
        idx += 1
    result_queue.put({"frames": frames, "elapsed": time.time() - start})


def _worker_dual(weights_old: str, weights_new: str, sample_images: list[str], imgsz: int,
                  device: str, duration: float, conf: float, result_queue: mp.Queue) -> None:
    import cv2
    model_old = YOLO(weights_old)
    model_new = YOLO(weights_new)
    imgs = [im for im in (cv2.imread(p) for p in sample_images) if im is not None]
    if not imgs:
        result_queue.put({"frames": 0, "elapsed": 0.0})
        return

    frames, idx = 0, 0
    start = time.time()
    while time.time() - start < duration:
        img = imgs[idx % len(imgs)]
        model_old.predict(img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        model_new.predict(img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        frames += 1
        idx += 1
    result_queue.put({"frames": frames, "elapsed": time.time() - start})


def _run_streams(worker_fn, worker_args: tuple, stream_counts: list[int]) -> dict:
    results = {}
    for n_streams in stream_counts:
        result_queue: mp.Queue = mp.Queue()
        procs = [mp.Process(target=worker_fn, args=(*worker_args, result_queue)) for _ in range(n_streams)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

        total_frames, max_elapsed = 0, 0.0
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
        print(f"  {n_streams} поток(ов): {aggregate_fps:.1f} FPS суммарно ({per_stream_fps:.1f} на поток)")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-old", required=True)
    p.add_argument("--weights-new", required=True)
    p.add_argument("--sample-images-dir", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-sample-images", type=int, default=50)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    sample_images = sorted(
        str(p_) for p_ in Path(args.sample_images_dir).glob("*")
        if p_.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:args.max_sample_images]
    if not sample_images:
        raise SystemExit(f"Нет картинок в {args.sample_images_dir}")

    stream_counts = [int(s.strip()) for s in args.streams.split(",") if s.strip()]
    result = {"imgsz": args.imgsz, "device": args.device}

    print("\n[bench] === old_only (базовая модель, для сравнения) ===")
    result["old_only"] = _run_streams(
        _worker_single,
        (args.weights_old, sample_images, args.imgsz, args.device, args.duration, args.conf),
        stream_counts,
    )

    print("\n[bench] === new_only (новая маленькая модель сама по себе) ===")
    result["new_only"] = _run_streams(
        _worker_single,
        (args.weights_new, sample_images, args.imgsz, args.device, args.duration, args.conf),
        stream_counts,
    )

    print("\n[bench] === dual_merged (обе модели на каждый кадр — то, что реально в проде) ===")
    result["dual_merged"] = _run_streams(
        _worker_dual,
        (args.weights_old, args.weights_new, sample_images, args.imgsz, args.device, args.duration, args.conf),
        stream_counts,
    )

    print("\n[bench] === Итог: во сколько раз dual медленнее old_only, по streams ===")
    for n in stream_counts:
        old_fps = result["old_only"][str(n)]["aggregate_fps"]
        dual_fps = result["dual_merged"][str(n)]["aggregate_fps"]
        ratio = old_fps / dual_fps if dual_fps else float("inf")
        print(f"  streams={n}: old={old_fps:.1f} FPS, dual={dual_fps:.1f} FPS, "
              f"замедление x{ratio:.2f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[bench] Результат сохранён: {args.output_json}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

