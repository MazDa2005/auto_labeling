"""
Обработка ОДНОЙ картинки — тонкая обёртка над batch_label.py для быстрых тестов.
Под капотом создаёт временную папку с одной картинкой и использует ту же пакетную логику

Пример:
    python auto_label.py --image frame.jpg --classes person,vest,helmet --out annotations/frame.json
    python auto_label.py --image frame.jpg --classes-file classes.json --out annotations/frame.json
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from batch_label import load_stages, run_stage_batch, save_annotated_image
from teachers.pipeline_utils import merge_new_detections, get_class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--classes", default=None,
                        help="comma-separated; если не указано — берётся из classes.json")
    parser.add_argument("--classes-file", default="classes.json")
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Классы: либо из CLI, либо из classes.json
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
    else:
        classes = get_class_names(args.classes_file)
        print(f"Классы не указаны явно — взято {len(classes)} из {args.classes_file}")

    stages = load_stages(args.config)
    if not stages:
        print("В конфиге нет активных stage'ей", file=sys.stderr)
        sys.exit(1)

    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    masks_root = out_path.parent / "masks"

    detect_stages = [s for s in stages if s["type"] == "detect"]
    refine_stages = [s for s in stages if s["type"] == "refine"]
    verify_stages = [s for s in stages if s["type"] == "verify"]
    qc_stages = [s for s in stages if s["type"] == "qc"]

    with tempfile.TemporaryDirectory() as tmp:
        images_dir = Path(tmp) / "images"
        images_dir.mkdir()
        shutil.copy(image_path, images_dir / image_path.name)

        stage_out_root = Path(tmp) / "stages"

        # 1. DETECT
        per_stage_dirs = []
        for stage in detect_stages:
            stage_dir = str(stage_out_root / stage["name"])
            extra_args = ["--classes", ",".join(classes), "--classes-file", args.classes_file]
            run_stage_batch(stage, str(images_dir), stage_dir, extra_args)
            per_stage_dirs.append(stage_dir)

        merged = []
        for stage_dir in per_stage_dirs:
            p = Path(stage_dir) / f"{image_path.stem}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    dets = json.load(f).get("detections", [])
                merged = merge_new_detections(merged, dets)

        current_dir = Path(tmp) / "current"
        current_dir.mkdir()
        with open(current_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"detections": merged}, f)

        # 2. REFINE
        for stage in refine_stages:
            stage_dir = str(stage_out_root / stage["name"])
            run_stage_batch(
                stage, str(images_dir), stage_dir,
                ["--boxes-dir", str(current_dir), "--masks-root", str(masks_root)],
            )
            current_dir = Path(stage_dir)

        # 3. VERIFY
        for stage in verify_stages:
            stage_dir = str(stage_out_root / stage["name"])
            threshold = str(stage.get("threshold", 0.25))
            # Передаём --classes-file для clip_verify.py
            run_stage_batch(
                stage, str(images_dir), stage_dir,
                ["--boxes-dir", str(current_dir), "--threshold", threshold,
                 "--classes-file", args.classes_file],
            )
            current_dir = Path(stage_dir)

        # 4. QC FILTER 
        if qc_stages:
            qc_stage = qc_stages[0]
            qc_out = out_path.parent  

            if "python_exe" in qc_stage:
                python_cmd = [qc_stage["python_exe"]]
            else:
                python_cmd = ["conda", "run", "-n", qc_stage["conda_env"], "python"]

            cmd = [
                *python_cmd, qc_stage["script"],
                "--annotations-dir", str(current_dir),
                "--output-dir", str(qc_out),
            ]
            print(f"\n [{qc_stage['name']}] ({qc_stage['type']}) окружение: {qc_stage['conda_env']}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[WARN] {qc_stage['name']} завершился с ошибкой (код {result.returncode})",
                      file=sys.stderr)

            # Определяем, куда попал результат
            if (qc_out / "clean" / f"{image_path.stem}.json").exists():
                result_path = qc_out / "clean" / f"{image_path.stem}.json"
                bucket = "clean"
            elif (qc_out / "review" / f"{image_path.stem}.json").exists():
                result_path = qc_out / "review" / f"{image_path.stem}.json"
                bucket = "review"
            else:
                result_path = None
                bucket = None
        else:
            result_path = current_dir / f"{image_path.stem}.json"
            bucket = None

        # Читаем итоговые детекции
        detections = []
        if result_path and result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                detections = json.load(f).get("detections", [])

    img = Image.open(image_path)
    try:
        rel_path = str(Path(image_path).relative_to(Path.cwd()))
    except ValueError:
        rel_path = str(image_path)
    output = {
        "image": rel_path,
        "width": img.width,
        "height": img.height,
        "classes_requested": classes,
        "detections": detections,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    annotated_path = str(out_path.with_suffix("")) + "_annotated.jpg"
    save_annotated_image(str(image_path), detections, annotated_path)

    status = f" ({bucket})" if bucket else ""
    print(f"\nИтого: {len(detections)} детекций{status}, сохранено в {out_path}")


if __name__ == "__main__":
    main()