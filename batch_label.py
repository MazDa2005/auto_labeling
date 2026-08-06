"""
Пакетный оркестратор — КАЖДЫЙ stage запускается ОДИН РАЗ на весь батч картинок
(модель грузится один раз внутри teacher-скрипта, не при каждой картинке).

Порядок: detect -> refine -> verify -> [qc].
После QC результаты раскладываются по clean/ и review/ для ручного ревью.
Пример:
    python batch_label.py --images-dir frames/ --config pipeline_config.yaml --out-dir annotations/
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from teachers.pipeline_utils import list_images, merge_new_detections, get_class_names


def get_python_cmd(stage: dict) -> list[str]:
    """Возвращает команду для запуска Python (оптимизировано для Docker)."""
    if "python_exe" in stage:
        return [stage["python_exe"]]
    return [f"/opt/conda/envs/{stage['conda_env']}/bin/python"]


def load_stages(config_path: str) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return [s for s in config.get("stages", []) if s.get("enabled", True)]


def run_stage_batch(stage: dict, images_dir: str, out_dir: str, extra_args: list[str]):
    """Запускает teacher-скрипт ОДИН РАЗ на всю папку с картинками."""
    python_cmd = get_python_cmd(stage)

    cmd = [
        *python_cmd, stage["script"],
        "--images-dir", images_dir,
        "--model", stage["model_path"],
        "--out-dir", out_dir,
        *extra_args,
    ]
    if "pretrained" in stage:
        cmd.extend(["--pretrained", str(stage["pretrained"])])

    print(f"\n=== [{stage['name']}] ({stage['type']}) окружение: {stage['conda_env']} ===")
    result = subprocess.run(cmd)  # без capture_output — чтобы видеть прогресс в реальном времени
    if result.returncode != 0:
        print(f"[WARN] {stage['name']} завершился с ошибкой (код {result.returncode})", file=sys.stderr)


def load_class_colors(classes_file: str = "classes.json") -> dict[str, tuple[int, int, int]]:
    """Загружает цвета классов из classes.json (BGR -> RGB для PIL)."""
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


def save_annotated_image(image_path: str, detections: list[dict], out_path: str,
                         classes_file: str = "classes.json"):
    """Рисует маски (цвет из classes.json) + боксы + подписи."""
    class_colors = load_class_colors(classes_file)
    default_color = (150, 150, 150)

    img = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    masks_drawn = 0
    masks_missing = 0

    # 1. Полупрозрачные МАСКИ (цвет из classes.json) 
    for det in detections:
        mask_path = det.get("mask_path")
        if not mask_path:
            continue
        if not Path(mask_path).exists():
            masks_missing += 1
            continue
        cls = det["class"]
        rgb = class_colors.get(cls, default_color)
        mask_img = Image.open(mask_path).convert("L").resize(img.size, resample=Image.NEAREST)
        mask_np = np.array(mask_img)
        threshold = 127 if mask_np.max() > 1 else 0
        mask_array = mask_np > threshold
        colored = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        colored[mask_array] = (*rgb, 60) 
        overlay = Image.alpha_composite(overlay, Image.fromarray(colored, mode="RGBA"))
        masks_drawn += 1

    if masks_missing:
        print(f"[batch][WARN] {Path(out_path).stem}: {masks_missing} масок не найдено", file=sys.stderr)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    #  2. Боксы и подписи 
    for det in detections:
        cls = det["class"]
        x1, y1, x2, y2 = det["bbox"]
        rgb = class_colors.get(cls, default_color)

        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)

        label = cls
        if det.get("confidence") is not None:
            label += f" {det['confidence']:.2f}"

        try:
            bbox = draw.textbbox((0, 0), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(label) * 7, 12

        bg_color = tuple(max(0, c - 60) for c in rgb)
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=bg_color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255))

    img.save(out_path)
    print(f"[batch] {Path(out_path).stem}: отрисовано {masks_drawn} масок, пропущено {masks_missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes", default=None, help="comma-separated; если не указано — берётся из classes.json")
    parser.add_argument("--classes-file", default="classes.json")
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
    else:
        classes = get_class_names(args.classes_file)
        print(f"Классы не указаны явно — взято {len(classes)} из {args.classes_file}")

    stages = load_stages(args.config)
    images = list_images(args.images_dir)

    if not images:
        print(f"Нет изображений в {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    stage_out_root = out_dir / "_stages"
    masks_root = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_stages = [s for s in stages if s["type"] == "detect"]
    refine_stages = [s for s in stages if s["type"] == "refine"]
    verify_stages = [s for s in stages if s["type"] == "verify"]
    qc_stages = [s for s in stages if s["type"] == "qc"]

    # 1. DETECT — запускается один раз на всю папку с картинками
    detect_out = stage_out_root / "detect_merged"
    detect_out.mkdir(parents=True, exist_ok=True)

    per_stage_dirs = []
    for stage in detect_stages:
        stage_dir = str(stage_out_root / stage["name"])
        extra_args = ["--classes", ",".join(classes), "--classes-file", args.classes_file]
        run_stage_batch(stage, args.images_dir, stage_dir, extra_args)
        per_stage_dirs.append(stage_dir)

    for image_path in images:
        merged = []
        for stage_dir in per_stage_dirs:
            p = Path(stage_dir) / f"{image_path.stem}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    dets = json.load(f).get("detections", [])
                merged = merge_new_detections(merged, dets)

        with open(detect_out / f"{image_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": str(Path(image_path).absolute()),
                "detections": merged
            }, f, indent=2, ensure_ascii=False)

    current_dir = detect_out

    # 2. REFINE — тоже один раз на всю папку, читает боксы из предыдущего этапа
    for stage in refine_stages:
        stage_dir = str(stage_out_root / stage["name"])
        run_stage_batch(
            stage, args.images_dir, stage_dir,
            ["--boxes-dir", str(current_dir), "--masks-root", str(masks_root), "--classes-file", args.classes_file],
        )
        current_dir = Path(stage_dir)

    # 3. VERIFY — один раз на всю папку
    for stage in verify_stages:
        stage_dir = str(stage_out_root / stage["name"])
        threshold = str(stage.get("threshold", 0.25))
        run_stage_batch(
            stage, args.images_dir, stage_dir,
            ["--boxes-dir", str(current_dir), "--threshold", threshold, "--classes-file", args.classes_file],
        )
        current_dir = Path(stage_dir)

    # 4. QC FILTER (опционально)
    if qc_stages:
        qc_stage = qc_stages[0]
        python_cmd = get_python_cmd(qc_stage)

        cmd = [
            *python_cmd, qc_stage["script"],
            "--annotations-dir", str(current_dir),
            "--output-dir", str(out_dir),
            "--masks-root", str(masks_root),
        ]
        print(f"\n=== [{qc_stage['name']}] ({qc_stage['type']}) окружение: {qc_stage['conda_env']} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARN] {qc_stage['name']} завершился с ошибкой (код {result.returncode})", file=sys.stderr)
            sys.exit(1)

        print(f"\n QC-фильтр завершил работу.")
        return

    # 5. Финальная сборка (если QC выключен)
    print(f"\n Финальная сборка (без QC-фильтра)")
    summary = []
    for image_path in images:
        result_path = current_dir / f"{image_path.stem}.json"
        detections = []
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                detections = json.load(f).get("detections", [])

        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"[WARN] {image_path.name}: не удалось открыть ({e})", file=sys.stderr)
            continue

        # Относительный путь относительно images_dir
        try:
            rel_path = str(Path(image_path).relative_to(Path(args.images_dir)))
        except ValueError:
            rel_path = str(Path(image_path).absolute())

        final_json = {
            "image": rel_path,
            "width": img.width,
            "height": img.height,
            "classes_requested": classes,
            "detections": detections,
        }
        with open(out_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        save_annotated_image(str(image_path), detections, str(out_dir / f"{image_path.stem}_annotated.jpg"), args.classes_file)
        summary.append({"image": image_path.name, "num_detections": len(detections)})
        print(f"{image_path.name}: {len(detections)} детекций")

    with open(out_dir / "_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n Готово. Результаты в {out_dir}")


if __name__ == "__main__":
    main()
