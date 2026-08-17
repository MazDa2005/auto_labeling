"""
Объединение нескольких YOLO-датасетов (каждый со своей структурой images/labels/data.yaml)
в один общий датасет.

ВАЖНО: все объединяемые датасеты должны использовать ОДИНАКОВЫЙ список классов в ОДИНАКОВОМ
порядке (одинаковый classes.json/data.yaml) 

При совпадении имён файлов между датасетами — файл переименовывается с префиксом
имени исходного датасета, чтобы не потерять данные 
"""
import argparse
import shutil
import sys
from pathlib import Path
import yaml

def normalize_names(names) -> list[str]:
    """
    Приводит names к списку строк.
    Поддерживает:
      names: [a, b, c]
      names: {0: a, 1: b, 2: c}
    """
    if isinstance(names, dict):
        return [
            str(names[k]).strip()
            for k in sorted(names, key=lambda x: int(x))
        ]

    if isinstance(names, list):
        return [str(x).strip() for x in names]

    return []


def load_names_and_colors(dataset_dir: str) -> tuple[list[str], dict]:
    data_yaml_path = Path(dataset_dir) / "data.yaml"

    if not data_yaml_path.exists():
        print(f"[ERROR] {dataset_dir}: нет data.yaml", file=sys.stderr)
        sys.exit(1)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = normalize_names(data.get("names", []))
    colors = data.get("colors", {}) or {}

    if not isinstance(colors, dict):
        colors = {}

    return names, colors

def check_classes_consistency(datasets: list[str]) -> tuple[list[str], dict]:
    reference_names, reference_colors = load_names_and_colors(datasets[0])

    for ds in datasets[1:]:
        names, colors = load_names_and_colors(ds)

        if names != reference_names:
            print(f"[ERROR] Несовпадение списка классов между датасетами:", file=sys.stderr)
            print(f"  {datasets[0]}: {reference_names}", file=sys.stderr)
            print(f"  {ds}: {names}", file=sys.stderr)
            print(
                "Объединение остановлено — сначала приведите classes.json/data.yaml к одному виду "
                "и пересоздайте датасеты через convert_to_yolo_seg.py",
                file=sys.stderr
            )
            sys.exit(1)

        if colors:
            if not reference_colors:
                reference_colors = colors
            elif colors != reference_colors:
                print(
                    f"[WARN] {ds}: цвета в data.yaml отличаются от эталона. "
                    f"Использую цвета из {datasets[0]}.",
                    file=sys.stderr
                )

    return reference_names, reference_colors

def merge(datasets: list[str], output_dir: str, split: str):
    names, colors = check_classes_consistency(datasets)
    print(f"Классы совпадают у всех {len(datasets)} датасетов ({len(names)} классов) — ок")

    split_out = "val" if split == "valid" else split

    images_out = Path(output_dir) / split_out / "images"
    labels_out = Path(output_dir) / split_out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_renamed = 0
    seen_stems: set[str] = set()

    for ds in datasets:
        ds_name = Path(ds).name
        ds_images = Path(ds) / split / "images"
        ds_labels = Path(ds) / split / "labels"

        if not ds_images.exists() and split_out != split:
            ds_images = Path(ds) / split_out / "images"
            ds_labels = Path(ds) / split_out / "labels"
        if not ds_images.exists():
            print(f"[WARN] {ds}: нет images/{split}, пропуск датасета", file=sys.stderr)
            continue

        image_files = sorted(ds_images.iterdir())
        for img_path in image_files:
            stem = img_path.stem
            label_path = ds_labels / f"{stem}.txt"

            final_stem = stem
            if stem in seen_stems:
                final_stem = f"{ds_name}__{stem}"
                total_renamed += 1
                print(f"[RENAME] {stem} -> {final_stem} (коллизия имён между датасетами)")

            seen_stems.add(final_stem)

            shutil.copy(img_path, images_out / f"{final_stem}{img_path.suffix}")
            if label_path.exists():
                shutil.copy(label_path, labels_out / f"{final_stem}.txt")
            else:
                # пустой label — картинка без объектов, это нормально для YOLO (fon)
                (labels_out / f"{final_stem}.txt").touch()

            total_copied += 1
    yaml_path = Path(output_dir) / "data.yaml"

    existing_yaml = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            existing_yaml = yaml.safe_load(f) or {}

    data_yaml = {
        "path": str(Path(output_dir).resolve()),
        "train": existing_yaml.get("train", "train/images"),
        "val": existing_yaml.get("val", "val/images"),
        "nc": len(names),
        "names": names,
    }

    if split_out == "train":
        data_yaml["train"] = "train/images"
    elif split_out == "val":
        data_yaml["val"] = "val/images"
    else:
        data_yaml[split_out] = f"{split_out}/images"

    if colors:
        data_yaml["colors"] = colors
    elif isinstance(existing_yaml.get("colors"), dict):
        data_yaml["colors"] = existing_yaml["colors"]

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nГотово: {total_copied} картинок объединено ({total_renamed} переименовано из-за коллизий).")
    print(f"Объединённый датасет: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, help="Пути к датасетам для объединения")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    merge(args.datasets, args.output_dir, args.split)
