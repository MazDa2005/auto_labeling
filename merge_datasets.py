"""
Объединение нескольких YOLO-датасетов (каждый со своей структурой images/labels/data.yaml)
в один общий датасет.

ВАЖНО: все объединяемые датасеты должны использовать ОДИНАКОВЫЙ список классов в ОДИНАКОВОМ
порядке (одинаковый classes.json/data.yaml) — иначе class_id будут указывать на разные классы
в разных датасетах, и разметка перепутается. Скрипт это проверяет и останавливается при
несовпадении.

При совпадении имён файлов между датасетами — файл переименовывается с префиксом
имени исходного датасета, чтобы не потерять данные (а не тихо перезаписать).

Пример:
    python merge_datasets.py \
        --datasets dataset_batch1/ dataset_batch2/ dataset_batch3/ \
        --output-dir dataset_merged/ \
        --split train
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml


def load_names(dataset_dir: str) -> list[str]:
    data_yaml_path = Path(dataset_dir) / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[ERROR] {dataset_dir}: нет data.yaml", file=sys.stderr)
        sys.exit(1)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def check_classes_consistency(datasets: list[str]) -> list[str]:
    reference_names = load_names(datasets[0])
    for ds in datasets[1:]:
        names = load_names(ds)
        if names != reference_names:
            print(f"[ERROR] Несовпадение списка классов между датасетами:", file=sys.stderr)
            print(f"  {datasets[0]}: {reference_names}", file=sys.stderr)
            print(f"  {ds}: {names}", file=sys.stderr)
            print("Объединение остановлено — сначала приведите classes.json к одному виду "
                  "и пересоздайте датасеты через convert_to_yolo_seg.py", file=sys.stderr)
            sys.exit(1)
    return reference_names


def merge(datasets: list[str], output_dir: str, split: str):
    names = check_classes_consistency(datasets)
    print(f"Классы совпадают у всех {len(datasets)} датасетов ({len(names)} классов) — ок")

    images_out = Path(output_dir) / "images" / split
    labels_out = Path(output_dir) / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_renamed = 0
    seen_stems: set[str] = set()

    for ds in datasets:
        ds_name = Path(ds).name
        ds_images = Path(ds) / "images" / split
        ds_labels = Path(ds) / "labels" / split

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

    data_yaml = {
        "path": str(Path(output_dir).resolve()),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "nc": len(names),
        "names": names,
    }
    with open(Path(output_dir) / "data.yaml", "w", encoding="utf-8") as f:
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
