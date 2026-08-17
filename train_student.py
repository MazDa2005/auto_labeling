"""
train_student.py — обучение student-модели (YOLO) на объединённом YOLO-датасете.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
import gc
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


class MemoryCallback:
    """Очистка памяти после каждой эпохи."""
    def on_train_epoch_end(self, trainer):
        gc.collect()
        torch.cuda.empty_cache()


def load_data_yaml(data_yaml_path: str) -> dict:
    path = Path(data_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"data.yaml должен быть словарём: {data_yaml_path}")

    return data


def get_class_names_from_yaml(data: dict) -> list[str]:
    """
    Поддерживает варианты:
      names: [class1, class2]
      names: {0: class1, 1: class2}
      classes: [class1, class2]
    """
    names = data.get("names", data.get("classes"))

    if names is None:
        raise ValueError("В data.yaml нет поля names или classes")

    if isinstance(names, dict):
        return [str(names[k]).strip() for k in sorted(names, key=lambda x: int(x))]

    if isinstance(names, list):
        return [str(x).strip() for x in names]

    raise ValueError("Поле names в data.yaml должно быть списком или словарём")


def find_split_dirs(dataset_dir: Path, split: str) -> tuple[Path | None, Path | None]:
    """
    Находит папки images и labels для сплита, поддерживая оба формата.
    Возвращает (images_dir, labels_dir) или (None, None).
    """
    # Новый формат: dataset_dir/split/images
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    if img_dir.exists():
        return img_dir, lbl_dir if lbl_dir.exists() else None
    
    # Старый формат: dataset_dir/images/split
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    if img_dir.exists():
        return img_dir, lbl_dir if lbl_dir.exists() else None
    
    return None, None


def merge_yolo_datasets(source_dirs: list[Path], target_dir: Path) -> None:
    """Объединяет несколько YOLO-датасетов в один с префиксами имён."""
    split_map = {
        "train": "train",
        "val": "val",
        "valid": "val",
    }

    # Создаем структуру: target_dir/split/images и target_dir/split/labels
    for dst_split in set(split_map.values()):
        (target_dir / dst_split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / dst_split / "labels").mkdir(parents=True, exist_ok=True)

    for src in source_dirs:
        project_name = src.parent.name

        for src_split, dst_split in split_map.items():
            img_src_dir, lbl_src_dir = find_split_dirs(src, src_split)
            if img_src_dir is None:
                continue

            for img_path in sorted(img_src_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                new_stem = f"{project_name}_{img_path.stem}"

                # Копируем в target_dir/dst_split/images/
                shutil.copy(
                    img_path,
                    target_dir / dst_split / "images" / f"{new_stem}{img_path.suffix}"
                )

                if lbl_src_dir:
                    lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
                    # Копируем в target_dir/dst_split/labels/
                    dst_lbl = target_dir / dst_split / "labels" / f"{new_stem}.txt"

                    if lbl_path.exists():
                        shutil.copy(lbl_path, dst_lbl)
                    else:
                        dst_lbl.touch()


def write_progress(progress_file: Path | None, payload: dict) -> None:
    if progress_file is None:
        return
    progress_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dirs", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--data-yaml", required=True, help="data.yaml с names и опционально colors")
    p.add_argument("--base-model", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--progress-file", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--plots", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--cache", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--patience", type=int, default=50)

    args = p.parse_args()

    source_dirs = [Path(s.strip()) for s in args.source_dirs.split(",") if s.strip()]
    target_dir = Path(args.target_dir)
    runs_dir = Path(args.runs_dir)
    progress_file = Path(args.progress_file) if args.progress_file else None

    write_progress(progress_file, {"stage": "merge", "progress": 0.0, "message": "Объединение датасетов..."})
    merge_yolo_datasets(source_dirs, target_dir)

    source_yaml = load_data_yaml(args.data_yaml)
    class_names = get_class_names_from_yaml(source_yaml)

    # Считаем количество картинок в train и val
    n_train = len(list((target_dir / "train" / "images").glob("*")))
    n_val = len(list((target_dir / "val" / "images").glob("*")))

    if n_train == 0:
        write_progress(progress_file, {"stage": "failed", "progress": 0.0, "message": "0 train-картинок"})
        sys.exit(1)

    # Если val пустой, используем train для валидации (fallback)
    val_path = "val/images" if n_val > 0 else "train/images"

    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "train/images",
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }

    # Если в исходном data.yaml были цвета, сохраняем их в итоговый data.yaml
    if "colors" in source_yaml:
        data_yaml["colors"] = source_yaml["colors"]

    # Записываем data.yaml ОДИН раз
    data_yaml_path = target_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    write_progress(progress_file, {
        "stage": "train",
        "progress": 0.05,
        "message": f"Старт: {n_train} train / {n_val} val | workers={args.workers}, cache={args.cache}"
    })

    model = YOLO(args.base_model)

    def on_epoch_end(trainer):
        current = trainer.epoch + 1
        total = trainer.epochs
        write_progress(progress_file, {
            "stage": "train",
            "progress": round(0.05 + 0.95 * current / max(total, 1), 4),
            "message": f"Эпоха {current}/{total}",
        })

    model.add_callback("on_train_epoch_end", on_epoch_end)

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir.resolve()),
        name=args.run_name,
        exist_ok=True,
        workers=args.workers,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    write_progress(progress_file, {
        "stage": "done",
        "progress": 1.0,
        "message": f"Готово. Веса: {best_weights}",
        "best_weights": str(best_weights)
    })


if __name__ == "__main__":
    main()