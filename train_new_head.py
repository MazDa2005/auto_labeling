"""
train_new_head.py — обучение НОВОЙ Segment-головы поверх ЗАМОРОЖЕННОГО backbone+neck
старой модели. Это первый шаг варианта A (настоящее слияние в одну модель) —
см. concat_segment_head.py и merge_model.py.

Логика 1-в-1 повторяет статью y-t-g, адаптированную под сегментацию:
  1. Стартуем с весов СТАРОЙ модели (не generic yolo11n-seg.pt!) — иначе backbone
     новой головы не будет соответствовать тому, что реально выдаёт backbone
     старой модели при мёрже.
  2. freeze=N, где N — индекс головы (все слои до головы замораживаются).
  3. Доп. callback переводит все замороженные BN-слои в eval + отключает
     track_running_stats — иначе BN running stats всё равно "уедут" в train mode
     (это не блокируется обычной заморозкой requires_grad=False!).
  4. ultralytics сам переинициализирует ТОЛЬКО последний слой (голову) под новое
     число классов из data.yaml — веса backbone/neck переносятся как есть.

После обучения новую голову НЕ нужно вручную вырезать/сохранять по слоям —
merge_model.py берёт `model.model.model[-1]` прямо из чекпоинта.

Пример:
    python train_new_head.py \
        --old-weights projects/my_project/runs/run_1/weights/best.pt \
        --dataset-dir dataset_new_classes/ \
        --classes-file new_classes.json \
        --runs-dir runs_new_head \
        --run-name welding_v1 \
        --epochs 100
"""
import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_class_names(classes_file: str) -> list[str]:
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [c["name"] for c in sorted(data["classes"], key=lambda c: c["index"])]


def put_in_eval_mode(trainer, n_layers: int):
    """BN-заморозка: переводит BN замороженных слоёв в eval + отключает track_running_stats.
    Нужно вызывать каждую эпоху, т.к. model.train() в начале эпохи снова переключает
    ВСЮ модель в train mode и сбрасывает этот эффект.

    on_pretrain_routine_start срабатывает ДО того, как trainer.model создан (в некоторых
    версиях ultralytics) — в этот момент просто ничего не делаем, колбэк на
    on_train_epoch_start всё равно применит заморозку перед первой же эпохой."""
    model = getattr(trainer, "model", None)
    if model is None:
        return
    for name, module in model.named_modules():
        if name.endswith("bn"):
            try:
                layer_idx = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            if layer_idx < n_layers:
                module.eval()
                module.track_running_stats = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-weights", required=True,
                    help="Веса ТЕКУЩЕЙ (старой) student-модели — та, поверх backbone которой учим новую голову")
    p.add_argument("--dataset-dir", required=True,
                    help="YOLO-seg датасет ТОЛЬКО новых классов (images/train,val + labels/train,val)")
    p.add_argument("--classes-file", required=True, help="classes.json только новых классов (нумерация с 0)")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--plots", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--cache", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--patience", type=int, default=50)
    args = p.parse_args()

    if not Path(args.old_weights).exists():
        raise FileNotFoundError(f"Не найдены веса старой модели: {args.old_weights}")

    dataset_dir = Path(args.dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        class_names = load_class_names(args.classes_file)
        data_yaml = {
            "path": str(dataset_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
        print(f"[data.yaml] сгенерирован из {args.classes_file}: {data_yaml_path}")

    n_train = len(list((dataset_dir / "train"/"images").glob("*")))
    if n_train == 0:
        raise SystemExit(f"[ERROR] 0 train-картинок в {dataset_dir / 'images' / 'train'}")
    print(f"[data] train={n_train}")

    # 1. Стартуем со старых весов — backbone/neck отсюда останутся нетронутыми
    model = YOLO(args.old_weights)

    # 2. Индекс головы = индекс последнего слоя. Всё, что раньше него, замораживаем.
    head_index = model.model.model[-1].i
    print(f"[freeze] голова на слое {head_index}, замораживаю слои 0..{head_index - 1}")

    # 3. BN callback — без него замороженные BN всё равно "уедут"
    model.add_callback("on_train_epoch_start", lambda trainer: put_in_eval_mode(trainer, head_index))
    model.add_callback("on_pretrain_routine_start", lambda trainer: put_in_eval_mode(trainer, head_index))

    # 4. Обучение: data.yaml с новыми классами -> ultralytics сам переинициализирует
    #    только последний слой под новое nc, остальное перенесётся из old_weights
    results = model.train(
        data=str(data_yaml_path),
        freeze=head_index,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.runs_dir).resolve()),
        name=args.run_name,
        exist_ok=True,
        workers=args.workers,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Готово. Веса новой головы (полный чекпоинт, backbone идентичен старому): {best_weights}")
    print("👉 Дальше: merge_model.py --old-weights <old best.pt> "
          f"--new-weights {best_weights} ...")
    print("👉 Перед использованием в проде — запустите sanity_check.py, чтобы убедиться,\n"
          "   что confidence на старых классах не изменились после мёржа.")


if __name__ == "__main__":
    main()
