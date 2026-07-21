"""
Помощник для ручной проверки детекций из папки review/.

Для каждой картинки из review/:
  1. Показывает _annotated.jpg в просмотрщике
  2. Показывает детекции с причинами (needs_review и rejected отдельно)
  3. Для каждой проблемной детекции спрашивает: [A]ccept / [R]eject / [S]kip
  4. Обновляет JSON, перерисовывает _annotated.jpg
  5. Если все детекции обработаны — переносит картинку в clean/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from qc_filter import draw_qc_annotated


def classify_final(detections: list[dict]) -> str:
    """Определяет, куда должна попасть картинка после ручной проверки."""
    has_review = any(d.get("qc_bucket") == "needs_review" for d in detections)
    return "review" if has_review else "clean"


def move_image_data(stem: str, src_bucket: str, dst_bucket: str, ann_dir: Path):
    """Переносит JSON, annotated.jpg и маски между clean/ и review/."""
    src = ann_dir / src_bucket
    dst = ann_dir / dst_bucket
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)

    # JSON
    json_src = src / f"{stem}.json"
    json_dst = dst / f"{stem}.json"
    if json_src.exists():
        shutil.move(str(json_src), str(json_dst))

    # annotated.jpg
    ann_src = src / f"{stem}_annotated.jpg"
    ann_dst = dst / f"{stem}_annotated.jpg"
    if ann_src.exists():
        shutil.move(str(ann_src), str(ann_dst))

    # Маски этой картинки
    src_masks = src / "masks"
    dst_masks = dst / "masks"
    if src_masks.exists():
        for mask_file in src_masks.glob(f"{stem}_*"):
            shutil.move(str(mask_file), str(dst_masks / mask_file.name))


def update_mask_paths(detections: list[dict], new_bucket: str, ann_dir: Path, stem: str):
    """Обновляет mask_path в JSON после переноса в другую папку."""
    masks_dir = ann_dir / new_bucket / "masks"
    for det in detections:
        old_path = det.get("mask_path")
        if old_path:
            old_name = Path(old_path).name
            new_path = masks_dir / old_name
            if new_path.exists():
                det["mask_path"] = str(new_path.resolve())


def process_image(json_path: Path, ann_dir: Path):
    """Обрабатывает одну картинку из review/."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections", [])
    stem = json_path.stem
    annotated_path = json_path.parent / f"{stem}_annotated.jpg"

    # Показываем картинку в просмотрщике (если есть GUI)
    if annotated_path.exists():
        try:
            Image.open(annotated_path).show()
        except Exception as e:
            print(f"[WARN] не удалось открыть просмотрщик: {e}")
    else:
        print(f"[WARN] нет {annotated_path.name}")

    print(f"\n{'='*70}")
    print(f"📷 {stem} ({len(detections)} детекций)")
    print(f"{'='*70}")

    # Собираем индексы проблемных детекций (needs_review + rejected)
    needs_review_indices = []
    rejected_indices = []

    for i, det in enumerate(detections):
        bucket = det.get("qc_bucket", "accepted")
        cls = det["class"]
        conf = det.get("confidence", 0)
        reason = det.get("qc_reason") or ""

        if bucket == "accepted":
            print(f"  [{i:2d}] ✅ {cls} (conf={conf:.2f})")
        elif bucket == "needs_review":
            needs_review_indices.append(i)
            print(f"  [{i:2d}] ⚠️  {cls} (conf={conf:.2f}) — {reason}")
        elif bucket == "rejected":
            rejected_indices.append(i)
            print(f"  [{i:2d}] ❌ {cls} (conf={conf:.2f}) — {reason}")

    # Все проблемные детекции
    problem_indices = needs_review_indices + rejected_indices

    if not problem_indices:
        print("\n✨ Все детекции уже accepted, ничего не нужно делать.")
        return

    print(f"\n🔍 Нужно решить по {len(problem_indices)} детекциям "
          f"(⚠️ {len(needs_review_indices)} на проверке, ❌ {len(rejected_indices)} отклонённых):")
    print("   [A]ccept — принять (зелёная рамка)")
    print("   [R]eject — отклонить (красная рамка, маска удалится)")
    print("   [S]kip — пропустить, оставить как есть\n")

    for idx in problem_indices:
        det = detections[idx]
        cls = det["class"]
        reason = det.get("qc_reason", "")
        current_bucket = det.get("qc_bucket")

        # Подсказка по умолчанию в зависимости от текущего статуса
        if current_bucket == "rejected":
            hint = "сейчас rejected, принять? [A/R/S, по ум. R]"
        else:
            hint = "сейчас needs_review [A/R/S, по ум. S]"

        while True:
            choice = input(f"  Детекция [{idx}] {cls} — {reason}\n"
                          f"  {hint}: ").strip().lower()

            # Если пустой ввод — используем дефолт
            if choice == "":
                choice = "r" if current_bucket == "rejected" else "s"

            if choice in ("a", "accept"):
                det["qc_bucket"] = "accepted"
                det["qc_reason"] = "accepted manually"
                print(f"    → ✅ accepted")
                break
            elif choice in ("r", "reject"):
                det["qc_bucket"] = "rejected"
                det["qc_reason"] = "rejected manually"
                print(f"    → ❌ rejected")
                break
            elif choice in ("s", "skip"):
                print(f"    → ⏭️  пропущено (осталось {current_bucket})")
                break
            else:
                print("    Неверный ввод, попробуйте снова.")

    # Сохраняем обновлённый JSON
    data["detections"] = detections
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Удаляем маски отклонённых детекций
    masks_dir = ann_dir / "review" / "masks"
    if masks_dir.exists():
        for i, det in enumerate(detections):
            if det.get("qc_bucket") == "rejected":
                mask_path = det.get("mask_path")
                if mask_path and Path(mask_path).exists():
                    try:
                        Path(mask_path).unlink()
                        print(f"    🗑️  Удалена маска: {Path(mask_path).name}")
                    except OSError as e:
                        print(f"    [WARN] не удалось удалить маску: {e}")

    # Перерисовываем _annotated.jpg с новыми цветами рамок
    image_path = data.get("image")
    if image_path:
        # Ищем картинку с fallback (абсолютный/относительный путь)
        image_file = Path(image_path)
        if not image_file.exists():
            image_file = Path.cwd() / image_path
        if not image_file.exists():
            image_file = ann_dir.parent.parent / image_path

        if image_file.exists():
            draw_qc_annotated(str(image_file), detections, str(annotated_path))
            print(f"    🎨 Перерисован {annotated_path.name}")

    # Определяем, куда перенести картинку
    final_bucket = classify_final(detections)

    if final_bucket == "clean":
        print(f"\n📦 Переношу {stem} в clean/")
        move_image_data(stem, "review", "clean", ann_dir)

        # Обновляем mask_path в JSON после переноса
        new_json_path = ann_dir / "clean" / f"{stem}.json"
        if new_json_path.exists():
            with open(new_json_path, "r", encoding="utf-8") as f:
                new_data = json.load(f)
            update_mask_paths(new_data["detections"], "clean", ann_dir, stem)
            with open(new_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
    else:
        remaining = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        print(f"\n⏸️  Оставляю {stem} в review/ (осталось {remaining} needs_review)")


def main():
    parser = argparse.ArgumentParser(description="Помощник для ручной проверки review/")
    parser.add_argument("--ann-dir", default="ann", help="Корневая папка аннотаций")
    parser.add_argument("--image", default=None, help="Обработать только эту картинку (stem)")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    review_dir = ann_dir / "review"

    if not review_dir.exists():
        print(f"Папка {review_dir} не найдена")
        sys.exit(1)

    if args.image:
        json_path = review_dir / f"{args.image}.json"
        if not json_path.exists():
            print(f"Файл {json_path} не найден")
            sys.exit(1)
        process_image(json_path, ann_dir)
    else:
        json_files = sorted(review_dir.glob("*.json"))
        if not json_files:
            print("🎉 Папка review/ пуста — все картинки уже в clean/")
            return

        print(f"📋 Найдено {len(json_files)} картинок для проверки в review/")
        for json_path in json_files:
            process_image(json_path, ann_dir)

        # Финальная статистика
        remaining = len(list(review_dir.glob("*.json")))
        clean_count = len(list((ann_dir / "clean").glob("*.json")))
        print(f"\n{'='*70}")
        print(f"✅ Готово! clean: {clean_count}, review: {remaining}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()