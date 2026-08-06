"""
Спот-чек для clean/ — выборочная ручная проверка N% картинок, которые прошли QC
автоматически (не имели needs_review/rejected). Отдельно от review_helper.py, потому что
тут решение не "по каждой детекции", а "картинка в целом ок / есть проблема".

Если картинка помечена как проблемная (F) — переносится обратно в review/ с пометкой
flagged_by_spotcheck, чтобы её потом разобрали через review_helper.py как обычно.
"""
import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"checked": {}}


def save_log(log_path: Path, log: dict):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def move_back_to_review(stem: str, clean_dir: Path, ann_dir: Path, reason: str):
    """Переносит картинку из clean/ обратно в review помечает все детекции этой картинки, требующие пересмотра."""
    review_dir = ann_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "masks").mkdir(parents=True, exist_ok=True)

    json_path = clean_dir / f"{stem}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for det in data.get("detections", []):
        if det.get("qc_bucket") == "accepted":
            det["qc_bucket"] = "needs_review"
            det["qc_reason"] = f"flagged by spot-check: {reason}"

    with open(review_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    json_path.unlink()

    ann_src = clean_dir / f"{stem}_annotated.jpg"
    if ann_src.exists():
        shutil.move(str(ann_src), str(review_dir / f"{stem}_annotated.jpg"))

    src_masks = clean_dir / "masks"
    if src_masks.exists():
        for mask_file in src_masks.glob(f"{stem}_*"):
            shutil.move(str(mask_file), str(review_dir / "masks" / mask_file.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True, help="Папка clean/, например ann-1/clean")
    parser.add_argument("--percent", type=float, default=None, help="Процент от непроверенных картинок")
    parser.add_argument("--count", type=int, default=None, help="Или фиксированное число картинок")
    parser.add_argument("--log", default=None, help="Файл лога (по умолчанию: <clean-dir>/../spot_check_log.json)")
    args = parser.parse_args()

    if not args.percent and not args.count:
        print("Укажите --percent или --count", file=sys.stderr)
        sys.exit(1)

    clean_dir = Path(args.clean_dir)
    ann_dir = clean_dir.parent
    log_path = Path(args.log) if args.log else ann_dir / "spot_check_log.json"
    log = load_log(log_path)

    all_json = [p for p in clean_dir.glob("*.json") if not p.name.startswith("_")]
    unchecked = [p for p in all_json if p.stem not in log["checked"]]

    if not unchecked:
        print("Все картинки в clean/ уже проходили спот-чек (по логу). "
              "Удалите/почистите лог, если хотите пересэмплировать.")
        return

    if args.count:
        n = min(args.count, len(unchecked))
    else:
        n = max(1, math.ceil(len(unchecked) * args.percent / 100))

    sample = random.sample(unchecked, n)
    print(f"Всего картинок в clean/: {len(all_json)}, ещё не проверено: {len(unchecked)}")
    print(f"Спот-чек: {n} картинок ({n / len(all_json) * 100:.1f}% от всей папки)\n")

    ok_count, flagged_count = 0, 0

    for json_path in sample:
        stem = json_path.stem
        annotated_path = clean_dir / f"{stem}_annotated.jpg"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_det = len(data.get("detections", []))

        print(f" {stem} ({n_det} детекций, все были accepted)")

        if annotated_path.exists():
            Image.open(annotated_path).show()
        else:
            print(f"[WARN] нет {annotated_path.name}")

        while True:
            choice = input("  [O]k / [F]lag (проблема, вернуть на пересмотр) / [S]kip (не считать проверенным): ").strip().lower()
            if choice in ("o", "ok", ""):
                log["checked"][stem] = {"verdict": "ok"}
                ok_count += 1
                break
            elif choice in ("f", "flag"):
                reason = input("  Кратко, в чём проблема: ").strip() or "не указано"
                move_back_to_review(stem, clean_dir, ann_dir, reason)
                log["checked"][stem] = {"verdict": "flagged", "reason": reason}
                flagged_count += 1
                print(f"    → перенесено в review/ для пересмотра")
                break
            elif choice in ("s", "skip"):
                print("    → пропущено, не отмечено в логе")
                break
            else:
                print("    Неверный ввод, попробуйте снова.")

    save_log(log_path, log)

    print(f"Спот-чек завершён: ok={ok_count}, flagged={flagged_count}")
    if flagged_count:
        print(f"Помеченные картинки перенесены в review/ — разберите их через review_helper.py")
    print(f"Лог сохранён: {log_path}")


if __name__ == "__main__":
    main()