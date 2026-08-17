"""
merge_project_annotations.py — объединяет JSON-аннотации нескольких проектов,
у которых ОДНИ И ТЕ ЖЕ кадры, но РАЗНЫЕ группы классов в detections
(например, пайплайн гонялся отдельно с classes_group_ppe.json и classes_group_other.json).

Берёт только картинки, которые ПОЛНОСТЬЮ прошли review (лежат в ann/clean/) во ВСЕХ
перечисленных проектах — если картинка ещё не разобрана в одной из групп, она пропускается
с предупреждением, а не берётся частично.

Автоматически собирает classes.json из всех проектов, если --classes-file не указан.

Пример:
    python merge_project_annotations.py \
        --projects projects/demo_ppe projects/demo_other \
        --output-project projects/demo_merged
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def load_clean(project_dir: Path) -> dict[str, dict]:
    clean_dir = project_dir / "ann" / "clean"
    result = {}
    if not clean_dir.exists():
        return result
    for json_path in clean_dir.glob("*.json"):
        if json_path.name.startswith("_"):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            result[json_path.stem] = json.load(f)
    return result


def collect_classes_from_projects(projects: list[Path]) -> list[dict]:
    """
    Собирает уникальный список классов из всех проектов.
    Пытается сохранить цвета и original_id из исходных classes.json, если они есть.
    """
    all_classes = {}  # name -> {index, original_id, color}
    current_index = 0
    
    for proj in projects:
        # Пытаемся прочитать classes.json проекта
        classes_file = proj / "classes.json"
        project_colors = {}
        
        if classes_file.exists():
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                classes_list = data.get("classes", data.get("classes ", []))
                for c in classes_list:
                    name = c.get("name", c.get("name ", "")).strip()
                    color = c.get("color", c.get("color ", "#808080")).strip()
                    original_id = c.get("original_id", c.get("original_id "))
                    if name:
                        project_colors[name] = {
                            "color": color,
                            "original_id": original_id
                        }
            except Exception:
                pass
        
        # Собираем классы из детекций
        clean_dir = proj / "ann" / "clean"
        if clean_dir.exists():
            for json_path in clean_dir.glob("*.json"):
                if json_path.name.startswith("_"):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for det in data.get("detections", []):
                        cls_name = det.get("class", "").strip()
                        if cls_name and cls_name not in all_classes:
                            all_classes[cls_name] = {
                                "index": current_index,
                                "original_id": project_colors.get(cls_name, {}).get("original_id"),
                                "color": project_colors.get(cls_name, {}).get("color", "#808080")
                            }
                            current_index += 1
                except Exception:
                    pass
    
    # Формируем итоговый список
    result = []
    for name, info in sorted(all_classes.items(), key=lambda x: x[1]["index"]):
        entry = {
            "index": info["index"],
            "name": name,
            "color": info["color"]
        }
        if info["original_id"] is not None:
            entry["original_id"] = info["original_id"]
        result.append(entry)
    
    return result


def save_classes_file(classes: list[dict], output_path: Path):
    """Сохраняет список классов в classes.json."""
    data = {"classes": classes}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Создан файл классов: {output_path} ({len(classes)} классов)")


def merge(project_dirs: list[str], output_project: str, classes_file: str | None = None):
    projects = [Path(p) for p in project_dirs]
    project_names = [p.name for p in projects]

    per_project_data = [load_clean(p) for p in projects]

    # только картинки, полностью прошедшие review во ВСЕХ группах
    common_stems = set(per_project_data[0].keys())
    for data in per_project_data[1:]:
        common_stems &= set(data.keys())

    skipped_partial = set()
    for data in per_project_data:
        skipped_partial |= set(data.keys())
    skipped_partial -= common_stems

    if skipped_partial:
        print(f"[WARN] {len(skipped_partial)} картинок пропущено — не готовы ('не в clean/') "
              f"хотя бы в одном из проектов: {sorted(skipped_partial)[:5]}...", file=sys.stderr)

    if not common_stems:
        print("[ERROR] Нет ни одной картинки, готовой во всех проектах сразу", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_project)
    out_ann = output_path / "ann" / "clean"
    out_frames = output_path / "frames"
    out_masks = out_ann / "masks"
    out_ann.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    merged_count = 0

    for stem in sorted(common_stems):
        merged_detections = []
        base_data = None

        for proj, proj_name, data in zip(projects, project_names, per_project_data):
            entry = data[stem]
            if base_data is None:
                base_data = entry  # берём width/height/image из первой группы

            src_masks_dir = proj / "ann" / "clean" / "masks"
            for det in entry.get("detections", []):
                det = dict(det)
                old_mask = det.get("mask_path")
                if old_mask:
                    old_mask_path = Path(old_mask)
                    if old_mask_path.exists():
                        # префикс именем проекта — на случай совпадения имён масок между группами
                        new_name = f"{proj_name}_{old_mask_path.name}"
                        new_path = out_masks / new_name
                        shutil.copy(old_mask_path, new_path)
                        det["mask_path"] = str(new_path.resolve())
                    else:
                        det["mask_path"] = None
                merged_detections.append(det)

        # копируем сам кадр один раз (картинки идентичны между группами)
        src_frame = None
        for proj in projects:
            candidate = proj / "frames" / f"{stem}.jpg"
            if candidate.exists():
                src_frame = candidate
                break
            for ext in (".jpeg", ".png"):
                candidate = proj / "frames" / f"{stem}{ext}"
                if candidate.exists():
                    src_frame = candidate
                    break
        if src_frame:
            shutil.copy(src_frame, out_frames / src_frame.name)

        merged_json = {
            "image": str((out_frames / (src_frame.name if src_frame else f"{stem}.jpg")).resolve()),
            "width": base_data.get("width"),
            "height": base_data.get("height"),
            "detections": merged_detections,
        }
        with open(out_ann / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=2, ensure_ascii=False)

        merged_count += 1

    print(f"Готово: {merged_count} картинок объединено из {len(projects)} проектов")
    print(f"Результат: {output_path} (ann/clean/, frames/)")
    
    # Автоматически собираем classes.json из всех проектов
    if classes_file and Path(classes_file).exists():
        # Если указан существующий файл — копируем его
        shutil.copy(classes_file, output_path / "classes.json")
        print(f"Скопирован файл классов: {classes_file}")
    else:
        # Иначе собираем автоматически
        classes = collect_classes_from_projects(projects)
        save_classes_file(classes, output_path / "classes.json")
    
    print(f"Дальше: python convert_to_yolo_seg.py --annotations-dir {out_ann} "
          f"--classes-file {output_path / 'classes.json'} --output-dir {output_path}/dataset_yolo")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--projects", nargs="+", required=True, help="Пути к проектам-группам (каждый со своими frames/, ann/clean/)")
    p.add_argument("--output-project", required=True)
    p.add_argument("--classes-file", default=None, help="Опционально: использовать существующий classes.json вместо автоматического сбора")
    args = p.parse_args()

    merge(args.projects, args.output_project, args.classes_file)