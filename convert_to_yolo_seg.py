"""
Конвертация внутреннего формата аннотаций в YOLO segmentation формат.

Автоматически разделяет датасет на train/val (80/20).
Поддерживает маски (polygon из контура) и fallback на bounding box.

"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path
import cv2
import yaml

def _get_key(d: dict, key: str, default=None):
    """
    Аккуратно читает ключи с учётом возможных пробелов:
    "classes" / "classes ", "name" / "name " и т.п.
    """
    if key in d:
        return d[key]
    if f"{key} " in d:
        return d[f"{key} "]
    return default


def load_names_and_colors(classes_file: str) -> tuple[list[str], dict[str, str]]:
    """
    Читает имена классов и цвета из classes.json.
    Возвращает (names, colors).
    """
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes_list = _get_key(data, "classes", [])

    classes_list = sorted(
        classes_list,
        key=lambda c: int(_get_key(c, "index", 0) or 0)
    )

    names = []
    colors = {}

    for c in classes_list:
        name = str(_get_key(c, "name", "")).strip()
        color = str(_get_key(c, "color", "")).strip()

        if not name:
            continue

        names.append(name)

        if color:
            colors[name] = color

    return names, colors

def load_class_mapping(classes_file: str) -> dict[str, int]:
    """Загружает маппинг классов, устойчив к пробелам в ключах JSON."""
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Поддержка пробелов в ключах ("classes " или "classes")
    classes_list = data.get("classes", data.get("classes ", []))
    return {
        c.get("name", c.get("name ", "")).strip(): c.get("index", c.get("index ", 0)) 
        for c in classes_list
    }

def mask_to_polygon(mask_path: str, epsilon_frac: float = 0.005) -> list[tuple[float, float]] | None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None
    
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) < 3:
        return None
    
    return [(float(p[0][0]), float(p[0][1])) for p in approx]

def bbox_to_polygon(bbox: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def resolve_mask_path(mask_path: str, annotations_dir: Path) -> Path | None:
    """
    Ищет файл маски даже если mask_path устарел (маска переехала при review→clean).
    Сначала проверяет путь как есть, потом ищет по имени файла в соседних папках масок.
    """
    p = Path(mask_path)
    if p.exists():
        return p
    # Маска могла переехать review/masks → clean/masks или наоборот
    name = p.name
    for base in [annotations_dir, annotations_dir.parent]:
        for subdir in ("masks", "clean/masks", "review/masks"):
            cand = base / subdir / name
            if cand.exists():
                return cand
    return None


# Глобальные счётчики для диагностики (сбрасываются при каждом запуске convert)
_mask_ok = 0
_mask_fallback = 0


def convert_detection(det: dict, img_w: int, img_h: int, class_mapping: dict[str, int],
                       annotations_dir: Path | None = None) -> str | None:
    global _mask_ok, _mask_fallback

    if det.get("qc_bucket") != "accepted":
        return None
    cls_name = det.get("class", "").strip()
    if not cls_name or cls_name not in class_mapping:
        print(f"[WARN] Класс '{cls_name}' отсутствует в маппинге — пропускаю", file=sys.stderr)
        return None

    class_id = class_mapping[cls_name]

    polygon = None
    mask_path = det.get("mask_path")
    if mask_path:
        resolved = resolve_mask_path(mask_path, annotations_dir) if annotations_dir else (
            Path(mask_path) if Path(mask_path).exists() else None
        )
        if resolved:
            polygon = mask_to_polygon(str(resolved))

    if polygon is not None:
        _mask_ok += 1
    else:
        _mask_fallback += 1
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        polygon = bbox_to_polygon(bbox)

    normalized = []
    for x, y in polygon:
        normalized.append(x / img_w)
        normalized.append(y / img_h)

    coords_str = " ".join(f"{v:.6f}" for v in normalized)
    return f"{class_id} {coords_str}"

def find_image(image_path: str, json_path: Path, annotations_dir: Path) -> Path | None:
    img_path = Path(image_path)
    if img_path.exists():
        return img_path
    alt_path = annotations_dir / img_path.name
    if alt_path.exists():
        return alt_path
    frames_path = annotations_dir.parent / "frames" / img_path.name
    if frames_path.exists():
        return frames_path
    same_dir = json_path.parent / img_path.name
    if same_dir.exists():
        return same_dir
    return None

def process_split(
    json_files: list[Path],
    class_mapping: dict[str, int],
    images_out: Path,
    labels_out: Path,
    annotations_dir: Path,
    split_name: str
) -> tuple[int, int]:
    converted, skipped = 0, 0
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] {json_path.name}: невалидный JSON — {e}", file=sys.stderr)
            skipped += 1
            continue
        
        image_path = data.get("image")
        width, height = data.get("width"), data.get("height")
        detections = data.get("detections", [])
        
        if not image_path:
            print(f"[WARN] {json_path.name}: отсутствует поле 'image'", file=sys.stderr)
            skipped += 1
            continue
        
        img_path = find_image(image_path, json_path, annotations_dir)
        if img_path is None:
            print(f"[WARN] {json_path.name}: картинка не найдена ({image_path})", file=sys.stderr)
            skipped += 1
            continue
        
        if not width or not height:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] {json_path.name}: не удалось прочитать картинку", file=sys.stderr)
                skipped += 1
                continue
            height, width = img.shape[:2]
        
        lines = []
        for det in detections:
            line = convert_detection(det, width, height, class_mapping, annotations_dir)
            if line:
                lines.append(line)
        
        stem = img_path.stem
        label_path = labels_out / f"{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        image_ext = img_path.suffix
        shutil.copy(img_path, images_out / f"{stem}{image_ext}")
        converted += 1
        
        if converted % 50 == 0:
            print(f"  Обработано: {converted} картинок...", file=sys.stderr)
    
    return converted, skipped

def split_dataset(json_files: list[Path], val_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    random.seed(42)  # Для воспроизводимости
    shuffled = json_files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    if total <= 2:
        print(f"[WARN] Очень мало файлов ({total}) — всё пойдёт в train", file=sys.stderr)
        return shuffled, []
    
    val_count = max(1, int(total * val_ratio))
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    
    return train_files, val_files

def convert(annotations_dir: str, classes_file: str, output_dir: str, val_ratio: float = 0.2):
    print("Начало конвертации в YOLO segmentation формат")

    
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        print(f"[ERROR] Папка аннотаций не найдена: {annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    classes_path = Path(classes_file)
    if not classes_path.exists():
        print(f"[ERROR] Файл классов не найден: {classes_file}", file=sys.stderr)
        sys.exit(1)
    
    class_mapping = load_class_mapping(classes_file)
    print(f"Загружено {len(class_mapping)} классов")
    
    json_files = [
        p for p in annotations_path.glob("*.json")
        if not p.name.startswith("_")
    ]
    
    if not json_files:
        print(f"[ERROR] Не найдено JSON-аннотаций в {annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Найдено {len(json_files)} файлов аннотаций")
    
    # Честный сплит
    train_files, val_files = split_dataset(json_files, val_ratio)
    print(f"Разделение: Train = {len(train_files)}, Val = {len(val_files)}")
    
    output_path = Path(output_dir)
    splits = {
        "train": train_files,
        "val": val_files,
    }
    
    total_converted = 0
    total_skipped = 0
    global _mask_ok, _mask_fallback
    _mask_ok = 0
    _mask_fallback = 0

    for split_name, files in splits.items():
        if not files:
            print(f"\n⚠️  Сплит '{split_name}' пуст — пропускаю", file=sys.stderr)
            continue
        
        images_out = output_path / split_name / "images"
        labels_out = output_path / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        print(f"\nОбработка сплита '{split_name}' ({len(files)} файлов)...")
        converted, skipped = process_split(
            files, class_mapping, images_out, labels_out, annotations_path, split_name
        )
        
        total_converted += converted
        total_skipped += skipped
        print(f"✅ {split_name}: {converted} сконвертировано, {skipped} пропущено")
    
    # Статистика масок — главный индикатор качества датасета
    total_dets = _mask_ok + _mask_fallback
    if total_dets > 0:
        pct = _mask_ok / total_dets * 100
        print(f"\n📊 Маски: {_mask_ok}/{total_dets} настоящих полигонов ({pct:.1f}%), "
              f"{_mask_fallback} bbox-прямоугольников")
        if pct < 50:
            print("[WARN] Больше половины детекций — прямоугольники. "
                  "Проверь, что mask_path в JSON актуальны перед конвертацией.", file=sys.stderr)

    # Генерация data.yaml с цветами
    names, colors = load_names_and_colors(classes_file)

    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }

    if colors:
        data_yaml["colors"] = colors
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    

    print(f"Всего: {total_converted} картинок сконвертировано, {total_skipped} пропущено")
    print(f"Путь к датасету: {output_dir}")
    print(f"data.yaml создан в: {yaml_path}")
    print("=" * 60)
    
    if total_converted == 0:
        print("[ERROR] Ни одна картинка не была сконвертирована!", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертация аннотаций в YOLO segmentation формат")
    parser.add_argument("--annotations-dir", required=True, help="Папка с JSON аннотациями")
    parser.add_argument("--classes-file", default="classes.json", help="Путь к файлу classes.json")
    parser.add_argument("--output-dir", required=True, help="Папка для выходного датасета")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Доля валидационной выборки (по умолчанию: 0.2)")
    
    args = parser.parse_args()
    
    convert(
        args.annotations_dir,
        args.classes_file,
        args.output_dir,
        args.val_ratio
    )