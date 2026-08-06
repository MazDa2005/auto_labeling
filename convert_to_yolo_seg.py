"""
Конвертация внутреннего формата аннотаций в YOLO segmentation формат.

Автоматически разделяет датасет на train/val с настраиваемой долей.
Поддерживает маски (polygon из контура) и fallback на bounding box.

Пример использования:
    python convert_to_yolo_seg.py \
        --annotations-dir annotations/ \
        --classes-file classes.json \
        --output-dir dataset_yolo/ \
        --val-ratio 0.2
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml


def load_class_mapping(classes_file: str) -> dict[str, int]:
    """Загружает маппинг классов из JSON файла."""
    with open(classes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "classes" not in data:
        raise ValueError(f"Неверный формат {classes_file}: отсутствует ключ 'classes'")
    
    return {c["name"]: c["index"] for c in data["classes"]}


def mask_to_polygon(mask_path: str, epsilon_frac: float = 0.005) -> list[tuple[float, float]] | None:
    """
    Извлекает контур из маски и упрощает его до полигона.
    
    Args:
        mask_path: Путь к маске (PNG в градациях серого)
        epsilon_frac: Коэффициент упрощения контура (0.005 = 0.5% от периметра)
    
    Returns:
        Список точек полигона [(x1, y1), (x2, y2), ...] или None
    """
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
    """Конвертирует bounding box [x1, y1, x2, y2] в полигон из 4 точек."""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def convert_detection(det: dict, img_w: int, img_h: int, class_mapping: dict[str, int]) -> str | None:
    """
    Конвертирует одну детекцию в формат YOLO segmentation.
    
    Формат: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    Координаты нормализованы в [0, 1].
    
    Returns:
        Строка в формате YOLO или None если класс не найден
    """
    cls_name = det.get("class")
    if not cls_name:
        return None
    
    if cls_name not in class_mapping:
        print(f"[WARN] Класс '{cls_name}' отсутствует в classes.json — пропускаю", file=sys.stderr)
        return None
    
    class_id = class_mapping[cls_name]
    
    # Пытаемся получить полигон из маски
    polygon = None
    mask_path = det.get("mask_path")
    if mask_path and Path(mask_path).exists():
        polygon = mask_to_polygon(mask_path)
    
    # Fallback на bounding box
    if polygon is None:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        polygon = bbox_to_polygon(bbox)
    
    # Нормализация координат
    normalized = []
    for x, y in polygon:
        normalized.append(x / img_w)
        normalized.append(y / img_h)
    
    coords_str = " ".join(f"{v:.6f}" for v in normalized)
    return f"{class_id} {coords_str}"


def find_image(image_path: str, json_path: Path, annotations_dir: Path) -> Path | None:
    """
    Ищет картинку по пути из JSON, с fallback на альтернативные расположения.
    
    Проверяет:
    1. Абсолютный путь из JSON
    2. Относительно папки с аннотациями
    3. В подпапке frames/ рядом с аннотациями
    """
    img_path = Path(image_path)
    
    # 1. Прямой путь
    if img_path.exists():
        return img_path
    
    # 2. Относительно папки с аннотациями
    alt_path = annotations_dir / img_path.name
    if alt_path.exists():
        return alt_path
    
    # 3. В папке frames/ рядом с аннотациями
    frames_path = annotations_dir.parent / "frames" / img_path.name
    if frames_path.exists():
        return frames_path
    
    # 4. В той же папке, что и JSON
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
    """
    Обрабатывает список JSON файлов и сохраняет в папку сплита.
    
    Returns:
        (converted, skipped) — количество сконвертированных и пропущенных файлов
    """
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
        
        # Проверяем наличие пути к картинке
        if not image_path:
            print(f"[WARN] {json_path.name}: отсутствует поле 'image'", file=sys.stderr)
            skipped += 1
            continue
        
        # Ищем картинку
        img_path = find_image(image_path, json_path, annotations_dir)
        if img_path is None:
            print(f"[WARN] {json_path.name}: картинка не найдена ({image_path})", file=sys.stderr)
            skipped += 1
            continue
        
        # Получаем размеры изображения
        if not width or not height:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] {json_path.name}: не удалось прочитать картинку", file=sys.stderr)
                skipped += 1
                continue
            height, width = img.shape[:2]
        
        # Конвертируем детекции
        lines = []
        for det in detections:
            line = convert_detection(det, width, height, class_mapping)
            if line:
                lines.append(line)
        
        # Сохраняем label
        stem = img_path.stem
        label_path = labels_out / f"{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        # Копируем изображение
        image_ext = img_path.suffix
        shutil.copy(img_path, images_out / f"{stem}{image_ext}")
        converted += 1
        
        if converted % 50 == 0:
            print(f"  Обработано: {converted} картинок...", file=sys.stderr)
    
    return converted, skipped


def split_dataset(
    json_files: list[Path],
    val_ratio: float
) -> tuple[list[Path], list[Path]]:
    """
    Разделяет файлы на train и val.
    
    Args:
        json_files: Список всех JSON файлов
        val_ratio: Доля валидационной выборки (0.0-1.0)
    
    Returns:
        (train_files, val_files)
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio должен быть между 0.0 и 1.0, получено {val_ratio}")
    
    # Фиксированный seed для воспроизводимости
    random.seed(42)
    shuffled = json_files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    
    # Обработка очень маленьких датасетов
    if total <= 2:
        print(f"[WARN] Очень мало файлов ({total}) — всё пойдёт в train", file=sys.stderr)
        return shuffled, []
    
    if total < 10:
        print(f"[WARN] Мало файлов ({total}) — используем сплит 70/30", file=sys.stderr)
        val_ratio = 0.3
    
    val_count = max(1, int(total * val_ratio))
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    
    return train_files, val_files


def generate_data_yaml(
    output_dir: Path,
    classes_file: str,
    train_count: int,
    val_count: int
) -> dict:
    """Генерирует data.yaml для YOLO."""
    with open(classes_file, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    
    names = [c["name"] for c in sorted(classes_data["classes"], key=lambda c: c["index"])]
    
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    
    return data_yaml


def convert(
    annotations_dir: str,
    classes_file: str,
    output_dir: str,
    val_ratio: float = 0.2
):
    """
    Основная функция конвертации.
    
    Args:
        annotations_dir: Папка с JSON аннотациями
        classes_file: Путь к classes.json
        output_dir: Папка для выходного датасета
        val_ratio: Доля валидационной выборки (0.0-1.0)
    """
    print("=" * 60)
    print("🚀 Начало конвертации в YOLO segmentation формат")
    print("=" * 60)
    
    # Валидация входных данных
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        print(f"[ERROR] Папка аннотаций не найдена: {annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    classes_path = Path(classes_file)
    if not classes_path.exists():
        print(f"[ERROR] Файл классов не найден: {classes_file}", file=sys.stderr)
        sys.exit(1)
    
    # Загружаем маппинг классов
    class_mapping = load_class_mapping(classes_file)
    print(f"📋 Загружено {len(class_mapping)} классов")
    
    # Собираем все JSON файлы
    json_files = [
        p for p in annotations_path.glob("*.json")
        if not p.name.startswith("_")  # Пропускаем служебные файлы
    ]
    
    if not json_files:
        print(f"[ERROR] Не найдено JSON-аннотаций в {annotations_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"📁 Найдено {len(json_files)} файлов аннотаций")
    
    # Разделяем на train/val
    train_files, val_files = split_dataset(json_files, val_ratio)
    print(f"📊 Разделение: Train = {len(train_files)}, Val = {len(val_files)}")
    
    # Создаём структуру папок
    output_path = Path(output_dir)
    splits = {
        "train": train_files,
        "val": val_files,
    }
    
    total_converted = 0
    total_skipped = 0
    
    # Обрабатываем каждый сплит
    for split_name, files in splits.items():
        if not files:
            print(f"\n⚠️  Сплит '{split_name}' пуст — пропускаю", file=sys.stderr)
            continue
        
        images_out = output_path / "images" / split_name
        labels_out = output_path / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Обработка сплита '{split_name}' ({len(files)} файлов)...")
        converted, skipped = process_split(
            files, class_mapping, images_out, labels_out, annotations_path, split_name
        )
        
        total_converted += converted
        total_skipped += skipped
        print(f"✅ {split_name}: {converted} сконвертировано, {skipped} пропущено")
    
    # Генерируем data.yaml
    print("\n📝 Генерация data.yaml...")
    data_yaml = generate_data_yaml(output_path, classes_file, len(train_files), len(val_files))
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print(f"🎉 Конвертация завершена!")
    print(f"📊 Всего: {total_converted} картинок сконвертировано, {total_skipped} пропущено")
    print(f"📂 Путь к датасету: {output_dir}")
    print(f"📄 data.yaml создан в: {output_path / 'data.yaml'}")
    print("=" * 60)
    
    if total_converted == 0:
        print("[ERROR] Ни одна картинка не была сконвертирована!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Конвертация аннотаций в YOLO segmentation формат"
    )
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Папка с JSON аннотациями"
    )
    parser.add_argument(
        "--classes-file",
        default="classes.json",
        help="Путь к файлу classes.json (по умолчанию: classes.json)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Папка для выходного датасета"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля валидационной выборки от 0.0 до 1.0 (по умолчанию: 0.2)"
    )
    
    args = parser.parse_args()
    
    convert(
        args.annotations_dir,
        args.classes_file,
        args.output_dir,
        args.val_ratio
    )
