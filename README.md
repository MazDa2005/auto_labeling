# Auto-Labeling Pipeline

Автоматическая разметка видео-кадров для обучения lightweight student-модели (real-time детекция по нескольким камерам). Использует связку teacher-моделей `detect → refine → verify → qc`, результат конвертируется в формат YOLO segment для обучения.

## Архитектура

```
detect  → LocateAnything-3B или Grounding DINO находят объекты по текстовым промптам из classes.json
refine  → SAM3 уточняет грубый bbox до точной сегментационной маски (box-prompted режим)
verify  → CLIP проверяет семантическое соответствие класса содержимому bbox (relative scoring)
qc      → QC-фильтр применяет эвристики (размер, пропорции, дубли, конфликты, контекст PPE) и сигнал CLIP, раскладывая результат на папки clean/ и review/
```

Каждый этап — независимый скрипт в своём conda-окружении (у моделей несовместимые версии
зависимостей). Оркестратор (`batch_label.py` / `auto_label.py`) не знает ничего о конкретных
моделях — только про три типа этапов и их порядок, заданный в `pipeline_config.yaml`.

## Структура проекта

```
auto_labeling/
├── auto_label.py            # обработка ОДНОЙ картинки 
├── batch_label.py           # обработка ПАПКИ картинок 
├── convert_to_yolo_seg.py   # наш JSON+маски -> YOLO segment 
├── qc_filter.py             # эвристический  фильтр качества 
├── merge_datasets.py        # объединение нескольких YOLO-датасетов в один
├── pipeline_config.yaml     # конфиг этапов для СЕРВЕРА
├── classes.json             # единый список классов с промптами для всех моделей
└── teachers/
    ├── pipeline_utils.py         # общие утилиты (iou, merge, mask_to_bbox, list_images, промпты)
    ├── locate_infer.py           # detect-этап: LocateAnything-3B 
    ├── dino_infer.py             # detect-этап: Grounding DINO 
    ├── sam3_refine.py            # refine-этап: SAM3 box-prompted 
    ├── clip_verify.py            # verify-этап: CLIP relative scoring 
    └── locate_anything_worker.py # класс-обёртка над LocateAnything 
```

## Установка

Два conda-окружения (у SAM3 и LocateAnything разные версии зависимостей — не смешивать):

```bash
conda create --name sam3 python==3.10 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U ultralytics
pip install --no-deps git+https://github.com/ultralytics/CLIP.git

conda create --name nvidia python==3.10 -y
conda activate nvidia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.25.0 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5
pip install -U huggingface_hub
```

Плюс `pyyaml` в окружении, из которого запускаете оркестратор(sam3):
```bash
pip install pyyaml
```

Версию `cuXXX` в URL подставлять под фактическую CUDA-версию на машине (`nvidia-smi`).

## Веса моделей

Не ставятся через pip, скачивать отдельно:

```bash
# LocateAnything-3B
huggingface-cli download nvidia/LocateAnything-3B --local-dir /path/to/models/LocateAnything-3B
``` 

```bash
# SAM3 
https://www.modelscope.cn/models/facebook/sam3 
``` 

```bash
import open_clip
import torch

# Скачивает модель и кэширует в ~/.cache/clip/
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', 
    pretrained='datacomp_xl_s13b_b90k'
)
print("Модель скачана и закэширована")
print(f"Кэш: {open_clip.get_pretrained_url('ViT-L-14', 'datacomp_xl_s13b_b90k')}")
```


## Использование

### Одна картинка (быстрый тест)

```bash
python auto_label.py --image frame.jpg --classes person,helmet --out annotations/frame.json
```

### Папка картинок (основной способ) — классы берутся из classes.json автоматически

```bash
python batch_label.py --images-dir frames/ --config pipeline_config.yaml --out-dir annotations/
```

Явно указать классы (переопределить `classes.json`):
```bash
python batch_label.py --images-dir frames/ --classes person,helmet --out-dir annotations/
```

Результат в `annotations/`:
```
annotations/
├── <имя>.json              # детекции: class, bbox, confidence, source, mask_path
├── <имя>_annotated.jpg     # визуализация с рамками и масками — ТОЛЬКО для проверки глазами
├── masks/<имя>/*.png       # сегментационные маски по каждому объекту
├── _stages/<имя_модели>/*.json       # промежуточные сводки по каждой модели
└── _batch_summary.json     # сводка по всему батчу (успех/ошибка на картинку)
```

### Конвертация в YOLO segment (для обучения)

```bash
python convert_to_yolo_seg.py --annotations-dir annotations/ --classes-file classes.json --output-dir dataset_yolo/
```

Даёт готовую структуру `images/train/`, `labels/train/`, `data.yaml`. Картинки в `images/`
— чистые оригиналы без разметки, вся разметка — в `.txt`-файлах (нормализованные координаты
полигона на строку, первое число — class_id из `classes.json`).

Если у детекции нет сохранённой маски (verify отбросил / refine выключен) — используется bbox-прямоугольник как fallback-полигон (не настоящая сегментация, это стоит иметь в виду при контроле качества).

### Объединение нескольких партий разметки

```bash
python merge_datasets.py --datasets dataset_batch1/ dataset_batch2/ --output-dir dataset_merged/
```

Останавливается с ошибкой, если у датасетов разный список классов (защита от путаницы
class_id между партиями) — пересоздайте датасеты через `convert_to_yolo_seg.py` с актуальным
`classes.json`, если список менялся.


streamlit run main.py --server.address 0.0.0.0 --server.port 8501
