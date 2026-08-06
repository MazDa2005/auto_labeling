# Auto-Labeling Pipeline

Автоматизированный multi-stage pipeline для авто-разметки данных (PPE + железнодорожные объекты) с последующим обучением компактной YOLO student-модели для real-time инференса (15–30 FPS) на нескольких одновременных камерах. Цель — замена legacy видео-аналитики, работающей на ~1 FPS.

## Содержание

- [Идея и архитектура](#идея-и-архитектура)
- [Классы](#классы)
- [Структура проекта](#структура-проекта)
- [Пайплайн детекции: detect → refine → verify → qc](#пайплайн-детекции-detect--refine--verify--qc)
- [Веб-интерфейс (Streamlit + FastAPI)](#веб-интерфейс-streamlit--fastapi)
- [Обучение и тестирование student-модели](#обучение-и-тестирование-student-модели)
- [Бенчмарк](#бенчмарк)
- [Docker-деплой](#docker-деплой)
- [Conda-окружения](#conda-окружения)
- [Разработка на Windows / деплой на iet-08](#разработка-на-windows--деплой-на-iet-08)
- [Известные особенности и решённые проблемы](#известные-особенности-и-решённые-проблемы)
- [Лицензирование](#лицензирование)
- [Roadmap](#roadmap)

## Идея и архитектура

Мы генерируем обучающие данные с помощью связки "тяжёлых" teacher-моделей (detect → refine → verify), фильтруем их через эвристический QC-фильтр и ручное ревью, конвертируем в YOLO-segmentation формат и на этом датасете обучаем лёгкую student YOLO-модель.

Это **не classical soft-target distillation** (teacher и student архитектурно несовместимы на уровне логитов) — teacher-модели используются только для генерации псевдо-разметки (bbox + маска + класс), а student учится с нуля на этой разметке в обычном режиме supervised-обучения.

```
                     ┌─────────────┐    ┌────────────┐    ┌───────────┐    ┌────────────┐
 кадры/картинки ───▶ │   DETECT    │──▶ │   REFINE   │──▶ │  VERIFY   │──▶ │     QC     │
                     │ Grounding   │    │   SAM3     │    │   CLIP    │    │ эвристики  │
                     │ DINO +      │    │ (маски,    │    │(деприори- │    │ + разбивка │
                     │ LocateAny.  │    │ уточнение  │    │ зирован)  │    │ clean/     │
                     │             │    │ bbox)      │    │           │    │ review/    │
                     └─────────────┘    └────────────┘    └───────────┘    └────────────┘
                                                                                    │
                                                                     ручное ревью (Streamlit)
                                                                                    │
                                                                                    ▼
                                                              convert_to_yolo_seg.py
                                                                                    │
                                                                                    ▼
                                                          train_student.py (YOLOv8n/YOLO26n)
                                                                                    │
                                                                                    ▼
                                                              benchmark_student.py (mAP + FPS)
```

Пайплайн **config-driven**: оркестратор (`batch_label.py`) не знает конкретных моделей — только типы этапов (`detect` / `refine` / `verify` / `qc`) из `pipeline_config.yaml`. Чтобы заменить модель — меняется `script`/`conda_env`/`model_path` в конфиге, код оркестратора не трогается. Каждый этап запускается **один раз на весь батч картинок** (модель грузится один раз), а не по одной картинке — это сильно экономит время на detect/refine/verify.

## Классы

11 канонических классов в `classes.json` (PPE + железнодорожные объекты), единый источник правды для всех моделей (промпты для Grounding DINO / LocateAnything / CLIP берутся оттуда же):

`welding_gloves`, `welding`, `gas_mask`, `gloves`, `protective_headphones`, `helmet`, `orange_vest`, `person`, `mask`, `welding_mask`, `glasses` (+ `railcar` для железнодорожного домена).

Каждый класс хранит `index`, `color` (для отрисовки), список `prompts` (текстовые запросы для zero-shot детекторов).

## Структура проекта

```
auto_labeling/
├── app.py                     # Streamlit UI (полностью, все страницы пайплайна)
├── server.py                  # FastAPI backend — фоновые задачи (extract/pipeline/train/test)
├── batch_label.py             # Оркестратор пайплайна detect→refine→verify→qc
├── pipeline_config.yaml       # Конфигурация этапов пайплайна (единственное место для правки моделей)
├── classes.json                # Классы + промпты + цвета (единый источник правды)
├── pipeline_utils.py           # Общие утилиты teacher-скриптов (IoU, работа с classes.json и т.д.)
├── teachers/                   # (в конфиге пути вида teachers/dino_infer.py и т.д.)
│   ├── dino_infer.py           # DETECT: Grounding DINO
│   ├── locate_infer.py         # DETECT: LocateAnything-3B (супплементарный)
│   ├── locate_anything_worker.py
│   ├── sam3_refine.py          # REFINE: SAM3 (маски + уточнение bbox)
│   └── clip_verify.py          # VERIFY: CLIP (деприоритизирован, enabled: false)
├── qc_filter.py                 # QC: эвристический фильтр, раскладка clean/ и review/
├── review_helper.py             # CLI-помощник для ручного ревью review/ (альтернатива Streamlit)
├── spot_check.py                 # Выборочная проверка качества clean/ (после QC)
├── convert_to_yolo_seg.py        # Конвертация в YOLO-segmentation формат + train/val split
├── merge_datasets.py             # Объединение нескольких YOLO-датасетов в один (CLI)
├── frame_out.py                  # Извлечение кадров из видео
├── train_student.py              # Обучение student YOLO (ultralytics)
├── test_student.py               # Инференс на картинке/видео
├── benchmark_student.py          # Бенчмарк качества (mAP) + скорости (multi-stream FPS)
├── docker/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── environments/             # conda-pack tar.gz окружений (dino/sam3/nvidia), не в git
├── docker-compose.yml
├── nvidia.yml / sam3.yml / dino.yml  # экспортированные conda environment файлы
└── requirements-api.txt
```

## Пайплайн детекции: detect → refine → verify → qc

### 1. DETECT
- **Grounding DINO** (`dino_infer.py`, env `dino`) — основной детектор, zero-shot по текстовым промптам из `classes.json`. Даёт меньше false positives, чем LocateAnything, в нашем домене — используется как primary.
- **LocateAnything-3B** (`locate_infer.py` + `locate_anything_worker.py`, env `nvidia`) — супплементарный детектор.

Детекции всех `detect`-этапов сливаются с dedup по IoU (`merge_new_detections` в `pipeline_utils.py`).

### 2. REFINE
- **SAM3** через Ultralytics (`sam3_refine.py`, env `sam3`) — уточняет bbox от detect-этапа до сегментационной маски, пересчитывает bbox из маски, добавляет `mask_path`, `mask_area`, `refine_iou` (IoU между исходным и уточнённым bbox — сигнал расхождения detect/refine).

### 3. VERIFY (опционально)
- **CLIP** (`clip_verify.py`, env `sam3`, open_clip + SigLIP) — сравнивает кроп со всеми классами, добавляет `clip_agrees`, `clip_predicted_class`, `clip_predicted_score`, `clip_own_class_score`. **Отключён в конфиге** (`enabled: false`) — cosine similarity кластеризуется в узком диапазоне для мелких PPE-объектов, недостаточно дискриминативен. Оставлен в пайплайне для дальнейших экспериментов.

### 4. QC (`qc_filter.py`)
Финальный эвристический фильтр на уровне каждой детекции и попарных конфликтов:
- confidence < 0.5 → `rejected`
- вырожденный bbox (относительная площадь < 0.000025) → `rejected`
- низкий `refine_iou` (< 0.6, расхождение detect/refine) → `needs_review`
- несогласие CLIP (если включён) → `needs_review`
- дубликаты одного класса с высоким IoU → отклоняется тот, что с меньшим confidence
- взаимоисключающие классы в одном месте (`gloves`/`welding_gloves`, `mask`/`gas_mask`/`welding_mask`) → `needs_review`
- человек без единого предмета СИЗ на кадре с высоким confidence → `needs_review`

Картинки раскладываются по `ann/clean/` (все детекции accepted) и `ann/review/` (есть хотя бы одна needs_review/rejected), маски копируются с префиксом `{image_stem}_`.

### Ручное ревью
- **Streamlit UI** (`app.py`, страница Review) — основной способ разбора `review/`: accept/reject по каждой детекции, при отсутствии needs_review — перенос в `clean/`.
- **`review_helper.py`** — CLI-альтернатива с тем же workflow (A/R/S по каждой проблемной детекции).
- **`spot_check.py`** / страница Spot Check в Streamlit — выборочная (%ـ или фикс. количество) проверка уже прошедших QC картинок из `clean/` для контроля качества самого пайплайна; при обнаружении ошибки картинка возвращается в `review/`.

### Конвертация и объединение
- **`convert_to_yolo_seg.py`** — конвертирует `clean/` в YOLO-segmentation формат (полигон из маски, fallback на bbox-полигон), делает train/val split, генерирует `data.yaml`.
- **`merge_datasets.py`** — CLI-объединение нескольких уже сконвертированных YOLO-датасетов с проверкой согласованности списка классов и защитой от коллизий имён файлов (префикс имени проекта). Дублирует логику объединения, встроенную в страницу "Объединение датасетов" в `app.py`.

## Веб-интерфейс (Streamlit + FastAPI)

`app.py` — единая точка входа (`streamlit run app.py`). FastAPI-бэкенд (`server.py`) поднимается **автоматически в фоновом потоке того же процесса** через `@st.cache_resource` + кастомный `_BackgroundUvicornServer(uvicorn.Server)`, у которого отключена установка signal handler'ов (обязательно для uvicorn вне главного потока). Отдельно поднимать `python server.py` не нужно.

Страницы:
1. **Домой** — обзор существующих проектов.
2. **Загрузка** — видео (несколько сразу, `video.mp4`/`video_N.mp4`) или готовые изображения.
3. **Извлечение кадров** — извлечение кадров из всех видео проекта с заданным FPS.
4. **Пайплайн** — выбор классов, запуск `batch_label.py` через `/run-pipeline`.
5. **Review** — ручная проверка `ann/review/` по каждой детекции (accept/reject), статистика, перенос в `clean/`.
6. **Spot Check** — выборочный контроль качества `ann/clean/` (% от датасета).
7. **Конвертация** — запуск `convert_to_yolo_seg.py`, скачивание ZIP.
8. **Объединение датасетов** — объединение ≥2 сконвертированных проектов в новый, генерация `data.yaml`, скачивание ZIP.
9. **Обучение модели** — выбор проектов-источников, гиперпараметры (epochs/imgsz/batch/device/base_model), расширенные настройки для стабильности в Docker (`workers=0`, `plots=False`, `cache=False` по умолчанию — предотвращают зависание DataLoader), прогресс в реальном времени через файл прогресса.
10. **Тест модели** — инференс на изображении/видео выбранным checkpoint'ом, таблица детекций, распределение по классам, метрики скорости.

Долгие задачи (`extract-frames`, `run-pipeline`, `convert-to-yolo`, `train-model`, `test-model`) выполняются через FastAPI `BackgroundTasks` с polling `/task/{task_id}`; обучение — через `subprocess.Popen` с чтением файла прогресса (не блокирующий `subprocess.run`, т.к. обучение может идти часами).

## Обучение и тестирование student-модели

- **`train_student.py`** — объединяет YOLO-датасеты нескольких проектов (с префиксами имён), обучает YOLO (`yolo11n/s/m.pt`, `yolo26n.pt` — выбор в UI) через ultralytics в env `sam3`. Пишет прогресс в JSON-файл (эпоха/стадия) для отображения в Streamlit. `MemoryCallback`/`gc.collect()` + `torch.cuda.empty_cache()` после каждой эпохи и `torch.multiprocessing.set_sharing_strategy('file_system')` — защита от shared-memory ошибок в Docker.
- **`test_student.py`** — инференс на картинке или видео, отрисовка боксов цветами из `classes.json`, JSON с детекциями/метриками.
- Веса моделей хранятся в `/home/iet/iet-share/models/`; чекпоинты обучения — в `projects/<project>/runs/<run_name>/weights/best.pt`.

## Бенчмарк

**`benchmark_student.py`** оценивает student-модель по двум осям:

1. **Качество** — precision/recall/mAP50/mAP50-95, общие и по каждому классу отдельно (`model.val()` на val-сплите).
2. **Скорость** — FPS на 1 потоке и при N параллельных потоках (симуляция N одновременных камер). Каждый "поток камеры" — **отдельный процесс** (`multiprocessing`, не threading) со своим экземпляром модели: безопаснее и реалистичнее отражает реальный деплой. Используется `spawn` start method (обязательно — `fork` может сломать CUDA-контекст в дочернем процессе внутри conda/CUDA окружений).

Первый прогон бенчмарка: **overall mAP50 = 0.678**. Дальнейший анализ по классам и оптимизация — в процессе.

## Docker-деплой

Разработка ведётся на Windows-машине (есть доступ в интернет), деплой — на сервере **iet-08** (Ubuntu 24.04, RTX 5060 Ti 16GB VRAM), у которого доступа в интернет нет.

**Схема:**
1. Образ собирается локально на Windows.
2. `docker save | gzip` → перенос через scp/PowerShell-скрипты на iet-08.
3. Офлайн `docker load` на iet-08.

**Multi-stage build** (три стадии) решает конфликт `.dockerignore` vs `ADD`/`COPY` и убирает bloat промежуточных слоёв:
- `app-src` (Alpine) — копирует код проекта, удаляет `docker/environments/` (не должны попасть в финальный образ этим путём).
- `envbuilder` (CUDA base) — через `ADD` распаковывает `conda-pack` tar.gz окружений (`dino.tar.gz`, `sam3.tar.gz`, `nvidia.tar.gz`) без оставления архивов на диске.
- финальная стадия — собирает чистые артефакты через `COPY --from=` из предыдущих стадий.

Базовый образ — `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (уже содержит conda в `/opt/conda`), что избавляет от отдельной установки Miniconda.

`entrypoint.sh` запускает Streamlit напрямую из `/opt/conda/envs/sam3/bin/streamlit` (activate недоступен без базового conda-инициализатора в shell).

`docker-compose.yml` монтирует модели, sam3, LocateAnything-3B и `projects/` как volumes, пробрасывает GPU (`gpus: all`), порт `8501`.

## Conda-окружения

Три окружения, адресуются напрямую по пути к бинарнику (`/opt/conda/envs/<env>/bin/python`), а не через `conda run` — быстрее (экономия 2-3 сек на запуск) и надёжнее в Docker:

| env      | назначение                                                        |
|----------|--------------------------------------------------------------------|
| `sam3`   | основное — ultralytics, всё обучение/инференс YOLO, SAM3, CLIP, Streamlit/FastAPI |
| `dino`   | Grounding DINO                                                     |
| `nvidia` | LocateAnything-3B                                                  |

Отдельного окружения под YOLO не нужно — ultralytics уже входит в `sam3`.

`conda env export --no-builds` **не сохраняет** `--extra-index-url` для CUDA-специфичных сборок torch (например `+cu128`) — при экспорте `.yml`-файлов это нужно дописывать вручную.

## Разработка на Windows / деплой на iet-08

Код пишется на Windows-машине, переносится на iet-08 через scp/PowerShell-скрипты. В `pipeline_config.yaml` для локальной Windows-отладки можно указать `python_exe` напрямую (закомментированные примеры вида `C:\Users\Admin\anaconda3\envs\dino\python.exe`), в проде — используется `conda_env` + прямой путь `/opt/conda/envs/<env>/bin/python` внутри контейнера.

## Известные особенности и решённые проблемы

- Синтаксическая ошибка (одиночный `>`) в `locate_anything_worker.py`, ломавшая импорт — исправлено.
- Отсутствующие поля `workers`, `plots`, `cache` в `TrainModelRequest` (Pydantic молча отбрасывал флаги стабильности обучения) — добавлены явно в модель.
- Неполная обработка HTTP-ошибок в `api_get` vs `api_post` в `app.py`.
- Баг очистки папки масок в `qc_filter.py` — использовался неверный путь (`annotations_dir.parent / "masks"` вместо `out_dir/masks`).
- Дублирование логики разрешения путей к картинке в `review_helper.py` — стоит переиспользовать `_resolve_existing_path()` из `qc_filter.py` вместо повторной реализации.
- Несогласованный вызов conda в `server.py`: часть кода использует `conda run`, часть — прямые пути `/opt/conda/envs/sam3/bin/python` (стоит унифицировать на прямые пути везде).
- `openai/CLIP` на PyPI опубликован под другим, не связанным пакетом — устанавливать через `pip install --no-deps git+https://github.com/openai/CLIP.git`.
- `multiprocessing` для FPS-бенчмарка требует `spawn` (не `fork`) во избежание поломки CUDA-контекста.
- uvicorn вне главного потока требует отключения установки signal handler'ов (`install_signal_handlers` переопределён в `_BackgroundUvicornServer`).

## Лицензирование

- **Ultralytics** — AGPL-3.0; для закрытого коммерческого использования нужна Enterprise-лицензия.
- **LocateAnything-3B** — NVIDIA Research License, есть коммерческие ограничения.

Эти пункты стоит учитывать перед коммерческим релизом.

## Roadmap

- Продолжение бенчмаркинга и анализа качества student-модели по классам.
- Возможная интеграция LocateAnything-3B как постоянного супплементарного детектора (сейчас в конфиге, но роль уточняется).
- Объединение датасетов из нескольких проектов для более крупных обучающих запусков.
- Дальнейшая доработка FastAPI-эндпоинтов и train/test workflow (унификация вызова conda-окружений, возможный вынос дублирующейся логики review/qc в общий модуль).