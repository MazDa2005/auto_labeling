# АВТОРАЗМЕТКА Отчёт по проекту 
---

## 1. Паспорт проекта

- **Название проекта:** Auto-Labeling: teacher-student пайплайн псевдо-разметки для real-time детекции СИЗ и ж/д объектов

- **Краткое описание:**
Проект реализует пайплайн автоматической разметки данных на базе тяжелых teacher-моделей, с последующей автоматической и ручной QC-фильтрацией результатов. Далее на основе размеченных результатов обучается student-модель YOLO. Целевой набор включает 14 классов: СИЗ — helmet, orange_vest, gloves и др.; железнодорожные объекты — railcar; а также визуальные события — fire, smoke, welding.

---

## 2. Постановка задачи и контекст

### 2.1 Предметная область и задача

Видеоаналитика на промышленном объекте: контроль соблюдения требований по СИЗ (helmet, orange_vest, gloves, gas_mask и др.), распознавание ж/д объектов (railcar) по видеопотокам с камер, а также визуальных событий (fire, smoke, welding) по видеопотокам с камер. Целевая задача - object detection и segmentation для 14 классов с формированием масок объектов.

### 2.2 Формулировка задачи в терминах ML/ИИ

- **Входные данные:**  Видео или отдельные изображения.
- **Выход модели (student):** список детекций на кадре — `class_id`, bounding box, cегментационная маска, confidence.

### 2.3 Целевые метрики качества

- **Качество детекции:** precision, recall, mAP50, mAP50-95, по каждому классу отдельно и агрегированно. Per-class разбивка важна, так как присутствует дисбаланс классов.
- **Скорость:** FPS на одном потоке и агрегированный/per-stream FPS при N параллельных потоках (симуляция N камер).

---

## 3. Структура проекта

```
auto-labeling/
├── classes.json                  # файл классов и промптов для teacher-моделей
├── pipeline_config.yaml          # конфигурация этапов пайплайна
├── batch_label.py                # оркестратор — запускает stages по конфигу
└── teachers/
│  ├── dino_infer.py              # DETECT: Grounding DINO
│  ├── locate_infer.py            # DETECT: LocateAnything-3B
│  ├── locate_anything_worker.py  # обёртка над LocateAnything-3B
│  ├── sam3_refine.py             # REFINE: SAM3 → маска + bbox
│  ├── clip_verify.py             # VERIFY: CLIP (временно отключён)
│  ├── qc_filter.py               # QC: фильтр, эвристики
│  └── pipeline_utils.py          # общие утилиты 
│
├── convert_to_yolo_seg.py        # конвертация в YOLO segmentation формат
├── merge_datasets.py             # объединение YOLO-датасетов
├── train_student.py              # обучение student-модели (YOLO)
├── test_student.py               # инференс на изображении/видео
├── benchmark_student.py          # качество (P/R/mAP) + скорость (multi-stream FPS)
│
├── review_helper.py              # CLI-ревью проблемных детекций (review/)
├── spot_check.py                 # выборочная проверка clean/
├── frame_out.py                  # извлечение кадров из видео
│
├── app.py                        # Streamlit UI (интерфейс сервиса)
├── server.py                     # FastAPI-бэкенд (фоновые задачи)
│
├── docker-compose.yml
├── docker/
│ ├── miniconda.sh                # скрипт установки Miniconda в Docker-образ(устанавливается отдельно)
│ ├── entrypoint.sh               # точка входа контейнера: подготовка окружений
│ ├── Dockerfile                  # сборка Docker-образа с Miniconda и conda-окружениями
│ └── environments/
│   ├── dino.tar.gz               # conda-pack окружения
│   ├── sam3.tar.gz               # conda-pack окружения
│   ├── nvidia.tar.gz             # conda-pack окружения
│   ├── dino.yml                  # спецификация conda-окружения dino
│   ├── sam3.yml                  # спецификация conda-окружения sam3
│   └── nvidia.yml                # спецификация conda-окружения nvidia
│
└── projects/                     # создаётся в рантайме, не хранится в репозитории
  └── <project_name>/
    ├── frames/                   # извлечённые кадры видео
    ├── ann/
    │   ├── clean/                # прошли QC автоматически
    │   └── review/               # требуют ручной проверки
    ├── dataset_yolo/             # images/, labels/, data.yaml
    └── runs/                     # веса и логи обучения
```

Отдельно скачиваются модели, пути к ним настраиваются в `pipelline_config.yaml`.

---

## 4. Данные

- Данные проекта: сырые видео/кадры + пайплайн, который сам генерирует разметку
- Источник: видео/кадры с камер; дополнительные изображения для недопредставленных классов 
- Исходные видео/кадры не хранятся в репозитории — кладутся в `projects/<project_name>/` в рантайме (через UI-загрузку если нужна авторазметка, или вручную если разметка уже готова).
- Веса моделей (Grounding DINO, SAM3, LocateAnything-3B) не хранятся в репозитории — монтируются c хоста как read-only volume или должны быть скачаны отдельно при развёртывании на новой машине; пути настраиваются в `pipeline_config.yaml`.
- Финальный формат для обучения — YOLO segmentation (`convert_to_yolo_seg.py`, `merge_dataset.py`): `train/{images,label}`,
`val/{images,label}`, `test/{images,label}`(полигоны из масок SAM3, нормализованные координаты) + `data.yaml`.

---

## 5. Архитектура решения и сервис

### 5.1 Архитектура пайплайна

```
кадры → DETECT (Grounding DINO [+ LocateAnything-3B]) → REFINE (SAM3, маска)
      → VERIFY (CLIP, отключён) → QC-фильтр → clean/ | review/ (ручное ревью)
      → CONVERTTOYOLO → TRAIN (YOLO student) → BENCHMARK (качество + FPS)
```

Итоговый teacher-пайплайн:

1. **Detect:** Grounding DINO + LocateAnything-3B. Оба работают по текстовым промптам из `classes.json`. DINO собирает промпты всех целевых классов в один текстовый запрос (`"a person. a safety helmet. work gloves."`), LocateAnything опрашивается похожим образом через `<ref>`-теги. Результаты обоих детекторов сливаются по одной картинке через `merge_new_detections`. 

2. **Refine:** SAM3 (`SAM3SemanticPredictor`) принимает на вход боксы от detect-этапа и уточняет их до сегментационной маски: для каждой детекции считается новая, более точная bbox-рамка, полученная из контура маски, а также сохраняется сама маска (PNG) для последующей конвертации в полигон. Ключевой побочный сигнал — `refine_iou`: IoU между исходным (грубым) и уточнённым боксом. Низкий `refine_iou` — признак того, что SAM3, возможно, сегментировал не тот объект, который имел в виду detect-этап, и такие детекции в дальнейшем помечаются QC-фильтром как «на проверку».

3. **Verify:** CLIP (open_clip, `ViT-SO400M-14-SigLIP`) —  этап, обогащающий каждую детекцию сигналом согласия: кроп объекта (с расширенным контекстным окном вокруг bbox) сравнивается со всеми текстовыми промптами всех классов, и если «лучшая догадка» CLIP не совпадает с классом от детектора — это потенциальный повод для ручной проверки. **Результат эксперимента отрицательный:** для мелких PPE-объектов CLIP не работает. Этап отключён в финальной конфигурации.

4. **QC-фильтр  эвристики):** финальный автоматический фильтр перед ручным ревью, работает на трёх уровнях:
   - порог confidence, вырожденная площадь bbox, низкий `refine_iou`;
   - дубликаты одного класса на одном месте (побеждает бокс с большей confidence), конфликты между взаимоисключающими классами, неожиданные пересечения разных классов;
   - например, человек с высокой confidence на кадре без единого признака СИЗ — флаг на ревью.
По итогам классификации детекций картинка целиком уходит в `clean/`, только если **все** детекции на ней получили статус `accepted`; иначе — в `review/`.

5. **Review:** ручная проверка кадров, которые попали в `review/`. Выполняется либо через Streamlit UI, либо через CLI-инструмент `review_helper.py`. После принятия решений json обновляется, а картинка, у которой не осталось детекций со статусом `needs_review`, автоматически переносится из `review/` в `clean/`.

6. **Spot-check:** выборочная ручная проверка картинок, которые уже попали в `clean/` автоматически — контроль качества самого автофильтра, а не отдельных детекций. Если человек находит ошибку пайплайна — картинка целиком возвращается в `review/` с пометкой `flagged by spot-check`, а все её `accepted`-детекции переоткрываются как `needs_review` для последующего разбора через `review_helper.py`.

7. **Convert to YOLO seg:** конвертация внутреннего JSON-формата аннотаций (бокс + маска) в формат YOLO segmentation. Полигон объекта извлекается из PNG-маски через + упрощение контура; если маски нет (например, класс не проходил через refine-этап) — используется fallback на bbox-как-полигон (4 угла). Датасет автоматически делится на train/val с фиксированным random seed.

8. **Merge dataset:** объединение YOLO-датасетов из нескольких разных проектов в один для обучения на большем объёме данных. Перед объединением скрипт (`merge_datasets.py`) сверяет `data.yaml` всех датасетов — список и порядок классов должны совпадать, иначе `class_id` в разных датасетах будут указывать на разные классы, и разметка перепутается; при несовпадении объединение останавливается с явной ошибкой. При коллизии имён файлов между датасетами файл переименовывается с префиксом имени исходного датасета.

9. **Train student:** обучение лёгкой YOLO-модели (Ultralytics) на объединённом датасете.

10. **Test student:** раздел для прогона обученной модели на произвольном изображении или видео — модель предсказывает детекции, рисует их поверх кадра/видео и дополнительно сохраняет результат отдельным JSON.

11. **Benchmark:** финальная оценка обученной модели. Качество — `model.val()` (Ultralytics) на val-сплите: precision/recall/mAP50/mAP50-95, общие и по каждому классу отдельно. Скорость — симуляция нескольких параллельных камер: каждый «поток» — отдельный **процесс** (`multiprocessing`, `spawn`). Тестируются заданные конфигурации числа потоков (например, 1/2/4/8), для каждой считается суммарный и per-stream FPS.

- **Архитектура:** YOLO (Ultralytics), сегментационная версия (`yolo26n/.pt` как база для transfer learning, опробован также `yolo8n.pt`).
- **Обучающие данные:** только детекции со статусом `accepted` из `clean/` (прошли и авто-, и при необходимости ручной QC).
- **Настройки обучения:** `imgsz=640` (по умолчанию), `batch` подбирается под 16GB VRAM, `workers=2`, `cache=False`, `plots=False`, early stopping по `patience`.

### 5.2 API и endpoints

| Endpoint | Назначение |
|---|---|
| `POST /extract-frames` | извлечение кадров из видео проекта |
| `POST /run-pipeline` | запуск teacher-пайплайна разметки |
| `POST /convert-to-yolo` | конвертация проверенной разметки в YOLO-формат |
| `POST /train-model` | запуск обучения student-модели |
| `POST /test-model` | инференс обученной модели на изображении/видео (ближайший аналог `/predict`) |
| `POST /benchmark` | тестирование модели, получение метрик и FPS |
| `GET /training-runs/{project}` | список обученных чекпоинтов проекта |
| `GET //benchmark/models` | получение моделей для benchmark |
| `GET //benchmark/dataset` | получение датасетов для benchmark |
| `GET /task/{task_id}` | статус фоновой задачи (прогресс, стадия, сообщение) |
| `GET /tasks` | список всех задач |

Все "тяжёлые" операции запускаются через `BackgroundTasks` и опрашиваются через `/task/{id}`.

### 5.3 Технологический стек

- **ML/CV:** PyTorch (CUDA), Ultralytics (YOLO), Transformers (Grounding DINO), SAM3, open_clip,
  OpenCV, Pillow.
- **Backend:** FastAPI, uvicorn, Pydantic.
- **Frontend:** Streamlit.
- **Оркестрация процессов:** `subprocess`/прямой запуск python в conda-окружении для изоляции
  teacher-моделей; `multiprocessing` (`spawn`) для multi-stream бенчмарка.
- **Docker:** multi-stage build (`nvidia/cuda:12.6.0-base-ubuntu24.04` база), conda-pack для
  переноса окружений, `docker-compose.yml` для запуска (GPU passthrough, volume-мапинг весов).

---

## 6. Экспериментальный протокол и результаты

| Модель | Описание | mAP50 | mAP50-95 | Комментарий |
|---|---|---|---|---|
| Student v1 | YOLO(`26n`)|  `0.678` | `0.533` | Первый рабочий прогон(но тут было только 6 классов) |
| Student v2 (финальный) | YOLO(`26n`) |  `0.674` | `0.512` | Финальная модель(все 14 классов) |

Per-class разбивка (важно из-за дисбаланса классов):

| Класс | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| welding_gloves | 0.896 | 0.882 | **0.933** | 0.847 |
| welding | 0.401 | 0.223 | 0.237 | 0.164 |
| gas_mask | 0.648 | 0.512 | 0.538 | 0.315 |
| gloves | 0.644 | 0.333 | 0.357 | 0.257 |
| protective_headphones | 0.965 | 1.000 | **0.995** | 0.991 |
| helmet | 0.923 | 0.917 | **0.989** | 0.651 |
| orange_vest | 0.914 | 0.917 | **0.852** | 0.349 |
| person | 0.798 | 0.914 | **0.872** | 0.657 |
| mask | 0.848 | 0.828 | **0.890**| 0.735 |
| welding_mask | 0.483 | 0.467 | 0.425 | 0.356 |
| glasses | | | | |
| railcar | 0.725 | 0.900 | 0.830 | 0.712 |
| smoke | 0.324 | 0.176 | 0.166 | 0.116 |
| fire | | | | |

Классы, у которых нет метрик:

- glasses, fire - в датасете они представлены, но цифры метрик не сохранились

Классы, у которых еще низкие метрики:

- "welding" - тяжело найти датасеты в которых именно сегментация искр сварки, а авторазметка не справляется
- "gloves" - в датасете представлены очень маленькими объектами
- "welding_mask" - слишком похож с mask 
- "smoke" - так же как и искры тяжело определяется 

Скорость (multi-stream):

| Потоков (камер) | Суммарный FPS | FPS на поток |
|---|---|---|
| 1 | 332.2 | 332.2|
| 2 | 456.3 | 228.1 |
| 4 | 490.7 | 122.7 |
| 8 | 383.7 | 48.8 |

---

## 7. Требования и установка

### 7.1 Требования

- Python `3.10` (все три conda-окружения зафиксированы на этой версии).
- NVIDIA GPU + драйверы (проект тестировался на RTX 5060 Ti, 16GB VRAM) + CUDA 12.x.
- Conda/Miniconda — окружения не переносятся через `pip`/`venv`, так как teacher-модели требуют несовместимых друг с другом версий `torch`,`transformers`/CUDA и изолированы в отдельные conda-окружения.
- Docker + `nvidia-container-toolkit` — для продакшен-развёртывания.

| Окружение | Файл спецификации | Используется в |
|---|---|---|
| `dino` | `docker/environments/dino.yml` | `teachers/dino_infer.py` (Grounding DINO, detect) |
| `sam3` | `docker/environments/sam3.yml` | `teachers/sam3_refine.py`, `teachers/qc_filter.py`, Streamlit (`app.py`), `batch_label.py` (запуск)|
| `nvidia` | `docker/environments/nvidia.yml` | `teachers/locate_infer.py` (LocateAnything-3B, detect) |

### 7.2 Установка

**Установка окружений и моделей**

```bash
conda env create -f docker/environments/dino.yml
conda env create -f docker/environments/sam3.yml
conda env create -f docker/environments/nvidia.yml
```

- **SAM 3**

https://www.modelscope.cn/models/facebook/sam3
скачать sam3.pt положить в папку моделей

- **LocateAnything**

https://huggingface.co/nvidia/LocateAnything-3B

```bash
conda activate nvidia
hf download nvidia/LocateAnything-3B
```

- **Grounding Dino**

https://huggingface.co/IDEA-Research/grounding-dino-base

```bash
conda activate nvidia
hf download IDEA-Research/grounding-dino-base
```

- Для LA и GD нужно, активируем окружение nvidia, так как там есть библиотека hugging face.
- Пути к весам моделей заданы абсолютными путями в `pipeline_config.yaml` (`model_path`) — при переносе на другую машину их нужно поправить.Если собираетесь запускать проект через Docker, то в `docker-compose.yml` надо поправить пути volume, через которые пробрасываются модели.
- `classes.json` — единственный файл, который нужно редактировать при изменении списка целевых классов или промптов.

**Вариант A — Docker (рекомендуемый, продакшен-путь)**

После установки окружений для Docker их надо запаковать:

```bash
conda-pack -n `name` -o docker/environments/`name`.tar.gz
```

Веса моделей и сторонние репозитории (SAM3, LocateAnything-3B) в образ не пакуются — монтируются как volume'ы с хоста (`docker-compose.yml`, секция `volumes`). (!Проверить перед сборкой)

```bash
docker compose build
docker compose up -d
```

**Вариант B — вручную через conda**

```bash
conda activate sam3
streamlit run app.py
```

---

## 8. Как запустить проект

### 8.1 Через UI (основной сценарий)

```bash
docker compose up -d
# либо, при установке по Варианту B:
streamlit run app.py
```

Открыть `http://localhost:8501` → FastAPI-бэкенд
поднимается автоматически в фоновом потоке того же процесса.

### 8.2 Через CLI (отладка отдельного этапа) (примеры)

```bash
# Извлечь кадры из видео
python frame_out.py --video path/to/video.mp4 --output projects/demo/frames --fps 5

# Прогнать teacher-пайплайн разметки
python batch_label.py --images-dir projects/demo/frames \
    --config pipeline_config.yaml --out-dir projects/demo/ann

# Сконвертировать проверенную разметку в YOLO-формат
python convert_to_yolo_seg.py --annotations-dir projects/demo/ann/clean \
    --classes-file classes.json --output-dir projects/demo/dataset_yolo

# Обучить student-модель
python train_student.py --source-dirs projects/demo/dataset_yolo \
    --target-dir projects/demo/dataset_yolo_merged --runs-dir projects/demo/runs \
    --run-name run_001 --classes-file classes.json --epochs 100

# Инференс на изображении/видео
python test_student.py --weights projects/demo/runs/run_001/weights/best.pt \
    --input path/to/test.jpg --output out.jpg --classes-file classes.json

# Бенчмарк качества и скорости
python benchmark_student.py --weights projects/demo/runs/run_001/weights/best.pt \
    --data-yaml projects/demo/dataset_yolo/data.yaml \
    --sample-images-dir projects/demo/dataset_yolo/images/val \
    --streams 1,2,4,8 --duration 15 --output-json benchmark_result.json
```

Проверка работоспособности сервиса: `curl http://localhost:8000/tasks` должен вернуть
`{"tasks": []}` (или список активных задач) без ошибки соединения.

---

## 9. Конфигурация и безопасность

### 9.1 Конфигурация

- `pipeline_config.yaml` — список этапов пайплайна, модели, conda-окружения, пути к весам, вкл/выкл каждого этапа без изменения кода.
- `classes.json` — список классов и промптов для teacher-моделей.
- Гиперпараметры обучения настраиваются через UI или API-запрос.
- `.env`/секретов в проекте нет — все модели локальные, внешние платные API не используются.

### 9.2 Безопасность

- Секретов/токенов в репозитории нет.
- Лицензионные ограничения используемых моделей — открытый вопрос: Ultralytics YOLO — AGPL-3.0, LocateAnything-3B — NVIDIA Research License (ограничения на коммерческое использование).

---
