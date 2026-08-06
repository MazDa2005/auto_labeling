"""
Streamlit интерфейс для авто-разметки.
Запуск: streamlit run app.py

FastAPI бэкенд (server.py) поднимается автоматически в фоновом потоке
внутри этого же процесса
"""
import json
import random
import shutil
import socket
import threading
import time
from pathlib import Path

import requests
import streamlit as st
import uvicorn
import yaml
import pandas as pd  # <-- ДОБАВЛЕНО для таблиц бенчмарка

from server import app as fastapi_app

st.set_page_config(page_title="Auto-Labeling", layout="wide", page_icon="🔍")

# ── Настройки API ────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
PROJECTS_DIR = Path("projects")


class _BackgroundUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        pass


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@st.cache_resource(show_spinner=False)
def start_backend() -> bool:
    if _is_port_open("localhost", 8000):
        return True

    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
    server = _BackgroundUvicornServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True, name="fastapi-backend")
    thread.start()

    for _ in range(20):
        if _is_port_open("localhost", 8000):
            return True
        time.sleep(0.25)

    st.error("❌ Не удалось поднять встроенный FastAPI сервер за отведённое время.")
    return False


start_backend()

# ── Вспомогательные функции ────────────────────────────────────────────────
def api_post(endpoint: str, data: dict) -> dict:
    """Отправить POST запрос к API."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(" Не удалось подключиться к API серверу. Запустите: `python server.py`")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP ошибка: {e}")
        return {}


def api_get(endpoint: str) -> dict:
    """Отправить GET запрос к API."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к API серверу.")
        st.stop()


def get_projects() -> list[str]:
    """Получить список проектов."""
    if not PROJECTS_DIR.exists():
        return []
    return [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]


def get_review_files(project: str) -> list[Path]:
    """Получить файлы для ручной проверки."""
    review_dir = PROJECTS_DIR / project / "ann" / "review"
    if review_dir.exists():
        return sorted([f for f in review_dir.glob("*.json") if not f.name.startswith("_")])
    return []


def get_project_stats(project: str) -> dict:
    """Подсчитывает статистику по проекту: сколько в review, clean, и детекций по статусам."""
    project_dir = PROJECTS_DIR / project

    review_dir = project_dir / "ann" / "review"
    clean_dir = project_dir / "ann" / "clean"

    review_files = []
    if review_dir.exists():
        review_files = [f for f in review_dir.glob("*.json") if not f.name.startswith("_")]

    clean_files = []
    if clean_dir.exists():
        clean_files = [f for f in clean_dir.glob("*.json") if not f.name.startswith("_")]

    total_images = len(review_files) + len(clean_files)

    total_accepted = 0
    total_needs_review = 0
    total_rejected = 0

    for json_file in review_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for det in data.get("detections", []):
            bucket = det.get("qc_bucket", "accepted")
            if bucket == "accepted":
                total_accepted += 1
            elif bucket == "needs_review":
                total_needs_review += 1
            elif bucket == "rejected":
                total_rejected += 1

    for json_file in clean_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for det in data.get("detections", []):
            bucket = det.get("qc_bucket", "accepted")
            if bucket == "accepted":
                total_accepted += 1
            elif bucket == "needs_review":
                total_needs_review += 1
            elif bucket == "rejected":
                total_rejected += 1

    return {
        "review_count": len(review_files),
        "clean_count": len(clean_files),
        "total_images": total_images,
        "accepted": total_accepted,
        "needs_review": total_needs_review,
        "rejected": total_rejected,
    }


def find_annotated_image(stem: str, project: str) -> Path | None:
    """Найти картинку с наложенными боксами/масками (после QC)."""
    project_dir = PROJECTS_DIR / project
    for bucket in ["review", "clean"]:
        annotated = project_dir / "ann" / bucket / f"{stem}_annotated.jpg"
        if annotated.exists():
            return annotated
    return None


def find_original_image(stem: str, project: str) -> Path | None:
    """Найти оригинальный кадр без разметки поверх."""
    frames_dir = PROJECTS_DIR / project / "frames"
    if frames_dir.exists():
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = frames_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def find_image(stem: str, project: str) -> Path | None:
    """Найти картинку для отображения (аннотированная в приоритете, иначе оригинал)."""
    return find_annotated_image(stem, project) or find_original_image(stem, project)


def move_to_clean(stem: str, project: str):
    """Перенести картинку из review в clean."""
    project_dir = PROJECTS_DIR / project
    review_dir = project_dir / "ann" / "review"
    clean_dir = project_dir / "ann" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / "masks").mkdir(parents=True, exist_ok=True)

    json_src = review_dir / f"{stem}.json"
    if json_src.exists():
        shutil.move(str(json_src), str(clean_dir / json_src.name))

    ann_src = review_dir / f"{stem}_annotated.jpg"
    if ann_src.exists():
        shutil.move(str(ann_src), str(clean_dir / ann_src.name))

    src_masks = review_dir / "masks"
    dst_masks = clean_dir / "masks"
    if src_masks.exists():
        for mask in src_masks.glob(f"{stem}_*"):
            shutil.move(str(mask), str(dst_masks / mask.name))


def find_yolo_split_dir(dataset_dir: Path, split: str) -> tuple[Path | None, Path | None]:
    """
    Определяет расположение images/labels для сплита — поддерживает ОБА макета:
      НОВЫЙ (после "Объединения датасетов"): dataset_dir/<split>/images, .../labels
      СТАРЫЙ (после convert_to_yolo_seg.py):  dataset_dir/images/<split>, .../labels
    Возвращает (images_dir, labels_dir) либо (None, None), если сплита нет.
    """
    new_images = dataset_dir / split / "images"
    new_labels = dataset_dir / split / "labels"
    if new_images.exists():
        return new_images, new_labels

    old_images = dataset_dir / "images" / split
    old_labels = dataset_dir / "labels" / split
    if old_images.exists():
        return old_images, old_labels

    return None, None


def has_yolo_dataset(project: str) -> bool:
    """Есть ли у проекта готовый YOLO-датасет (в любом из двух макетов)."""
    dataset_dir = PROJECTS_DIR / project / "dataset_yolo"
    if not dataset_dir.exists():
        return False
    train_images, _ = find_yolo_split_dir(dataset_dir, "train")
    return train_images is not None


# ── Загрузка цветов классов ──────────────────────────────────────────────────
def load_class_colors() -> dict[str, str]:
    """Загрузить цвета классов из classes.json."""
    if not Path("classes.json").exists():
        return {}
    with open("classes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Исправление пробелов в ключах, если они есть
    classes_list = data.get("classes", data.get("classes ", []))
    return {
        c.get("name", c.get("name ", "")).strip(): c.get("color", "#808080").strip()
        for c in classes_list
    }


CLASS_COLORS = load_class_colors()

# ── Навигация ────────────────────────────────────────────────────────────────
PAGE_HOME = "Домой"
PAGE_UPLOAD = "Загрузка"
PAGE_EXTRACT = "Извлечение кадров"
PAGE_PIPELINE = "Пайплайн"
PAGE_REVIEW = "Review"
PAGE_SPOTCHECK = "Spot Check"
PAGE_CONVERT = "Конвертация"
PAGE_MERGE = "Объединение датасетов"
PAGE_TRAIN = " Обучение"
PAGE_TEST = "🧪 Тест модели"
PAGE_BENCHMARK = "📊 Бенчмарк"  # <-- ДОБАВЛЕНО

st.sidebar.title("Auto-Labeling")
page = st.sidebar.radio(
    "Навигация",
    [PAGE_HOME, PAGE_UPLOAD, PAGE_EXTRACT, PAGE_PIPELINE, PAGE_REVIEW, PAGE_SPOTCHECK,
     PAGE_CONVERT, PAGE_MERGE, PAGE_TRAIN, PAGE_TEST, PAGE_BENCHMARK],
    index=0,
)

# ── Страницы ────────────────────────────────────────────────────────────────

if page == PAGE_HOME:
    st.title("Добро пожаловать в Auto-Labeling!")

    st.markdown("""
    ### Как работает пайплайн:
    1. Загрузите видео или готовые изображения
    2. Извлеките кадры из видео
    3. Запустите пайплайн детекции и сегментации
    4. Проверьте результаты вручную
    5. Конвертируйте в YOLO для обучения
    6. Объедините датасеты из разных проектов
    7. Обучите student-модель и протестируйте её

    ### Быстрый старт:
    """)

    projects = get_projects()

    if projects:
        st.success(f"✅ Найдено проектов: {len(projects)}")
        for proj in projects:
            st.markdown(f"- **{proj}**")
    else:
        st.warning("⚠️ Нет проектов. Создайте новый, загрузив видео!")
    st.caption(" Убедитесь, что FastAPI сервер запущен: `python server.py`")

elif page == PAGE_UPLOAD:
    st.title(" Загрузка видео или изображений")

    source_type = st.radio("Источник данных", ["Видео", "Готовые изображения"])

    project_name = st.text_input("Имя проекта", "my_project")
    project_dir = PROJECTS_DIR / project_name

    if source_type == "Видео":
        uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])

        if uploaded_file:
            st.info(f" Загружен файл: {uploaded_file.name}")

            if st.button("💾 Сохранить видео"):
                if not project_dir.exists():
                    project_dir.mkdir(parents=True)
                    (project_dir / "frames").mkdir()

                video_path = project_dir / "video.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"✅ Видео сохранено в {video_path}")
                st.video(str(video_path))

    else:  # Готовые изображения
        uploaded_files = st.file_uploader(
            "Загрузите изображения",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"📷 Загружено файлов: {len(uploaded_files)}")

            if st.button("💾 Сохранить изображения"):
                if not project_dir.exists():
                    project_dir.mkdir(parents=True)
                    (project_dir / "images").mkdir()

                images_dir = project_dir / "images"
                for file in uploaded_files:
                    with open(images_dir / file.name, "wb") as f:
                        f.write(file.getbuffer())

                st.success(f"✅ Загружено {len(uploaded_files)} изображений в {images_dir}")

                st.image(str(images_dir / uploaded_files[0].name), caption="Пример изображения")

elif page == PAGE_EXTRACT:
    st.title("🎬 Извлечение кадров из видео")

    projects = get_projects()
    if not projects:
        st.warning("️ Нет проектов. Создайте проект на вкладке 'Загрузка'.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    video_path = project_dir / "video.mp4"
    if not video_path.exists():
        st.error(f" Видео не найдено в {video_path}")
        st.stop()

    fps = st.slider("Кадров в секунду", 1, 30, 5)

    if st.button("🎬 Извлечь кадры", width="stretch"):
        with st.spinner("Извлечение кадров..."):
            result = api_post("/extract-frames", {"project": selected_project, "fps": fps})

            if "task_id" in result:
                task_id = result["task_id"]
                st.info(f"Задача запущена: {task_id}")

                progress_bar = st.progress(0)
                status_text = st.empty()

                while True:
                    status = api_get(f"/task/{task_id}")
                    progress_bar.progress(status.get("progress", 0))
                    status_text.text(f"Статус: {status.get('status')} - {status.get('message')}")

                    if status.get("status") in ["done", "failed"]:
                        break
                    time.sleep(2)

                if status.get("status") == "done":
                    st.success("✅ Кадры успешно извлечены!")
                else:
                    st.error(f" Ошибка: {status.get('message')}")

elif page == PAGE_PIPELINE:
    st.title("️ Запуск пайплайна детекции")

    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    frames_dir = project_dir / "frames"
    if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
        st.error(f" Кадры не найдены в {frames_dir}")
        st.stop()

    with open("classes.json", "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    # Исправление пробелов в ключах
    classes_list = classes_data.get("classes", classes_data.get("classes ", []))
    class_names = [c.get("name", c.get("name ", "")).strip() for c in classes_list]

    if "pipeline_classes_select" not in st.session_state:
        st.session_state.pipeline_classes_select = list(class_names)

    col_reset, col_count = st.columns([1, 3])
    with col_reset:
        if st.button("Выбрать все классы"):
            st.session_state.pipeline_classes_select = list(class_names)
            st.rerun()

    selected_classes = st.multiselect(
        "Классы для детекции",
        class_names,
        key="pipeline_classes_select",
    )

    with col_count:
        if len(selected_classes) < len(class_names):
            missing = [c for c in class_names if c not in selected_classes]
            st.warning(
                f"Выбрано {len(selected_classes)} из {len(class_names)} классов. "
                f"Не выбраны: {', '.join(missing)}"
            )
        else:
            st.success(f"Выбраны все {len(class_names)} классов")

    if st.button("⚙️ Запустить пайплайн", width="stretch", disabled=not selected_classes):
        with st.spinner("Запуск пайплайна..."):
            result = api_post("/run-pipeline", {"project": selected_project, "classes": selected_classes})

            if "task_id" in result:
                task_id = result["task_id"]
                st.info(f"Задача запущена: {task_id}")

                progress_bar = st.progress(0)
                status_text = st.empty()

                while True:
                    status = api_get(f"/task/{task_id}")
                    progress_bar.progress(status.get("progress", 0))
                    status_text.text(f"Статус: {status.get('status')} - {status.get('message')}")

                    if status.get("status") in ["done", "failed"]:
                        break
                    time.sleep(2)

                if status.get("status") == "done":
                    st.success("✅ Пайплайн завершен!")
                else:
                    st.error(f"❌ Ошибка: {status.get('message')}")

elif page == PAGE_REVIEW:
    st.title("🔍 Ручная проверка детекций")

    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    stats = get_project_stats(selected_project)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Статистика проекта")

    if stats["total_images"] > 0:
        progress = stats["clean_count"] / stats["total_images"]
        st.sidebar.progress(progress, text=f"Проверено: {stats['clean_count']} из {stats['total_images']}")
    else:
        st.sidebar.info("Нет данных для отображения")

    st.sidebar.metric("📋 В review", stats["review_count"])
    st.sidebar.metric("✅ В clean", stats["clean_count"])
    st.sidebar.markdown("---")
    st.sidebar.metric("✅ Принято детекций", stats["accepted"])
    st.sidebar.metric("⚠️ На проверке", stats["needs_review"])
    st.sidebar.metric("❌ Отклонено", stats["rejected"])

    json_files = get_review_files(selected_project)

    if not json_files:
        st.success("🎉 Все детекции проверены!")
        st.stop()

    if "review_index" not in st.session_state:
        st.session_state.review_index = 0

    if st.session_state.review_index >= len(json_files):
        st.session_state.review_index = 0

    current_file = json_files[st.session_state.review_index]
    stem = current_file.stem

    with open(current_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    detections = data.get("detections", [])

    # Переключатель: аннотированная картинка (с масками/боксами) vs оригинал.
    view_mode = st.radio(
        "Показать",
        ["🖍️ Аннотированная", "🖼️ Оригинал"],
        horizontal=True,
        key="review_view_mode",
    )

    if view_mode == "🖍️ Аннотированная":
        image_path = find_annotated_image(stem, selected_project) or find_original_image(stem, selected_project)
    else:
        image_path = find_original_image(stem, selected_project) or find_annotated_image(stem, selected_project)

    col_img, col_det = st.columns([1.5, 1])

    with col_img:
        if image_path:
            st.image(str(image_path), width="stretch")
            st.caption(f"📷 {stem}")
        else:
            st.error("❌ Картинка не найдена!")

    def _write_annotations():
        with open(current_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _accept(det_idx: int):
        detections[det_idx]["qc_bucket"] = "accepted"
        detections[det_idx]["qc_reason"] = "принята ручной проверкой"
        _write_annotations()

    def _reject(det_idx: int):
        det = detections[det_idx]
        det["qc_bucket"] = "rejected"
        det["qc_reason"] = "отклонена ручной проверкой"
        mask_path = det.get("mask_path")
        if mask_path and Path(mask_path).exists():
            Path(mask_path).unlink()
        _write_annotations()

    DETECTIONS_PANEL_HEIGHT = 400

    with col_det:
        st.subheader(" Детекции")

        with st.container(height=DETECTIONS_PANEL_HEIGHT, border=True):
            for idx, det in enumerate(detections):
                bucket = det.get("qc_bucket", "accepted")
                cls = det["class"]
                conf = det.get("confidence", 0)
                reason = det.get("qc_reason", "")

                if bucket == "accepted":
                    emoji = "✅"
                    conf_color = "green"
                elif bucket == "needs_review":
                    emoji = "⚠️"
                    conf_color = "orange"
                else:
                    emoji = "❌"
                    conf_color = "red"

                cls_color = CLASS_COLORS.get(cls, "#808080")

                info_col, accept_col, reject_col = st.columns([4, 1, 1])

                with info_col:
                    st.markdown(f"**{emoji} [{idx}] {cls}**")
                    st.markdown(f":{conf_color}[{conf:.2f}]")
                    if reason:
                        st.caption(f"📝 {reason[:40]}{'...' if len(reason) > 40 else ''}")
                    st.markdown(
                        f"<div style='width:24px;height:24px;background:{cls_color};border:2px solid #333;border-radius:3px;display:inline-block;margin:2px 0'></div>",
                        unsafe_allow_html=True,
                    )

                with accept_col:
                    st.button(
                        "✅",
                        key=f"accept_{stem}_{idx}",
                        help="Принять",
                        disabled=bucket == "accepted",
                        on_click=_accept,
                        args=(idx,),
                    )

                with reject_col:
                    st.button(
                        "❌",
                        key=f"reject_{stem}_{idx}",
                        help="Отклонить",
                        disabled=bucket == "rejected",
                        on_click=_reject,
                        args=(idx,),
                    )

                st.divider()

        accepted = sum(1 for d in detections if d.get("qc_bucket") == "accepted")
        review = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        rejected = sum(1 for d in detections if d.get("qc_bucket") == "rejected")

        st.markdown(f"**Итого:** ✅ {accepted} | ⚠️ {review} | ❌ {rejected}")

    st.markdown("---")
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 2])

    with col_nav1:
        st.caption(f" {st.session_state.review_index + 1}/{len(json_files)}")

    with col_nav2:
        if st.button("⬅️ Назад", disabled=st.session_state.review_index == 0, use_container_width=True):
            st.session_state.review_index -= 1
            st.rerun()

    with col_nav3:
        if st.button("Вперёд ➡️", disabled=st.session_state.review_index == len(json_files) - 1, use_container_width=True):
            st.session_state.review_index += 1
            st.rerun()

    with col_nav4:
        has_review = any(d.get("qc_bucket") == "needs_review" for d in detections)
        if st.button("✅ В clean", disabled=has_review, use_container_width=True):
            move_to_clean(stem, selected_project)
            st.success(f"✅ {stem} перенесен в clean/")
            st.rerun()

elif page == PAGE_SPOTCHECK:
    st.title(" Spot Check — быстрая проверка")

    st.markdown("""
    **Spot Check** — режим для быстрой проверки случайных кадров из review.
    Идеально подходит для первичной оценки качества детекций.
    """)

    projects = get_projects()
    if not projects:
        st.warning("️ Нет проектов.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    json_files = get_review_files(selected_project)

    if not json_files:
        st.success(" Все детекции проверены! Нечего проверять.")
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🎲 Случайный кадр", use_container_width=True, type="primary"):
            st.session_state.spotcheck_file = random.choice(json_files)
            st.session_state.spotcheck_active = True

    if "spotcheck_active" in st.session_state and st.session_state.spotcheck_active:
        current_file = st.session_state.spotcheck_file
        stem = current_file.stem

        with open(current_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        detections = data.get("detections", [])

        view_mode = st.radio(
            "Показать",
            ["️ Аннотированная", "️ Оригинал"],
            horizontal=True,
            key="spotcheck_view_mode",
        )

        if view_mode == "🖍️ Аннотированная":
            image_path = find_annotated_image(stem, selected_project) or find_original_image(stem, selected_project)
        else:
            image_path = find_original_image(stem, selected_project) or find_annotated_image(stem, selected_project)

        col_img, col_det = st.columns([1.5, 1])

        with col_img:
            if image_path:
                st.image(str(image_path), width="stretch")
                st.caption(f"📷 {stem}")
            else:
                st.error("❌ Картинка не найдена!")

        def _write_spotcheck_annotations():
            with open(current_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        def _accept_spotcheck(det_idx: int):
            detections[det_idx]["qc_bucket"] = "accepted"
            detections[det_idx]["qc_reason"] = "принята в spot check"
            _write_spotcheck_annotations()
            st.rerun()

        def _reject_spotcheck(det_idx: int):
            det = detections[det_idx]
            det["qc_bucket"] = "rejected"
            det["qc_reason"] = "отклонена в spot check"
            mask_path = det.get("mask_path")
            if mask_path and Path(mask_path).exists():
                Path(mask_path).unlink()
            _write_spotcheck_annotations()
            st.rerun()

        DETECTIONS_PANEL_HEIGHT = 400

        with col_det:
            st.subheader(" Детекции")

            with st.container(height=DETECTIONS_PANEL_HEIGHT, border=True):
                for idx, det in enumerate(detections):
                    bucket = det.get("qc_bucket", "accepted")
                    cls = det["class"]
                    conf = det.get("confidence", 0)

                    if bucket == "accepted":
                        emoji = "✅"
                        conf_color = "green"
                    elif bucket == "needs_review":
                        emoji = "⚠️"
                        conf_color = "orange"
                    else:
                        emoji = ""
                        conf_color = "red"

                    cls_color = CLASS_COLORS.get(cls, "#808080")

                    info_col, accept_col, reject_col = st.columns([4, 1, 1])

                    with info_col:
                        st.markdown(f"**{emoji} [{idx}] {cls}**")
                        st.markdown(f":{conf_color}[{conf:.2f}]")
                        st.markdown(
                            f"<div style='width:24px;height:24px;background:{cls_color};border:2px solid #333;border-radius:3px;display:inline-block;margin:2px 0'></div>",
                            unsafe_allow_html=True,
                        )

                    with accept_col:
                        st.button("✅", key=f"spot_accept_{stem}_{idx}",
                                 disabled=bucket == "accepted",
                                 on_click=_accept_spotcheck, args=(idx,))

                    with reject_col:
                        st.button("❌", key=f"spot_reject_{stem}_{idx}",
                                 disabled=bucket == "rejected",
                                 on_click=_reject_spotcheck, args=(idx,))

                    st.divider()

            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("🎲 Ещё случайный", use_container_width=True):
                    st.session_state.spotcheck_file = random.choice(json_files)
                    st.rerun()

            with col_btn2:
                if st.button("🔍 В полный review", use_container_width=True):
                    try:
                        idx = json_files.index(current_file)
                        st.session_state.review_index = idx
                    except ValueError:
                        st.session_state.review_index = 0
                    st.session_state.spotcheck_active = False
                    st.rerun()

        accepted = sum(1 for d in detections if d.get("qc_bucket") == "accepted")
        review = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        rejected = sum(1 for d in detections if d.get("qc_bucket") == "rejected")

        st.markdown(f"**Детекций:** ✅ {accepted} | ⚠️ {review} | ❌ {rejected}")

    else:
        st.info(" Нажмите **'🎲 Случайный кадр'**, чтобы начать проверку")
        st.markdown(f"**Всего кадров в review:** {len(json_files)}")

elif page == PAGE_CONVERT:
    st.title("📦 Конвертация в YOLO формат")

    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    clean_dir = project_dir / "ann" / "clean"
    if not clean_dir.exists() or not list(clean_dir.glob("*.json")):
        st.warning("⚠️ Нет данных для конвертации. Проверьте все детекции в Review.")
        st.stop()

    if st.button("📦 Конвертировать в YOLO", width="stretch"):
        with st.spinner("Конвертация..."):
            result = api_post("/convert-to-yolo", {"project": selected_project})

            if "task_id" in result:
                task_id = result["task_id"]
                st.info(f"Задача запущена: {task_id}")

                progress_bar = st.progress(0)
                status_text = st.empty()

                while True:
                    status = api_get(f"/task/{task_id}")
                    progress_bar.progress(status.get("progress", 0))
                    status_text.text(f"Статус: {status.get('status')} - {status.get('message')}")

                    if status.get("status") in ["done", "failed"]:
                        break
                    time.sleep(2)

                if status.get("status") == "done":
                    st.success("✅ Конвертация завершена!")

                    dataset_dir = project_dir / "dataset_yolo"
                    if dataset_dir.exists():
                        zip_path = dataset_dir / f"{selected_project}_dataset.zip"
                        shutil.make_archive(str(zip_path.with_suffix("")), 'zip', dataset_dir)

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "📥 Скачать датасет",
                                data=f.read(),
                                file_name=f"{selected_project}_dataset.zip",
                                mime="application/zip"
                            )
                else:
                    st.error(f"❌ Ошибка: {status.get('message')}")

elif page == PAGE_MERGE:
    st.title("📦 Объединение YOLO-датасетов")

    st.markdown("""
    Выберите несколько проектов, у которых уже выполнена конвертация в YOLO,
    чтобы объединить их в один датасет с **честным train/val сплитом**
    (картинки не пересекаются между train и val — в отличие от старой
    версии, где val дублировал train).

    Итоговый датасет использует макет `train/images`, `train/labels`,
    `val/images`, `val/labels` (и `test/`, если задать долю test > 0).
    """)

    projects = get_projects()
    if not projects:
        st.warning("️ Нет проектов.")
        st.stop()

    # Проекты с готовым YOLO-датасетом — в ЛЮБОМ из двух макетов
    projects_with_dataset = [p for p in projects if has_yolo_dataset(p)]

    if len(projects_with_dataset) < 2:
        st.warning("⚠️ Для объединения нужно как минимум **2** проекта с готовым YOLO-датасетом. Сначала завершите конвертацию в соответствующих проектах.")
        st.stop()

    st.subheader("1. Выберите исходные проекты")
    selected_projects = st.multiselect(
        "Проекты для объединения",
        options=projects_with_dataset,
        help="Берутся ВСЕ картинки (train+val) каждого проекта — дальше пересобираются в новый честный сплит",
    )

    st.subheader("2. Имя нового проекта")
    new_project_name = st.text_input("Введите имя для объединенного проекта", "merged_dataset")
    new_project_dir = PROJECTS_DIR / new_project_name

    st.subheader("3. Доли val / test")
    col_val, col_test = st.columns(2)
    with col_val:
        val_ratio = st.slider("Доля val", 0.05, 0.5, 0.2, 0.05)
    with col_test:
        test_ratio = st.slider("Доля test (0 — без test-сплита)", 0.0, 0.3, 0.0, 0.05)

    ratio_ok = (val_ratio + test_ratio) < 0.9
    if not ratio_ok:
        st.error("❌ val + test слишком большие — на train почти ничего не останется.")

    if new_project_name in projects:
        st.error(f" Проект с именем '{new_project_name}' уже существует! Выберите другое имя.")
        can_merge = False
    elif not new_project_name.strip():
        st.error("❌ Имя проекта не может быть пустым.")
        can_merge = False
    elif len(selected_projects) < 2:
        st.warning("️ Выберите минимум 2 проекта для объединения.")
        can_merge = False
    else:
        can_merge = ratio_ok

    if st.button(" Объединить датасеты", width="stretch", disabled=not can_merge):
        with st.spinner("Объединение датасетов... Это может занять некоторое время."):
            try:
                # ─ 1. Собираем ВСЕ пары (картинка, лейбл) из всех сплитов каждого проекта ──
                all_pairs = []  # (image_path, label_path_or_None, project_name)
                for proj in selected_projects:
                    src_dataset = PROJECTS_DIR / proj / "dataset_yolo"
                    for split in ("train", "val", "test"):
                        src_images, src_labels = find_yolo_split_dir(src_dataset, split)
                        if src_images is None:
                            continue
                        for img_path in sorted(src_images.iterdir()):
                            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                                continue
                            label_path = src_labels / f"{img_path.stem}.txt"
                            all_pairs.append((img_path, label_path if label_path.exists() else None, proj))

                if not all_pairs:
                    st.error(" Не найдено ни одной картинки в выбранных проектах.")
                    st.stop()

                # ── 2. Честный shuffle + split (train/val/test НЕ пересекаются) ──
                random.seed(42)  # воспроизводимость между запусками
                shuffled = all_pairs.copy()
                random.shuffle(shuffled)

                total = len(shuffled)
                n_val = max(1, int(total * val_ratio))
                n_test = int(total * test_ratio)
                n_train = total - n_val - n_test

                train_pairs = shuffled[:n_train]
                val_pairs = shuffled[n_train:n_train + n_val]
                test_pairs = shuffled[n_train + n_val:]

                splits = {"train": train_pairs, "val": val_pairs}
                if test_pairs:
                    splits["test"] = test_pairs

                # ── 3. Копируем в НОВЫЙ макет: <split>/images/, <split>/labels/ ──
                new_dataset_dir = new_project_dir / "dataset_yolo"
                progress_bar = st.progress(0)
                total_copied = 0
                total_labels = 0
                total_to_copy = sum(len(v) for v in splits.values())

                for split_name, pairs in splits.items():
                    images_out = new_dataset_dir / split_name / "images"
                    labels_out = new_dataset_dir / split_name / "labels"
                    images_out.mkdir(parents=True, exist_ok=True)
                    labels_out.mkdir(parents=True, exist_ok=True)

                    for img_path, label_path, proj in pairs:
                        # Префикс именем проекта — чтобы избежать коллизий имён
                        new_stem = f"{proj}_{img_path.stem}"

                        shutil.copy(img_path, images_out / f"{new_stem}{img_path.suffix}")
                        total_copied += 1

                        if label_path is not None:
                            shutil.copy(label_path, labels_out / f"{new_stem}.txt")
                            total_labels += 1
                        else:
                            # картинка без объектов — пустой .txt, чтобы не выпасть из датасета
                            (labels_out / f"{new_stem}.txt").touch()

                        if total_to_copy:
                            progress_bar.progress(min(1.0, total_copied / total_to_copy))

                # ─ 4. data.yaml под новый макет ─
                with open("classes.json", "r", encoding="utf-8") as f:
                    classes_data = json.load(f)
                classes_list = classes_data.get("classes", classes_data.get("classes ", []))
                names = [c.get("name", c.get("name ", "")).strip() for c in classes_list]

                data_yaml = {
                    "path": str(new_dataset_dir.resolve()),
                    "train": "train/images",
                    "val": "val/images",
                    "nc": len(names),
                    "names": names,
                }
                if "test" in splits:
                    data_yaml["test"] = "test/images"

                with open(new_dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

                st.success(f"✅ Датасеты успешно объединены в проект: **{new_project_name}**!")
                split_summary = f"train={len(train_pairs)}, val={len(val_pairs)}"
                if test_pairs:
                    split_summary += f", test={len(test_pairs)}"
                st.markdown(f"""
                **Статистика объединения:**
                - 🖼️ Изображений скопировано: **{total_copied}** ({split_summary})
                - 🏷️ Файлов аннотаций скопировано: **{total_labels}**
                - 📁 Путь к новому датасету: `{new_dataset_dir}`
                """)
                st.caption("✅ train/val/test — честные непересекающиеся сплиты (не дублируют друг друга).")

                zip_path = new_dataset_dir / f"{new_project_name}_merged.zip"
                with st.spinner("Создание ZIP-архива..."):
                    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', new_dataset_dir)

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📥 Скачать объединенный YOLO-датасет",
                        data=f.read(),
                        file_name=f"{new_project_name}_merged.zip",
                        mime="application/zip"
                    )

            except Exception as e:
                st.error(f" Произошла ошибка при объединении: {str(e)}")

elif page == PAGE_TRAIN:
    st.title("🎓 Обучение student-модели")

    projects = get_projects()
    projects_with_dataset = [p for p in projects if has_yolo_dataset(p)]

    if not projects_with_dataset:
        st.warning("⚠️ Нет проектов с готовым YOLO-датасетом. Сначала пройди 'Конвертация' (или 'Объединение датасетов' для нескольких проектов).")
        st.stop()

    selected_project = st.selectbox(
        "Проект для обучения",
        projects_with_dataset,
        help="Можно выбрать как обычный проект после 'Конвертации', так и уже объединённый через 'Объединение датасетов'",
    )

    val_images, _ = find_yolo_split_dir(PROJECTS_DIR / selected_project / "dataset_yolo", "val")
    if val_images is None:
        st.caption("️ У этого проекта нет отдельного val-сплита — val будет совпадать с train, "
                   "метрики валидации будут оптимистичными.")

    run_name = st.text_input("Название запуска (run name)", value="run_1")

    col1, col2 = st.columns(2)
    with col1:
        base_model = st.selectbox("Базовая модель (transfer learning)", ["yolo26n.pt", "yolov8n.pt"])
        epochs = st.number_input("Эпохи", min_value=1, max_value=1000, value=100)
        imgsz = st.selectbox("Размер изображения", [416, 512, 640, 768, 960], index=2)
    with col2:
        batch = st.number_input("Batch size", min_value=1, max_value=256, value=16)
        device = st.selectbox("Устройство", ["0", "cpu"], help="'0' — первая GPU, 'cpu' — без GPU")

    if st.button("🎓 Запустить обучение", width="stretch"):
        result = api_post("/train-model", {
            "project": selected_project,
            "run_name": run_name,
            "base_model": base_model,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "device": device,
        })

        if "task_id" in result:
            st.session_state["train_task_id"] = result["task_id"]
            st.session_state["train_task_project"] = selected_project
            st.rerun()

    # Прогресс обучения показываем отдельным блоком
    if "train_task_id" in st.session_state:
        st.markdown("---")
        st.subheader(f"📊 Прогресс: {st.session_state.get('train_task_project', '')}")

        status = api_get(f"/task/{st.session_state['train_task_id']}")
        st.progress(min(1.0, max(0.0, status.get("progress", 0.0))))
        st.text(f"Статус: {status.get('status')} — {status.get('message')}")

        if status.get("status") in ("done", "failed"):
            if status.get("status") == "done":
                st.success("✅ Обучение завершено!")
            else:
                st.error("❌ Обучение завершилось с ошибкой")
            if st.button("Очистить статус"):
                del st.session_state["train_task_id"]
                st.session_state.pop("train_task_project", None)
                st.rerun()
        else:
            time.sleep(3)
            st.rerun()

elif page == PAGE_TEST:
    st.title("🧪 Тест обученной модели")

    projects = get_projects()
    projects_with_runs = [p for p in projects if (PROJECTS_DIR / p / "runs").exists()]

    if not projects_with_runs:
        st.warning("️ Нет обученных моделей. Сначала обучи модель на вкладке 'Обучение'.")
        st.stop()

    selected_project = st.selectbox("Проект с обученной моделью", projects_with_runs)

    runs_info = api_get(f"/training-runs/{selected_project}")
    runs = runs_info.get("runs", [])

    if not runs:
        st.warning("⚠️ В этом проекте нет завершённых запусков обучения (best.pt не найден).")
        st.stop()

    run_names = [r["name"] for r in runs]
    selected_run = st.selectbox("Запуск (run)", run_names)
    weights_path = next(r["weights_path"] for r in runs if r["name"] == selected_run)

    conf = st.slider("Порог уверенности (confidence)", 0.05, 0.95, 0.25, 0.05)

    uploaded_file = st.file_uploader(
        "Загрузи картинку или видео для теста",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if uploaded_file and st.button("🧪 Запустить тест", width="stretch"):
        test_dir = PROJECTS_DIR / selected_project / "test_uploads"
        test_dir.mkdir(parents=True, exist_ok=True)

        input_path = test_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_path = test_dir / f"result_{Path(uploaded_file.name).stem}{Path(uploaded_file.name).suffix}"
        if output_path.suffix.lower() in (".mp4", ".avi", ".mov"):
            output_path = output_path.with_suffix(".mp4")

        with st.spinner("Инференс..."):
            result = api_post("/test-model", {
                "weights_path": weights_path,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "classes_file": "classes.json",
                "conf": conf,
            })

            if "task_id" in result:
                task_id = result["task_id"]
                status = {}
                while True:
                    status = api_get(f"/task/{task_id}")
                    if status.get("status") in ("done", "failed"):
                        break
                    time.sleep(2)

                if status.get("status") == "done":
                    st.success("✅ Готово!")
                    if output_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        st.image(str(output_path), width="stretch")
                    else:
                        st.video(str(output_path))

                    result_json_path = output_path.with_suffix(".json")
                    if result_json_path.exists():
                        with open(result_json_path, "r", encoding="utf-8") as f:
                            result_data = json.load(f)
                        st.json(result_data)
                else:
                    st.error(f"❌ Ошибка: {status.get('message')}")


# ═══════════════════════════════════════════════════════════════
# 📊 НОВАЯ СТРАНИЦА: БЕНЧМАРК МОДЕЛИ
# ═══════════════════════════════════════════════════════════════
elif page == PAGE_BENCHMARK:
    st.title("📊 Бенчмарк модели")
    st.markdown("Оценка качества (mAP) и скорости (FPS) обученной модели.")

    # 1. Получаем списки с сервера
    models_resp = api_get("/benchmark/models")
    datasets_resp = api_get("/benchmark/datasets")
    models = models_resp.get("models", [])
    datasets = datasets_resp.get("datasets", [])

    if not models:
        st.warning("⚠️ Обученные модели не найдены. Сначала обучите модель.")
        st.stop()
    if not datasets:
        st.warning("⚠️ Датасеты не найдены. Сначала выполните конвертацию или объединение.")
        st.stop()

    # 2. Селекторы
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("🧠 Модель", options=models, format_func=lambda x: x["name"])
    with col2:
        selected_dataset = st.selectbox("📁 Датасет", options=datasets, format_func=lambda x: x["name"])

    # 3. Параметры
    with st.expander("️ Параметры бенчмарка", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        streams = c1.text_input("Потоки (через запятую)", "1,2,4,8")
        duration = c2.number_input("Длительность (сек)", 5, 120, 15)
        conf = c3.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
        imgsz = c4.selectbox("Размер изображения", [320, 416, 512, 640, 768], index=3)

    # 4. Кнопка запуска
    if st.button("🚀 Запустить бенчмарк", type="primary", use_container_width=True):
        ds_path = Path(selected_dataset["data_yaml"]).parent
        # Пробуем оба формата расположения папки val
        images_dir = str(ds_path / "val" / "images")
        if not Path(images_dir).exists():
            images_dir = str(ds_path / "images" / "val")

        payload = {
            "weights": selected_model["path"],
            "data_yaml": selected_dataset["data_yaml"],
            "images_dir": images_dir,
            "streams": streams,
            "duration": duration,
            "conf": conf,
            "imgsz": imgsz
        }

        res = api_post("/benchmark", payload)
        if "task_id" in res:
            st.session_state["bench_task_id"] = res["task_id"]
            st.rerun()

    # 5. Отображение прогресса и результатов
    if "bench_task_id" in st.session_state:
        task_id = st.session_state["bench_task_id"]
        status = api_get(f"/task/{task_id}")

        st.progress(status.get("progress", 0.0), text=status.get("message", "Выполняется..."))

        if status.get("status") == "done":
            st.success("✅ Бенчмарк завершён!")
            result = status.get("result", {})

            # --- Отрисовка результатов: Качество ---
            if "quality" in result:
                st.subheader("📈 Метрики качества")
                overall = result["quality"].get("overall", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precision", f"{overall.get('precision', 0):.3f}")
                c2.metric("Recall", f"{overall.get('recall', 0):.3f}")
                c3.metric("mAP50", f"{overall.get('mAP50', 0):.3f}")
                c4.metric("mAP50-95", f"{overall.get('mAP50-95', 0):.3f}")

                st.markdown("#### По классам:")
                class_data = []
                for cls, metrics in result["quality"].get("per_class", {}).items():
                    class_data.append({
                        "Класс": cls,
                        "Precision": f"{metrics.get('precision', 0):.3f}",
                        "Recall": f"{metrics.get('recall', 0):.3f}",
                        "mAP50": f"{metrics.get('mAP50', 0):.3f}",
                        "mAP50-95": f"{metrics.get('mAP50-95', 0):.3f}"
                    })
                st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)

            # --- Отрисовка результатов: Скорость ---
            if "speed" in result:
                st.subheader("⚡ Скорость (FPS)")
                speed_data = []
                for streams_count, metrics in result["speed"].items():
                    speed_data.append({
                        "Потоков": int(streams_count),
                        "Суммарный FPS": metrics.get("aggregate_fps", 0),
                        "На поток FPS": metrics.get("per_stream_fps", 0),
                        "Всего кадров": metrics.get("total_frames", 0)
                    })
                df_speed = pd.DataFrame(speed_data).sort_values("Потоков")
                st.dataframe(df_speed, use_container_width=True, hide_index=True)
                
                st.markdown("#### График производительности:")
                st.line_chart(df_speed.set_index("Потоков")[["Суммарный FPS", "На поток FPS"]])

            # Кнопка очистки
            if st.button(" Очистить результат и запустить новый"):
                del st.session_state["bench_task_id"]
                st.rerun()

        elif status.get("status") == "failed":
            st.error(f"❌ Ошибка: {status.get('message')}")
            if st.button("🔄 Попробовать снова"):
                del st.session_state["bench_task_id"]
                st.rerun()
        else:
            # Если running или pending, делаем rerun через 3 секунды для обновления прогресса
            time.sleep(3)
            st.rerun()
