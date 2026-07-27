"""
Streamlit интерфейс для авто-разметки.
Запуск: streamlit run app.py

FastAPI бэкенд (server.py) поднимается автоматически в фоновом потоке
внутри этого же процесса
"""
import json
import shutil
import socket
import threading
import time
from pathlib import Path
import yaml 

import requests
import streamlit as st
import uvicorn

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

# ── Вспомогательные функции ─────────────────────────────────────────────────
def api_post(endpoint: str, data: dict) -> dict:
    """Отправить POST запрос к API."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к API серверу. Запустите: `python server.py`")
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
    
    # Считаем детекции по статусам
    total_accepted = 0
    total_needs_review = 0
    total_rejected = 0
    
    # Из review/
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
    
    # Из clean/ (там все accepted или rejected после проверки)
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

def find_image(stem: str, project: str) -> Path | None:
    """Найти картинку для отображения."""
    project_dir = PROJECTS_DIR / project
    review_dir = project_dir / "ann" / "review"
    clean_dir = project_dir / "ann" / "clean"

    # Annotated версия
    for bucket in [review_dir, clean_dir]:
        annotated = bucket / f"{stem}_annotated.jpg"
        if annotated.exists():
            return annotated

    # Оригинал
    frames_dir = project_dir / "frames"
    if frames_dir.exists():
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = frames_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def move_to_clean(stem: str, project: str):
    """Перенести картинку из review в clean."""
    project_dir = PROJECTS_DIR / project
    review_dir = project_dir / "ann" / "review"
    clean_dir = project_dir / "ann" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / "masks").mkdir(parents=True, exist_ok=True)

    # JSON
    json_src = review_dir / f"{stem}.json"
    if json_src.exists():
        shutil.move(str(json_src), str(clean_dir / json_src.name))

    # Annotated
    ann_src = review_dir / f"{stem}_annotated.jpg"
    if ann_src.exists():
        shutil.move(str(ann_src), str(clean_dir / ann_src.name))

    # Маски
    src_masks = review_dir / "masks"
    dst_masks = clean_dir / "masks"
    if src_masks.exists():
        for mask in src_masks.glob(f"{stem}_*"):
            shutil.move(str(mask), str(dst_masks / mask.name))


# ── Загрузка цветов классов ──────────────────────────────────────────────────
def load_class_colors() -> dict[str, str]:
    """Загрузить цвета классов из classes.json."""
    if not Path("classes.json").exists():
        return {}
    with open("classes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["name"]: c.get("color", "#808080") for c in data["classes"]}


CLASS_COLORS = load_class_colors()

# ── Навигация ────────────────────────────────────────────────────────────────
PAGE_HOME = "Домой"
PAGE_UPLOAD = "Загрузка"
PAGE_EXTRACT = "Извлечение кадров"
PAGE_PIPELINE = "Пайплайн"
PAGE_REVIEW = "Review"
PAGE_CONVERT = "Конвертация"
PAGE_MERGE = "Объединение датасетов"

st.sidebar.title("Auto-Labeling")
page = st.sidebar.radio(
    "Навигация",
    [PAGE_HOME, PAGE_UPLOAD, PAGE_EXTRACT, PAGE_PIPELINE, PAGE_REVIEW, PAGE_CONVERT, PAGE_MERGE],
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
        st.error(f"❌ Видео не найдено в {video_path}")
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
                    st.error(f"❌ Ошибка: {status.get('message')}")

elif page == PAGE_PIPELINE:
    st.title("⚙️ Запуск пайплайна детекции")

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

    # Загрузить классы
    with open("classes.json", "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    class_names = [c["name"] for c in classes_data["classes"]]

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
    st.title(" Ручная проверка детекций")
 
    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов.")
        st.stop()
 
    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project
    
    # Статистика по проекту
    stats = get_project_stats(selected_project)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Статистика проекта")
    
    # Прогресс-бар
    if stats["total_images"] > 0:
        progress = stats["clean_count"] / stats["total_images"]
        st.sidebar.progress(progress, text=f"Проверено: {stats['clean_count']} из {stats['total_images']}")
    else:
        st.sidebar.info("Нет данных для отображения")
    
    # Счетчики
    st.sidebar.metric(" В review", stats["review_count"])
    st.sidebar.metric("✅ В clean", stats["clean_count"])
    st.sidebar.markdown("---")
    st.sidebar.metric("✅ Принято детекций", stats["accepted"])
    st.sidebar.metric("️ На проверке", stats["needs_review"])
    st.sidebar.metric("❌ Отклонено", stats["rejected"])
    
    json_files = get_review_files(selected_project)
 
    if not json_files:
        st.success("🎉 Все детекции проверены!")
        st.stop()
 
    # Навигация
    if "review_index" not in st.session_state:
        st.session_state.review_index = 0
 
    if st.session_state.review_index >= len(json_files):
        st.session_state.review_index = 0
 
    current_file = json_files[st.session_state.review_index]
    stem = current_file.stem
 
    # Загрузить данные
    with open(current_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    detections = data.get("detections", [])
 
    # Найти картинку
    image_path = find_image(stem, selected_project)
 
    # Две колонки
    col_img, col_det = st.columns([2, 1])
 
    with col_img:
        if image_path:
            st.image(str(image_path), width="stretch")
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
 
    DETECTIONS_PANEL_HEIGHT = 650
 
    with col_det:
        st.subheader("📋 Детекции")
 
        with st.container(height=DETECTIONS_PANEL_HEIGHT, border=True):
            for idx, det in enumerate(detections):
                bucket = det.get("qc_bucket", "accepted")
                cls = det["class"]
                conf = det.get("confidence", 0)
                reason = det.get("qc_reason", "")
 
                if bucket == "accepted":
                    emoji = "✅"
                    color = "green"
                    conf_color = "green"
                elif bucket == "needs_review":
                    emoji = "⚠️"
                    color = "orange"
                    conf_color = "orange"
                else:
                    emoji = "❌"
                    color = "red"
                    conf_color = "red"
 
                cls_color = CLASS_COLORS.get(cls, "#808080")
 
                info_col, accept_col, reject_col = st.columns([4, 1, 1])
 
                with info_col:
                    st.markdown(f"**{emoji} [{idx}] {cls}**")
                    st.markdown(f":{conf_color}[Confidence: {conf:.2f}]")
                    if reason:
                        st.caption(f" {reason}")
                    st.markdown(
                        f"<div style='width:30px;height:30px;background:{cls_color};border:2px solid #333;border-radius:4px;display:inline-block'></div>",
                        unsafe_allow_html=True,
                    )
 
                with accept_col:
                    st.button(
                        "✅",
                        key=f"accept_{stem}_{idx}",
                        help="Принять эту детекцию",
                        disabled=bucket == "accepted",
                        on_click=_accept,
                        args=(idx,),
                    )
 
                with reject_col:
                    st.button(
                        "❌",
                        key=f"reject_{stem}_{idx}",
                        help="Отклонить эту детекцию",
                        disabled=bucket == "rejected",
                        on_click=_reject,
                        args=(idx,),
                    )
 
                st.divider()
 
        accepted = sum(1 for d in detections if d.get("qc_bucket") == "accepted")
        review = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        rejected = sum(1 for d in detections if d.get("qc_bucket") == "rejected")
 
        st.markdown(f"**Итого на картинке:** ✅ {accepted} | ⚠️ {review} | ❌ {rejected}")
 
    # Кнопки навигации
    st.markdown("---")
    st.caption(f"📍 Картинка {st.session_state.review_index + 1} из {len(json_files)} в review")
    
    col1, col2, col3 = st.columns([1, 1, 1])
 
    with col1:
        if st.button("️ Назад", disabled=st.session_state.review_index == 0):
            st.session_state.review_index -= 1
            st.rerun()
 
    with col2:
        if st.button("️ Вперёд", disabled=st.session_state.review_index == len(json_files) - 1):
            st.session_state.review_index += 1
            st.rerun()
 
    with col3:
        has_review = any(d.get("qc_bucket") == "needs_review" for d in detections)
        if st.button("📦 Перенести в clean", disabled=has_review):
            move_to_clean(stem, selected_project)
            st.success(f"✅ {stem} перенесен в clean/")
            st.rerun()

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

                    # Скачать датасет
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
    чтобы объединить их изображения и аннотации в один новый проект.
    """)
    
    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов.")
        st.stop()
    
    # Фильтруем только те проекты, где уже есть dataset_yolo
    projects_with_dataset = []
    for proj in projects:
        dataset_dir = PROJECTS_DIR / proj / "dataset_yolo"
        if dataset_dir.exists() and (dataset_dir / "images" / "train").exists():
            projects_with_dataset.append(proj)
    
    if len(projects_with_dataset) < 2:
        st.warning("⚠️ Для объединения нужно как минимум **2** проекта с готовым YOLO-датасетом. Сначала завершите конвертацию в соответствующих проектах.")
        st.stop()
    
    st.subheader("1. Выберите исходные проекты")
    selected_projects = st.multiselect(
        "Проекты для объединения",
        options=projects_with_dataset,
        help="Будут взяты изображения и метки из папок dataset_yolo/images/train и dataset_yolo/labels/train"
    )
    
    st.subheader("2. Имя нового проекта")
    new_project_name = st.text_input("Введите имя для объединенного проекта", "merged_dataset")
    new_project_dir = PROJECTS_DIR / new_project_name
    
    if new_project_name in projects:
        st.error(f" Проект с именем '{new_project_name}' уже существует! Выберите другое имя.")
        can_merge = False
    elif not new_project_name.strip():
        st.error("❌ Имя проекта не может быть пустым.")
        can_merge = False
    elif len(selected_projects) < 2:
        st.warning("⚠️ Выберите минимум 2 проекта для объединения.")
        can_merge = False
    else:
        can_merge = True
    
    if st.button("🚀 Объединить датасеты", width="stretch", disabled=not can_merge):
        with st.spinner("Объединение датасетов... Это может занять некоторое время."):
            try:
                new_dataset_dir = new_project_dir / "dataset_yolo"
                new_images_dir = new_dataset_dir / "images" / "train"
                new_labels_dir = new_dataset_dir / "labels" / "train"
                
                new_images_dir.mkdir(parents=True, exist_ok=True)
                new_labels_dir.mkdir(parents=True, exist_ok=True)
                
                total_images = 0
                total_labels = 0
                
                progress_bar = st.progress(0)
                
                for i, proj in enumerate(selected_projects):
                    src_dataset = PROJECTS_DIR / proj / "dataset_yolo"
                    src_images = src_dataset / "images" / "train"
                    src_labels = src_dataset / "labels" / "train"
                    
                    if not src_images.exists():
                        continue
                        
                    img_files = list(src_images.glob("*"))
                    for img_path in img_files:
                        stem = img_path.stem
                        ext = img_path.suffix
                        
                        # Добавляем префикс имени проекта, чтобы избежать коллизий имен файлов
                        # (например, frame_000001.jpg из разных проектов)
                        new_stem = f"{proj}_{stem}"
                        
                        # Копируем изображение
                        shutil.copy(img_path, new_images_dir / f"{new_stem}{ext}")
                        total_images += 1
                        
                        # Копируем соответствующую аннотацию, если она есть
                        label_path = src_labels / f"{stem}.txt"
                        if label_path.exists():
                            shutil.copy(label_path, new_labels_dir / f"{new_stem}.txt")
                            total_labels += 1
                    
                    # Обновляем прогресс-бар
                    progress_bar.progress((i + 1) / len(selected_projects))
                
                # Создаем data.yaml для нового объединенного датасета
                with open("classes.json", "r", encoding="utf-8") as f:
                    classes_data = json.load(f)
                
                names = [c["name"] for c in sorted(classes_data["classes"], key=lambda c: c["index"])]
                
                data_yaml = {
                    "path": str(new_dataset_dir.resolve()),
                    "train": "images/train",
                    "val": "images/train", # Пока используем train как val (можно поправить при реальном разделении)
                    "nc": len(names),
                    "names": names,
                }
                
                with open(new_dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
                
                st.success(f"✅ Датасеты успешно объединены в проект: **{new_project_name}**!")
                st.markdown(f"""
                **Статистика объединения:**
                - 🖼️ Изображений скопировано: **{total_images}**
                - 🏷️ Файлов аннотаций скопировано: **{total_labels}**
                - 📁 Путь к новому датасету: `{new_dataset_dir}`
                """)
                
                # Предлагаем скачать объединенный датасет
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
                st.error(f"❌ Произошла ошибка при объединении: {str(e)}")

