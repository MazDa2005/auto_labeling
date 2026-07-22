"""
Streamlit интерфейс для авто-разметки.
Запуск: streamlit run app.py

Требует запущенный FastAPI сервер (server.py) на порту 8000.
"""
import json
import shutil
import time
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Auto-Labeling", layout="wide", page_icon="🔍")

# ── Настройки API ────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
PROJECTS_DIR = Path("projects")

# ── Вспомогательные функции ──────────────────────────────────────────────────

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
# ВАЖНО: строки здесь должны СИМВОЛ В СИМВОЛ совпадать со строками в elif ниже.
PAGE_HOME = "🏠 Домой"
PAGE_UPLOAD = "📁 Загрузка"
PAGE_EXTRACT = "🎬 Извлечение кадров"
PAGE_PIPELINE = "⚙️ Пайплайн"
PAGE_REVIEW = "🔍 Review"
PAGE_CONVERT = "📦 Конвертация"

st.sidebar.title("🔍 Auto-Labeling")
page = st.sidebar.radio(
    "Навигация",
    [PAGE_HOME, PAGE_UPLOAD, PAGE_EXTRACT, PAGE_PIPELINE, PAGE_REVIEW, PAGE_CONVERT],
    index=0,
)

# ── Страницы ─────────────────────────────────────────────────────────────────

if page == PAGE_HOME:
    st.title("🏠 Добро пожаловать в Auto-Labeling!")

    st.markdown("""
    ### 📌 Как работает пайплайн:
    1. **Загрузите видео** или готовые изображения
    2. **Извлеките кадры** из видео
    3. **Запустите пайплайн** детекции и сегментации
    4. **Проверьте результаты** вручную
    5. **Конвертируйте в YOLO** для обучения

    ### 🚀 Быстрый старт:
    """)

    projects = get_projects()

    if projects:
        st.success(f"✅ Найдено проектов: {len(projects)}")
        for proj in projects:
            st.markdown(f"- **{proj}**")
    else:
        st.warning("⚠️ Нет проектов. Создайте новый, загрузив видео!")

    st.markdown("---")
    st.caption("💡 Убедитесь, что FastAPI сервер запущен: `python server.py`")

elif page == PAGE_UPLOAD:
    st.title("📁 Загрузка видео или изображений")

    source_type = st.radio("Источник данных", ["Видео", "Готовые изображения"])

    project_name = st.text_input("Имя проекта", "my_project")
    project_dir = PROJECTS_DIR / project_name

    if source_type == "Видео":
        uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])

        if uploaded_file:
            st.info(f"📹 Загружен файл: {uploaded_file.name}")

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

                # Показать пример
                st.image(str(images_dir / uploaded_files[0].name), caption="Пример изображения")

elif page == PAGE_EXTRACT:
    st.title("🎬 Извлечение кадров из видео")

    projects = get_projects()
    if not projects:
        st.warning("⚠️ Нет проектов. Создайте проект на вкладке 'Загрузка'.")
        st.stop()

    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = PROJECTS_DIR / selected_project

    video_path = project_dir / "video.mp4"
    if not video_path.exists():
        st.error(f"❌ Видео не найдено в {video_path}")
        st.stop()

    fps = st.slider("Кадров в секунду", 1, 30, 5)

    if st.button("🎬 Извлечь кадры", use_container_width=True):
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
        st.error(f"❌ Кадры не найдены в {frames_dir}")
        st.stop()

    # Загрузить классы
    with open("classes.json", "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    class_names = [c["name"] for c in classes_data["classes"]]

    selected_classes = st.multiselect("Классы для детекции", class_names, default=class_names)

    if st.button("⚙️ Запустить пайплайн", use_container_width=True):
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
            st.image(str(image_path), use_container_width=True)
        else:
            st.error("❌ Картинка не найдена!")

    with col_det:
        st.subheader("📋 Детекции")

        for idx, det in enumerate(detections):
            bucket = det.get("qc_bucket", "accepted")
            cls = det["class"]
            conf = det.get("confidence", 0)
            reason = det.get("qc_reason", "")

            if bucket == "accepted":
                emoji = "✅"
                color = "green"
            elif bucket == "needs_review":
                emoji = "⚠️"
                color = "orange"
            else:
                emoji = "❌"
                color = "red"

            cls_color = CLASS_COLORS.get(cls, "#808080")

            st.markdown(f"**{emoji} [{idx}] {cls}**")
            st.markdown(f":{color}[Confidence: {conf:.2f}]")
            if reason:
                st.caption(f"📝 {reason}")
            st.markdown(f"<div style='width:20px;height:20px;background:{cls_color};border:1px solid #333'></div>",
                        unsafe_allow_html=True)
            st.divider()

        # Статистика
        accepted = sum(1 for d in detections if d.get("qc_bucket") == "accepted")
        review = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        rejected = sum(1 for d in detections if d.get("qc_bucket") == "rejected")

        st.markdown(f"**Итого:** ✅ {accepted} | ⚠️ {review} | ❌ {rejected}")

    # Кнопки навигации
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Назад", disabled=st.session_state.review_index == 0):
            st.session_state.review_index -= 1
            st.rerun()

    with col2:
        if st.button("➡️ Вперёд", disabled=st.session_state.review_index == len(json_files) - 1):
            st.session_state.review_index += 1
            st.rerun()

    with col3:
        has_review = any(d.get("qc_bucket") == "needs_review" for d in detections)
        if st.button("📦 Перенести в clean", disabled=has_review):
            move_to_clean(stem, selected_project)
            st.success(f"✅ {stem} перенесен в clean/")
            st.rerun()

    # Массовые действия
    st.markdown("### ⚡ Массовые действия")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Принять все needs_review"):
            for det in detections:
                if det.get("qc_bucket") == "needs_review":
                    det["qc_bucket"] = "accepted"
                    det["qc_reason"] = "accepted via UI"
            with open(current_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            st.rerun()

    with col2:
        if st.button("❌ Отклонить все needs_review"):
            for det in detections:
                if det.get("qc_bucket") == "needs_review":
                    det["qc_bucket"] = "rejected"
                    det["qc_reason"] = "rejected via UI"
                    mask_path = det.get("mask_path")
                    if mask_path and Path(mask_path).exists():
                        Path(mask_path).unlink()
            with open(current_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
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

    if st.button("📦 Конвертировать в YOLO", use_container_width=True):
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

# ─ Футер ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Auto-Labeling")