"""
Единый интерфейс для всего пайплайна авто-разметки.
Запуск: streamlit run main.py
"""
import streamlit as st
from pathlib import Path
import os
import json
import shutil
import subprocess
from review_app import load_class_colors, find_image

# ── Настройки ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Auto-Labeling", layout="wide", page_icon="🔍")
st.title("Авто-разметка промышленных объектов")

# ── Папки проекта ───────────────────────────────────────────────────────────
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# ── Навигация ───────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Домой": "home",
    "📁 Загрузка видео": "upload",
    "🎬 Извлечение кадров": "extract",
    "⚙️ Запуск пайплайна": "pipeline",
    "🔍 Ручная проверка": "review",
    "📦 Конвертация в YOLO": "convert",
}

selected_page = st.sidebar.radio("Навигация", list(PAGES.keys()), index=0)
page = PAGES[selected_page]

# ── Вспомогательные функции ───────────────────────────────────────────────
def run_subprocess(cmd, **kwargs):
    """Запускает subprocess и возвращает вывод."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return result.stdout, result.stderr, result.returncode

def get_projects():
    """Возвращает список проектов."""
    return [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]

def get_project_dir(project_name):
    """Возвращает путь к проекту."""
    return PROJECTS_DIR / project_name

def get_review_files(project_name):
    """Возвращает файлы для ручной проверки."""
    project_dir = get_project_dir(project_name)
    review_dir = project_dir / "ann" / "review"
    if review_dir.exists():
        return sorted([f for f in review_dir.glob("*.json") if not f.name.startswith("_")])
    return []

# ── Страницы приложения ───────────────────────────────────────────────────
if page == "home":
    st.subheader("Добро пожаловать в Auto-Labeling!")
    
    st.markdown("""
    ### 📌 Как работает пайплайн:
    1. **Загрузите видео** или готовые изображения
    2. **Извлеките кадры** из видео (если нужно)
    3. **Запустите пайплайн** детекции и сегментации
    4. **Проверьте результаты** вручную
    5. **Конвертируйте в YOLO** для обучения
    
    ### 🚀 Быстрый старт:
    """)
    
    projects = get_projects()
    
    if projects:
        st.success(f"Найдено проектов: {len(projects)}")
        st.dataframe(projects)
    else:
        st.warning("Нет проектов. Создайте новый, загрузив видео!")
    
    st.markdown("""
    **Совет:** Начните с загрузки видео на вкладке `📁 Загрузка видео`.
    """)

elif page == "upload":
    st.subheader("Загрузка видео или изображений")
    
    # Выбор типа источника
    source_type = st.radio("Источник данных", ["Видео", "Готовые изображения"])
    
    project_name = st.text_input("Имя проекта", "my_project")
    project_dir = get_project_dir(project_name)
    
    if source_type == "Видео":
        uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            st.info(f"Загружен файл: {uploaded_file.name}")
            
            # Создаем проект
            if not project_dir.exists():
                project_dir.mkdir()
                (project_dir / "frames").mkdir()
                st.success("Создан новый проект!")
            
            # Сохраняем видео
            video_path = project_dir / "video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Видео сохранено в {video_path}")
            st.video(str(video_path))
    
    else:  # Готовые изображения
        uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"Загружено файлов: {len(uploaded_files)}")
            
            # Создаем проект
            if not project_dir.exists():
                project_dir.mkdir()
                (project_dir / "images").mkdir()
                st.success("Создан новый проект!")
            
            # Сохраняем изображения
            images_dir = project_dir / "images"
            for file in uploaded_files:
                with open(images_dir / file.name, "wb") as f:
                    f.write(file.getbuffer())
            
            st.success(f"Изображения сохранены в {images_dir}")
            
            # Показываем пример
            if uploaded_files:
                st.image(str(images_dir / uploaded_files[0].name), caption="Пример изображения")

elif page == "extract":
    st.subheader("Извлечение кадров из видео")
    
    projects = get_projects()
    if not projects:
        st.warning("Нет проектов. Создайте проект, загрузив видео!")
        st.stop()
    
    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = get_project_dir(selected_project)
    
    # Проверяем наличие видео
    video_path = project_dir / "video.mp4"
    if not video_path.exists():
        st.error(f"Видео не найдено в {video_path}. Загрузите видео на вкладке 'Загрузка видео'.")
        st.stop()
    
    # Настройки
    fps = st.slider("Кадров в секунду", 1, 30, 5)
    
    if st.button("Извлечь кадры", use_container_width=True):
        with st.spinner("Извлекаем кадры... Это может занять несколько минут"):
            # Запускаем frame_out.py
            cmd = [
                "python", "frame_out.py",
                "--video", str(video_path),
                "--output", str(project_dir / "frames"),
                "--fps", str(fps)
            ]
            
            stdout, stderr, returncode = run_subprocess(cmd)
            
            if returncode == 0:
                st.success("Кадры успешно извлечены!")
                st.balloons()
                
                # Показываем пример
                frames_dir = project_dir / "frames"
                frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                if frame_files:
                    st.image(str(frame_files[0]), caption="Пример кадра")
            else:
                st.error("Ошибка при извлечении кадров!")
                st.code(stderr)

elif page == "pipeline":
    st.subheader("Запуск пайплайна детекции и сегментации")
    
    projects = get_projects()
    if not projects:
        st.warning("Нет проектов. Создайте проект, загрузив видео!")
        st.stop()
    
    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = get_project_dir(selected_project)
    
    # Проверяем наличие кадров
    frames_dir = project_dir / "frames"
    if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
        st.error(f"Кадры не найдены в {frames_dir}. Извлеките кадры на вкладке 'Извлечение кадров'.")
        st.stop()
    
    # Настройки пайплайна
    classes_file = "classes.json"
    with open(classes_file, "r") as f:
        classes_data = json.load(f)
    class_names = [c["name"] for c in classes_data["classes"]]
    
    selected_classes = st.multiselect("Классы для детекции", class_names, default=class_names)
    
    # Запуск пайплайна
    if st.button("Запустить пайплайн", use_container_width=True):
        with st.spinner("Запускаем пайплайн... Это может занять несколько минут"):
            # Запускаем batch_label.py
            cmd = [
                "python", "batch_label.py",
                "--images-dir", str(frames_dir),
                "--classes", ",".join(selected_classes),
                "--out-dir", str(project_dir / "ann"),
                "--config", "pipeline_config.yaml"
            ]
            
            stdout, stderr, returncode = run_subprocess(cmd)
            
            if returncode == 0:
                st.success("Пайплайн успешно завершен!")
                st.balloons()
                
                # Показываем статистику
                ann_dir = project_dir / "ann"
                if ann_dir.exists():
                    with open(ann_dir / "_batch_summary.json") as f:
                        summary = json.load(f)
                    st.metric("Обработано кадров", len(summary))
                    st.metric("Всего детекций", sum(s["num_detections"] for s in summary))
            else:
                st.error("Ошибка при запуске пайплайна!")
                st.code(stderr)

elif page == "review":
    st.subheader("Ручная проверка детекций")
    
    projects = get_projects()
    if not projects:
        st.warning("Нет проектов. Создайте проект, загрузив видео!")
        st.stop()
    
    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = get_project_dir(selected_project)
    
    review_dir = project_dir / "ann" / "review"
    if not review_dir.exists() or not list(review_dir.glob("*.json")):
        st.warning("Нет данных для проверки. Запустите пайплайн на вкладке 'Запуск пайплайна'.")
        st.stop()
    
    # Загружаем цвета классов
    class_colors = load_class_colors()
    
    # Список файлов для проверки
    json_files = get_review_files(selected_project)
    if not json_files:
        st.success("Все детекции проверены! Можно конвертировать в YOLO.")
        st.stop()
    
    # Навигация
    current_index = st.session_state.get("review_index", 0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Назад", use_container_width=True, disabled=current_index == 0):
            st.session_state.review_index = current_index - 1
            st.rerun()
    
    with col2:
        if st.button("Вперёд", use_container_width=True, disabled=current_index == len(json_files) - 1):
            st.session_state.review_index = current_index + 1
            st.rerun()
    
    # Текущий файл
    current_file = json_files[current_index]
    stem = current_file.stem
    
    # Загружаем данные
    with open(current_file, "r") as f:
        data = json.load(f)
    detections = data.get("detections", [])
    
    # Поиск изображения
    image_path = find_image(stem, review_dir, project_dir)
    
    # Две колонки: картинка и детекции
    col_img, col_det = st.columns([2, 1])
    
    with col_img:
        if image_path:
            st.image(str(image_path), use_column_width=True)
            st.caption("Annotated-версия (с масками и рамками QC)")
        else:
            st.error("Картинка не найдена! Проверьте структуру проекта.")
    
    with col_det:
        st.subheader("Детекции")
        
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
            
            # Цвет класса
            cls_color = class_colors.get(cls, "#808080")
            
            st.markdown(f"**{emoji} [{idx}] {cls}**")
            st.markdown(f":{color}[Confidence: {conf:.2f}]")
            if reason:
                st.caption(f"📝 {reason}")
            st.markdown(f"<div style='width:20px;height:20px;background:{cls_color};border:1px solid #333'></div>",
                        unsafe_allow_html=True)
            st.divider()
        
        # Статистика
        accepted_count = sum(1 for d in detections if d.get("qc_bucket") == "accepted")
        review_count = sum(1 for d in detections if d.get("qc_bucket") == "needs_review")
        rejected_count = sum(1 for d in detections if d.get("qc_bucket") == "rejected")
        
        st.markdown(f"**Итого:** ✅ {accepted_count} | ⚠️ {review_count} | ❌ {rejected_count}")
    
    # Кнопки управления
    st.markdown("---")
    st.subheader("Управление детекциями")
    
    # Массовые действия
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Принять все needs_review", use_container_width=True):
            # Обновляем JSON
            for det in detections:
                if det.get("qc_bucket") == "needs_review":
                    det["qc_bucket"] = "accepted"
                    det["qc_reason"] = "accepted via UI"
            
            with open(current_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            st.success("Все детекции приняты!")
            st.rerun()
    
    with col2:
        if st.button("❌ Отклонить все needs_review", use_container_width=True):
            # Обновляем JSON
            for det in detections:
                if det.get("qc_bucket") == "needs_review":
                    det["qc_bucket"] = "rejected"
                    det["qc_reason"] = "rejected via UI"
                    mask_path = det.get("mask_path")
                    if mask_path and Path(mask_path).exists():
                        Path(mask_path).unlink()
            
            with open(current_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            st.success("Все детекции отклонены!")
            st.rerun()
    
    # Перенос в clean
    has_review = any(d.get("qc_bucket") == "needs_review" for d in detections)
    
    if st.button("📦 Перенести в clean", use_container_width=True, disabled=has_review):
        # Переносим файлы
        clean_dir = project_dir / "ann" / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        (clean_dir / "masks").mkdir(parents=True, exist_ok=True)
        
        # JSON
        shutil.move(str(current_file), str(clean_dir / current_file.name))
        
        # Annotated
        ann_jpg = review_dir / f"{stem}_annotated.jpg"
        if ann_jpg.exists():
            shutil.move(str(ann_jpg), str(clean_dir / ann_jpg.name))
        
        # Маски
        src_masks = review_dir / "masks"
        dst_masks = clean_dir / "masks"
        if src_masks.exists():
            for mask in src_masks.glob(f"{stem}_*"):
                shutil.move(str(mask), str(dst_masks / mask.name))
        
        st.success(f"Картинка {stem} перенесена в clean/")
        st.rerun()

elif page == "convert":
    st.subheader("Конвертация в YOLO-формат")
    
    projects = get_projects()
    if not projects:
        st.warning("Нет проектов. Создайте проект, загрузив видео!")
        st.stop()
    
    selected_project = st.selectbox("Выберите проект", projects)
    project_dir = get_project_dir(selected_project)
    
    clean_dir = project_dir / "ann" / "clean"
    if not clean_dir.exists() or not list(clean_dir.glob("*.json")):
        st.warning("Нет данных для конвертации. Проверьте все детекции на вкладке 'Ручная проверка'.")
        st.stop()
    
    # Конвертация
    if st.button("Конвертировать в YOLO", use_container_width=True):
        with st.spinner("Конвертируем в YOLO-формат..."):
            # Запускаем convert_to_yolo_seg.py
            cmd = [
                "python", "convert_to_yolo_seg.py",
                "--annotations-dir", str(clean_dir),
                "--classes-file", "classes.json",
                "--output-dir", str(project_dir / "dataset_yolo")
            ]
            
            stdout, stderr, returncode = run_subprocess(cmd)
            
            if returncode == 0:
                st.success("Конвертация успешно завершена!")
                st.balloons()
                
                # Показываем структуру датасета
                dataset_dir = project_dir / "dataset_yolo"
                st.subheader("Структура датасета:")
                st.code(f"""
{dataset_dir}/
├── images/train/
├── labels/train/
└── data.yaml
                """)
                
                # Скачиваем архив
                zip_path = dataset_dir / f"{selected_project}_dataset.zip"
                with st.spinner("Создаем архив..."):
                    # Создаем ZIP-архив
                    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', dataset_dir)
                
                st.download_button(
                    "Скачать датасет",
                    data=open(zip_path, "rb").read(),
                    file_name=f"{selected_project}_dataset.zip",
                    mime="application/zip"
                )
            else:
                st.error("Ошибка при конвертации!")
                st.code(stderr)

# ── Футер ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Auto-Labeling. Все права защищены.")
st.caption("Это приложение использует вашу локальную машину для обработки данных.")
