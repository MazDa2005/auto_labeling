"""
FastAPI сервер для фоновых задач пайплайна.
Запуск: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Auto-Labeling Backend")

# CORS для Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASKS = {}

PROJECTS_DIR = Path("projects")


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, done, failed
    stage: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None  # Для хранения результатов бенчмарка


class ExtractFramesRequest(BaseModel):
    project: str
    fps: int = 5


class PipelineRequest(BaseModel):
    project: str
    classes: Optional[list[str]] = None


class ConvertRequest(BaseModel):
    project: str


class TrainModelRequest(BaseModel):
    project: str
    run_name: str
    base_model: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"


class TestModelRequest(BaseModel):
    weights_path: str
    input_path: str
    output_path: str
    classes_file: str = "classes.json"
    conf: float = 0.25


class BenchmarkRequest(BaseModel):
    weights: str
    data_yaml: str
    images_dir: str
    streams: str = "1,2,4,8"
    duration: int = 15
    conf: float = 0.25
    imgsz: int = 640


def run_extract_frames(task_id: str, project: str, fps: int):
    project_dir = PROJECTS_DIR / project
    video_path = project_dir / "video.mp4"

    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "extract_frames"
    TASKS[task_id].message = "Извлечение кадров..."

    cmd = [sys.executable, "frame_out.py", "--video", str(video_path), "--output", str(project_dir / "frames"), "--fps", str(fps)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Кадры извлечены"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


def run_pipeline(task_id: str, project: str, classes: list[str]):
    project_dir = PROJECTS_DIR / project
    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "batch_label"
    TASKS[task_id].message = "Запуск детекции..."

    cmd = [sys.executable, "batch_label.py", "--images-dir", str(project_dir / "frames"), "--classes", ",".join(classes), "--out-dir", str(project_dir / "ann"), "--config", "pipeline_config.yaml"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Пайплайн завершен"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


def run_convert(task_id: str, project: str):
    project_dir = PROJECTS_DIR / project
    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "convert"
    TASKS[task_id].message = "Конвертация в YOLO..."

    cmd = [sys.executable, "convert_to_yolo_seg.py", "--annotations-dir", str(project_dir / "ann" / "clean"), "--classes-file", "classes.json", "--output-dir", str(project_dir / "dataset_yolo")]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Конвертация завершена"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)

def run_train_model(task_id: str, req: TrainModelRequest):
    """
    Фоновая задача: обучение student-модели.
    """
    # ИСПРАВЛЕНО: было req.target_project, стало req.project
    project_dir = PROJECTS_DIR / req.project
    dataset_dir = project_dir / "dataset_yolo"
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    progress_file = runs_dir / f"{req.run_name}_progress.json"

    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "train_student"
    TASKS[task_id].message = "Запуск обучения..."

    cmd = [
        "conda", "run", "-n", "sam3", "--no-capture-output", "python", "train_student.py",
        "--source-dirs", str(dataset_dir),
        "--target-dir", str(runs_dir),
        "--runs-dir", str(runs_dir),
        "--run-name", req.run_name,
        "--classes-file", "classes.json",
        "--base-model", req.base_model,
        "--epochs", str(req.epochs),
        "--imgsz", str(req.imgsz),
        "--batch", str(req.batch),
        "--device", req.device,
        "--progress-file", str(progress_file),
        "--workers", "2",
        "--plots", "false",
        "--cache", "false",
        "--patience", "50"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        while proc.poll() is None:
            if progress_file.exists():
                try:
                    data = json.loads(progress_file.read_text(encoding="utf-8"))
                    TASKS[task_id].progress = data.get("progress", 0.0)
                    TASKS[task_id].message = data.get("message", "")
                    TASKS[task_id].stage = data.get("stage", "train")
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(3)

        stdout_tail = (proc.stdout.read() or "") if proc.stdout else ""

        if proc.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка обучения: {stdout_tail[-500:]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            if progress_file.exists():
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                TASKS[task_id].message = data.get("message", "Обучение завершено")
            else:
                TASKS[task_id].message = "Обучение завершено"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)

def run_test_model(task_id: str, req: TestModelRequest):
    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "test_student"
    TASKS[task_id].message = "Инференс..."

    result_json_path = Path(req.output_path).with_suffix(".json")
    cmd = [
        "conda", "run", "-n", "sam3", "--no-capture-output", "python", "test_student.py",
        "--weights", req.weights_path, "--input", req.input_path, "--output", req.output_path,
        "--classes-file", req.classes_file, "--conf", str(req.conf), "--result-json", str(result_json_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500]}"
        else:
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Готово"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


def run_benchmark_task(task_id: str, req: BenchmarkRequest):
    """Фоновая задача: запуск бенчмарка модели."""
    TASKS[task_id].status = "running"
    TASKS[task_id].stage = "benchmark"
    TASKS[task_id].message = "Запуск бенчмарка..."
    TASKS[task_id].progress = 0.1

    output_json = f"benchmark_{task_id}.json"
    cmd = [
        sys.executable, "benchmark_student.py",
        "--weights", req.weights,
        "--data-yaml", req.data_yaml,
        "--sample-images-dir", req.images_dir,
        "--streams", req.streams,
        "--duration", str(req.duration),
        "--conf", str(req.conf),
        "--imgsz", str(req.imgsz),
        "--output-json", output_json
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and Path(output_json).exists():
            with open(output_json, "r", encoding="utf-8") as f:
                TASKS[task_id].result = json.load(f)
            TASKS[task_id].status = "done"
            TASKS[task_id].progress = 1.0
            TASKS[task_id].message = "Бенчмарк завершён"
        else:
            TASKS[task_id].status = "failed"
            TASKS[task_id].message = f"Ошибка: {result.stderr[:500] if result.stderr else 'Неизвестная ошибка'}"
    except subprocess.TimeoutExpired:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = "Превышено время ожидания (10 минут)"
    except Exception as e:
        TASKS[task_id].status = "failed"
        TASKS[task_id].message = str(e)


# ── Эндпоинты ──────────────────────────────────────────────────────────────

@app.post("/extract-frames", response_model=TaskStatus)
async def extract_frames(req: ExtractFramesRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    background_tasks.add_task(run_extract_frames, task_id, req.project, req.fps)
    return TASKS[task_id]


@app.post("/run-pipeline", response_model=TaskStatus)
async def run_pipeline_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    if req.classes is None:
        with open("classes.json", encoding="utf-8") as f:
            classes = [c["name"] for c in json.load(f)["classes"]]
    else:
        classes = req.classes
    background_tasks.add_task(run_pipeline, task_id, req.project, classes)
    return TASKS[task_id]


@app.post("/convert-to-yolo", response_model=TaskStatus)
async def convert_to_yolo(req: ConvertRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    background_tasks.add_task(run_convert, task_id, req.project)
    return TASKS[task_id]


@app.post("/train-model", response_model=TaskStatus)
async def train_model(req: TrainModelRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    background_tasks.add_task(run_train_model, task_id, req)
    return TASKS[task_id]


@app.post("/test-model", response_model=TaskStatus)
async def test_model(req: TestModelRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    background_tasks.add_task(run_test_model, task_id, req)
    return TASKS[task_id]


@app.post("/benchmark", response_model=TaskStatus)
async def start_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = TaskStatus(task_id=task_id, status="pending", message="В очереди")
    background_tasks.add_task(run_benchmark_task, task_id, req)
    return TASKS[task_id]


@app.get("/training-runs/{project}")
async def list_training_runs(project: str):
    runs_dir = PROJECTS_DIR / project / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    result = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            result.append({"name": run_dir.name, "weights_path": str(best.resolve())})
    return {"runs": result}


@app.get("/benchmark/models")
async def get_benchmark_models():
    models = []
    for proj_dir in PROJECTS_DIR.iterdir():
        if proj_dir.is_dir():
            runs_dir = proj_dir / "runs"
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    best_pt = run_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        models.append({"name": f"{proj_dir.name} / {run_dir.name}", "path": str(best_pt.resolve())})
    return {"models": models}


@app.get("/benchmark/datasets")
async def get_benchmark_datasets():
    datasets = []
    for proj_dir in PROJECTS_DIR.iterdir():
        if proj_dir.is_dir():
            ds_dir = proj_dir / "dataset_yolo"
            if ds_dir.exists() and (ds_dir / "data.yaml").exists():
                datasets.append({
                    "name": proj_dir.name,
                    "data_yaml": str((ds_dir / "data.yaml").resolve()),
                })
    return {"datasets": datasets}


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Задача не найдена")
    return TASKS[task_id]


@app.get("/tasks")
async def list_tasks():
    return {"tasks": list(TASKS.values())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
