"""Обёртка над subprocess для запуска пайплайна."""
import subprocess
import sys
from pathlib import Path
from api.services.task_manager import task_manager
from api.schemas.models import TaskStatus


def run_extract_frames_sync(task_id: str, project: str, fps: int):
    """Извлечение кадров из видео."""
    project_dir = Path("projects") / project
    
    cmd = [
        sys.executable, "frame_out.py",
        "--video", str(project_dir / "video.mp4"),
        "--output", str(project_dir / "frames"),
        "--fps", str(fps)
    ]
    
    task_manager.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        stage="extract_frames",
        progress=0.0,
        message="Extracting frames from video..."
    )
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"Extract frames failed: {result.stderr[:500]}"
            )
            return
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.DONE,
            progress=1.0,
            message="Frames extracted successfully"
        )
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=str(e)
        )


def run_pipeline_sync(task_id: str, project: str, classes: list[str]):
    """Запуск пайплайна (detect → refine → qc_filter) БЕЗ извлечения кадров."""
    project_dir = Path("projects") / project
    
    stages = [
        ("batch_label", [
            sys.executable, "batch_label.py",
            "--images-dir", str(project_dir / "frames"),
            "--classes", ",".join(classes),
            "--out-dir", str(project_dir / "ann")
        ]),
        ("qc_filter", [
            sys.executable, "qc_filter.py",
            "--annotations-dir", str(project_dir / "ann"),
            "--output-dir", str(project_dir / "ann"),
            "--classes-file", "classes.json"
        ]),
    ]
    
    for i, (stage_name, cmd) in enumerate(stages):
        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            stage=stage_name,
            progress=i / len(stages),
            message=f"Running {stage_name}..."
        )
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"{stage_name} failed: {result.stderr[:500]}"
                )
                return
        except Exception as e:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=str(e)
            )
            return
    
    task_manager.update_task(
        task_id,
        status=TaskStatus.DONE,
        progress=1.0,
        message="Pipeline completed"
    )


def run_convert_sync(task_id: str, project: str):
    """Конвертация в YOLO формат."""
    project_dir = Path("projects") / project
    
    cmd = [
        sys.executable, "convert_to_yolo_seg.py",
        "--annotations-dir", str(project_dir / "ann" / "clean"),
        "--classes-file", "classes.json",
        "--output-dir", str(project_dir / "dataset_yolo")
    ]
    
    task_manager.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        stage="convert",
        progress=0.0,
        message="Converting to YOLO format..."
    )
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"Convert failed: {result.stderr[:500]}"
            )
            return
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.DONE,
            progress=1.0,
            message="Conversion completed"
        )
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=str(e)
        )