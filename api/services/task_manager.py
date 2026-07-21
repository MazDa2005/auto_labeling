"""Менеджер задач для отслеживания прогресса."""
import uuid
from datetime import datetime
from typing import Optional
from api.schemas.models import TaskStatus, TaskStatusResponse


class TaskManager:
    """In-memory менеджер задач (в продакшене заменить на Redis/БД)."""
    
    def __init__(self):
        self.tasks: dict[str, TaskStatusResponse] = {}
    
    def create_task(self) -> str:
        """Создаёт новую задачу и возвращает task_id."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            stage=None,
            progress=0.0,
            message="Queued"
        )
        return task_id
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ):
        """Обновляет статус задачи."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        if status is not None:
            task.status = status
        if stage is not None:
            task.stage = stage
        if progress is not None:
            task.progress = progress
        if message is not None:
            task.message = message
    
    def get_task(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Возвращает задачу по ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> list[TaskStatusResponse]:
        """Возвращает список всех задач."""
        return list(self.tasks.values())


# Глобальный экземпляр
task_manager = TaskManager()