"""
FastAPI приложение для авто-разметки.
Запуск: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from api.routers import projects, pipeline, review, results

app = FastAPI(
    title="Auto-Labeling API",
    description="Пайплайн авто-разметки для промышленной безопасности",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(review.router, prefix="/api/review", tags=["review"])
app.include_router(results.router, prefix="/api/results", tags=["results"])

# Статические файлы (картинки, датасеты)
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def root():
    return {
        "message": "Auto-Labeling API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}