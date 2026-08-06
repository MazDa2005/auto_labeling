#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Проверка GPU внутри контейнера:"
nvidia-smi || echo "[entrypoint][WARN] nvidia-smi недоступен — GPU не проброшен"

export PATH=/opt/conda/envs/sam3/bin:$PATH

echo "[entrypoint] Запуск Streamlit из окружения sam3..."
exec /opt/conda/envs/sam3/bin/streamlit run /app/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
