# syntax=docker/dockerfile:1

# ── Stage 1: чистый код приложения ──
FROM alpine AS app-src
WORKDIR /app
COPY . /app
RUN rm -rf /app/docker/environments /app/docker/miniconda.sh

# ── Stage 2: сборка conda-окружений ──
# Используем Ubuntu 22.04, он стабильнее и лучше работает с apt
FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS envbuilder
ENV DEBIAN_FRONTEND=noninteractive

COPY docker/miniconda.sh /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

ADD docker/environments/dino.tar.gz /opt/conda/envs/dino
RUN /opt/conda/envs/dino/bin/conda-unpack

ADD docker/environments/sam3.tar.gz /opt/conda/envs/sam3
RUN /opt/conda/envs/sam3/bin/conda-unpack

ADD docker/environments/nvidia.tar.gz /opt/conda/envs/nvidia
RUN /opt/conda/envs/nvidia/bin/conda-unpack

# ── Stage 3: финальный образ ──
FROM nvidia/cuda:12.6.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 🔥 ПРИНУДИТЕЛЬНАЯ СМЕНА ЗЕРКАЛА НА ЯНДЕКС (обход блокировок archive.ubuntu.com)
RUN sed -i 's/archive.ubuntu.com/mirror.yandex.ru/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.yandex.ru/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libglx-mesa0 \
        libxxf86vm1 \
        libx11-6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=envbuilder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

COPY --from=app-src /app /app

RUN mkdir -p \
    /home/iet/iet-share/models \
    /home/iet/iet-share/sam3 \
    /home/iet/iet-share/LocateAnything-3B \
    /app/projects

EXPOSE 8501
EXPOSE 8000

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
