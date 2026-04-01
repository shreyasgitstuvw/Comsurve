# ── MCEI Production Dockerfile ────────────────────────────────────────────────
# Multi-stage: builder installs Python deps, runtime is a slim final image.
# Both API (uvicorn) and scheduler are served from the same image;
# the SERVICE env var controls which process starts (see entrypoint.sh).

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools needed for some native Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="MCEI"
LABEL org.opencontainers.image.description="Multimodal Commodity Event Intelligence Engine"

# Non-root user for security
RUN useradd --uid 1001 --create-home mcei

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=mcei:mcei . /app

# Runtime defaults — override with env vars in docker-compose.yml or .env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/mcei.db \
    QDRANT_PATH=/data/qdrant_data \
    LOG_FILE=/data/logs/mcei.log \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    SERVICE=api

# Persistent data lives on a mounted volume
VOLUME ["/data"]

EXPOSE 8000

USER mcei

ENTRYPOINT ["bash", "/app/scripts/entrypoint.sh"]
