# ============================================================================
# SmartDelay AI 360 - Dockerfile (imagen base de la plataforma)
# ============================================================================
# Imagen multipropósito utilizada por los servicios `api` y `app`.
# Cada servicio sobrescribe el CMD desde docker-compose.yml.
# ============================================================================

FROM python:3.11-slim AS base

# --- Metadatos -----------------------------------------------------------------
LABEL maintainer="SmartDelay AI 360"
LABEL description="Plataforma de IA para predicción de retrasos y rentabilidad logística"

# --- Variables de entorno del runtime -----------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# --- Dependencias del sistema --------------------------------------------------
# build-essential: compilación de paquetes nativos (xgboost, prophet, etc.)
# libgomp1: requerido por XGBoost/LightGBM en runtime
# curl: healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Directorio de trabajo -----------------------------------------------------
WORKDIR /app

# --- Instalación de dependencias Python ---------------------------------------
# Copiamos primero requirements.txt para aprovechar la cache de Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- Copiar código fuente del proyecto ----------------------------------------
COPY . .

# --- Usuario no root para producción ------------------------------------------
RUN useradd --create-home --shell /bin/bash smartdelay && \
    chown -R smartdelay:smartdelay /app
USER smartdelay

# --- Puertos expuestos ---------------------------------------------------------
# 8000: API FastAPI
# 8501: Dashboard Streamlit
EXPOSE 8000 8501

# --- Comando por defecto (sobreescrito por docker-compose) --------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
