"""
Punto de entrada de la API SmartDelay AI 360 (FastAPI).

Registra los routers de health, predicciones y recomendaciones,
y configura CORS, logging y metadatos OpenAPI.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, predictions, recommendations
from src.config import settings
from src.utils import configurar_logger


configurar_logger("api")


# ----------------------------------------------------------------------------
# Aplicación principal
# ----------------------------------------------------------------------------
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "API enterprise para predicción de retrasos logísticos, análisis de "
        "rentabilidad, forecast de ventas y recomendaciones ejecutivas con "
        "IA generativa (Claude vía Amazon Bedrock)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ----------------------------------------------------------------------------
# Middlewares
# ----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Registro de routers
# ----------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(recommendations.router)


# ----------------------------------------------------------------------------
# Endpoint raíz
# ----------------------------------------------------------------------------
@app.get("/", tags=["root"], summary="Información general del servicio")
def root() -> dict:
    """Información general y enlaces útiles del API."""
    return {
        "servicio": settings.api_title,
        "version": settings.api_version,
        "documentacion": "/docs",
        "redoc": "/redoc",
        "healthcheck": "/health/",
    }
