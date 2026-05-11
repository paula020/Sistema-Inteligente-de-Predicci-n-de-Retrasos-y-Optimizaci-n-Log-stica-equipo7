"""Endpoint de salud (healthcheck) para monitoreo y orquestadores."""

from fastapi import APIRouter

from src.config import settings
from src.utils import ahora_iso


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", summary="Verifica que la API está viva")
def health_check() -> dict:
    """Endpoint mínimo de healthcheck."""
    return {
        "status": "ok",
        "service": settings.api_title,
        "version": settings.api_version,
        "env": settings.app_env,
        "timestamp": ahora_iso(),
    }
