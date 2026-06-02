"""
Utilidades transversales del proyecto SmartDelay AI 360.

Funciones auxiliares para logging, manejo de archivos, métricas y
operaciones repetitivas en los pipelines de datos y modelos.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from src.config import settings


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def configurar_logger(nombre: str = "smartdelay") -> None:
    """Configura el logger global con formato uniforme para todo el stack."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.info(f"Logger '{nombre}' inicializado en nivel {settings.log_level}")


# ----------------------------------------------------------------------------
# Lectura y escritura de datos
# ----------------------------------------------------------------------------
def cargar_dataset(nombre_archivo: str, capa: str = "raw") -> pd.DataFrame:
    """
    Carga un dataset desde una de las capas de datos (raw/processed/external).

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo (con extensión: .csv o .parquet).
    capa : str
        Capa de datos: 'raw', 'processed' o 'external'.
    """
    mapa_capas = {
        "raw": settings.data_raw_path,
        "processed": settings.data_processed_path,
        "external": settings.data_external_path,
    }
    if capa not in mapa_capas:
        raise ValueError(f"Capa inválida '{capa}'. Use: {list(mapa_capas.keys())}")

    ruta = settings.resolve_path(mapa_capas[capa]) / nombre_archivo
    logger.info(f"Cargando dataset: {ruta}")

    if ruta.suffix == ".csv":
        return pd.read_csv(ruta)
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    raise ValueError(f"Formato no soportado: {ruta.suffix}")


def guardar_dataset(
    df: pd.DataFrame, nombre_archivo: str, capa: str = "processed"
) -> Path:
    """Persiste un DataFrame en la capa indicada y devuelve la ruta."""
    mapa_capas = {
        "raw": settings.data_raw_path,
        "processed": settings.data_processed_path,
        "external": settings.data_external_path,
    }
    base = settings.resolve_path(mapa_capas[capa])
    base.mkdir(parents=True, exist_ok=True)
    ruta = base / nombre_archivo

    if ruta.suffix == ".csv":
        df.to_csv(ruta, index=False)
    elif ruta.suffix == ".parquet":
        df.to_parquet(ruta, index=False)
    else:
        raise ValueError(f"Formato no soportado: {ruta.suffix}")

    logger.success(f"Dataset guardado: {ruta} ({len(df):,} filas)")
    return ruta


# ----------------------------------------------------------------------------
# Persistencia de modelos
# ----------------------------------------------------------------------------
def guardar_modelo(modelo: Any, nombre: str) -> Path:
    """Serializa un modelo entrenado en el directorio de modelos."""
    base = settings.resolve_path(settings.models_path)
    base.mkdir(parents=True, exist_ok=True)
    ruta = base / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    logger.success(f"Modelo guardado en {ruta}")
    return ruta


def cargar_modelo(nombre: str) -> Any:
    """Carga un modelo previamente serializado desde disco."""
    ruta = settings.resolve_path(settings.models_path) / f"{nombre}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
    logger.info(f"Cargando modelo: {ruta}")
    return joblib.load(ruta)


# ----------------------------------------------------------------------------
# Reportes y artefactos
# ----------------------------------------------------------------------------
def guardar_reporte_json(data: dict, nombre: str) -> Path:
    """Persiste un diccionario como reporte JSON con timestamp."""
    base = settings.resolve_path(settings.reports_path)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = base / f"{nombre}_{timestamp}.json"
    with ruta.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"Reporte guardado: {ruta}")
    return ruta


# ----------------------------------------------------------------------------
# Helpers de fecha
# ----------------------------------------------------------------------------
def ahora_iso() -> str:
    """Timestamp ISO-8601 del momento actual."""
    return datetime.now().isoformat(timespec="seconds")
