"""
Inferencia del modelo de predicción de retrasos.

Carga el modelo entrenado desde disco y expone funciones de predicción
para una guía individual o un lote completo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import cargar_modelo


# Cache simple en memoria para evitar recargar el modelo en cada llamada
_modelo_cache: dict[str, Any] = {}


def _obtener_modelo(nombre_modelo: str) -> Any:
    """Carga el modelo desde caché o desde disco si no está en memoria."""
    if nombre_modelo not in _modelo_cache:
        logger.info(f"Cargando modelo '{nombre_modelo}' a memoria")
        _modelo_cache[nombre_modelo] = cargar_modelo(nombre_modelo)
    return _modelo_cache[nombre_modelo]


def predecir_retraso(
    features: dict | pd.DataFrame,
    nombre_modelo: str = "modelo_retrasos_xgb",
    umbral: float = 0.5,
) -> dict:
    """
    Predice probabilidad de retraso para una guía o lote.

    Parameters
    ----------
    features : dict | pd.DataFrame
        Diccionario (una guía) o DataFrame (lote) con las features esperadas.
    nombre_modelo : str
        Nombre del modelo serializado en `models/`.
    umbral : float
        Umbral de decisión para clasificar como retraso.

    Returns
    -------
    dict
        Diccionario con probabilidades, etiquetas y nivel de riesgo.
    """
    modelo = _obtener_modelo(nombre_modelo)

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features.copy()

    probabilidades = modelo.predict_proba(df)[:, 1]
    predicciones = (probabilidades >= umbral).astype(int)
    niveles_riesgo = [_clasificar_riesgo(p) for p in probabilidades]

    logger.info(f"Predicciones generadas: {len(df):,} registros")

    return {
        "probabilidad_retraso": probabilidades.tolist(),
        "prediccion": predicciones.tolist(),
        "nivel_riesgo": niveles_riesgo,
        "umbral_usado": umbral,
    }


def _clasificar_riesgo(probabilidad: float) -> str:
    """Convierte una probabilidad en una etiqueta de riesgo de negocio."""
    if probabilidad < 0.25:
        return "BAJO"
    if probabilidad < 0.5:
        return "MEDIO"
    if probabilidad < 0.75:
        return "ALTO"
    return "CRITICO"


def predecir_lote(
    df: pd.DataFrame,
    nombre_modelo: str = "modelo_retrasos_xgb",
    umbral: float = 0.5,
) -> pd.DataFrame:
    """Predice para un DataFrame completo y devuelve un DataFrame enriquecido."""
    resultado = predecir_retraso(df, nombre_modelo, umbral)
    df_out = df.copy()
    df_out["probabilidad_retraso"] = resultado["probabilidad_retraso"]
    df_out["prediccion"] = resultado["prediccion"]
    df_out["nivel_riesgo"] = resultado["nivel_riesgo"]
    return df_out
