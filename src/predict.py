"""
Inferencia del modelo de predicción de retrasos logísticos — MVP.

Arquitectura
------------
Para el MVP usamos un único `Pipeline` sklearn que encapsula TODO el
preprocesamiento (imputación, escalado, one-hot encoding) y el clasificador.
Esto garantiza que la inferencia desde Streamlit/API sea idéntica al
entrenamiento sin tener que re-aplicar transformaciones a mano.

Artefactos esperados en `models/`:
    - delay_model.pkl       (joblib.dump del Pipeline completo)
    - feature_columns.json  (metadata: columnas esperadas, métricas, etc.)

Funciones expuestas:
    - `cargar_pipeline_mvp()`            → (pipeline, info)
    - `predecir_desde_formulario(form)`  → dict con resultado de negocio
    - `predecir_lote(df)`                → DataFrame enriquecido
    - `clasificar_riesgo(probabilidad)`  → 'BAJO' | 'MEDIO' | 'ALTO'
    - `recomendacion_para(nivel)`        → texto operativo

Umbrales de negocio (MVP):
    - BAJO  : probabilidad < 0.40
    - MEDIO : 0.40 ≤ probabilidad ≤ 0.70
    - ALTO  : probabilidad > 0.70
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from src.config import settings


# ============================================================================
# Constantes
# ============================================================================
NOMBRE_MODELO_DEFAULT = "delay_model.pkl"
NOMBRE_COLUMNAS_DEFAULT = "feature_columns.json"

# Umbrales del MVP
UMBRAL_BAJO = 0.40
UMBRAL_ALTO = 0.70

# Recomendaciones operativas asociadas a cada nivel de riesgo
RECOMENDACIONES = {
    "BAJO": (
        "Continuar con el flujo logístico normal. "
        "No se requieren acciones adicionales."
    ),
    "MEDIO": (
        "Monitorear el envío de cerca y validar la capacidad operativa "
        "del operador asignado. Revisar disponibilidad de la flota."
    ),
    "ALTO": (
        "Priorizar el despacho, revisar la ruta planeada y "
        "notificar preventivamente al cliente. Considerar reasignación "
        "del transportista o cambio de modo de envío."
    ),
}


# Cache simple en memoria (evita recargar el pipeline en cada request)
_cache: dict[str, Any] = {}


# ============================================================================
# Carga del pipeline y metadata
# ============================================================================
def _ruta_artefacto(nombre: str) -> Path:
    """Resuelve la ruta de un archivo dentro de `models/`."""
    return settings.resolve_path(settings.models_path) / nombre


def cargar_pipeline_mvp(
    nombre_modelo: str = NOMBRE_MODELO_DEFAULT,
    nombre_columnas: str = NOMBRE_COLUMNAS_DEFAULT,
    forzar_recarga: bool = False,
) -> tuple[Any, dict]:
    """
    Carga el pipeline MVP y la metadata de columnas desde `models/`.

    Returns
    -------
    (pipeline, info) : tuple
        - pipeline : sklearn Pipeline con preprocesamiento + clasificador
        - info     : dict con `feature_columns`, métricas, modelo, etc.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra `delay_model.pkl` o `feature_columns.json`.
    """
    if not forzar_recarga and "pipeline" in _cache:
        return _cache["pipeline"], _cache["info"]

    ruta_modelo = _ruta_artefacto(nombre_modelo)
    ruta_columnas = _ruta_artefacto(nombre_columnas)

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline MVP en {ruta_modelo}.\n"
            f"Ejecuta primero `python scripts/train_mvp.py` o llama a "
            f"`entrenar_modelo_delay_mvp()` para entrenar y persistir el modelo."
        )
    if not ruta_columnas.exists():
        raise FileNotFoundError(
            f"No se encontró la metadata de columnas en {ruta_columnas}."
        )

    logger.info(f"Cargando pipeline desde {ruta_modelo}")
    pipeline = joblib.load(ruta_modelo)
    with ruta_columnas.open(encoding="utf-8") as f:
        info = json.load(f)

    _cache["pipeline"] = pipeline
    _cache["info"] = info
    logger.success(
        f"Pipeline '{info.get('modelo', '?')}' cargado | "
        f"{len(info.get('feature_columns', []))} features"
    )
    return pipeline, info


def limpiar_cache_modelos() -> None:
    """Limpia el cache en memoria (útil tras re-entrenar)."""
    _cache.clear()
    logger.info("Cache de modelos limpiado")


# ============================================================================
# Clasificación de riesgo y recomendaciones
# ============================================================================
def clasificar_riesgo(probabilidad: float) -> str:
    """
    Convierte una probabilidad en una etiqueta de riesgo de negocio.

    Umbrales del MVP:
        - BAJO  : < 40 %
        - MEDIO : 40 % – 70 %
        - ALTO  : > 70 %
    """
    if probabilidad < UMBRAL_BAJO:
        return "BAJO"
    if probabilidad <= UMBRAL_ALTO:
        return "MEDIO"
    return "ALTO"


def recomendacion_para(nivel_riesgo: str) -> str:
    """Devuelve la recomendación operativa asociada al nivel de riesgo."""
    return RECOMENDACIONES.get(nivel_riesgo, "Sin recomendación disponible.")


# ============================================================================
# Construcción del DataFrame desde el formulario
# ============================================================================
def _construir_dataframe_form(form: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Construye un DataFrame con las EXACTAS columnas esperadas por el pipeline.

    - Si `order_date` viene en el form, deriva `order_month` y `order_dayofweek`.
    - Verifica que estén presentes todas las `feature_columns`.
    - Devuelve el DataFrame con el orden de columnas que espera el pipeline.

    Lanza KeyError si falta algún campo requerido.
    """
    df = pd.DataFrame([form])

    # Derivar variables temporales si el form trae una fecha
    if "order_date" in df.columns:
        fecha = pd.to_datetime(df["order_date"], errors="coerce")
        if "order_month" not in df.columns:
            df["order_month"] = fecha.dt.month
        if "order_dayofweek" not in df.columns:
            df["order_dayofweek"] = fecha.dt.dayofweek
        df = df.drop(columns=["order_date"])

    # Validar que tenemos TODAS las features esperadas
    faltantes = [c for c in feature_columns if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Faltan campos en el formulario: {faltantes}. "
            f"El modelo requiere: {feature_columns}"
        )

    # Asegurar el orden y descartar columnas extra
    return df[feature_columns].copy()


# ============================================================================
# Predicción principal — entrada para Streamlit / API
# ============================================================================
def predecir_desde_formulario(
    form: dict[str, Any],
    nombre_modelo: str = NOMBRE_MODELO_DEFAULT,
) -> dict[str, Any]:
    """
    Entrada principal de predicción para Streamlit y la API.

    Recibe el dict del formulario, valida las columnas, transforma fechas
    si es necesario y delega TODO el preprocesamiento al Pipeline sklearn
    ya entrenado.

    Parameters
    ----------
    form : dict
        Campos del formulario (las 13 features del MVP + opcional `order_date`).
    nombre_modelo : str
        Archivo del pipeline a usar dentro de `models/`.

    Returns
    -------
    dict con:
        - 'ok' (bool)
        - 'probabilidad' (float, 0..1)
        - 'probabilidad_pct' (float, 0..100)
        - 'etiqueta' (str)           # 'Puede retrasarse' / 'Bajo riesgo de retraso'
        - 'nivel_riesgo' (str)       # BAJO / MEDIO / ALTO
        - 'recomendacion' (str)
        - 'modelo_usado' (str)
        - 'features_aplicadas' (int)
        - 'error' (str | None)
    """
    try:
        pipeline, info = cargar_pipeline_mvp(nombre_modelo=nombre_modelo)
        feature_columns = info["feature_columns"]

        df = _construir_dataframe_form(form, feature_columns)
        probabilidad = float(pipeline.predict_proba(df)[0, 1])
        nivel = clasificar_riesgo(probabilidad)
        etiqueta = (
            "Puede retrasarse"
            if probabilidad >= UMBRAL_BAJO
            else "Bajo riesgo de retraso"
        )

        return {
            "ok": True,
            "probabilidad": probabilidad,
            "probabilidad_pct": round(probabilidad * 100, 2),
            "etiqueta": etiqueta,
            "nivel_riesgo": nivel,
            "recomendacion": recomendacion_para(nivel),
            "modelo_usado": info.get("modelo", "delay_model"),
            "features_aplicadas": len(feature_columns),
            "error": None,
        }

    except FileNotFoundError as exc:
        logger.error(f"Modelo no encontrado: {exc}")
        return {
            "ok": False,
            "error": (
                "El modelo aún no ha sido entrenado.\n"
                "Ejecuta `python scripts/train_mvp.py` o llama a "
                "`entrenar_modelo_delay_mvp()` desde un notebook."
            ),
            "detalle": str(exc),
        }

    except KeyError as exc:
        logger.error(f"Campos faltantes en el formulario: {exc}")
        return {
            "ok": False,
            "error": (
                f"El formulario no contiene todos los campos requeridos. {exc}"
            ),
            "detalle": str(exc),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado en predecir_desde_formulario")
        return {
            "ok": False,
            "error": f"Error procesando el formulario: {exc}",
            "detalle": str(exc),
        }


# ============================================================================
# Predicción por lote (DataFrame con varias filas)
# ============================================================================
def predecir_lote(
    df: pd.DataFrame, nombre_modelo: str = NOMBRE_MODELO_DEFAULT
) -> pd.DataFrame:
    """
    Predice retraso para un DataFrame completo y devuelve uno enriquecido.

    Espera que `df` contenga (al menos) las columnas listadas en
    `info['feature_columns']`. Las columnas extra se ignoran.
    """
    pipeline, info = cargar_pipeline_mvp(nombre_modelo=nombre_modelo)
    feature_columns = info["feature_columns"]

    faltantes = [c for c in feature_columns if c not in df.columns]
    if faltantes:
        raise KeyError(f"Columnas faltantes en el lote: {faltantes}")

    X = df[feature_columns]
    probabilidades = pipeline.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["probabilidad_retraso"] = probabilidades
    df_out["prediccion"] = (probabilidades >= 0.5).astype(int)
    df_out["nivel_riesgo"] = [clasificar_riesgo(p) for p in probabilidades]
    return df_out


# ============================================================================
# Compatibilidad con el flujo "legacy" (DataFrame ya pre-procesado)
# ============================================================================
def predecir_retraso(
    features: dict | pd.DataFrame,
    nombre_modelo: str = NOMBRE_MODELO_DEFAULT,
    umbral_decision: float = 0.5,
) -> dict:
    """
    Predice probabilidad de retraso para un input. Acepta:
        - dict: se trata como una fila del formulario y delega en
          `predecir_desde_formulario`.
        - DataFrame: se asume que ya tiene las `feature_columns` y se
          aplica el pipeline directamente.
    """
    if isinstance(features, dict):
        resultado = predecir_desde_formulario(features, nombre_modelo)
        if not resultado.get("ok"):
            raise RuntimeError(resultado.get("error"))
        return {
            "probabilidad_retraso": [resultado["probabilidad"]],
            "prediccion": [int(resultado["probabilidad"] >= umbral_decision)],
            "nivel_riesgo": [resultado["nivel_riesgo"]],
            "umbral_usado": umbral_decision,
            "modelo_usado": resultado["modelo_usado"],
        }

    pipeline, info = cargar_pipeline_mvp(nombre_modelo=nombre_modelo)
    X = features[info["feature_columns"]]
    probabilidades = pipeline.predict_proba(X)[:, 1]
    predicciones = (probabilidades >= umbral_decision).astype(int)
    niveles = [clasificar_riesgo(p) for p in probabilidades]
    return {
        "probabilidad_retraso": probabilidades.tolist(),
        "prediccion": predicciones.tolist(),
        "nivel_riesgo": niveles,
        "umbral_usado": umbral_decision,
        "modelo_usado": info.get("modelo", "delay_model"),
    }
