"""
Módulo de inferencia para SmartDelay AI 360.

Encapsula la lógica que transforma un *input crudo* (lo que llega
desde el formulario de Streamlit o un POST a la API) en un vector
de features con la misma forma que `X_train`, listo para
`modelo.predict_proba`.

Componentes:
    - `MetadataInferencia`: dataclass con el encoder de target encoding,
      la lista de columnas finales del modelo y los nombres de las
      columnas categóricas one-hot.
    - `guardar_metadata_inferencia` / `cargar_metadata_inferencia`:
      persistencia de la metadata junto al modelo en `models/`.
    - `transformar_para_inferencia`: aplica el mismo pipeline de
      `construir_features_dataco` pero sobre una fila/lote nuevo,
      reusando el encoder entrenado (sin re-fit) y alineando las
      columnas one-hot con la lista guardada (rellenando con 0).

las features ex-post (`days_for_shipping_real`,
`delay_days`, `delivery_status`, etc.) NUNCA llegan a la inferencia
porque fueron eliminadas del set de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings
from src.features import (
    TARGET,
    TargetEncoderSuavizado,
    codificar_categoricas,
    variables_economicas,
    variables_temporales,
)


# Nombre del artefacto persistido junto al modelo ganador
NOMBRE_METADATA = "metadata_inferencia"

# Categóricas one-hot por defecto (deben coincidir con las del entrenamiento)
CATEGORICAS_DEFAULT = (
    "shipping_mode",
    "customer_segment",
    "market",
    "order_region",
    "category_name",
    "department_name",
)


# ============================================================================
# Dataclass de metadata
# ============================================================================
@dataclass
class MetadataInferencia:
    """Conjunto de objetos necesarios para reproducir el preprocesamiento."""

    encoder: TargetEncoderSuavizado
    feature_columns: list[str]
    columnas_onehot: list[str] = field(default_factory=list)
    media_global_target: float = 0.5

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns no puede estar vacío")


# ============================================================================
# Persistencia
# ============================================================================
def _ruta_metadata(nombre: str = NOMBRE_METADATA) -> Path:
    base = settings.resolve_path(settings.models_path)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{nombre}.joblib"


def guardar_metadata_inferencia(
    encoder: TargetEncoderSuavizado,
    feature_columns: list[str],
    columnas_onehot: list[str] | None = None,
    nombre: str = NOMBRE_METADATA,
) -> Path:
    """Persiste la metadata necesaria para inferencia en `models/`."""
    columnas_onehot = list(columnas_onehot or CATEGORICAS_DEFAULT)
    metadata = MetadataInferencia(
        encoder=encoder,
        feature_columns=list(feature_columns),
        columnas_onehot=columnas_onehot,
        media_global_target=float(encoder.media_global_),
    )
    ruta = _ruta_metadata(nombre)
    joblib.dump(metadata, ruta)
    logger.success(
        f"Metadata de inferencia guardada: {ruta} | "
        f"{len(metadata.feature_columns)} features, "
        f"media_global={metadata.media_global_target:.4f}"
    )
    return ruta


def cargar_metadata_inferencia(nombre: str = NOMBRE_METADATA) -> MetadataInferencia:
    """Carga la metadata persistida o lanza FileNotFoundError."""
    ruta = _ruta_metadata(nombre)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró metadata de inferencia en {ruta}. "
            "Entrena un modelo y llama a `seleccionar_mejor_modelo(..., "
            "encoder=..., feature_columns=...)` primero."
        )
    metadata: MetadataInferencia = joblib.load(ruta)
    logger.info(
        f"Metadata cargada | {len(metadata.feature_columns)} features | "
        f"media_global={metadata.media_global_target:.4f}"
    )
    return metadata


# ============================================================================
# Transformación de un input crudo en X listo para el modelo
# ============================================================================
def _enriquecer_input(form: dict[str, Any]) -> pd.DataFrame:
    """
    Construye un DataFrame de una fila a partir del dict del formulario.

    Añade campos derivados que el modelo espera pero el formulario no
    expone directamente (con valores neutros / calculados):
        - `order_city`     -> "DESCONOCIDA"   (target encoder caerá a media global)
        - `customer_id`    -> -1              (idem)
        - `route`          -> market + "__" + order_region
        - `order_year`, `order_month`, `order_dayofweek`, `order_quarter`,
          `order_weekofyear`, `order_is_weekend`, `order_hour`
        - `order_item_profit_ratio` ≈ 1 - discount_pct
        - `discount_pct`
    """
    df = pd.DataFrame([form])

    # Variables temporales: si llega `order_date` (datetime) las derivamos
    if "order_date" in df.columns:
        df["order_date_dateorders"] = pd.to_datetime(df["order_date"])
        df = variables_temporales(df, "order_date_dateorders")
        df = df.drop(columns=["order_date"])
    else:
        # Si no llega la fecha, usamos los campos explícitos del formulario
        df["order_year"] = df.get("order_year", pd.Timestamp.now().year)
        df["order_month"] = df.get("order_month", 1)
        df["order_dayofweek"] = df.get("order_dayofweek", 0)
        df["order_day"] = df.get("order_day", 1)
        df["order_quarter"] = ((df["order_month"] - 1) // 3 + 1).astype(int)
        df["order_weekofyear"] = df.get("order_weekofyear", 1)
        df["order_is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
        df["order_hour"] = df.get("order_hour", 0)

    # Campos para target encoding (no presentes en el form → categoría no vista)
    if "order_city" not in df.columns:
        df["order_city"] = "DESCONOCIDA"
    if "customer_id" not in df.columns:
        df["customer_id"] = -1
    if "route" not in df.columns and {"market", "order_region"}.issubset(df.columns):
        df["route"] = df["market"].astype(str) + "__" + df["order_region"].astype(str)

    # Variables económicas — usar `order_item_discount_rate` si está
    if "order_item_discount" in df.columns and "sales" in df.columns:
        ventas = df["sales"].replace(0, np.nan)
        df["order_item_discount_rate"] = (df["order_item_discount"] / ventas).fillna(0)

    # Aproximación ex-ante del profit ratio (proxy razonable)
    if "order_item_profit_ratio" not in df.columns:
        descuento = df.get("order_item_discount_rate", pd.Series([0.0]))
        df["order_item_profit_ratio"] = 1.0 - descuento

    # Pipelines reutilizados (solo los pasos ex-ante)
    df = variables_economicas(df)
    return df


def transformar_para_inferencia(
    form: dict[str, Any] | pd.DataFrame,
    metadata: MetadataInferencia,
) -> pd.DataFrame:
    """
    Convierte un input crudo (dict o DataFrame) en un X listo para el modelo.

    Pasos aplicados:
        1. Enriquecimiento con campos derivados.
        2. Target encoding con el encoder pre-entrenado (sin re-fit).
        3. One-hot encoding alineado a `metadata.feature_columns`.
        4. Reindex a `metadata.feature_columns` rellenando huecos con 0.

    Parameters
    ----------
    form : dict | pd.DataFrame
        Diccionario con los campos del formulario o DataFrame con una/N filas.
    metadata : MetadataInferencia

    Returns
    -------
    pd.DataFrame con shape `(n_filas, len(metadata.feature_columns))`.
    """
    if isinstance(form, dict):
        df = _enriquecer_input(form)
    else:
        df = form.copy()
        # Si recibimos un DataFrame, asumimos que ya viene enriquecido o
        # bien que es una sola fila por usuario. Aplicamos enriquecimiento
        # si faltan columnas temporales mínimas.
        if "order_month" not in df.columns and "order_date_dateorders" in df.columns:
            df = variables_temporales(df, "order_date_dateorders")
        df = variables_economicas(df)
        if "order_city" not in df.columns:
            df["order_city"] = "DESCONOCIDA"
        if "customer_id" not in df.columns:
            df["customer_id"] = -1
        if "route" not in df.columns and {"market", "order_region"}.issubset(df.columns):
            df["route"] = df["market"].astype(str) + "__" + df["order_region"].astype(str)

    # 2. Target encoding (sin entrenar de nuevo)
    df = metadata.encoder.transform(df)
    df = df.rename(
        columns={
            "order_city_risk_score": "city_risk_score",
            "customer_id_risk_score": "customer_delay_rate",
        }
    )

    # 3. One-hot encoding solo de las categóricas conocidas
    categoricas_presentes = [c for c in metadata.columnas_onehot if c in df.columns]
    if categoricas_presentes:
        df = codificar_categoricas(df, categoricas_presentes, metodo="onehot")

    # 4. Eliminar columnas no numéricas residuales (texto que sobró)
    for col in df.select_dtypes(exclude=[np.number, "bool"]).columns:
        df = df.drop(columns=col)

    # 5. Alinear con feature_columns esperadas por el modelo
    X = df.reindex(columns=metadata.feature_columns, fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# ============================================================================
# Helper para construir metadata desde el resultado del feature engineering
# ============================================================================
def construir_y_persistir_metadata(
    resultado_features: dict[str, Any],
    columnas_onehot: list[str] | None = None,
    nombre: str = NOMBRE_METADATA,
) -> Path:
    """
    Toma el dict que devuelve `construir_features_dataco` y persiste la
    metadata para inferencia.

    Parameters
    ----------
    resultado_features : dict
        Debe contener al menos las claves `encoder` y `features_finales`.
    columnas_onehot : list[str] | None
        Lista de columnas categóricas que se aplicaron one-hot encoding.
        Si es None, se usa la lista por defecto.
    """
    encoder = resultado_features.get("encoder")
    feature_columns = resultado_features.get("features_finales") or resultado_features.get(
        "X_train"
    )
    if encoder is None or feature_columns is None:
        raise ValueError(
            "resultado_features debe incluir 'encoder' y 'features_finales' (o 'X_train')"
        )

    # Si nos pasaron el DataFrame X_train, sacamos las columnas
    if isinstance(feature_columns, pd.DataFrame):
        feature_columns = feature_columns.columns.tolist()

    return guardar_metadata_inferencia(
        encoder=encoder,
        feature_columns=list(feature_columns),
        columnas_onehot=columnas_onehot,
        nombre=nombre,
    )
