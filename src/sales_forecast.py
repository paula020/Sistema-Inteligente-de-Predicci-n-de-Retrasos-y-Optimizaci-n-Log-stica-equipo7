"""
Forecast de ventas y demanda logística.

Proyecta volúmenes futuros de guías o ingresos por ciudad/cliente/ruta
usando Prophet (Facebook) como modelo base y XGBoost como alternativa
para datasets con muchas features exógenas.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_DISPONIBLE = True
except ImportError:  # pragma: no cover
    PROPHET_DISPONIBLE = False
    logger.warning("Prophet no disponible. Solo se podrá usar la baseline.")


# ----------------------------------------------------------------------------
# Preparación de series temporales
# ----------------------------------------------------------------------------
def preparar_serie_temporal(
    df: pd.DataFrame,
    columna_fecha: str = "fecha_despacho",
    columna_valor: str = "valor_flete",
    frecuencia: str = "D",
) -> pd.DataFrame:
    """
    Construye una serie temporal agregada lista para forecasting.

    Devuelve un DataFrame con columnas `ds` (fecha) y `y` (valor)
    en el formato esperado por Prophet.
    """
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    df = df.dropna(subset=[columna_fecha])

    serie = (
        df.set_index(columna_fecha)[columna_valor]
        .resample(frecuencia)
        .sum()
        .reset_index()
        .rename(columns={columna_fecha: "ds", columna_valor: "y"})
    )
    logger.info(f"Serie temporal creada: {len(serie):,} puntos ({frecuencia})")
    return serie


# ----------------------------------------------------------------------------
# Forecast con Prophet
# ----------------------------------------------------------------------------
def forecast_prophet(
    serie: pd.DataFrame,
    horizonte_dias: int = 90,
    estacionalidad_semanal: bool = True,
    estacionalidad_anual: bool = True,
) -> pd.DataFrame:
    """
    Genera un forecast con Prophet a `horizonte_dias` días.

    Returns
    -------
    DataFrame con columnas: ds, yhat, yhat_lower, yhat_upper.
    """
    if not PROPHET_DISPONIBLE:
        raise ImportError("Prophet no está instalado. Use forecast_baseline().")

    logger.info(f"Entrenando Prophet | horizonte: {horizonte_dias} días")
    modelo = Prophet(
        weekly_seasonality=estacionalidad_semanal,
        yearly_seasonality=estacionalidad_anual,
        daily_seasonality=False,
        interval_width=0.95,
    )
    modelo.fit(serie)

    futuro = modelo.make_future_dataframe(periods=horizonte_dias)
    forecast = modelo.predict(futuro)

    columnas = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    return forecast[columnas].tail(horizonte_dias).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Forecast baseline (media móvil)
# ----------------------------------------------------------------------------
def forecast_baseline(
    serie: pd.DataFrame, horizonte_dias: int = 30, ventana: int = 14
) -> pd.DataFrame:
    """
    Forecast simple basado en media móvil de los últimos `ventana` días.

    Útil como benchmark frente a Prophet o XGBoost.
    """
    media = serie["y"].tail(ventana).mean()
    desv = serie["y"].tail(ventana).std()
    ultima_fecha = pd.to_datetime(serie["ds"].max())

    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=horizonte_dias,
        freq="D",
    )

    return pd.DataFrame({
        "ds": fechas_futuras,
        "yhat": media,
        "yhat_lower": media - 1.96 * desv,
        "yhat_upper": media + 1.96 * desv,
    })


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def evaluar_forecast(real: pd.Series, predicho: pd.Series) -> dict:
    """Calcula MAE, RMSE y MAPE para evaluar la calidad del forecast."""
    real = real.to_numpy()
    predicho = predicho.to_numpy()

    mae = float(np.mean(np.abs(real - predicho)))
    rmse = float(np.sqrt(np.mean((real - predicho) ** 2)))
    mape = float(np.mean(np.abs((real - predicho) / np.where(real == 0, 1, real))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}
