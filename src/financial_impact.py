"""
Análisis de impacto financiero de los retrasos logísticos.

Cuantifica el costo monetario asociado a retrasos: penalizaciones,
reprocesos, devoluciones, pérdida de cliente y costo operativo extra.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Parámetros financieros por defecto (ajustables por la empresa)
# ----------------------------------------------------------------------------
COSTOS_DEFAULT = {
    "penalizacion_por_dia": 50_000,        # COP por día de retraso
    "costo_reproceso": 120_000,            # COP por evento
    "tasa_devolucion": 0.08,               # 8% de las guías retrasadas se devuelven
    "costo_devolucion": 250_000,           # COP por devolución
    "lucro_cesante_pct": 0.05,             # 5% del valor del flete por mala experiencia
}


def calcular_costo_retraso(
    fila: pd.Series, costos: dict = COSTOS_DEFAULT
) -> float:
    """Calcula el costo financiero esperado de un retraso para una guía."""
    dias = max(int(fila.get("dias_retraso", 0)), 0)
    valor_flete = float(fila.get("valor_flete", 0.0))

    costo_penalizacion = dias * costos["penalizacion_por_dia"]
    costo_reproceso = costos["costo_reproceso"] if dias > 0 else 0.0
    costo_devolucion = (
        costos["tasa_devolucion"] * costos["costo_devolucion"] if dias > 0 else 0.0
    )
    lucro_cesante = costos["lucro_cesante_pct"] * valor_flete if dias > 0 else 0.0

    return float(
        costo_penalizacion + costo_reproceso + costo_devolucion + lucro_cesante
    )


def calcular_impacto_dataframe(
    df: pd.DataFrame, costos: dict = COSTOS_DEFAULT
) -> pd.DataFrame:
    """Aplica el cálculo de impacto financiero a un DataFrame completo."""
    df = df.copy()
    df["costo_retraso_estimado"] = df.apply(
        lambda fila: calcular_costo_retraso(fila, costos), axis=1
    )
    logger.info(
        f"Impacto financiero calculado | Total estimado: "
        f"${df['costo_retraso_estimado'].sum():,.0f}"
    )
    return df


def costo_esperado_por_probabilidad(
    probabilidad_retraso: float,
    valor_flete: float,
    dias_estimados: int = 2,
    costos: dict = COSTOS_DEFAULT,
) -> float:
    """
    Costo esperado dado el riesgo del modelo: P(retraso) * costo si retrasa.

    Útil para priorizar guías de alto riesgo en tiempo real desde la API.
    """
    fila = pd.Series(
        {"dias_retraso": dias_estimados, "valor_flete": valor_flete}
    )
    costo_si_retrasa = calcular_costo_retraso(fila, costos)
    return float(probabilidad_retraso * costo_si_retrasa)


def resumen_impacto_por_dimension(
    df: pd.DataFrame, dimension: str
) -> pd.DataFrame:
    """Agrupa el impacto financiero por una dimensión (ruta, cliente, ciudad)."""
    if dimension not in df.columns:
        raise ValueError(f"La columna '{dimension}' no existe en el dataframe")

    agg = (
        df.groupby(dimension)
        .agg(
            guias=("costo_retraso_estimado", "count"),
            costo_total=("costo_retraso_estimado", "sum"),
            costo_promedio=("costo_retraso_estimado", "mean"),
            dias_retraso_promedio=("dias_retraso", "mean"),
        )
        .sort_values("costo_total", ascending=False)
        .reset_index()
    )
    return agg
