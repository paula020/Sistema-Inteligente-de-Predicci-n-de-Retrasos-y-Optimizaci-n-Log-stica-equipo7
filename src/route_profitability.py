"""
Análisis de rentabilidad de rutas logísticas.

Calcula costos directos, costos indirectos, margen y ROI por ruta
para identificar oportunidades de optimización y decisiones de pricing.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Estructuras de costos por defecto
# ----------------------------------------------------------------------------
COSTOS_DIRECTOS_DEFAULT = {
    "combustible_por_km": 1_200,
    "peajes_por_km": 180,
    "salario_conductor_por_dia": 180_000,
    "mantenimiento_por_km": 350,
}

COSTOS_INDIRECTOS_DEFAULT = {
    "administracion_pct": 0.08,        # 8% sobre el ingreso
    "tasa_reproceso": 0.05,            # 5% de las guías generan reproceso
    "costo_reproceso": 100_000,
    "tasa_devolucion": 0.03,           # 3% se devuelven
    "costo_devolucion": 250_000,
}


# ----------------------------------------------------------------------------
# Cálculos directos
# ----------------------------------------------------------------------------
def costos_directos(
    df: pd.DataFrame, costos: dict = COSTOS_DIRECTOS_DEFAULT
) -> pd.DataFrame:
    """Calcula los costos directos por guía/ruta."""
    df = df.copy()
    km = df.get("distancia_km", 0)
    dias = df.get("dias_transito", 1).clip(lower=1)

    df["costo_combustible"] = km * costos["combustible_por_km"]
    df["costo_peajes"] = km * costos["peajes_por_km"]
    df["costo_conductor"] = dias * costos["salario_conductor_por_dia"]
    df["costo_mantenimiento"] = km * costos["mantenimiento_por_km"]

    df["costo_directo_total"] = (
        df["costo_combustible"]
        + df["costo_peajes"]
        + df["costo_conductor"]
        + df["costo_mantenimiento"]
    )
    return df


# ----------------------------------------------------------------------------
# Cálculos indirectos
# ----------------------------------------------------------------------------
def costos_indirectos(
    df: pd.DataFrame, costos: dict = COSTOS_INDIRECTOS_DEFAULT
) -> pd.DataFrame:
    """Calcula los costos indirectos por guía/ruta."""
    df = df.copy()
    valor_flete = df.get("valor_flete", 0)

    df["costo_administracion"] = valor_flete * costos["administracion_pct"]
    df["costo_reproceso_esperado"] = (
        costos["tasa_reproceso"] * costos["costo_reproceso"]
    )
    df["costo_devolucion_esperado"] = (
        costos["tasa_devolucion"] * costos["costo_devolucion"]
    )

    df["costo_indirecto_total"] = (
        df["costo_administracion"]
        + df["costo_reproceso_esperado"]
        + df["costo_devolucion_esperado"]
    )
    return df


# ----------------------------------------------------------------------------
# Rentabilidad
# ----------------------------------------------------------------------------
def calcular_rentabilidad(
    df: pd.DataFrame,
    costos_dir: dict = COSTOS_DIRECTOS_DEFAULT,
    costos_ind: dict = COSTOS_INDIRECTOS_DEFAULT,
) -> pd.DataFrame:
    """
    Pipeline completo de rentabilidad: ingresos, costos, margen, ROI.

    Requiere columnas mínimas: `valor_flete`, `distancia_km`, `dias_transito`.
    """
    logger.info("Calculando rentabilidad por guía")
    df = costos_directos(df, costos_dir)
    df = costos_indirectos(df, costos_ind)

    df["costo_total"] = df["costo_directo_total"] + df["costo_indirecto_total"]
    df["margen_absoluto"] = df["valor_flete"] - df["costo_total"]
    df["margen_pct"] = np.where(
        df["valor_flete"] > 0, df["margen_absoluto"] / df["valor_flete"], np.nan
    )
    df["roi"] = np.where(
        df["costo_total"] > 0, df["margen_absoluto"] / df["costo_total"], np.nan
    )
    return df


# ----------------------------------------------------------------------------
# Reportes
# ----------------------------------------------------------------------------
def ranking_rutas(
    df: pd.DataFrame, columna_ruta: str = "ruta", top_n: int = 20
) -> pd.DataFrame:
    """Top N de rutas por rentabilidad (margen absoluto)."""
    if columna_ruta not in df.columns:
        raise ValueError(f"Columna '{columna_ruta}' no encontrada")

    agg = (
        df.groupby(columna_ruta)
        .agg(
            guias=("valor_flete", "count"),
            ingreso_total=("valor_flete", "sum"),
            costo_total=("costo_total", "sum"),
            margen_total=("margen_absoluto", "sum"),
            margen_promedio_pct=("margen_pct", "mean"),
            roi_promedio=("roi", "mean"),
        )
        .sort_values("margen_total", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return agg


def rutas_no_rentables(df: pd.DataFrame, columna_ruta: str = "ruta") -> pd.DataFrame:
    """Rutas con margen negativo: candidatas a revisión o suspensión."""
    ranking = ranking_rutas(df, columna_ruta, top_n=10_000)
    no_rentables = ranking[ranking["margen_total"] < 0].copy()
    logger.warning(f"Rutas no rentables identificadas: {len(no_rentables)}")
    return no_rentables
