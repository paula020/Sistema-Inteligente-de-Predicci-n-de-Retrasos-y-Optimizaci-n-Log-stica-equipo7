"""
Scoring de mercados potenciales.

Calcula un puntaje compuesto por ciudad/región basado en demanda,
rentabilidad, crecimiento y competencia, para priorizar inversión
comercial y expansión geográfica.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Pesos por defecto del score (suman 1.0)
# ----------------------------------------------------------------------------
PESOS_DEFAULT = {
    "demanda": 0.30,
    "rentabilidad": 0.30,
    "crecimiento": 0.25,
    "competencia": 0.15,   # mayor competencia => penalización
}


# ----------------------------------------------------------------------------
# Normalización
# ----------------------------------------------------------------------------
def _normalizar_minmax(serie: pd.Series) -> pd.Series:
    """Escala una serie al rango [0, 1]."""
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(0.5, index=serie.index)
    return (serie - serie.min()) / rango


# ----------------------------------------------------------------------------
# Cálculo de score
# ----------------------------------------------------------------------------
def calcular_score_mercado(
    df: pd.DataFrame,
    columna_ciudad: str = "ciudad_destino",
    columna_demanda: str = "guias",
    columna_rentabilidad: str = "margen_pct",
    columna_crecimiento: str = "crecimiento_pct",
    columna_competencia: str = "competidores",
    pesos: dict | None = None,
) -> pd.DataFrame:
    """
    Calcula un score compuesto de atractivo de mercado por ciudad.

    Cada componente se normaliza a [0,1]. La competencia se invierte
    para que más competencia reduzca el score.
    """
    pesos = pesos or PESOS_DEFAULT
    logger.info(f"Calculando score de mercado para {df[columna_ciudad].nunique()} ciudades")

    agg = (
        df.groupby(columna_ciudad)
        .agg(
            demanda=(columna_demanda, "sum"),
            rentabilidad=(columna_rentabilidad, "mean"),
            crecimiento=(columna_crecimiento, "mean")
            if columna_crecimiento in df.columns
            else (columna_demanda, "mean"),
            competencia=(columna_competencia, "mean")
            if columna_competencia in df.columns
            else (columna_demanda, "count"),
        )
        .reset_index()
    )

    agg["demanda_norm"] = _normalizar_minmax(agg["demanda"])
    agg["rentabilidad_norm"] = _normalizar_minmax(agg["rentabilidad"])
    agg["crecimiento_norm"] = _normalizar_minmax(agg["crecimiento"])
    agg["competencia_norm"] = 1 - _normalizar_minmax(agg["competencia"])

    agg["score_mercado"] = (
        pesos["demanda"] * agg["demanda_norm"]
        + pesos["rentabilidad"] * agg["rentabilidad_norm"]
        + pesos["crecimiento"] * agg["crecimiento_norm"]
        + pesos["competencia"] * agg["competencia_norm"]
    )

    agg["categoria"] = pd.cut(
        agg["score_mercado"],
        bins=[-np.inf, 0.4, 0.6, 0.8, np.inf],
        labels=["EVITAR", "OBSERVAR", "OPORTUNIDAD", "ESTRATEGICO"],
    )

    return agg.sort_values("score_mercado", ascending=False).reset_index(drop=True)


def top_mercados_potenciales(
    df_score: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Devuelve el top N de ciudades con mejor score."""
    return df_score.head(top_n).reset_index(drop=True)
