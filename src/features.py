"""
Ingeniería de variables (feature engineering) para SmartDelay AI 360.

Crea variables derivadas a partir de los datos limpios:
fechas, distancias, costos, indicadores operacionales y de cliente.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Variables temporales
# ----------------------------------------------------------------------------
def variables_de_fecha(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    """Genera variables de calendario a partir de una columna de fecha."""
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")

    df[f"{columna_fecha}_anio"] = df[columna_fecha].dt.year
    df[f"{columna_fecha}_mes"] = df[columna_fecha].dt.month
    df[f"{columna_fecha}_dia"] = df[columna_fecha].dt.day
    df[f"{columna_fecha}_dia_semana"] = df[columna_fecha].dt.dayofweek
    df[f"{columna_fecha}_es_fin_semana"] = (
        df[columna_fecha].dt.dayofweek >= 5
    ).astype(int)
    df[f"{columna_fecha}_trimestre"] = df[columna_fecha].dt.quarter
    return df


# ----------------------------------------------------------------------------
# Variables operacionales
# ----------------------------------------------------------------------------
def calcular_dias_transito(
    df: pd.DataFrame, col_origen: str, col_entrega: str
) -> pd.DataFrame:
    """Calcula los días entre fecha de despacho y fecha de entrega."""
    df = df.copy()
    df[col_origen] = pd.to_datetime(df[col_origen], errors="coerce")
    df[col_entrega] = pd.to_datetime(df[col_entrega], errors="coerce")
    df["dias_transito"] = (df[col_entrega] - df[col_origen]).dt.days
    return df


def variable_retraso(
    df: pd.DataFrame,
    col_promesa: str = "fecha_promesa",
    col_entrega: str = "fecha_entrega",
    umbral_dias: int = 0,
) -> pd.DataFrame:
    """
    Define la variable objetivo `retraso` (1 si llegó tarde, 0 si no).

    Un envío se considera retrasado si la diferencia entre la fecha real
    de entrega y la promesa supera `umbral_dias`.
    """
    df = df.copy()
    df[col_promesa] = pd.to_datetime(df[col_promesa], errors="coerce")
    df[col_entrega] = pd.to_datetime(df[col_entrega], errors="coerce")
    diferencia = (df[col_entrega] - df[col_promesa]).dt.days
    df["dias_retraso"] = diferencia
    df["retraso"] = (diferencia > umbral_dias).astype(int)
    return df


# ----------------------------------------------------------------------------
# Variables económicas
# ----------------------------------------------------------------------------
def costo_por_kilometro(
    df: pd.DataFrame, col_costo: str = "costo_total", col_km: str = "distancia_km"
) -> pd.DataFrame:
    """Calcula el costo por kilómetro como indicador de eficiencia."""
    df = df.copy()
    df["costo_por_km"] = np.where(
        df[col_km] > 0, df[col_costo] / df[col_km], np.nan
    )
    return df


def ingreso_por_kilo(
    df: pd.DataFrame, col_ingreso: str = "valor_flete", col_peso: str = "peso_kg"
) -> pd.DataFrame:
    """Calcula el ingreso por kilo transportado."""
    df = df.copy()
    df["ingreso_por_kilo"] = np.where(
        df[col_peso] > 0, df[col_ingreso] / df[col_peso], np.nan
    )
    return df


# ----------------------------------------------------------------------------
# Codificación de categóricas
# ----------------------------------------------------------------------------
def codificar_categoricas(
    df: pd.DataFrame, columnas: list[str], metodo: str = "onehot"
) -> pd.DataFrame:
    """Codifica variables categóricas a numéricas."""
    df = df.copy()
    if metodo == "onehot":
        return pd.get_dummies(df, columns=columnas, drop_first=True)
    if metodo == "frecuencia":
        for col in columnas:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq)
        return df
    raise ValueError(f"Método de codificación no soportado: {metodo}")


# ----------------------------------------------------------------------------
# Pipeline orquestador
# ----------------------------------------------------------------------------
def construir_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Pipeline de generación de features.

    `config` permite controlar columnas de fecha, columnas categóricas
    y columnas usadas para cálculos económicos.
    """
    config = config or {}
    logger.info("Iniciando ingeniería de variables")

    for col in config.get("columnas_fecha", []):
        if col in df.columns:
            df = variables_de_fecha(df, col)

    if "fecha_despacho" in df.columns and "fecha_entrega" in df.columns:
        df = calcular_dias_transito(df, "fecha_despacho", "fecha_entrega")

    if "fecha_promesa" in df.columns and "fecha_entrega" in df.columns:
        df = variable_retraso(df)

    if {"costo_total", "distancia_km"}.issubset(df.columns):
        df = costo_por_kilometro(df)

    if {"valor_flete", "peso_kg"}.issubset(df.columns):
        df = ingreso_por_kilo(df)

    categoricas = config.get("categoricas", [])
    if categoricas:
        df = codificar_categoricas(df, categoricas)

    logger.success(f"Features generadas. Shape final: {df.shape}")
    return df
