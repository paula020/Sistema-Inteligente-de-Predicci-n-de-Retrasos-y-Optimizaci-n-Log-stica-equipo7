"""
Preprocesamiento de datos logísticos.

Pipeline de limpieza, normalización y validación de los datasets crudos
antes de la ingeniería de variables y el entrenamiento.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Limpieza básica
# ----------------------------------------------------------------------------
def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte nombres de columnas a snake_case sin espacios ni acentos."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[áä]", "a", regex=True)
        .str.replace(r"[éë]", "e", regex=True)
        .str.replace(r"[íï]", "i", regex=True)
        .str.replace(r"[óö]", "o", regex=True)
        .str.replace(r"[úü]", "u", regex=True)
        .str.replace(r"[ñ]", "n", regex=True)
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def eliminar_duplicados(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Elimina filas duplicadas e informa cuántas se removieron."""
    antes = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    logger.info(f"Duplicados eliminados: {antes - len(df):,}")
    return df


def imputar_nulos(
    df: pd.DataFrame,
    estrategia_numerica: str = "median",
    estrategia_categorica: str = "moda",
) -> pd.DataFrame:
    """Imputa valores nulos según el tipo de dato de cada columna."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            valor = (
                df[col].median()
                if estrategia_numerica == "median"
                else df[col].mean()
            )
        else:
            valor = (
                df[col].mode().iloc[0]
                if estrategia_categorica == "moda" and not df[col].mode().empty
                else "desconocido"
            )
        df[col] = df[col].fillna(valor)
    return df


# ----------------------------------------------------------------------------
# Manejo de outliers
# ----------------------------------------------------------------------------
def filtrar_outliers_iqr(
    df: pd.DataFrame, columnas: list[str], factor: float = 1.5
) -> pd.DataFrame:
    """Filtra outliers mediante el método del rango intercuartílico (IQR)."""
    df = df.copy()
    for col in columnas:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf = q1 - factor * iqr
        lim_sup = q3 + factor * iqr
        antes = len(df)
        df = df[(df[col] >= lim_inf) & (df[col] <= lim_sup)]
        logger.info(f"Outliers removidos en '{col}': {antes - len(df):,}")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Validaciones
# ----------------------------------------------------------------------------
def validar_columnas_requeridas(df: pd.DataFrame, requeridas: list[str]) -> None:
    """Verifica que el dataframe contenga las columnas esperadas."""
    faltantes = set(requeridas) - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {sorted(faltantes)}")


# ----------------------------------------------------------------------------
# Pipeline orquestador
# ----------------------------------------------------------------------------
def preprocesar_dataframe(
    df: pd.DataFrame,
    columnas_outliers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pipeline estándar de preprocesamiento.

    Aplica en orden:
        1. Normalización de nombres de columnas
        2. Eliminación de duplicados
        3. Imputación de nulos
        4. (Opcional) Filtrado de outliers por IQR
    """
    logger.info(f"Iniciando preprocesamiento de {len(df):,} filas")
    df = normalizar_nombres_columnas(df)
    df = eliminar_duplicados(df)
    df = imputar_nulos(df)
    if columnas_outliers:
        df = filtrar_outliers_iqr(df, columnas_outliers)
    logger.success(f"Preprocesamiento finalizado: {len(df):,} filas resultantes")
    return df
