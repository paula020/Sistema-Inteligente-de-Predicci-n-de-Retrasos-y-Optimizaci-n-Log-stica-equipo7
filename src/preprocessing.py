"""
Módulo de carga y limpieza de datos para SmartDelay AI 360.

Especializado en el dataset *DataCo Smart Supply Chain* (Kaggle):
    https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Características que hacen este dataset particular:
    - Encoding `latin-1` (NO utf-8) — falla la lectura con utf-8 estándar.
    - Columnas con espacios, mayúsculas y caracteres especiales.
    - Dos columnas de fecha tipo string:
        * `order date (DateOrders)`
        * `shipping date (DateOrders)`
    - Variable objetivo del módulo de retrasos: `Late_delivery_risk` (0/1).

Además, las funciones genéricas (`normalizar_nombres_columnas`,
`imputar_nulos`, `filtrar_outliers_iqr`, etc.) se conservan como
utilidades reutilizables para otros datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings


# ============================================================================
# Constantes específicas del dataset DataCo
# ============================================================================

# Nombre estándar del archivo descargado de Kaggle
ARCHIVO_DATACO_DEFAULT = "DataCoSupplyChainDataset.csv"

# DataCo viene con encoding latin-1; intentamos varios por robustez
ENCODINGS_INTENTAR = ("latin-1", "ISO-8859-1", "cp1252", "utf-8")

# Columnas de fecha del dataset
COLUMNAS_FECHA_DATACO = ("order date (DateOrders)", "shipping date (DateOrders)")

# Columna objetivo del modelo de predicción de retrasos
TARGET_RETRASO = "Late_delivery_risk"

# Columnas con información sensible que se eliminan por privacidad/GDPR
COLUMNAS_SENSIBLES = (
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
    "Product Image",
    "Product Description",
)

# Columnas mínimas que deben existir tras la carga para considerar válido el dataset
COLUMNAS_REQUERIDAS_DATACO = (
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Quantity",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
)


# ============================================================================
# 1. LECTURA DEL CSV
# ============================================================================
def cargar_dataco(
    nombre_archivo: str = ARCHIVO_DATACO_DEFAULT,
    capa: str = "raw",
) -> pd.DataFrame:
    """
    Carga el CSV DataCo desde la capa de datos indicada.

    Prueba en orden varios encodings comunes para evitar fallos
    típicos al cargar el dataset en Windows (cp1252) o Linux (utf-8).

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo CSV en la capa correspondiente.
    capa : str
        Capa de datos: 'raw', 'processed' o 'external'.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos originales (sin transformar).
    """
    mapa_capas = {
        "raw": settings.data_raw_path,
        "processed": settings.data_processed_path,
        "external": settings.data_external_path,
    }
    if capa not in mapa_capas:
        raise ValueError(
            f"Capa inválida '{capa}'. Use: {list(mapa_capas.keys())}"
        )

    ruta = settings.resolve_path(mapa_capas[capa]) / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}\n"
            f"Descarga el dataset desde Kaggle y colócalo en data/raw/."
        )

    logger.info(f"Cargando dataset DataCo desde: {ruta}")

    ultimo_error: Exception | None = None
    for encoding in ENCODINGS_INTENTAR:
        try:
            df = pd.read_csv(ruta, encoding=encoding, low_memory=False)
            logger.success(
                f"Dataset cargado con encoding '{encoding}' | "
                f"shape={df.shape}"
            )
            return df
        except UnicodeDecodeError as exc:
            logger.warning(f"Encoding '{encoding}' falló: {exc}")
            ultimo_error = exc

    raise UnicodeDecodeError(  # type: ignore[misc]
        "latin-1", b"", 0, 1,
        f"No se pudo leer el archivo con ningún encoding probado. "
        f"Último error: {ultimo_error}"
    )


# ============================================================================
# 2. NORMALIZACIÓN DE NOMBRES DE COLUMNAS
# ============================================================================
def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte nombres de columnas a snake_case sin espacios, paréntesis ni
    caracteres especiales.

    Ejemplo:
        'Order Item Profit Ratio'      -> 'order_item_profit_ratio'
        'order date (DateOrders)'      -> 'order_date_dateorders'
        'Late_delivery_risk'           -> 'late_delivery_risk'
    """
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
    logger.info(f"Columnas normalizadas ({len(df.columns)} columnas)")
    return df


# ============================================================================
# 3. DETECCIÓN DE NULOS
# ============================================================================
def reporte_nulos(df: pd.DataFrame, umbral_pct: float = 0.0) -> pd.DataFrame:
    """
    Genera un reporte de nulos por columna.

    Parameters
    ----------
    df : pd.DataFrame
    umbral_pct : float
        Solo retorna columnas con porcentaje de nulos >= umbral_pct (0-100).

    Returns
    -------
    pd.DataFrame con columnas: ['columna', 'nulos', 'pct_nulos', 'dtype'].
    """
    total = len(df)
    nulos = df.isna().sum()
    reporte = pd.DataFrame({
        "columna": nulos.index,
        "nulos": nulos.values,
        "pct_nulos": (nulos.values / total * 100).round(2),
        "dtype": [str(df[c].dtype) for c in nulos.index],
    })
    reporte = reporte[reporte["pct_nulos"] >= umbral_pct]
    reporte = reporte.sort_values("pct_nulos", ascending=False).reset_index(drop=True)

    if not reporte.empty:
        logger.info(
            f"Reporte de nulos: {len(reporte)} columnas con nulos "
            f"(>= {umbral_pct}%)"
        )
    else:
        logger.success("No se detectaron columnas con nulos significativos")
    return reporte


def imputar_nulos(
    df: pd.DataFrame,
    estrategia_numerica: str = "median",
    estrategia_categorica: str = "moda",
) -> pd.DataFrame:
    """
    Imputa valores nulos según el tipo de dato.

    - Numéricas: mediana (por defecto) o media.
    - Categóricas: moda (por defecto) o literal 'desconocido'.
    """
    df = df.copy()
    columnas_imputadas = []
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
        columnas_imputadas.append(col)

    if columnas_imputadas:
        logger.info(f"Columnas imputadas: {len(columnas_imputadas)}")
    return df


# ============================================================================
# 4. CONVERSIÓN DE FECHAS
# ============================================================================
def convertir_fechas(
    df: pd.DataFrame,
    columnas: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Convierte columnas de fecha de texto a `datetime64[ns]`.

    Si `columnas` es None, intenta convertir las dos columnas de fecha
    típicas de DataCo, ya normalizadas (`order_date_dateorders` y
    `shipping_date_dateorders`).
    """
    df = df.copy()
    if columnas is None:
        columnas = ["order_date_dateorders", "shipping_date_dateorders"]

    for col in columnas:
        if col not in df.columns:
            logger.warning(f"Columna de fecha '{col}' no encontrada, se omite")
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
        invalidas = df[col].isna().sum()
        if invalidas:
            logger.warning(
                f"Columna '{col}': {invalidas:,} fechas inválidas convertidas a NaT"
            )
        else:
            logger.info(f"Columna '{col}' convertida a datetime correctamente")
    return df


# ============================================================================
# 5. ELIMINACIÓN DE DUPLICADOS
# ============================================================================
def eliminar_duplicados(
    df: pd.DataFrame, subset: list[str] | None = None
) -> pd.DataFrame:
    """Elimina filas duplicadas y registra cuántas se removieron."""
    antes = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removidos = antes - len(df)
    if removidos > 0:
        logger.info(f"Duplicados eliminados: {removidos:,}")
    else:
        logger.info("No se detectaron duplicados")
    return df


# ============================================================================
# 6. ELIMINACIÓN DE COLUMNAS SENSIBLES / VACÍAS
# ============================================================================
def eliminar_columnas_sensibles(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con información personal/sensible o irrelevante."""
    df = df.copy()
    a_eliminar = []
    for col_original in COLUMNAS_SENSIBLES:
        # Comparar contra nombres normalizados
        col_normalizada = (
            col_original.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        if col_normalizada in df.columns:
            a_eliminar.append(col_normalizada)

    if a_eliminar:
        df = df.drop(columns=a_eliminar)
        logger.info(f"Columnas sensibles eliminadas: {a_eliminar}")
    return df


def eliminar_columnas_totalmente_nulas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas que están 100% vacías (común con `Product Description`)."""
    columnas_vacias = [c for c in df.columns if df[c].isna().all()]
    if columnas_vacias:
        df = df.drop(columns=columnas_vacias)
        logger.warning(f"Columnas 100% nulas eliminadas: {columnas_vacias}")
    return df


# ============================================================================
# 7. VALIDACIONES
# ============================================================================
def validar_columnas_requeridas(
    df: pd.DataFrame, requeridas: list[str] | tuple[str, ...]
) -> None:
    """Verifica que el dataframe contenga las columnas esperadas."""
    faltantes = set(requeridas) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Columnas requeridas faltantes: {sorted(faltantes)}"
        )
    logger.success(f"Validación OK: {len(requeridas)} columnas requeridas presentes")


def validar_target(df: pd.DataFrame, target: str = "late_delivery_risk") -> None:
    """Valida que la variable objetivo exista y sea binaria."""
    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no está en el dataset")

    valores_unicos = set(df[target].dropna().unique())
    if not valores_unicos.issubset({0, 1}):
        raise ValueError(
            f"La variable '{target}' debe ser binaria (0/1). "
            f"Valores encontrados: {valores_unicos}"
        )
    tasa = df[target].mean()
    logger.success(
        f"Variable objetivo '{target}' válida | tasa de positivos: {tasa:.2%}"
    )


# ============================================================================
# 8. PERSISTENCIA
# ============================================================================
def guardar_dataset_procesado(
    df: pd.DataFrame,
    nombre: str = "dataco_clean.parquet",
) -> Path:
    """Guarda el dataframe limpio en `data/processed/`."""
    base = settings.resolve_path(settings.data_processed_path)
    base.mkdir(parents=True, exist_ok=True)
    ruta = base / nombre

    if ruta.suffix == ".parquet":
        df.to_parquet(ruta, index=False)
    elif ruta.suffix == ".csv":
        df.to_csv(ruta, index=False)
    else:
        raise ValueError(f"Formato no soportado: {ruta.suffix}")

    tamano_mb = ruta.stat().st_size / 1024 / 1024
    logger.success(
        f"Dataset limpio guardado: {ruta} "
        f"({len(df):,} filas, {len(df.columns)} columnas, {tamano_mb:.2f} MB)"
    )
    return ruta


# ============================================================================
# 9. PIPELINE PRINCIPAL — DATACO
# ============================================================================
def limpiar_dataco(
    nombre_archivo: str = ARCHIVO_DATACO_DEFAULT,
    guardar: bool = True,
    nombre_salida: str = "dataco_clean.parquet",
    imputar: bool = False,
) -> pd.DataFrame:
    """
    Pipeline orquestador de limpieza para el dataset DataCo.

    Pasos:
        1. Cargar CSV (latin-1 con fallback de encodings)
        2. Normalizar nombres de columnas
        3. Eliminar columnas 100% nulas
        4. Eliminar columnas sensibles (privacidad)
        5. Convertir columnas de fecha a datetime
        6. Eliminar duplicados
        7. (Opcional) Imputar nulos
        8. Validar estructura y variable objetivo
        9. (Opcional) Guardar en `data/processed/`

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo crudo (en `data/raw/`).
    guardar : bool
        Si True, persiste el resultado en `data/processed/`.
    nombre_salida : str
        Nombre del archivo de salida.
    imputar : bool
        Si True, imputa nulos automáticamente al final del pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio listo para feature engineering.
    """
    logger.info("=" * 60)
    logger.info("INICIO DEL PIPELINE DE LIMPIEZA — DATACO")
    logger.info("=" * 60)

    # 1. Carga
    df = cargar_dataco(nombre_archivo)
    shape_inicial = df.shape

    # 2. Normalización de columnas
    df = normalizar_nombres_columnas(df)

    # 3. Eliminar columnas 100% nulas
    df = eliminar_columnas_totalmente_nulas(df)

    # 4. Eliminar columnas sensibles
    df = eliminar_columnas_sensibles(df)

    # 5. Conversión de fechas
    df = convertir_fechas(df)

    # 6. Eliminar duplicados
    df = eliminar_duplicados(df)

    # 7. Imputación opcional
    if imputar:
        df = imputar_nulos(df)

    # 8. Validaciones
    columnas_requeridas_norm = [
        c.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        for c in COLUMNAS_REQUERIDAS_DATACO
    ]
    validar_columnas_requeridas(df, columnas_requeridas_norm)
    validar_target(df, "late_delivery_risk")

    # 9. Persistencia
    if guardar:
        guardar_dataset_procesado(df, nombre_salida)

    logger.info("=" * 60)
    logger.success(
        f"PIPELINE COMPLETADO | "
        f"{shape_inicial} -> {df.shape}"
    )
    logger.info("=" * 60)
    return df


# ============================================================================
# UTILIDADES GENÉRICAS (reutilizables en otros datasets)
# ============================================================================
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


def preprocesar_dataframe(
    df: pd.DataFrame,
    columnas_outliers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pipeline genérico (no específico de DataCo) para preprocesar
    cualquier dataframe tabular.
    """
    logger.info(f"Iniciando preprocesamiento genérico de {len(df):,} filas")
    df = normalizar_nombres_columnas(df)
    df = eliminar_duplicados(df)
    df = imputar_nulos(df)
    if columnas_outliers:
        df = filtrar_outliers_iqr(df, columnas_outliers)
    logger.success(f"Preprocesamiento finalizado: {len(df):,} filas resultantes")
    return df
