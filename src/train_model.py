"""
Módulo de entrenamiento de modelos ML para SmartDelay AI 360.

Entrena, evalúa y compara cuatro clasificadores para predecir
`late_delivery_risk`:

    1. Logistic Regression  (con StandardScaler dentro de un Pipeline)
    2. Random Forest        (sklearn)
    3. XGBoost              (xgboost.XGBClassifier)
    4. LightGBM             (lightgbm.LGBMClassifier)

Cada entrenamiento registra en MLflow:
    - parámetros e hiperparámetros
    - métricas: accuracy, precision, recall, F1, ROC-AUC
    - matriz de confusión (artefacto PNG)
    - feature importance (artefacto PNG + CSV)
    - el modelo serializado

Al final, el orquestador `entrenar_todos_los_modelos` ejecuta los 4
clasificadores como *nested runs* de un experimento padre y devuelve
un diccionario con los resultados, listo para que
`seleccionar_mejor_modelo` elija el ganador según la métrica deseada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend no interactivo para guardar PNGs sin display
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.utils import guardar_modelo, guardar_reporte_json


# ============================================================================
# Configuraciones por defecto de cada modelo
# ============================================================================
PARAMS_LOGISTIC = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

PARAMS_RANDOM_FOREST = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

PARAMS_XGBOOST = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": 42,
}

PARAMS_LIGHTGBM = {
    "n_estimators": 400,
    "max_depth": -1,
    "num_leaves": 64,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


# ============================================================================
# Estructura de resultados
# ============================================================================
@dataclass
class ResultadoEntrenamiento:
    """Encapsula todo el output de entrenar un modelo."""

    nombre: str
    modelo: Any
    metricas: dict[str, float]
    matriz_confusion: np.ndarray
    feature_importance: pd.Series | None
    ruta_modelo: Path
    ruta_cm: Path | None = None
    ruta_fi: Path | None = None
    run_id: str | None = None
    hiperparametros: dict = field(default_factory=dict)


# ============================================================================
# Funciones auxiliares (métricas, visualizaciones, MLflow)
# ============================================================================
def _calcular_metricas(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula el set estándar de métricas para clasificación binaria."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _guardar_matriz_confusion(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
) -> tuple[np.ndarray, Path]:
    """Genera y guarda la matriz de confusión como imagen PNG."""
    cm = confusion_matrix(y_true, y_pred)
    base = settings.resolve_path(settings.reports_path)
    base.mkdir(parents=True, exist_ok=True)
    ruta = base / f"confusion_matrix_{nombre_modelo}.png"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["A tiempo (0)", "Retraso (1)"],
        yticklabels=["A tiempo (0)", "Retraso (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Matriz de confusión guardada: {ruta}")
    return cm, ruta


def _extraer_feature_importance(
    modelo: Any, columnas: list[str], nombre_modelo: str
) -> tuple[pd.Series | None, Path | None]:
    """
    Extrae y persiste la importancia de variables del modelo.

    - Para árboles: `feature_importances_`
    - Para Logistic Regression (dentro de Pipeline): coeficientes absolutos
    """
    importancia: pd.Series | None = None

    if hasattr(modelo, "feature_importances_"):
        importancia = pd.Series(modelo.feature_importances_, index=columnas)
    elif isinstance(modelo, Pipeline) and hasattr(modelo.named_steps.get("clf"), "coef_"):
        coef = modelo.named_steps["clf"].coef_[0]
        importancia = pd.Series(np.abs(coef), index=columnas)
    elif hasattr(modelo, "coef_"):
        coef = modelo.coef_[0]
        importancia = pd.Series(np.abs(coef), index=columnas)

    if importancia is None:
        logger.warning(f"No se pudo extraer feature importance para {nombre_modelo}")
        return None, None

    importancia = importancia.sort_values(ascending=False)
    base = settings.resolve_path(settings.reports_path)
    base.mkdir(parents=True, exist_ok=True)

    # Guardar CSV completo
    ruta_csv = base / f"feature_importance_{nombre_modelo}.csv"
    importancia.to_csv(ruta_csv, header=["importancia"])

    # Guardar gráfico top 20
    ruta_png = base / f"feature_importance_{nombre_modelo}.png"
    top = importancia.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="#4c8acc")
    ax.set_title(f"Top 20 features — {nombre_modelo}")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Feature importance guardada: {ruta_csv} / {ruta_png}")
    return importancia, ruta_png


def _configurar_mlflow() -> None:
    """Apunta MLflow al tracking URI y al experimento configurados."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _loggear_resultados_mlflow(
    nombre_modelo: str,
    modelo: Any,
    params: dict,
    metricas: dict[str, float],
    ruta_cm: Path | None,
    ruta_fi: Path | None,
    flavor: str = "sklearn",
) -> str:
    """Registra parámetros, métricas, artefactos y el modelo en MLflow."""
    mlflow.log_params(params)
    mlflow.log_metrics(metricas)

    if ruta_cm and ruta_cm.exists():
        mlflow.log_artifact(str(ruta_cm), artifact_path="plots")
    if ruta_fi and ruta_fi.exists():
        mlflow.log_artifact(str(ruta_fi), artifact_path="plots")

    try:
        if flavor == "xgboost":
            import mlflow.xgboost as mlflow_xgboost

            mlflow_xgboost.log_model(modelo, artifact_path="model")
        elif flavor == "lightgbm":
            import mlflow.lightgbm as mlflow_lightgbm

            mlflow_lightgbm.log_model(modelo, artifact_path="model")
        else:
            mlflow.sklearn.log_model(modelo, artifact_path="model")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"No se pudo loggear el modelo en MLflow: {exc}")

    return mlflow.active_run().info.run_id


# ============================================================================
# Función genérica de entrenamiento
# ============================================================================
def _entrenar_y_evaluar(
    modelo: Any,
    nombre: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    flavor: str = "sklearn",
) -> ResultadoEntrenamiento:
    """Entrena un modelo cualquiera y devuelve métricas + artefactos."""
    logger.info(f"--- Entrenando {nombre} ---")

    with mlflow.start_run(run_name=nombre, nested=True):
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        metricas = _calcular_metricas(y_test, y_pred, y_proba)

        cm, ruta_cm = _guardar_matriz_confusion(y_test, y_pred, nombre)
        fi, ruta_fi = _extraer_feature_importance(
            modelo, X_train.columns.tolist(), nombre
        )

        ruta_modelo = guardar_modelo(modelo, f"modelo_{nombre}")
        run_id = _loggear_resultados_mlflow(
            nombre, modelo, params, metricas, ruta_cm, ruta_fi, flavor
        )

    logger.success(
        f"{nombre} | "
        f"ROC-AUC={metricas['roc_auc']:.4f} | "
        f"F1={metricas['f1_score']:.4f} | "
        f"Recall={metricas['recall']:.4f}"
    )

    return ResultadoEntrenamiento(
        nombre=nombre,
        modelo=modelo,
        metricas=metricas,
        matriz_confusion=cm,
        feature_importance=fi,
        ruta_modelo=ruta_modelo,
        ruta_cm=ruta_cm,
        ruta_fi=ruta_fi,
        run_id=run_id,
        hiperparametros=params,
    )


# ============================================================================
# Entrenadores específicos por modelo
# ============================================================================
def entrenar_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> ResultadoEntrenamiento:
    """
    Entrena Logistic Regression con `StandardScaler` dentro de un Pipeline.

    LR es sensible a la escala de las features; el scaler asegura
    que coeficientes y convergencia sean correctos.
    """
    params = params or PARAMS_LOGISTIC
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(**params)),
        ]
    )
    return _entrenar_y_evaluar(
        pipeline, "logistic_regression", X_train, y_train, X_test, y_test, params
    )


def entrenar_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> ResultadoEntrenamiento:
    """Entrena Random Forest. No requiere escalado."""
    params = params or PARAMS_RANDOM_FOREST
    modelo = RandomForestClassifier(**params)
    return _entrenar_y_evaluar(
        modelo, "random_forest", X_train, y_train, X_test, y_test, params
    )


def entrenar_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> ResultadoEntrenamiento:
    """Entrena XGBoost. Maneja desbalance con `scale_pos_weight` auto."""
    from xgboost import XGBClassifier

    params = dict(params or PARAMS_XGBOOST)
    if "scale_pos_weight" not in params:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        params["scale_pos_weight"] = float(neg / pos) if pos > 0 else 1.0

    modelo = XGBClassifier(**params)
    return _entrenar_y_evaluar(
        modelo, "xgboost", X_train, y_train, X_test, y_test, params, flavor="xgboost"
    )


def entrenar_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> ResultadoEntrenamiento:
    """Entrena LightGBM. Rápido y eficiente con muchos features y filas."""
    from lightgbm import LGBMClassifier

    params = params or PARAMS_LIGHTGBM
    modelo = LGBMClassifier(**params)
    return _entrenar_y_evaluar(
        modelo, "lightgbm", X_train, y_train, X_test, y_test, params, flavor="lightgbm"
    )


# ============================================================================
# Orquestador y comparación
# ============================================================================
ENTRENADORES = {
    "logistic_regression": entrenar_logistic_regression,
    "random_forest": entrenar_random_forest,
    "xgboost": entrenar_xgboost,
    "lightgbm": entrenar_lightgbm,
}


def entrenar_todos_los_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    modelos: list[str] | None = None,
    nombre_experimento: str | None = None,
) -> dict[str, ResultadoEntrenamiento]:
    """
    Entrena todos los modelos solicitados dentro de un run padre en MLflow.

    Parameters
    ----------
    modelos : list[str] | None
        Lista de nombres de modelos a entrenar (ver `ENTRENADORES`).
        Si es None, entrena los 4.
    nombre_experimento : str | None
        Sobrescribe el nombre del experimento de MLflow para esta corrida.

    Returns
    -------
    dict[str, ResultadoEntrenamiento]
        Diccionario con los resultados de cada modelo.
    """
    modelos = modelos or list(ENTRENADORES.keys())
    desconocidos = set(modelos) - set(ENTRENADORES.keys())
    if desconocidos:
        raise ValueError(f"Modelos no soportados: {desconocidos}")

    if nombre_experimento:
        mlflow.set_experiment(nombre_experimento)
    else:
        _configurar_mlflow()

    logger.info("=" * 60)
    logger.info(f"ENTRENAMIENTO DE {len(modelos)} MODELOS")
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Tasa de retraso (train): {y_train.mean():.2%}")
    logger.info("=" * 60)

    resultados: dict[str, ResultadoEntrenamiento] = {}

    with mlflow.start_run(run_name="comparativa_modelos") as parent_run:
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("tasa_positivos_train", float(y_train.mean()))

        for nombre in modelos:
            try:
                resultados[nombre] = ENTRENADORES[nombre](
                    X_train, y_train, X_test, y_test
                )
            except Exception as exc:  # pragma: no cover
                logger.error(f"Error entrenando '{nombre}': {exc}")

        # Loggear comparativa en el run padre
        if resultados:
            df_comp = comparar_modelos(resultados)
            ruta = settings.resolve_path(settings.reports_path) / "comparativa_modelos.csv"
            df_comp.to_csv(ruta, index=False)
            mlflow.log_artifact(str(ruta))
            logger.info(f"Comparativa registrada en MLflow run: {parent_run.info.run_id}")

    return resultados


def comparar_modelos(resultados: dict[str, ResultadoEntrenamiento]) -> pd.DataFrame:
    """Genera un DataFrame comparando las métricas de cada modelo."""
    filas = []
    for nombre, r in resultados.items():
        filas.append({"modelo": nombre, **r.metricas, "run_id": r.run_id})
    df = pd.DataFrame(filas).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


def seleccionar_mejor_modelo(
    resultados: dict[str, ResultadoEntrenamiento],
    metrica: str = "roc_auc",
    encoder: Any | None = None,
    feature_columns: list[str] | None = None,
    columnas_onehot: list[str] | None = None,
) -> ResultadoEntrenamiento:
    """
    Selecciona el modelo con mejor valor de la métrica indicada.

    Métricas soportadas: 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'.

    Si se proporcionan `encoder` y `feature_columns` (típicamente
    obtenidos de `construir_features_dataco`), también persiste la
    metadata de inferencia para que la app Streamlit y la API puedan
    aplicar las mismas transformaciones a inputs nuevos.
    """
    if not resultados:
        raise ValueError("No hay resultados para seleccionar mejor modelo")

    mejor_nombre, mejor = max(
        resultados.items(), key=lambda kv: kv[1].metricas.get(metrica, -np.inf)
    )
    logger.success(
        f"Mejor modelo: '{mejor_nombre}' "
        f"({metrica}={mejor.metricas[metrica]:.4f})"
    )

    # Persistir un alias 'modelo_mejor.joblib' para uso por la API/app
    ruta_alias = guardar_modelo(mejor.modelo, "modelo_mejor")
    logger.info(f"Modelo ganador persistido como: {ruta_alias}")

    # Persistir metadata de inferencia si se proporcionó
    ruta_metadata: str | None = None
    if encoder is not None and feature_columns is not None:
        from src.inference import guardar_metadata_inferencia

        ruta_metadata = str(
            guardar_metadata_inferencia(
                encoder=encoder,
                feature_columns=feature_columns,
                columnas_onehot=columnas_onehot,
            )
        )

    # Reporte final con el resumen
    guardar_reporte_json(
        {
            "mejor_modelo": mejor_nombre,
            "metrica_seleccion": metrica,
            "metricas": mejor.metricas,
            "hiperparametros": mejor.hiperparametros,
            "ruta_modelo": str(mejor.ruta_modelo),
            "ruta_alias_mejor": str(ruta_alias),
            "ruta_metadata_inferencia": ruta_metadata,
            "run_id": mejor.run_id,
            "comparativa": comparar_modelos(resultados).to_dict(orient="records"),
        },
        nombre="resultado_seleccion_modelo",
    )

    return mejor


# ============================================================================
# COMPATIBILIDAD: función original del pipeline anterior
# ============================================================================
def entrenar_modelo_retrasos(
    df: pd.DataFrame,
    columna_objetivo: str = "retraso",
    hiperparametros: dict | None = None,
    nombre_modelo: str = "modelo_retrasos_xgb",
) -> dict[str, Any]:
    """
    Wrapper de compatibilidad: entrena XGBoost recibiendo un DataFrame
    completo (con la columna objetivo dentro) y haciendo split estratificado.

    Para flujos nuevos, usar `entrenar_todos_los_modelos` o el flujo MVP
    `entrenar_modelo_delay_mvp` definido más abajo.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    resultado = entrenar_xgboost(X_train, y_train, X_test, y_test, hiperparametros)
    return {
        "modelo": resultado.modelo,
        "metricas": resultado.metricas,
        "ruta_modelo": str(resultado.ruta_modelo),
        "run_id": resultado.run_id,
    }


# ============================================================================
# ============= PIPELINE MVP — Predicción de retrasos (Streamlit)  ===========
# ============================================================================
# Pipeline simplificado, autocontenido y compatible con el formulario de
# la app demo. Entrena un único Pipeline sklearn (preprocesamiento +
# clasificador) usando SOLO las 13 features ex-ante definidas en
# `src.features.FEATURES_MVP_TODAS`.
#
# Output esperado en disco:
#   - models/delay_model.pkl       (Pipeline serializado con joblib)
#   - models/feature_columns.json  (metadata legible de columnas y métricas)
# ============================================================================

NOMBRE_MODELO_MVP_PKL = "delay_model.pkl"
NOMBRE_FEATURES_JSON = "feature_columns.json"


def _construir_clasificador_mvp(nombre: str, y_train: pd.Series):
    """Devuelve una instancia del clasificador solicitado, ya parametrizado."""
    if nombre == "logistic_regression":
        return LogisticRegression(**PARAMS_LOGISTIC)

    if nombre == "random_forest":
        return RandomForestClassifier(**PARAMS_RANDOM_FOREST)

    if nombre == "xgboost":
        from xgboost import XGBClassifier

        params = dict(PARAMS_XGBOOST)
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        params["scale_pos_weight"] = float(neg / pos) if pos > 0 else 1.0
        return XGBClassifier(**params)

    if nombre == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**PARAMS_LIGHTGBM)

    raise ValueError(f"Clasificador no soportado: '{nombre}'")


def entrenar_modelo_delay_mvp(
    df: pd.DataFrame,
    modelos_a_probar: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = "late_delivery_risk",
    guardar: bool = True,
    nombre_archivo_modelo: str = NOMBRE_MODELO_MVP_PKL,
    nombre_archivo_columnas: str = NOMBRE_FEATURES_JSON,
) -> dict[str, Any]:
    """
    Pipeline MVP de entrenamiento para predicción de retrasos.

    Pasos:
        1. Valida la columna objetivo.
        2. Extrae las 13 features ex-ante con `preparar_features_mvp`.
        3. Split train/test estratificado.
        4. Construye un Pipeline sklearn:
               [ColumnTransformer(num + cat)] -> [clasificador]
        5. Entrena varios candidatos (LR, RF, XGB, LGBM por defecto).
        6. Selecciona el mejor por ROC-AUC en test.
        7. Guarda el pipeline en `models/delay_model.pkl` y la
           metadata en `models/feature_columns.json`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataCo limpio (output de `limpiar_dataco`). Debe contener
        la columna objetivo y las 13 columnas del MVP (o `order_date_dateorders`
        para derivar mes/día).
    modelos_a_probar : list[str] | None
        Subconjunto de {'logistic_regression', 'random_forest',
        'xgboost', 'lightgbm'}. Por defecto los entrena todos.
    test_size : float
    random_state : int
    target : str
        Nombre de la columna objetivo binaria (0/1).
    guardar : bool
        Si True, persiste pipeline y metadata.

    Returns
    -------
    dict con:
        - 'pipeline'         : el Pipeline sklearn ganador
        - 'nombre_modelo'    : str
        - 'metricas'         : dict de métricas en test
        - 'feature_columns'  : list[str]
        - 'comparativa'      : pd.DataFrame con todos los candidatos
        - 'ruta_modelo'      : Path | None
        - 'ruta_columnas'    : Path | None
    """
    import json
    from pathlib import Path

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SKPipeline

    from src.features import (
        FEATURES_MVP_CATEGORICAS,
        FEATURES_MVP_NUMERICAS,
        FEATURES_MVP_TODAS,
        construir_preprocesador_mvp,
        preparar_features_mvp,
    )

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO MVP — DELAY MODEL")
    logger.info("=" * 60)

    # --- 1. Validación ------------------------------------------------------
    if target not in df.columns:
        raise KeyError(
            f"El dataset no contiene la columna objetivo '{target}'. "
            f"Asegúrate de cargar `dataco_clean.parquet`."
        )

    # --- 2. Preparar X / y --------------------------------------------------
    df_clean = df.copy()
    # Si las columnas temporales no existen pero sí la fecha, las derivamos
    if {"order_month", "order_dayofweek"}.isdisjoint(df_clean.columns) and (
        "order_date_dateorders" in df_clean.columns
    ):
        fecha = pd.to_datetime(df_clean["order_date_dateorders"], errors="coerce")
        df_clean["order_month"] = fecha.dt.month
        df_clean["order_dayofweek"] = fecha.dt.dayofweek

    X = preparar_features_mvp(df_clean)
    y = df_clean[target].astype(int)

    # Eliminar filas con target nulo (defensa)
    mask_validas = ~y.isna()
    X = X.loc[mask_validas].reset_index(drop=True)
    y = y.loc[mask_validas].reset_index(drop=True)

    logger.info(
        f"Dataset preparado: X={X.shape} | tasa de retraso={y.mean():.2%}"
    )

    # --- 3. Split -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # --- 4-6. Probar candidatos --------------------------------------------
    modelos_a_probar = modelos_a_probar or [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    ]
    candidatos: dict[str, dict] = {}

    for nombre in modelos_a_probar:
        try:
            logger.info(f"--- Entrenando candidato: {nombre} ---")
            clf = _construir_clasificador_mvp(nombre, y_train)
            pipeline = SKPipeline(
                steps=[
                    ("preproc", construir_preprocesador_mvp()),
                    ("clf", clf),
                ]
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metricas = _calcular_metricas(y_test, y_pred, y_proba)

            candidatos[nombre] = {"pipeline": pipeline, "metricas": metricas}
            logger.success(
                f"  {nombre} -> ROC-AUC={metricas['roc_auc']:.4f} "
                f"F1={metricas['f1_score']:.4f} "
                f"Recall={metricas['recall']:.4f}"
            )
        except ImportError as exc:
            logger.warning(f"  Se omite {nombre} (dependencia no instalada): {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  Error entrenando {nombre}: {exc}")

    if not candidatos:
        raise RuntimeError(
            "Ningún candidato se entrenó correctamente. Revisa logs y dependencias."
        )

    # Elegir mejor por ROC-AUC
    mejor_nombre = max(candidatos, key=lambda n: candidatos[n]["metricas"]["roc_auc"])
    mejor = candidatos[mejor_nombre]
    logger.success(
        f"Mejor candidato: '{mejor_nombre}' "
        f"(ROC-AUC={mejor['metricas']['roc_auc']:.4f})"
    )

    # --- 7. Comparativa -----------------------------------------------------
    df_comp = pd.DataFrame(
        [{"modelo": n, **c["metricas"]} for n, c in candidatos.items()]
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # --- 8. Persistencia ----------------------------------------------------
    ruta_modelo: Path | None = None
    ruta_columnas: Path | None = None

    if guardar:
        base = settings.resolve_path(settings.models_path)
        base.mkdir(parents=True, exist_ok=True)

        ruta_modelo = base / nombre_archivo_modelo
        ruta_columnas = base / nombre_archivo_columnas

        # Pipeline completo serializado con joblib (extension .pkl por convención)
        import joblib

        joblib.dump(mejor["pipeline"], ruta_modelo)

        feature_info = {
            "feature_columns": FEATURES_MVP_TODAS,
            "categorical_columns": FEATURES_MVP_CATEGORICAS,
            "numerical_columns": FEATURES_MVP_NUMERICAS,
            "target": target,
            "modelo": mejor_nombre,
            "metricas_test": mejor["metricas"],
            "comparativa": df_comp.to_dict(orient="records"),
            "ruta_modelo": str(ruta_modelo),
            "version_features": "mvp-v1",
        }
        with ruta_columnas.open("w", encoding="utf-8") as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)

        logger.success(f"Pipeline guardado: {ruta_modelo}")
        logger.success(f"Feature columns guardado: {ruta_columnas}")

    logger.info("=" * 60)

    return {
        "pipeline": mejor["pipeline"],
        "nombre_modelo": mejor_nombre,
        "metricas": mejor["metricas"],
        "feature_columns": FEATURES_MVP_TODAS,
        "comparativa": df_comp,
        "ruta_modelo": ruta_modelo,
        "ruta_columnas": ruta_columnas,
    }
