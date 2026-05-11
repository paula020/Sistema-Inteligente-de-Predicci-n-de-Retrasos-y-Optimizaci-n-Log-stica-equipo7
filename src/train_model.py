"""
Entrenamiento del modelo de predicción de retrasos.

Entrena un clasificador XGBoost con tracking de MLflow,
guarda métricas, parámetros y el modelo final en `models/`.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import settings
from src.utils import guardar_modelo, guardar_reporte_json


# ----------------------------------------------------------------------------
# Configuración por defecto del modelo
# ----------------------------------------------------------------------------
HIPERPARAMETROS_DEFAULT: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


# ----------------------------------------------------------------------------
# División de datos
# ----------------------------------------------------------------------------
def dividir_train_test(
    df: pd.DataFrame,
    columna_objetivo: str = "retraso",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide el dataset en train/test estratificado por la variable objetivo."""
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Train: {X_train.shape} | Test: {X_test.shape} | "
        f"Tasa retraso: {y.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def calcular_metricas(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcula el set de métricas estándar del modelo."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def entrenar_modelo_retrasos(
    df: pd.DataFrame,
    columna_objetivo: str = "retraso",
    hiperparametros: dict | None = None,
    nombre_modelo: str = "modelo_retrasos_xgb",
) -> dict[str, Any]:
    """
    Entrena el modelo de clasificación de retrasos con tracking en MLflow.

    Returns
    -------
    dict
        Diccionario con el modelo entrenado, métricas y rutas de artefactos.
    """
    hiperparametros = hiperparametros or HIPERPARAMETROS_DEFAULT
    logger.info("Iniciando entrenamiento del modelo de retrasos")

    X_train, X_test, y_train, y_test = dividir_train_test(df, columna_objetivo)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=nombre_modelo) as run:
        mlflow.log_params(hiperparametros)

        modelo = XGBClassifier(**hiperparametros)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        metricas = calcular_metricas(y_test, y_pred, y_proba)

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, artifact_path="model")

        ruta_modelo = guardar_modelo(modelo, nombre_modelo)
        reporte = {
            "run_id": run.info.run_id,
            "modelo": nombre_modelo,
            "metricas": metricas,
            "hiperparametros": hiperparametros,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }
        guardar_reporte_json(reporte, f"{nombre_modelo}_reporte")

        logger.success(
            f"Modelo entrenado | ROC-AUC: {metricas['roc_auc']:.4f} | "
            f"F1: {metricas['f1_score']:.4f} | Recall: {metricas['recall']:.4f}"
        )

    return {
        "modelo": modelo,
        "metricas": metricas,
        "ruta_modelo": str(ruta_modelo),
        "run_id": run.info.run_id,
    }
