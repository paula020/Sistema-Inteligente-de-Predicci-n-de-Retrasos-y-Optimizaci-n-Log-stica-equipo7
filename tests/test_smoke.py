"""
Pruebas smoke: validan que los módulos principales importan correctamente
y que las funciones de cálculo de negocio responden sin errores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_import_config():
    from src.config import settings

    assert settings.app_env in {"development", "staging", "production"}


def test_preprocesar_dataframe():
    from src.preprocessing import preprocesar_dataframe

    df = pd.DataFrame(
        {
            "Ciudad Origen": ["Bogotá", "Medellín", "Bogotá"],
            "Valor Flete": [100, 200, None],
        }
    )
    out = preprocesar_dataframe(df)
    assert "ciudad_origen" in out.columns
    assert out["valor_flete"].isna().sum() == 0


def test_calcular_rentabilidad():
    from src.route_profitability import calcular_rentabilidad

    df = pd.DataFrame(
        {
            "ruta": ["A-B", "A-C"],
            "valor_flete": [1_000_000, 500_000],
            "distancia_km": [400, 200],
            "dias_transito": [2, 1],
        }
    )
    out = calcular_rentabilidad(df)
    assert {"costo_total", "margen_absoluto", "roi"}.issubset(out.columns)


def test_score_mercado():
    from src.market_scoring import calcular_score_mercado

    df = pd.DataFrame(
        {
            "ciudad_destino": ["Bogota", "Medellin", "Cali"],
            "guias": [1000, 800, 500],
            "margen_pct": [0.15, 0.10, 0.08],
        }
    )
    out = calcular_score_mercado(df)
    assert "score_mercado" in out.columns
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Tests específicos del módulo de preprocesamiento DataCo
# ---------------------------------------------------------------------------
def test_normalizar_nombres_columnas_dataco():
    """Verifica que los nombres típicos de DataCo se normalicen correctamente."""
    from src.preprocessing import normalizar_nombres_columnas

    df = pd.DataFrame(
        columns=[
            "Late_delivery_risk",
            "Order Item Profit Ratio",
            "order date (DateOrders)",
            "shipping date (DateOrders)",
            "Customer Fname",
        ]
    )
    out = normalizar_nombres_columnas(df)
    assert "late_delivery_risk" in out.columns
    assert "order_item_profit_ratio" in out.columns
    assert "order_date_dateorders" in out.columns
    assert "shipping_date_dateorders" in out.columns
    assert "customer_fname" in out.columns


def test_reporte_nulos():
    from src.preprocessing import reporte_nulos

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, None],
            "b": [None, None, "x", "y"],
            "c": [1, 2, 3, 4],
        }
    )
    rep = reporte_nulos(df)
    assert {"columna", "nulos", "pct_nulos", "dtype"}.issubset(rep.columns)
    pct_b = rep.loc[rep["columna"] == "b", "pct_nulos"].iloc[0]
    assert pct_b == 50.0


def test_convertir_fechas():
    from src.preprocessing import convertir_fechas

    df = pd.DataFrame(
        {
            "order_date_dateorders": ["1/1/2018 0:00", "2/15/2018 3:30", "bad-date"],
            "shipping_date_dateorders": ["1/3/2018 0:00", "2/18/2018 4:00", "1/4/2018"],
        }
    )
    out = convertir_fechas(df)
    assert pd.api.types.is_datetime64_any_dtype(out["order_date_dateorders"])
    assert out["order_date_dateorders"].isna().sum() == 1


def test_validar_target():
    from src.preprocessing import validar_target

    df_ok = pd.DataFrame({"late_delivery_risk": [0, 1, 1, 0]})
    validar_target(df_ok)

    df_malo = pd.DataFrame({"late_delivery_risk": [0, 1, 2]})
    import pytest

    with pytest.raises(ValueError):
        validar_target(df_malo)


def test_eliminar_columnas_totalmente_nulas():
    from src.preprocessing import eliminar_columnas_totalmente_nulas

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "vacia": [None, None, None],
        }
    )
    out = eliminar_columnas_totalmente_nulas(df)
    assert "vacia" not in out.columns
    assert "a" in out.columns


# ---------------------------------------------------------------------------
# Tests del módulo de feature engineering (DataCo)
# ---------------------------------------------------------------------------
def _df_dataco_sintetico(n: int = 200):
    """Construye un DataFrame sintético con la forma del DataCo limpio."""
    import numpy as np

    rng = np.random.default_rng(42)
    fechas = pd.date_range("2017-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "order_date_dateorders": fechas,
            "days_for_shipping_real": rng.integers(0, 8, n),
            "days_for_shipment_scheduled": rng.integers(0, 6, n),
            "sales": rng.uniform(50, 500, n),
            "order_profit_per_order": rng.uniform(-20, 100, n),
            "order_item_profit_ratio": rng.uniform(-0.2, 0.5, n),
            "order_item_discount_rate": rng.uniform(0, 0.3, n),
            "order_item_quantity": rng.integers(1, 5, n),
            "order_city": rng.choice(["Bogota", "Medellin", "Cali", "Barranquilla"], n),
            "market": rng.choice(["LATAM", "EUR", "USA"], n),
            "order_region": rng.choice(["Norte", "Sur", "Centro"], n),
            "customer_id": rng.integers(1, 30, n),
            "shipping_mode": rng.choice(["Standard", "Express", "First Class"], n),
            "customer_segment": rng.choice(["Consumer", "Corporate", "Home"], n),
            "category_name": rng.choice(["A", "B", "C"], n),
            "department_name": rng.choice(["X", "Y"], n),
            "late_delivery_risk": rng.integers(0, 2, n),
        }
    )
    return df


def test_variables_temporales():
    from src.features import variables_temporales

    df = _df_dataco_sintetico(50)
    out = variables_temporales(df)
    for col in ("order_year", "order_month", "order_dayofweek", "order_is_weekend"):
        assert col in out.columns
    assert out["order_is_weekend"].isin([0, 1]).all()


def test_variables_retraso_e_is_late():
    from src.features import variables_retraso

    df = _df_dataco_sintetico(100)
    out = variables_retraso(df)
    assert "delay_days" in out.columns and "is_late" in out.columns
    assert out["is_late"].isin([0, 1]).all()
    diferencia = out["days_for_shipping_real"] - out["days_for_shipment_scheduled"]
    assert (out["delay_days"] == diferencia).all()


def test_target_encoder_no_leakage():
    """
    Validación CRÍTICA: el encoder debe usar SOLO valores del train.
    Si vemos una categoría únicamente en test, debe caer en la media global.
    """
    from src.features import TargetEncoderSuavizado

    X_train = pd.DataFrame({"ciudad": ["A", "A", "B", "B", "B"]})
    y_train = pd.Series([1, 1, 0, 0, 0])
    X_test = pd.DataFrame({"ciudad": ["A", "B", "C"]})  # 'C' es nueva

    enc = TargetEncoderSuavizado(m=1, columnas=["ciudad"]).fit(X_train, y_train)
    out = enc.transform(X_test)

    assert "ciudad_risk_score" in out.columns
    # 'C' nunca se vio: cae a la media global
    score_c = out.loc[out["ciudad"] == "C", "ciudad_risk_score"].iloc[0]
    assert abs(score_c - enc.media_global_) < 1e-9


def test_split_temporal_es_estrictamente_cronologico():
    from src.features import split_temporal

    df = _df_dataco_sintetico(100)
    train, test = split_temporal(df, "order_date_dateorders", test_size=0.2)
    assert train["order_date_dateorders"].max() <= test["order_date_dateorders"].min()
    assert len(train) + len(test) == len(df)


def test_eliminar_features_leakage_y_seleccionar_X_y():
    from src.features import (
        eliminar_features_leakage,
        seleccionar_X_y,
        variables_temporales,
        variables_economicas,
    )

    df = _df_dataco_sintetico(100)
    df = variables_temporales(df)
    df = variables_economicas(df)
    df = eliminar_features_leakage(df)

    columnas_prohibidas = {
        "days_for_shipping_real", "delay_days", "is_late",
        "shipping_efficiency", "delivery_status",
    }
    assert columnas_prohibidas.isdisjoint(df.columns)

    # Convertir categóricas residuales para poder hacer X / y
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    X, y = seleccionar_X_y(df)
    assert "late_delivery_risk" not in X.columns
    assert len(X) == len(y)


def test_pipeline_completo_dataco():
    """Smoke test del pipeline orquestador completo."""
    from src.features import construir_features_dataco

    df = _df_dataco_sintetico(300)
    resultado = construir_features_dataco(df, test_size=0.25)
    assert resultado["X_train"].shape[0] > 0
    assert resultado["X_test"].shape[0] > 0
    assert resultado["X_train"].shape[1] == resultado["X_test"].shape[1]
    # Ninguna columna prohibida sobrevivió al pipeline
    for col in ("days_for_shipping_real", "is_late", "delay_days"):
        assert col not in resultado["X_train"].columns


# ---------------------------------------------------------------------------
# Tests del módulo de entrenamiento
# ---------------------------------------------------------------------------
def _datos_ml_sinteticos(n_train: int = 500, n_test: int = 200, n_features: int = 10):
    """Genera X/y sintéticos con señal suficiente para que el modelo aprenda."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        weights=[0.6, 0.4],
        random_state=42,
    )
    columnas = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=columnas)
    y = pd.Series(y, name="late_delivery_risk")
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]


def test_calcular_metricas():
    from src.train_model import _calcular_metricas
    import numpy as np

    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.3])

    m = _calcular_metricas(y_true, y_pred, y_proba)
    for clave in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert clave in m
        assert 0.0 <= m[clave] <= 1.0


def test_entrenar_logistic_regression(monkeypatch, tmp_path):
    """Verifica entrenamiento de LR sin tocar el tracking real de MLflow."""
    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)

    from src.train_model import entrenar_logistic_regression

    X_train, y_train, X_test, y_test = _datos_ml_sinteticos()

    with mlflow.start_run(run_name="parent"):
        resultado = entrenar_logistic_regression(X_train, y_train, X_test, y_test)

    assert resultado.nombre == "logistic_regression"
    assert 0.0 <= resultado.metricas["roc_auc"] <= 1.0
    assert resultado.matriz_confusion.shape == (2, 2)
    # Pipeline → feature importance debe extraerse de los coeficientes
    assert resultado.feature_importance is not None
    assert len(resultado.feature_importance) == X_train.shape[1]


def test_entrenar_random_forest(monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)

    from src.train_model import entrenar_random_forest

    X_train, y_train, X_test, y_test = _datos_ml_sinteticos()
    with mlflow.start_run(run_name="parent"):
        resultado = entrenar_random_forest(X_train, y_train, X_test, y_test)

    assert resultado.feature_importance is not None
    assert resultado.metricas["roc_auc"] > 0.5  # mejor que random


def test_comparar_y_seleccionar_mejor():
    from src.train_model import (
        ResultadoEntrenamiento, comparar_modelos, seleccionar_mejor_modelo,
    )
    import numpy as np

    # Mock de dos resultados con distintas métricas
    class _DummyModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

    r1 = ResultadoEntrenamiento(
        nombre="modelo_a", modelo=_DummyModel(),
        metricas={"roc_auc": 0.80, "f1_score": 0.70, "accuracy": 0.75,
                  "precision": 0.7, "recall": 0.7},
        matriz_confusion=np.array([[10, 2], [3, 15]]),
        feature_importance=None,
        ruta_modelo=Path("dummy_a"),
    )
    r2 = ResultadoEntrenamiento(
        nombre="modelo_b", modelo=_DummyModel(),
        metricas={"roc_auc": 0.90, "f1_score": 0.85, "accuracy": 0.88,
                  "precision": 0.8, "recall": 0.9},
        matriz_confusion=np.array([[12, 1], [2, 16]]),
        feature_importance=None,
        ruta_modelo=Path("dummy_b"),
    )
    resultados = {"modelo_a": r1, "modelo_b": r2}

    df = comparar_modelos(resultados)
    assert df.iloc[0]["modelo"] == "modelo_b"  # mejor por ROC-AUC

    mejor = seleccionar_mejor_modelo(resultados, metrica="roc_auc")
    assert mejor.nombre == "modelo_b"


# ---------------------------------------------------------------------------
# Tests del módulo de inferencia y predicción del MVP
# ---------------------------------------------------------------------------
def test_clasificar_riesgo():
    from src.predict import clasificar_riesgo

    assert clasificar_riesgo(0.10) == "BAJO"
    assert clasificar_riesgo(0.39) == "BAJO"
    assert clasificar_riesgo(0.40) == "MEDIO"
    assert clasificar_riesgo(0.55) == "MEDIO"
    assert clasificar_riesgo(0.70) == "MEDIO"
    assert clasificar_riesgo(0.71) == "ALTO"
    assert clasificar_riesgo(0.99) == "ALTO"


def test_recomendacion_para_cada_nivel():
    from src.predict import recomendacion_para, RECOMENDACIONES

    for nivel in ("BAJO", "MEDIO", "ALTO"):
        reco = recomendacion_para(nivel)
        assert reco == RECOMENDACIONES[nivel]
        assert len(reco) > 10  # texto no vacío


def _dataset_dataco_para_mvp(n: int = 600):
    """Genera un DataFrame DataCo-like con TODAS las columnas que el MVP necesita."""
    import numpy as np

    rng = np.random.default_rng(123)
    fechas = pd.date_range("2017-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            # categóricas MVP
            "shipping_mode": rng.choice(
                ["Standard Class", "First Class", "Second Class", "Same Day"], n
            ),
            "customer_segment": rng.choice(
                ["Consumer", "Corporate", "Home Office"], n
            ),
            "market": rng.choice(["LATAM", "Europe", "USCA", "Pacific Asia"], n),
            "order_region": rng.choice(
                ["Central America", "South America", "Western Europe", "South Asia"], n
            ),
            "order_country": rng.choice(
                ["Estados Unidos", "México", "Brasil", "Colombia"], n
            ),
            "category_name": rng.choice(["Electronics", "Apparel", "Sporting Goods"], n),
            # numéricas MVP
            "sales": rng.uniform(50, 500, n),
            "order_item_quantity": rng.integers(1, 5, n),
            "order_item_discount": rng.uniform(0, 50, n),
            "order_item_product_price": rng.uniform(20, 400, n),
            "days_for_shipment_scheduled": rng.integers(0, 6, n),
            # variables temporales
            "order_date_dateorders": fechas,
            # target
            "late_delivery_risk": rng.integers(0, 2, n),
        }
    )
    return df


def test_features_mvp_presentes():
    """Verifica que las constantes MVP existan y sean coherentes."""
    from src.features import (
        FEATURES_MVP_CATEGORICAS,
        FEATURES_MVP_NUMERICAS,
        FEATURES_MVP_TODAS,
    )

    assert len(FEATURES_MVP_CATEGORICAS) == 6
    assert len(FEATURES_MVP_NUMERICAS) == 7
    assert len(FEATURES_MVP_TODAS) == 13
    # No deben mezclarse
    assert set(FEATURES_MVP_CATEGORICAS).isdisjoint(set(FEATURES_MVP_NUMERICAS))


def test_preparar_features_mvp_deriva_fecha():
    """`preparar_features_mvp` debe derivar order_month/dayofweek de la fecha."""
    from src.features import preparar_features_mvp

    df = _dataset_dataco_para_mvp(30)
    out = preparar_features_mvp(df)

    assert "order_month" in out.columns
    assert "order_dayofweek" in out.columns
    assert out.shape[1] == 13
    assert out["order_month"].between(1, 12).all()
    assert out["order_dayofweek"].between(0, 6).all()


def test_entrenar_modelo_delay_mvp_end_to_end(tmp_path, monkeypatch):
    """
    End-to-end: entrena un Pipeline MVP, lo persiste, lo recarga y predice
    a partir del formulario.
    """
    from src.config import settings
    from src import predict as predict_module
    from src.train_model import entrenar_modelo_delay_mvp

    # Redirigir el directorio de modelos a un temporal aislado
    monkeypatch.setattr(settings, "models_path", tmp_path, raising=False)
    predict_module.limpiar_cache_modelos()

    df = _dataset_dataco_para_mvp(400)

    # Usamos solo LR + RF para que la prueba sea rápida y no dependa de
    # XGBoost/LightGBM si no están instalados.
    resultado = entrenar_modelo_delay_mvp(
        df,
        modelos_a_probar=["logistic_regression", "random_forest"],
        guardar=True,
    )

    assert resultado["nombre_modelo"] in {"logistic_regression", "random_forest"}
    assert len(resultado["feature_columns"]) == 13
    assert (tmp_path / "delay_model.pkl").exists()
    assert (tmp_path / "feature_columns.json").exists()

    # Recargar metadata desde JSON
    import json

    with (tmp_path / "feature_columns.json").open(encoding="utf-8") as f:
        info = json.load(f)
    assert info["feature_columns"] == resultado["feature_columns"]
    assert info["target"] == "late_delivery_risk"
    assert "metricas_test" in info

    # Probar el flujo de inferencia desde el formulario
    form = {
        "shipping_mode": "First Class",
        "customer_segment": "Consumer",
        "market": "LATAM",
        "order_region": "Central America",
        "order_country": "Colombia",
        "category_name": "Electronics",
        "sales": 250.0,
        "order_item_quantity": 2,
        "order_item_discount": 10.0,
        "order_item_product_price": 200.0,
        "days_for_shipment_scheduled": 4,
        "order_month": 6,
        "order_dayofweek": 2,
    }
    res = predict_module.predecir_desde_formulario(form)
    assert res["ok"] is True
    assert 0.0 <= res["probabilidad"] <= 1.0
    assert res["nivel_riesgo"] in {"BAJO", "MEDIO", "ALTO"}
    assert res["features_aplicadas"] == 13
    assert res["recomendacion"]  # texto no vacío


def test_predecir_desde_formulario_sin_modelo_devuelve_error_amigable(
    tmp_path, monkeypatch
):
    """Si no existe el modelo, debe devolver `ok=False` con mensaje claro."""
    from src.config import settings
    from src import predict as predict_module

    monkeypatch.setattr(settings, "models_path", tmp_path, raising=False)
    predict_module.limpiar_cache_modelos()

    resultado = predict_module.predecir_desde_formulario(
        {"shipping_mode": "Standard Class", "market": "LATAM"}
    )
    assert resultado["ok"] is False
    assert "error" in resultado and len(resultado["error"]) > 0


def test_predecir_desde_formulario_campos_faltantes(tmp_path, monkeypatch):
    """Si el formulario no envía todos los campos, error amigable."""
    from src.config import settings
    from src import predict as predict_module
    from src.train_model import entrenar_modelo_delay_mvp

    monkeypatch.setattr(settings, "models_path", tmp_path, raising=False)
    predict_module.limpiar_cache_modelos()

    df = _dataset_dataco_para_mvp(300)
    entrenar_modelo_delay_mvp(
        df,
        modelos_a_probar=["logistic_regression"],
        guardar=True,
    )

    # Formulario incompleto: solo 2 campos
    res = predict_module.predecir_desde_formulario(
        {"shipping_mode": "Standard Class", "market": "LATAM"}
    )
    assert res["ok"] is False
    assert "faltan" in res["error"].lower() or "campos" in res["error"].lower()
