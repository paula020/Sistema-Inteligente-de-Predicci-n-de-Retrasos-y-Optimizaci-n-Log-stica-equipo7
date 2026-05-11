"""
Pruebas smoke: validan que los módulos principales importan correctamente
y que las funciones de cálculo de negocio responden sin errores.
"""

from __future__ import annotations

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
