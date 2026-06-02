"""
Página: Predicción de retrasos.

Permite ingresar datos de una guía y obtener la probabilidad
de retraso, nivel de riesgo y costo esperado.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.financial_impact import costo_esperado_por_probabilidad


st.set_page_config(page_title="Predicción de Retrasos", page_icon="🚚", layout="wide")
st.title("Predicción de Retrasos")
st.caption("Scoring de guías en tiempo real con XGBoost.")


# ----------------------------------------------------------------------------
# Formulario de entrada
# ----------------------------------------------------------------------------
with st.form("form_guia"):
    col1, col2, col3 = st.columns(3)

    distancia_km = col1.number_input("Distancia (km)", min_value=1.0, value=450.0)
    peso_kg = col2.number_input("Peso (kg)", min_value=0.0, value=120.0)
    valor_flete = col3.number_input("Valor del flete (COP)", min_value=0.0, value=850_000.0)

    dias_estimados = col1.number_input("Días de tránsito estimados", min_value=1, value=2)
    ciudad_origen = col2.text_input("Ciudad origen", "Bogota")
    ciudad_destino = col3.text_input("Ciudad destino", "Medellin")

    tipo_servicio = st.selectbox(
        "Tipo de servicio", ["estandar", "express", "refrigerado", "carga_pesada"]
    )

    enviado = st.form_submit_button("Predecir riesgo")


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
if enviado:
    try:
        from src.predict import predecir_retraso

        features = {
            "distancia_km": distancia_km,
            "peso_kg": peso_kg,
            "valor_flete": valor_flete,
            "dias_transito_estimado": dias_estimados,
            "ciudad_origen": ciudad_origen,
            "ciudad_destino": ciudad_destino,
            "tipo_servicio": tipo_servicio,
        }
        resultado = predecir_retraso(features)
        prob = resultado["probabilidad_retraso"][0]
        riesgo = resultado["nivel_riesgo"][0]
        costo = costo_esperado_por_probabilidad(prob, valor_flete, dias_estimados)

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de retraso", f"{prob:.1%}")
        c2.metric("Nivel de riesgo", riesgo)
        c3.metric("Costo esperado", f"${costo:,.0f}")

    except FileNotFoundError:
        st.warning(
            "El modelo aún no ha sido entrenado. Entrene el modelo con "
            "`src.train_model.entrenar_modelo_retrasos` antes de predecir."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error al predecir: {exc}")
