"""
Página: Recomendaciones IA (Claude vía Bedrock).

Genera diagnósticos y recomendaciones ejecutivas a partir
de KPIs ingresados manualmente o calculados del dataset.
"""

from __future__ import annotations

import streamlit as st

from src.bedrock_client import get_bedrock_client
from src.config import settings


st.set_page_config(page_title="Recomendaciones IA", page_icon="🤖", layout="wide")
st.title("Recomendaciones Ejecutivas con IA")
st.caption(f"Modelo: `{settings.bedrock_model_id}`")


with st.form("form_kpis"):
    col1, col2 = st.columns(2)
    tasa_retraso = col1.slider("Tasa de retraso actual", 0.0, 1.0, 0.18, 0.01)
    margen = col2.slider("Margen promedio (%)", 0.0, 1.0, 0.12, 0.01)
    rutas_no_rentables = col1.number_input("Rutas no rentables", 0, 500, 7)
    impacto = col2.number_input("Impacto financiero mensual (COP)", 0, 10_000_000_000, 145_000_000)
    top_ciudades = st.text_input("Top ciudades (separadas por coma)", "Bogota, Medellin, Cali")

    enviado = st.form_submit_button("Generar recomendación")


if enviado:
    contexto = {
        "tasa_retraso_actual": tasa_retraso,
        "margen_promedio_pct": margen,
        "rutas_no_rentables": int(rutas_no_rentables),
        "top_ciudades": [c.strip() for c in top_ciudades.split(",") if c.strip()],
        "impacto_financiero_mensual": float(impacto),
    }
    try:
        with st.spinner("Consultando a Claude vía Amazon Bedrock..."):
            cliente = get_bedrock_client()
            respuesta = cliente.recomendacion_ejecutiva(contexto)
        st.success("Recomendación generada")
        st.markdown(respuesta)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No fue posible consultar Bedrock: {exc}")
        st.info(
            "Verifica las credenciales AWS y que el modelo de Bedrock "
            "esté habilitado en tu cuenta y región."
        )
