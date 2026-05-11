"""
Dashboard ejecutivo de SmartDelay AI 360 (Streamlit).

Página principal que orquesta la navegación entre los distintos
módulos analíticos: KPIs, retrasos, rentabilidad, forecast,
mercados potenciales y recomendaciones IA.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings


# ----------------------------------------------------------------------------
# Configuración global de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartDelay AI 360 - Dashboard Ejecutivo",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("SmartDelay AI 360")
st.caption(
    "Plataforma Inteligente para Predicción de Retrasos, Rentabilidad "
    "Logística y Toma de Decisiones Empresariales."
)

st.markdown("---")


# ----------------------------------------------------------------------------
# KPIs principales (placeholder hasta conectar datos reales)
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasa de Retraso", "18.4%", "-2.1% vs mes anterior")
col2.metric("Margen Promedio", "12.7%", "+0.8%")
col3.metric("Rutas No Rentables", "7", "-3")
col4.metric("Impacto Financiero", "$145M COP", "-$22M")

st.markdown("---")


# ----------------------------------------------------------------------------
# Mapa de navegación
# ----------------------------------------------------------------------------
st.subheader("Módulos disponibles")

st.markdown(
    """
    Use el menú lateral para acceder a cada módulo analítico:

    - **Predicción de Retrasos** — Scoring de guías en tiempo real (XGBoost + MLflow)
    - **Rentabilidad de Rutas** — Costos directos, indirectos, margen y ROI
    - **Forecast de Ventas** — Proyección de demanda por ciudad, cliente y ruta
    - **Mercados Potenciales** — Scoring compuesto de ciudades para expansión
    - **Recomendaciones IA** — Análisis ejecutivo generado por Claude (Bedrock)
    """
)


# ----------------------------------------------------------------------------
# Sidebar: información del entorno
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Información del sistema")
    st.write(f"**Entorno:** `{settings.app_env}`")
    st.write(f"**API:** `{settings.api_host}:{settings.api_port}`")
    st.write(f"**MLflow:** `{settings.mlflow_tracking_uri}`")
    st.write(f"**Modelo LLM:** `{settings.bedrock_model_id}`")

    st.markdown("---")
    st.caption("© SmartDelay AI 360")
