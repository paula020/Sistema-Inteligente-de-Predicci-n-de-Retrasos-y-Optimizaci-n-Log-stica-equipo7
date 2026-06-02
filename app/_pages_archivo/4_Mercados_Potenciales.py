"""
Página: Mercados potenciales.

Calcula un score por ciudad combinando demanda, rentabilidad,
crecimiento y competencia para priorizar expansión comercial.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.market_scoring import calcular_score_mercado, top_mercados_potenciales


st.set_page_config(page_title="Mercados Potenciales", page_icon="🌎", layout="wide")
st.title("Mercados Potenciales")
st.caption("Scoring compuesto de ciudades para priorizar expansión.")


archivo = st.file_uploader(
    "Sube un CSV con columnas: ciudad_destino, guias, margen_pct, "
    "crecimiento_pct (opcional), competidores (opcional)",
    type=["csv"],
)

if archivo is not None:
    df = pd.read_csv(archivo)
    df_score = calcular_score_mercado(df)

    st.subheader("Top mercados potenciales")
    st.dataframe(top_mercados_potenciales(df_score, top_n=15), use_container_width=True)

    fig = px.bar(
        df_score.head(20),
        x="ciudad_destino",
        y="score_mercado",
        color="categoria",
        title="Score de atractivo por ciudad",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Carga un dataset agregado por ciudad para iniciar el análisis.")
