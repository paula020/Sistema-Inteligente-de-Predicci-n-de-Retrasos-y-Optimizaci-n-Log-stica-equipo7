"""
Página: Forecast de ventas.

Permite proyectar la demanda futura a partir de una serie histórica
de guías o ingresos, usando Prophet (o baseline si no está disponible).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sales_forecast import (
    PROPHET_DISPONIBLE,
    forecast_baseline,
    forecast_prophet,
    preparar_serie_temporal,
)


st.set_page_config(page_title="Forecast de Ventas", page_icon="📈", layout="wide")
st.title("Forecast de Ventas")
st.caption("Proyección de demanda futura por día.")


archivo = st.file_uploader(
    "Sube un CSV con columnas: fecha_despacho, valor_flete",
    type=["csv"],
)
horizonte = st.slider("Horizonte de pronóstico (días)", 7, 180, 60)
modelo = st.radio(
    "Modelo",
    ["Prophet", "Baseline (media móvil)"],
    index=0 if PROPHET_DISPONIBLE else 1,
    horizontal=True,
)


if archivo is not None:
    df = pd.read_csv(archivo)
    serie = preparar_serie_temporal(df)

    if modelo == "Prophet" and PROPHET_DISPONIBLE:
        forecast = forecast_prophet(serie, horizonte_dias=horizonte)
    else:
        forecast = forecast_baseline(serie, horizonte_dias=horizonte)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie["ds"], y=serie["y"], name="Histórico"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            name="IC superior",
            line={"dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            name="IC inferior",
            line={"dash": "dot"},
        )
    )
    fig.update_layout(title="Proyección de ventas", xaxis_title="Fecha", yaxis_title="Valor")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast, use_container_width=True)
else:
    st.info("Carga un dataset histórico para generar el forecast.")
