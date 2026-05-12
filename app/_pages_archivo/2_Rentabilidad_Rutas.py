"""
Página: Rentabilidad de rutas.

Carga datos procesados, calcula costos directos/indirectos, margen y ROI,
y muestra el ranking de rutas más y menos rentables.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.route_profitability import (
    calcular_rentabilidad,
    ranking_rutas,
    rutas_no_rentables,
)


st.set_page_config(page_title="Rentabilidad de Rutas", page_icon="💰", layout="wide")
st.title("Rentabilidad de Rutas")
st.caption("Costos directos, indirectos, margen y ROI por ruta.")


archivo = st.file_uploader(
    "Sube un CSV/Parquet con columnas: ruta, valor_flete, distancia_km, dias_transito",
    type=["csv", "parquet"],
)

if archivo is not None:
    df = (
        pd.read_csv(archivo)
        if archivo.name.endswith(".csv")
        else pd.read_parquet(archivo)
    )
    st.write(f"Registros cargados: **{len(df):,}**")

    df_rent = calcular_rentabilidad(df)

    tab1, tab2 = st.tabs(["Top rutas rentables", "Rutas no rentables"])

    with tab1:
        st.subheader("Top 20 rutas por margen total")
        st.dataframe(ranking_rutas(df_rent, top_n=20), use_container_width=True)

    with tab2:
        st.subheader("Rutas con margen negativo")
        st.dataframe(rutas_no_rentables(df_rent), use_container_width=True)
else:
    st.info("Carga un dataset para iniciar el análisis.")
