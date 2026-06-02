"""
SmartDelay AI 360 — Aplicación Streamlit (MVP)
==============================================

Demo ejecutivo para predecir retrasos logísticos a partir de los datos
de un envío. Tres secciones:

    1. Inicio       — presentación del producto y del problema.
    2. Dashboard    — KPIs e insights del dataset histórico (DataCo).
    3. Predicción   — formulario de un envío + resultado del modelo.

Ejecución:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegura que el paquete `src` sea importable cuando Streamlit lanza el script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import (
    UMBRAL_ALTO,
    UMBRAL_BAJO,
    clasificar_riesgo,
    limpiar_cache_modelos,
    predecir_desde_formulario,
)


# ============================================================================
# Configuración global de la página
# ============================================================================
st.set_page_config(
    page_title="SmartDelay AI 360",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Paleta y estilos
# ============================================================================
COLORES_RIESGO = {
    "BAJO": "#10b981",   # verde esmeralda
    "MEDIO": "#f59e0b",  # ámbar
    "ALTO": "#ef4444",   # rojo
}

CSS = """
<style>
/* Tipografía y espaciado general */
.main .block-container {padding-top: 2rem; padding-bottom: 3rem;}

/* Header del producto */
.brand-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.brand-subtitle {
    font-size: 1.1rem;
    color: #475569;
    margin-bottom: 1.5rem;
}

/* Tarjetas KPI */
.kpi-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    height: 100%;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-card .kpi-value {
    font-size: 2.0rem;
    font-weight: 700;
    color: #0f172a;
}
.kpi-card .kpi-delta {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.4rem;
}

/* Tarjeta de resultado de predicción */
.result-card {
    border-radius: 16px;
    padding: 28px;
    color: white;
    margin-top: 1rem;
}
.result-card h2 {
    color: white;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}
.result-card .prob {
    font-size: 3.0rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-card .reco {
    font-size: 1.0rem;
    background: rgba(255, 255, 255, 0.18);
    padding: 14px;
    border-radius: 10px;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {background: #0f172a;}
section[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {color: white !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# Carga del dataset histórico (cacheada)
# ============================================================================
@st.cache_data(show_spinner=False)
def cargar_dataset_historico() -> pd.DataFrame | None:
    """Carga el parquet limpio de DataCo si está disponible."""
    ruta = PROJECT_ROOT / "data" / "processed" / "dataco_clean.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    return None


# ============================================================================
# Helper para renderizar KPIs estilizados
# ============================================================================
def kpi(col, etiqueta: str, valor: str, delta: str = "") -> None:
    """Renderiza una tarjeta KPI con estilo ejecutivo."""
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{etiqueta}</div>
            <div class="kpi-value">{valor}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Sidebar — Navegación
# ============================================================================
with st.sidebar:
    st.markdown("# SmartDelay AI 360")
    st.caption("MVP · Predicción de retrasos")
    st.divider()

    seccion = st.radio(
        "Navegación",
        options=["🏠 Inicio", "📊 Dashboard", "🤖 Predicción"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Modelo activo: `modelo_mejor.joblib`")

    if st.button("🔄 Recargar modelo", use_container_width=True):
        limpiar_cache_modelos()
        cargar_dataset_historico.clear()
        st.success("Caché limpiado. Recarga la página.")

    st.markdown("---")
    st.caption("© 2026 SmartDelay AI 360")


# ============================================================================
# Sección: INICIO
# ============================================================================
def render_inicio() -> None:
    st.markdown(
        '<div class="brand-title">SmartDelay AI 360</div>'
        '<div class="brand-subtitle">Predicción inteligente de retrasos logísticos</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown(
            """
            ### El problema

            En la cadena logística, **los retrasos en las entregas son la principal causa de
            insatisfacción del cliente, sobrecostos operativos y pérdida de competitividad**.
            Una sola guía retrasada puede generar penalizaciones, reprocesos y devoluciones
            por cientos de miles de pesos.

            ### La solución

            **SmartDelay AI 360** es una plataforma que utiliza algoritmos de Machine Learning
            entrenados sobre el histórico de envíos (dataset *DataCo Smart Supply Chain*)
            para **predecir, antes del despacho, la probabilidad de que un envío se retrase**.

            ### Cómo funciona el modelo

            1. Procesa variables ex-ante: modo de envío, segmento de cliente, mercado, región,
               país, categoría del producto, ventas, cantidad, descuento, precio, días
               programados y temporalidad de la orden.
            2. Aplica *target encoding* con suavizado bayesiano para incorporar
               el historial de riesgo por ciudad, ruta y cliente.
            3. Evalúa la guía con un modelo ganador seleccionado entre **Logistic Regression,
               Random Forest, XGBoost y LightGBM** según el ROC-AUC.
            4. Entrega una probabilidad, un nivel de riesgo y una recomendación operativa.
            """
        )

    with col_b:
        st.info(
            "**Stack técnico**\n\n"
            "- Python · Pandas · Scikit-learn\n"
            "- XGBoost · LightGBM\n"
            "- MLflow tracking\n"
            "- Streamlit · Plotly\n"
            "- FastAPI · Docker · AWS\n"
            "- Claude vía Amazon Bedrock"
        )
        st.success(
            "**Variable objetivo**\n\n`Late_delivery_risk` (binaria: 0 = a tiempo, "
            "1 = retraso)"
        )


# ============================================================================
# Sección: DASHBOARD
# ============================================================================
def render_dashboard() -> None:
    st.markdown(
        '<div class="brand-title">Dashboard histórico</div>'
        '<div class="brand-subtitle">KPIs e insights del dataset DataCo Smart Supply Chain</div>',
        unsafe_allow_html=True,
    )

    df = cargar_dataset_historico()
    if df is None:
        st.warning(
            "No se encontró `data/processed/dataco_clean.parquet`. "
            "Ejecuta primero el notebook `01_eda_dataco.ipynb` para "
            "generar el dataset limpio."
        )
        return

    # ---- KPIs principales ---------------------------------------------------
    total = len(df)
    tasa = df["late_delivery_risk"].mean()
    retrasos = int(df["late_delivery_risk"].sum())
    a_tiempo = total - retrasos

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total de envíos", f"{total:,}")
    kpi(c2, "Retrasos registrados", f"{retrasos:,}", f"{tasa:.1%} del total")
    kpi(c3, "Entregas a tiempo", f"{a_tiempo:,}", f"{(1-tasa):.1%} del total")
    kpi(
        c4,
        "Modos de envío",
        f"{df['shipping_mode'].nunique() if 'shipping_mode' in df.columns else 0}",
        "categorías únicas",
    )

    st.markdown("")

    # ---- Distribución retraso vs sin retraso --------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        df_dist = pd.DataFrame(
            {
                "estado": ["A tiempo", "Retrasado"],
                "envios": [a_tiempo, retrasos],
            }
        )
        fig = px.pie(
            df_dist,
            values="envios",
            names="estado",
            color="estado",
            color_discrete_map={"A tiempo": "#10b981", "Retrasado": "#ef4444"},
            hole=0.5,
            title="Distribución de entregas",
        )
        fig.update_layout(showlegend=True, margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Distribución por modo de envío -------------------------------------
    with col_b:
        if "shipping_mode" in df.columns:
            agg = (
                df.groupby("shipping_mode")
                .agg(
                    envios=("late_delivery_risk", "count"),
                    tasa_retraso=("late_delivery_risk", "mean"),
                )
                .sort_values("tasa_retraso", ascending=False)
                .reset_index()
            )
            fig = px.bar(
                agg,
                x="shipping_mode",
                y="tasa_retraso",
                text=agg["tasa_retraso"].apply(lambda x: f"{x:.1%}"),
                color="tasa_retraso",
                color_continuous_scale="RdYlGn_r",
                title="Tasa de retraso por modo de envío",
                labels={"shipping_mode": "Modo de envío", "tasa_retraso": "% retraso"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Por mercado y región -----------------------------------------------
    col_c, col_d = st.columns(2)

    with col_c:
        if "market" in df.columns:
            agg_m = (
                df.groupby("market")
                .agg(
                    envios=("late_delivery_risk", "count"),
                    tasa_retraso=("late_delivery_risk", "mean"),
                )
                .reset_index()
                .sort_values("tasa_retraso", ascending=False)
            )
            fig = px.bar(
                agg_m,
                x="market",
                y="tasa_retraso",
                text=agg_m["tasa_retraso"].apply(lambda x: f"{x:.1%}"),
                color="tasa_retraso",
                color_continuous_scale="RdYlGn_r",
                title="Tasa de retraso por mercado",
                labels={"market": "Mercado", "tasa_retraso": "% retraso"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if "order_region" in df.columns:
            agg_r = (
                df.groupby("order_region")
                .agg(
                    envios=("late_delivery_risk", "count"),
                    tasa_retraso=("late_delivery_risk", "mean"),
                )
                .reset_index()
                .sort_values("tasa_retraso", ascending=False)
                .head(15)
            )
            fig = px.bar(
                agg_r,
                x="tasa_retraso",
                y="order_region",
                orientation="h",
                color="tasa_retraso",
                color_continuous_scale="RdYlGn_r",
                title="Top 15 regiones por tasa de retraso",
                labels={"order_region": "Región", "tasa_retraso": "% retraso"},
            )
            fig.update_layout(
                xaxis_tickformat=".0%",
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Sección: PREDICCIÓN
# ============================================================================
def _opciones_desde_df(df: pd.DataFrame | None, columna: str, fallback: list[str]) -> list[str]:
    """Devuelve las opciones únicas de una columna o un fallback razonable."""
    if df is not None and columna in df.columns:
        opciones = sorted([x for x in df[columna].dropna().unique().tolist() if str(x).strip()])
        if opciones:
            return opciones
    return fallback


def render_prediccion() -> None:
    st.markdown(
        '<div class="brand-title">Predicción de retraso</div>'
        '<div class="brand-subtitle">Diligencia los datos del envío para evaluar el riesgo</div>',
        unsafe_allow_html=True,
    )

    df_hist = cargar_dataset_historico()

    # Opciones dinámicas (del dataset si está, fallback en otro caso)
    opciones_shipping = _opciones_desde_df(
        df_hist, "shipping_mode",
        ["Standard Class", "First Class", "Second Class", "Same Day"],
    )
    opciones_segment = _opciones_desde_df(
        df_hist, "customer_segment", ["Consumer", "Corporate", "Home Office"]
    )
    opciones_market = _opciones_desde_df(
        df_hist, "market", ["LATAM", "Europe", "USCA", "Pacific Asia", "Africa"]
    )
    opciones_region = _opciones_desde_df(
        df_hist, "order_region",
        ["Central America", "South America", "Western Europe", "South Asia"],
    )
    opciones_country = _opciones_desde_df(
        df_hist, "order_country",
        ["Estados Unidos", "México", "Brasil", "Colombia", "Francia", "España"],
    )
    opciones_category = _opciones_desde_df(
        df_hist, "category_name",
        ["Electronics", "Apparel", "Cleats", "Sporting Goods", "Cardio Equipment"],
    )

    # ------------------------------------------------------------------------
    # Formulario
    # ------------------------------------------------------------------------
    with st.form("form_envio", clear_on_submit=False):
        st.subheader("Datos del envío")

        col1, col2, col3 = st.columns(3)
        shipping_mode = col1.selectbox("Modo de envío", opciones_shipping)
        customer_segment = col2.selectbox("Segmento de cliente", opciones_segment)
        market = col3.selectbox("Mercado", opciones_market)

        col4, col5, col6 = st.columns(3)
        order_region = col4.selectbox("Región del pedido", opciones_region)
        order_country = col5.selectbox("País del pedido", opciones_country)
        category_name = col6.selectbox("Categoría del producto", opciones_category)

        st.subheader("Datos económicos y operativos")
        col7, col8, col9, col10 = st.columns(4)
        sales = col7.number_input("Ventas (Sales)", min_value=0.0, value=250.0, step=10.0)
        order_item_quantity = col8.number_input(
            "Cantidad (Order Item Quantity)", min_value=1, value=1, step=1
        )
        order_item_discount = col9.number_input(
            "Descuento (Order Item Discount)", min_value=0.0, value=10.0, step=1.0
        )
        order_item_product_price = col10.number_input(
            "Precio del producto", min_value=0.0, value=200.0, step=10.0
        )

        col11, col12 = st.columns(2)
        days_for_shipment_scheduled = col11.number_input(
            "Días de envío programados", min_value=0, max_value=20, value=4, step=1
        )
        order_date = col12.date_input(
            "Fecha del pedido",
            value=dt.date.today(),
            min_value=dt.date(2015, 1, 1),
            max_value=dt.date(2030, 12, 31),
        )

        st.caption(
            "ℹ️ El modelo deriva automáticamente *Order Month*, *Order Day of Week* y otras "
            "variables temporales a partir de la fecha del pedido."
        )

        enviado = st.form_submit_button(
            "Evaluar riesgo de retraso", type="primary", use_container_width=True
        )

    if not enviado:
        st.info(
            "Completa el formulario y presiona **Evaluar riesgo de retraso** para "
            "consultar al modelo entrenado."
        )
        return

    # ------------------------------------------------------------------------
    # Llamada al modelo
    # ------------------------------------------------------------------------
    # Construimos el dict con las 13 features EXACTAS del MVP.
    # `order_date` se convierte en `order_month` + `order_dayofweek` para
    # respetar la lista guardada en `models/feature_columns.json`.
    fecha = pd.Timestamp(order_date)
    form = {
        # --- Categóricas (6) ---
        "shipping_mode": shipping_mode,
        "customer_segment": customer_segment,
        "market": market,
        "order_region": order_region,
        "order_country": order_country,
        "category_name": category_name,
        # --- Numéricas (7) ---
        "sales": float(sales),
        "order_item_quantity": int(order_item_quantity),
        "order_item_discount": float(order_item_discount),
        "order_item_product_price": float(order_item_product_price),
        "days_for_shipment_scheduled": int(days_for_shipment_scheduled),
        "order_month": int(fecha.month),
        "order_dayofweek": int(fecha.dayofweek),
    }

    with st.spinner("Consultando el modelo..."):
        resultado = predecir_desde_formulario(form)

    if not resultado.get("ok"):
        st.error(resultado.get("error", "Error desconocido."))
        with st.expander("Detalle técnico"):
            st.code(resultado.get("detalle", "—"))
        return

    # ------------------------------------------------------------------------
    # Visualización del resultado
    # ------------------------------------------------------------------------
    prob = resultado["probabilidad"]
    nivel = resultado["nivel_riesgo"]
    color = COLORES_RIESGO[nivel]
    pct = resultado["probabilidad_pct"]

    st.markdown(
        f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%);">
            <h2>{resultado['etiqueta']}</h2>
            <div class="prob">{pct:.1f}%</div>
            <div>Nivel de riesgo: <b>{nivel}</b></div>
            <div class="reco"><b>Recomendación operativa:</b><br>{resultado['recomendacion']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ---- Detalles de soporte ------------------------------------------------
    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Gauge / barra de probabilidad
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, UMBRAL_BAJO * 100], "color": "#dcfce7"},
                        {"range": [UMBRAL_BAJO * 100, UMBRAL_ALTO * 100], "color": "#fef3c7"},
                        {"range": [UMBRAL_ALTO * 100, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 3},
                        "thickness": 0.8,
                        "value": pct,
                    },
                },
                title={"text": "Probabilidad de retraso", "font": {"size": 16}},
            )
        )
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Resumen del modelo")
        st.metric("Modelo utilizado", resultado["modelo_usado"])
        st.metric("Features procesadas", resultado["features_aplicadas"])
        st.metric("Probabilidad", f"{pct:.2f}%")
        st.metric("Nivel de riesgo", nivel)

    # ---- Tabla con los datos enviados ---------------------------------------
    with st.expander("Ver datos del envío evaluado"):
        # Mostramos un dict ampliado (incluye la fecha humanizada como contexto)
        form_display = {
            **form,
            "_fecha_pedido": fecha.strftime("%Y-%m-%d (%A)"),
        }
        df_form = pd.DataFrame([form_display]).T.rename(columns={0: "valor"})
        st.dataframe(df_form, use_container_width=True)


# ============================================================================
# Router principal
# ============================================================================
if seccion.endswith("Inicio"):
    render_inicio()
elif seccion.endswith("Dashboard"):
    render_dashboard()
elif seccion.endswith("Predicción"):
    render_prediccion()
