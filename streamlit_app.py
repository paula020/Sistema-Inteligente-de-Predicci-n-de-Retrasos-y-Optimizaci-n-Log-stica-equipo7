"""Entrypoint para Streamlit Cloud.

Intenta ejecutar la app completa en app/streamlit_app.py.
Si faltan dependencias en el entorno de cloud, muestra un fallback ligero
para mantener la app desplegada y operativa.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import traceback


APP_FILE = Path(__file__).resolve().parent / "app" / "streamlit_app.py"


def _run_fallback(error: Exception) -> None:
	import streamlit as st

	st.set_page_config(page_title="SmartDelay AI 360", page_icon="🚚", layout="wide")
	st.title("SmartDelay AI 360")
	st.warning("La version completa no pudo cargarse en este entorno.")
	st.caption(f"Detalle tecnico: {type(error).__name__}: {error}")

	st.markdown("### Estimador rapido de riesgo")
	col1, col2, col3 = st.columns(3)
	with col1:
		distancia_km = st.number_input("Distancia (km)", min_value=0.0, value=200.0)
	with col2:
		peso_kg = st.number_input("Peso (kg)", min_value=0.0, value=50.0)
	with col3:
		dias_estimados = st.number_input("Dias estimados", min_value=0, value=2)

	score = 0
	if distancia_km > 500:
		score += 1
	if peso_kg > 150:
		score += 1
	if dias_estimados > 3:
		score += 1

	if score == 0:
		nivel = "BAJO"
	elif score == 1:
		nivel = "MEDIO"
	else:
		nivel = "ALTO"

	st.metric("Riesgo estimado", nivel)
	st.info(
		"Este modo es de contingencia para despliegue cloud. "
		"La logica ML completa permanece disponible en entornos con dependencias completas."
	)

	with st.expander("Ver stack trace"):
		st.code("".join(traceback.format_exception(error)), language="text")


try:
	runpy.run_path(str(APP_FILE), run_name="__main__")
except Exception as exc:  # pragma: no cover - solo para despliegue cloud
	_run_fallback(exc)
