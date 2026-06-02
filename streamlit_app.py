"""Entrypoint para Streamlit Cloud.

Permite desplegar usando `streamlit_app.py` en la raiz del repositorio,
redirigiendo la ejecucion al script real ubicado en `app/streamlit_app.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy


APP_FILE = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(str(APP_FILE), run_name="__main__")
