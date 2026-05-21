"""
Prefect flow for Phase 3 training pipeline.

Usage:
    python flows/prefect_training_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from prefect import flow, task
except Exception:  # pragma: no cover
    # Fallback mode: allow local execution even when Prefect is unavailable.
    def task(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    def flow(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.preprocessing import limpiar_dataco
from src.train_model import entrenar_modelo_delay_mvp
from src.utils import guardar_reporte_json


@task(name="build_clean_dataset")
def build_clean_dataset_task() -> tuple[int, int]:
    """Create/refresh clean dataset from raw DataCo source."""
    df = limpiar_dataco(guardar=True)
    return df.shape


@task(name="train_mvp_model")
def train_mvp_model_task() -> dict:
    """Train MVP models and return winner metadata."""
    df = limpiar_dataco(guardar=True)
    result = entrenar_modelo_delay_mvp(df, guardar=True)
    return {
        "winner": result["nombre_modelo"],
        "metrics": result["metricas"],
        "model_path": str(result["ruta_modelo"]),
        "metadata_path": str(result["ruta_columnas"]),
    }


@task(name="save_training_report")
def save_training_report_task(payload: dict) -> str:
    """Persist flow output into reports folder."""
    out = guardar_reporte_json(payload, "prefect_training_run")
    return str(out)


@flow(name="smartdelay-training-flow")
def smartdelay_training_flow() -> str:
    """Phase 3 orchestration: data prep + training + reporting."""
    rows, cols = build_clean_dataset_task()
    training = train_mvp_model_task()

    payload = {
        "phase": "Fase 3 - Pipeline de Entrenamiento",
        "dataset_shape": {"rows": rows, "cols": cols},
        "training": training,
    }

    return save_training_report_task(payload)


if __name__ == "__main__":
    report_path = smartdelay_training_flow()
    print(f"Flow completed. Report: {report_path}")
