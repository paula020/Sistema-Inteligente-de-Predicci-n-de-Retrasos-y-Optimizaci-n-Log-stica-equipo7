"""
CLI script for Phase 5 monitoring.

Usage:
    python scripts/run_monitoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring import build_drift_report, build_quality_report
from src.preprocessing import limpiar_dataco
from src.utils import configurar_logger, guardar_reporte_json


def _load_dataset() -> pd.DataFrame:
    processed_path = ROOT / "data" / "processed" / "dataco_clean.parquet"
    if processed_path.exists():
        logger.info(f"Loading processed dataset: {processed_path}")
        return pd.read_parquet(processed_path)

    logger.warning("Processed dataset not found. Building from raw...")
    return limpiar_dataco(guardar=True)


def _split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col = "order_date_dateorders"
    if date_col in df.columns:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.sort_values(date_col)
        split = int(len(tmp) * 0.7)
        return tmp.iloc[:split].copy(), tmp.iloc[split:].copy()

    split = int(len(df) * 0.7)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def main() -> int:
    configurar_logger("monitoring")

    df = _load_dataset()
    reference_df, current_df = _split_reference_current(df)

    quality = build_quality_report(df)
    drift = build_drift_report(reference_df, current_df)

    report = {
        "phase": "Fase 5 - Monitoreo",
        "quality": quality,
        "drift": drift,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
    }

    path = guardar_reporte_json(report, "monitoring_report")
    logger.success(f"Monitoring report saved: {path}")

    if drift["high_drift_features"]:
        logger.warning(
            "High drift detected in features: "
            + ", ".join(drift["high_drift_features"])
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
