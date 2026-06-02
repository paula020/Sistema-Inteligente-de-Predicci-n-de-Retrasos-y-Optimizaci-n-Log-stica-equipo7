"""
Monitoring utilities for SmartDelay AI 360.

Phase 5 support:
- Basic data quality KPIs
- Feature drift estimation using PSI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    missing_cells_pct: float
    duplicated_rows_pct: float
    target_rate: float | None


def build_quality_report(df: pd.DataFrame, target_col: str = "late_delivery_risk") -> dict[str, Any]:
    """Return basic quality metrics for a dataframe."""
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)
    missing_cells = int(df.isna().sum().sum())
    duplicated_rows = int(df.duplicated().sum())

    target_rate = None
    if target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        if target.notna().any():
            target_rate = float(target.mean())

    report = QualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_cells_pct=round(missing_cells / total_cells * 100, 4),
        duplicated_rows_pct=round(duplicated_rows / max(n_rows, 1) * 100, 4),
        target_rate=target_rate,
    )
    return report.__dict__


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) for one numeric feature."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(ref, quantiles))

    # Defensive fallback when data has too few unique values.
    if len(cut_points) < 3:
        cut_points = np.array([ref.min() - 1e-9, ref.max() + 1e-9])

    ref_hist, _ = np.histogram(ref, bins=cut_points)
    cur_hist, _ = np.histogram(cur, bins=cut_points)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(value)


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    psi_alert_threshold: float = 0.2,
) -> dict[str, Any]:
    """Build PSI drift report for numeric columns.

    PSI guide:
    - < 0.1: low drift
    - 0.1 to 0.2: moderate drift
    - >= 0.2: high drift
    """
    if numeric_columns is None:
        numeric_columns = [
            c
            for c in reference_df.columns
            if c in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[c])
        ]

    per_feature: dict[str, float] = {}
    high_drift: list[str] = []

    for col in numeric_columns:
        psi_value = round(_psi(reference_df[col], current_df[col]), 6)
        per_feature[col] = psi_value
        if psi_value >= psi_alert_threshold:
            high_drift.append(col)

    return {
        "n_features_checked": len(numeric_columns),
        "psi_alert_threshold": psi_alert_threshold,
        "high_drift_features": high_drift,
        "psi_by_feature": per_feature,
    }
