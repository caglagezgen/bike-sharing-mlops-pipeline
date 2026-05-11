"""
Statistical drift detection utilities.
Provides KS test, Chi-squared test, and PSI computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ks_drift(ref: pd.Series, cur: pd.Series, threshold: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test for distributional shift."""
    stat, p = stats.ks_2samp(ref.dropna(), cur.dropna())
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < threshold),
    }


def chi2_drift(ref: pd.Series, cur: pd.Series, threshold: float = 0.05) -> dict:
    """Chi-squared test for categorical drift."""
    categories = set(ref.dropna().unique()) | set(cur.dropna().unique())
    ref_counts = {c: (ref == c).sum() for c in categories}
    cur_counts = {c: (cur == c).sum() for c in categories}
    observed = [cur_counts.get(c, 0) for c in categories]
    expected = [ref_counts.get(c, 0) for c in categories]
    if sum(expected) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}
    stat, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < threshold),
    }


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. PSI > 0.2 indicates significant drift."""
    breakpoints = np.percentile(ref.dropna(), np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref.dropna(), bins=breakpoints)
    cur_counts, _ = np.histogram(cur.dropna(), bins=breakpoints)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_report(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str] | None = None,
) -> dict:
    """Run a full drift report across numerical and categorical columns."""
    report: dict = {"numerical": {}, "categorical": {}}
    for col in num_cols:
        if col in ref.columns and col in cur.columns:
            report["numerical"][col] = {
                "ks": ks_drift(ref[col], cur[col]),
                "psi": round(psi(ref[col], cur[col]), 4),
            }
    for col in (cat_cols or []):
        if col in ref.columns and col in cur.columns:
            report["categorical"][col] = chi2_drift(ref[col], cur[col])
    return report
