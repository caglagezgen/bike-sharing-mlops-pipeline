"""
Drift detection entry point.

Compares a reference processed dataset against a current (new batch) dataset
using KS test, Chi-squared test, and PSI. Writes a drift report to
artifacts/drift_report.json and exits with code 1 if any drift is detected
(so CI can flag the result as a warning).

Usage:
    python monitoring/run_drift.py \
        --reference data/processed/train_processed.csv \
        --current   /tmp/current_processed.csv \
        [--fail-on-drift]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift import run_report  # noqa: E402

# Feature columns defined in feature_config.json
NUM_COLS = ["temp", "humidity", "windspeed", "year", "month", "hour", "dayofweek"]
CAT_COLS = ["season", "holiday", "weather"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature drift report")
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference (historical baseline) processed CSV",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current (new batch) processed CSV",
    )
    parser.add_argument(
        "--output",
        default="artifacts/drift_report.json",
        help="Path to write the drift report JSON",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 when the drift ratio threshold is exceeded",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.20,
        help="PSI value above which a numerical feature is considered drifted (default: 0.20)",
    )
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.05,
        help="p-value below which KS / Chi2 test signals drift (default: 0.05)",
    )
    parser.add_argument(
        "--min-drift-ratio",
        type=float,
        default=0.30,
        help=(
            "Minimum fraction of features that must drift before CT is triggered "
            "(default: 0.30 = 30%%). Set lower to be more sensitive."
        ),
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    cur_path = Path(args.current)
    out_path = Path(args.output)

    if not ref_path.exists():
        print(f"[drift] ERROR: reference file not found: {ref_path}", flush=True)
        sys.exit(2)
    if not cur_path.exists():
        print(f"[drift] ERROR: current file not found: {cur_path}", flush=True)
        sys.exit(2)

    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    print(f"[drift] Reference rows : {len(ref_df)}", flush=True)
    print(f"[drift] Current rows   : {len(cur_df)}", flush=True)

    # Keep only columns that exist in both DataFrames
    available_num = [c for c in NUM_COLS if c in ref_df.columns and c in cur_df.columns]
    available_cat = [c for c in CAT_COLS if c in ref_df.columns and c in cur_df.columns]

    report = run_report(ref_df, cur_df, num_cols=available_num, cat_cols=available_cat)

    # Summarise findings
    drifted_features: list[str] = []
    print("\n[drift] --- Numerical features ---", flush=True)
    for col, stats in report["numerical"].items():
        ks = stats["ks"]
        psi_val = stats["psi"]
        flag = "DRIFT" if ks["p_value"] < args.ks_threshold or psi_val > args.psi_threshold else "ok"
        print(
            f"  {col:15s}  KS p={ks['p_value']:.4f}  PSI={psi_val:.4f}  [{flag}]",
            flush=True,
        )
        if flag == "DRIFT":
            drifted_features.append(col)

    print("\n[drift] --- Categorical features ---", flush=True)
    for col, stats in report["categorical"].items():
        flag = "DRIFT" if stats["p_value"] < args.ks_threshold else "ok"
        print(
            f"  {col:15s}  Chi2 p={stats['p_value']:.4f}  [{flag}]",
            flush=True,
        )
        if flag == "DRIFT":
            drifted_features.append(col)

    total_features = len(available_num) + len(available_cat)
    drift_ratio = len(drifted_features) / max(total_features, 1)
    drift_triggered = drift_ratio >= args.min_drift_ratio

    report["summary"] = {
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "drift_ratio": round(drift_ratio, 3),
        "drift_triggered": drift_triggered,
        "thresholds": {
            "psi": args.psi_threshold,
            "ks_p_value": args.ks_threshold,
            "min_drift_ratio": args.min_drift_ratio,
        },
        "reference_rows": len(ref_df),
        "current_rows": len(cur_df),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[drift] Report written to {out_path}", flush=True)
    print(
        f"[drift] Thresholds — PSI>{args.psi_threshold}  KS-p<{args.ks_threshold}  "
        f"min-drift-ratio={args.min_drift_ratio}",
        flush=True,
    )
    print(
        f"[drift] Result — {len(drifted_features)}/{total_features} features drifted "
        f"(ratio={drift_ratio:.2f}, triggered={drift_triggered})",
        flush=True,
    )

    if drift_triggered:
        print(
            f"[drift] ACTION: drift ratio {drift_ratio:.2f} >= threshold {args.min_drift_ratio} "
            f"— retraining recommended. Drifted: {drifted_features}",
            flush=True,
        )
        if args.fail_on_drift:
            sys.exit(1)
    elif len(drifted_features) > 0:
        print(
            f"[drift] INFO: {len(drifted_features)} feature(s) drifted but ratio "
            f"{drift_ratio:.2f} < threshold {args.min_drift_ratio} — monitoring only, no CT triggered.",
            flush=True,
        )
    else:
        print("[drift] No significant drift detected.", flush=True)


if __name__ == "__main__":
    main()
