"""
Systematic hyperparameter search for the XGBoost bike-sharing demand model.

Performs a grid search over:
  - learning_rate  × n_estimators  (the core lr/shrinkage trade-off)
  - max_depth                       (tree complexity)

subsample=0.9 and colsample_bytree=0.9 are held fixed — prior experiments
(Notebook 03) established these as the best stochastic sub-sampling values.

Results are saved to artifacts/hyperparameter_search_results.json so they
can be referenced in the assessment report and are reproducible without
re-running the notebook.

Usage:
    python scripts/hyperparameter_search.py
    python scripts/hyperparameter_search.py --data data/processed/train_processed.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, PROCESSED_DIR  # noqa: E402
from src.features.engineering import get_feature_columns  # noqa: E402

# ── Search space ──────────────────────────────────────────────────────────────
# The learning_rate / n_estimators relationship is the primary axis of the
# search. Lower learning rates require proportionally more trees to reach the
# same training loss but generalise better (each step is smaller, less prone
# to overfitting). The pairs below sample this trade-off curve:
#   lr=0.20 → converges fast, fewer trees needed
#   lr=0.10 → standard default in most XGBoost tutorials
#   lr=0.05 → half the default; roughly doubles the optimal tree count
#   lr=0.02 → aggressive shrinkage; needs many more trees, diminishing return
SEARCH_GRID = {
    "learning_rate":    [0.02, 0.05, 0.10, 0.20],
    "n_estimators":     [300, 600, 900, 1300],
    "max_depth":        [4, 5, 6],
}

# Fixed values — held constant to keep the search tractable.
FIXED_PARAMS = {
    "subsample":        0.9,
    "colsample_bytree": 0.9,
    "objective":        "reg:squarederror",
    "n_jobs":           4,
    "random_state":     42,
}

VAL_RATIO = 0.20  # chronological 80/20 split — matches the production pipeline


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def run_search(data_path: Path) -> dict:
    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns()

    x = df[feature_cols].astype(float)
    y = df["count"].astype(float).to_numpy()

    split_idx = int(len(df) * (1 - VAL_RATIO))
    x_train, x_val = x.iloc[:split_idx].values, x.iloc[split_idx:].values
    y_train, y_val = y[:split_idx], y[split_idx:]
    y_train_log = np.log1p(y_train)

    keys = list(SEARCH_GRID.keys())
    combos = list(product(*SEARCH_GRID.values()))
    total = len(combos)
    print(f"Search space: {total} combinations")
    print(f"Grid: {SEARCH_GRID}")
    print(f"Fixed: {FIXED_PARAMS}\n")

    trials: list[dict] = []
    search_start = time.time()

    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        model = xgb.XGBRegressor(**params, **FIXED_PARAMS, verbosity=0)

        t0 = time.time()
        model.fit(x_train, y_train_log, verbose=False)
        duration = round(time.time() - t0, 3)

        preds = np.expm1(model.predict(x_val))
        trial: dict = {
            **params,
            **{k: v for k, v in FIXED_PARAMS.items() if k not in ("objective", "n_jobs", "random_state")},
            "val_rmsle":  round(rmsle(y_val, preds), 6),
            "val_mae":    round(mae(y_val, preds), 4),
            "val_r2":     round(r2(y_val, preds), 6),
            "fit_time_s": duration,
        }
        trials.append(trial)

        marker = " ◀ best so far" if trial["val_rmsle"] == min(t["val_rmsle"] for t in trials) else ""
        print(
            f"[{i:>3}/{total}] lr={params['learning_rate']:.2f} "
            f"n={params['n_estimators']:>4} depth={params['max_depth']}  "
            f"RMSLE={trial['val_rmsle']:.4f}  R²={trial['val_r2']:.4f}  "
            f"({duration:.2f}s){marker}"
        )

    total_time = round(time.time() - search_start, 1)
    trials_sorted = sorted(trials, key=lambda t: t["val_rmsle"])
    best = trials_sorted[0]

    print(f"\n{'='*65}")
    print(f"Search complete in {total_time}s")
    print(f"Best params  : lr={best['learning_rate']}, n_estimators={best['n_estimators']}, max_depth={best['max_depth']}")
    print(f"Best RMSLE   : {best['val_rmsle']:.4f}")
    print(f"Best R²      : {best['val_r2']:.4f}")
    print(f"{'='*65}")

    result = {
        "search_completed_at": datetime.utcnow().isoformat() + "Z",
        "total_trials": total,
        "total_search_time_s": total_time,
        "data_path": str(data_path),
        "val_ratio": VAL_RATIO,
        "search_grid": SEARCH_GRID,
        "fixed_params": {k: v for k, v in FIXED_PARAMS.items() if k not in ("objective", "n_jobs", "random_state")},
        "best_params": {
            "learning_rate":    best["learning_rate"],
            "n_estimators":     best["n_estimators"],
            "max_depth":        best["max_depth"],
            "subsample":        best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
        },
        "best_metrics": {
            "val_rmsle": best["val_rmsle"],
            "val_mae":   best["val_mae"],
            "val_r2":    best["val_r2"],
        },
        "all_trials": trials_sorted,
    }

    out_path = ARTIFACTS_DIR / "hyperparameter_search_results.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost hyperparameter grid search")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DIR / "train_processed.csv",
        help="Path to processed training CSV",
    )
    args = parser.parse_args()
    run_search(args.data)


if __name__ == "__main__":
    main()
