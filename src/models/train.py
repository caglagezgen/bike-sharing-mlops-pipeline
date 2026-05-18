import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in CI/Docker
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from codecarbon import EmissionsTracker

from src.config import ARTIFACTS_DIR, MODEL_DIR, PROCESSED_DIR, ensure_dirs
from src.features.engineering import get_feature_columns
from src.models.registry import ModelRegistry
from src.models.version import ModelVersionManager

DATASET_META_PATH = ARTIFACTS_DIR / "dataset_meta.json"
DATA_PROFILE_PATH = ARTIFACTS_DIR / "data_profile.json"


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def load_feature_columns() -> list[str]:
    config_path = ARTIFACTS_DIR / "feature_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config.get("feature_columns", get_feature_columns())
    return get_feature_columns()


def train_model(data_path: Path, val_ratio: float, seed: int) -> dict:
    frame = pd.read_csv(data_path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.sort_values("datetime").reset_index(drop=True)

    feature_cols = load_feature_columns()
    x = frame[feature_cols].astype(float)
    y = frame["count"].astype(float).to_numpy()

    split_index = int(len(frame) * (1 - val_ratio))
    x_train, x_val = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Log-transform the target (log1p to handle count=0 safely).
    # Bike count is right-skewed with heavy-tailed outliers at peak hours.
    # Compressing the scale with log1p:
    #   1. Makes the residual distribution more symmetric, which aligns with
    #      the squared-error loss assumption of regression models.
    #   2. Directly optimises RMSLE — the Kaggle competition metric — because
    #      RMSLE itself measures error in log space. Training on log(count)
    #      with MSE loss is equivalent to minimising RMSLE on the original scale.
    #   3. Reduces the leverage of extreme peak-hour values that would otherwise
    #      dominate the gradient updates.
    y_train_log = np.log1p(y_train)

    # --- XGBoost hyperparameter rationale ---
    #
    # n_estimators=1300 + learning_rate=0.05:
    #   In gradient boosting, the learning rate shrinks each tree's contribution
    #   by a factor η (eta). Lower η requires more trees to reach the same
    #   training loss, but generalises better because each step is smaller and
    #   less prone to overfitting. The rule of thumb is: halving the learning
    #   rate roughly doubles the optimal number of trees. The baseline of
    #   lr=0.1 / n=300 was scaled to lr=0.05 / n=1300 following the tuning
    #   experiments from the reference analysis, which reduced MAPE from ~46%
    #   to ~24% on the validation set.
    #
    # max_depth=5:
    #   Controls the maximum depth of each tree. Depth 5 allows up to 32 leaf
    #   nodes, sufficient to capture multi-way feature interactions (e.g.
    #   hour × season × weather) without memorising training noise. Deeper
    #   trees (depth 6+) showed marginal accuracy gain but higher variance on
    #   the validation set, a sign of overfitting.
    #
    # subsample=0.9:
    #   At each boosting round, 90% of training rows are sampled without
    #   replacement (stochastic gradient boosting). This introduces variance
    #   reduction analogous to bagging — each tree sees a slightly different
    #   view of the data, reducing correlation between trees and improving
    #   generalisation. Setting this below 1.0 also speeds up training.
    #
    # colsample_bytree=0.9:
    #   Each tree is built using a random 90% subset of features. This further
    #   decorrelates trees (similar to the random subspace method in Random
    #   Forests) and prevents any single dominant feature from appearing in
    #   every tree, improving robustness.
    #
    # objective=reg:squarederror:
    #   Standard MSE loss. Combined with the log1p-transformed target, this
    #   effectively minimises RMSLE on the original count scale.
    model = xgb.XGBRegressor(
        n_estimators=1300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=seed,
    )
    # --- Sustainability: track CO₂ emissions and wall-clock time for model.fit() ---
    # EmissionsTracker measures CPU/GPU power draw and converts to kg CO₂ using
    # the energy grid mix of the host region (falls back to global average when
    # the region cannot be detected). This follows the methodology of
    # Lottick et al. (2019) and is the approach recommended by Patterson et al.
    # (2021) for reporting ML training emissions.
    # save_to_file=False keeps all output in-memory; we write our own artifact.
    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    tracker.start()
    t0 = time.time()
    model.fit(x_train, y_train_log)
    training_duration_s = round(time.time() - t0, 2)
    emissions_kg = tracker.stop()  # returns kg CO₂ equivalent
    if emissions_kg is None:
        emissions_kg = 0.0

    # --- SHAP feature importance ---
    # TreeExplainer is the exact (not approximate) explainer for XGBoost.
    # We compute SHAP values on the validation set so importance reflects
    # generalisation behaviour, not training memorisation.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_val)  # shape: (n_val, n_features)
    # Mean absolute SHAP value per feature — a model-agnostic importance measure
    shap_importance = dict(
        zip(feature_cols, np.abs(shap_values).mean(axis=0).tolist())
    )

    val_pred_log = model.predict(x_val)
    val_pred = np.expm1(val_pred_log)
    val_pred = np.maximum(val_pred, 0)

    # --- Evaluation metrics rationale ---
    #
    # RMSLE: Primary competition metric. Penalises under-prediction more than
    #   over-prediction on a logarithmic scale. For a service provider it is
    #   worse to have too few bikes (lost revenue) than too many (idle assets).
    #
    # MAE: Mean absolute error in original bike units. Business-interpretable —
    #   "on average, we are off by N bikes per hour". Robust to outliers
    #   compared to RMSE because it does not square the residuals.
    #
    # MAPE: Mean absolute percentage error. Allows comparison across different
    #   demand magnitudes (e.g., low-demand night hours vs peak commute hours).
    #   Undefined when actual count = 0, so we apply a zero-guard mask before
    #   computing. The article benchmark target is ~22–25%.
    #
    # R²: Coefficient of determination. Measures the proportion of demand
    #   variance explained by the model relative to a naive mean predictor.
    #   R² = 0.95 means the model explains 95% of variability — the benchmark
    #   achieved in the reference analysis after tuning.
    val_rmsle = rmsle(y_val, val_pred)
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    mask = y_val > 0
    val_mape = float(np.mean(np.abs((y_val[mask] - val_pred[mask]) / y_val[mask])) * 100) if mask.any() else 0.0
    ss_res = float(np.sum((y_val - val_pred) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    val_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "model": model,
        "feature_columns": feature_cols,
        "val_rmsle": val_rmsle,
        "val_mae": val_mae,
        "val_mape": val_mape,
        "val_r2": val_r2,
        "val_rows": len(y_val),
        "shap_importance": shap_importance,
        "shap_values": shap_values,
        "x_val": x_val,
        "training_duration_s": training_duration_s,
        "emissions_kg_co2": round(emissions_kg, 8),
    }


def persist_artifacts(result: dict) -> dict:
    ensure_dirs()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(result["model"], model_path)

    metrics = {
        "val_rmsle": result["val_rmsle"],
        "val_mae": result["val_mae"],
        "val_mape": result["val_mape"],
        "val_r2": result["val_r2"],
        "val_rows": result["val_rows"],
        "training_duration_s": result["training_duration_s"],
        "emissions_kg_co2": result["emissions_kg_co2"],
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    meta = {
        "feature_columns": result["feature_columns"],
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "val_rmsle": result["val_rmsle"],
        "val_mae": result["val_mae"],
        "val_mape": result["val_mape"],
        "val_r2": result["val_r2"],
        "training_duration_s": result["training_duration_s"],
        "emissions_kg_co2": result["emissions_kg_co2"],
    }
    meta_path = ARTIFACTS_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # --- SHAP importance JSON ---
    shap_json_path = ARTIFACTS_DIR / "shap_importance.json"
    sorted_importance = dict(
        sorted(result["shap_importance"].items(), key=lambda x: x[1], reverse=True)
    )
    shap_json_path.write_text(json.dumps(sorted_importance, indent=2))

    # --- SHAP bar chart ---
    shap_plot_path = ARTIFACTS_DIR / "shap_importance.png"
    features = list(sorted_importance.keys())
    values = list(sorted_importance.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(shap_plot_path, dpi=120)
    plt.close(fig)

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "meta_path": meta_path,
        "shap_json_path": shap_json_path,
        "shap_plot_path": shap_plot_path,
    }


def log_to_mlflow(result: dict, artifacts: dict) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("bike-sharing-demand")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "xgboost",
                "n_estimators": 1300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            }
        )
        mlflow.log_metric("val_rmsle", result["val_rmsle"])
        mlflow.log_metric("val_mae", result["val_mae"])
        mlflow.log_metric("val_mape", result["val_mape"])
        mlflow.log_metric("val_r2", result["val_r2"])
        mlflow.log_metric("training_duration_s", result["training_duration_s"])
        mlflow.log_metric("emissions_kg_co2", result["emissions_kg_co2"])
        mlflow.log_artifact(str(artifacts["metrics_path"]))
        mlflow.log_artifact(str(artifacts["meta_path"]))
        mlflow.log_artifact(str(artifacts["shap_json_path"]))
        mlflow.log_artifact(str(artifacts["shap_plot_path"]))
        if DATASET_META_PATH.exists():
            mlflow.log_artifact(str(DATASET_META_PATH))
        if DATA_PROFILE_PATH.exists():
            mlflow.log_artifact(str(DATA_PROFILE_PATH))

        try:
            mlflow.xgboost.log_model(result["model"], artifact_path="model")
        except Exception:
            mlflow.log_artifact(str(artifacts["model_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bike sharing model")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DIR / "train_processed.csv",
        help="Path to processed training data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage",
        default=os.getenv("MODEL_STAGE", "staging"),
        help="Registry stage for the trained model (staging|production)",
    )
    args = parser.parse_args()

    ensure_dirs()
    result = train_model(args.data, args.val_ratio, args.seed)
    artifacts = persist_artifacts(result)
    log_to_mlflow(result, artifacts)

    dataset_meta = {}
    if DATASET_META_PATH.exists():
        dataset_meta = json.loads(DATASET_META_PATH.read_text())

    version_manager = ModelVersionManager()
    version = version_manager.register(
        metrics={
            "val_rmsle": result["val_rmsle"],
            "val_mae": result["val_mae"],
            "val_mape": result["val_mape"],
            "val_r2": result["val_r2"],
            "val_rows": result["val_rows"],
        },
        params={
            "model": "xgboost",
            "n_estimators": 1300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        feature_columns=result["feature_columns"],
        dataset_meta=dataset_meta,
    )
    registry = ModelRegistry()
    registry.register(
        version=version,
        model_path=str(artifacts["model_path"]),
        metrics={
            "val_rmsle": result["val_rmsle"],
            "val_mae": result["val_mae"],
            "val_mape": result["val_mape"],
            "val_r2": result["val_r2"],
            "val_rows": result["val_rows"],
        },
        params={
            "model": "xgboost",
            "n_estimators": 1300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        feature_columns=result["feature_columns"],
        dataset_meta=dataset_meta,
        stage=args.stage,
    )
    print(f"Model version: {version}")


if __name__ == "__main__":
    main()
