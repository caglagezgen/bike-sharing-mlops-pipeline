"""
Tests for src/models/train.py
Covers: rmsle metric, train_model integration (with mocked heavy deps),
and that all expected metric keys are returned.
"""
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# rmsle unit tests (pure function — no mocking needed)
# ---------------------------------------------------------------------------
from src.models.train import rmsle


def test_rmsle_perfect_predictions():
    """rmsle must be 0 when predictions exactly match actuals."""
    y = np.array([1.0, 10.0, 100.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_positive_when_error():
    """rmsle must be strictly positive when predictions differ from actuals."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    assert rmsle(y_true, y_pred) > 0


def test_rmsle_penalises_under_prediction_more():
    """
    RMSLE penalises under-prediction more than over-prediction of the same
    magnitude. Under-predicting by k gives a larger log ratio than
    over-predicting by k (for k > 0, y > 0).
    """
    y_true = np.array([100.0])
    over  = np.array([110.0])   # 10% over
    under = np.array([90.0])    # 10% under
    # Under-prediction → smaller log1p(pred) → larger squared difference
    assert rmsle(y_true, under) > rmsle(y_true, over)


# ---------------------------------------------------------------------------
# train_model integration test (mocked I/O)
# ---------------------------------------------------------------------------

def _make_processed_csv(tmp_path, n_rows=200):
    """Build a minimal processed CSV that satisfies train_model requirements."""
    rng = np.random.default_rng(0)
    records = []
    for i in range(n_rows):
        records.append({
            "datetime":   pd.Timestamp("2011-01-01") + pd.Timedelta(hours=i),
            "season":     (i % 4) + 1,
            "holiday":    0,
            "weather":    1,
            "temp":       rng.uniform(5, 35),
            "humidity":   rng.uniform(30, 90),
            "windspeed":  rng.uniform(0, 30),
            "year":       2011,
            "month":      (i % 12) + 1,
            "hour":       i % 24,
            "dayofweek":  i % 7,
            "count":      max(1, int(rng.integers(5, 200))),
        })
    path = tmp_path / "train_processed.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_train_model_returns_required_keys(tmp_path, monkeypatch):
    """train_model must return a dict with all expected metric keys."""
    import src.models.train as train_mod

    # --- Mock EmissionsTracker to avoid codecarbon overhead ---
    class _FakeTracker:
        def start(self): pass
        def stop(self): return 1e-6

    monkeypatch.setattr(train_mod, "EmissionsTracker", lambda **kw: _FakeTracker())

    # --- Mock SHAP to avoid slow tree explainer ---
    import types

    fake_shap = types.SimpleNamespace(
        TreeExplainer=lambda model: types.SimpleNamespace(
            shap_values=lambda X: np.zeros((len(X), X.shape[1]))
        )
    )
    monkeypatch.setattr(train_mod, "shap", fake_shap)

    data_path = _make_processed_csv(tmp_path)
    result = train_mod.train_model(data_path, val_ratio=0.2, seed=42)

    required_keys = {
        "model", "feature_columns",
        "val_rmsle", "val_mae", "val_mape", "val_r2",
        "val_rows", "training_duration_s", "emissions_kg_co2",
    }
    assert required_keys.issubset(result.keys())


def test_train_model_metrics_are_valid(tmp_path, monkeypatch):
    """Trained model metrics must be finite and in plausible ranges."""
    import src.models.train as train_mod

    class _FakeTracker:
        def start(self): pass
        def stop(self): return 1e-6

    monkeypatch.setattr(train_mod, "EmissionsTracker", lambda **kw: _FakeTracker())

    import types
    fake_shap = types.SimpleNamespace(
        TreeExplainer=lambda model: types.SimpleNamespace(
            shap_values=lambda X: np.zeros((len(X), X.shape[1]))
        )
    )
    monkeypatch.setattr(train_mod, "shap", fake_shap)

    data_path = _make_processed_csv(tmp_path)
    result = train_mod.train_model(data_path, val_ratio=0.2, seed=42)

    assert np.isfinite(result["val_rmsle"]), "val_rmsle must be finite"
    assert result["val_rmsle"] >= 0,         "val_rmsle must be non-negative"
    assert np.isfinite(result["val_mae"]),    "val_mae must be finite"
    assert result["val_r2"] <= 1.0,            "val_r2 must not exceed 1"
    assert result["val_rows"] > 0,            "val_rows must be positive"
    assert result["training_duration_s"] >= 0
