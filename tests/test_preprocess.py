"""
Tests for src/preprocess.py
Covers: column dropping, time-feature enrichment, output CSV creation,
and feature_config.json artifact writing.
"""
import json

import pandas as pd

from src.data.preprocess import build_data_profile, preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_csv(tmp_path, rows=5):
    """Write a minimal raw CSV that mimics the Kaggle bike-sharing dataset."""
    records = []
    for i in range(rows):
        records.append({
            "datetime":   f"2011-01-01 {i:02d}:00:00",
            "season":     1,
            "holiday":    0,
            "weather":    1,
            "temp":       9.84,
            "atemp":      14.395,
            "humidity":   81,
            "windspeed":  0.0,
            "casual":     3,
            "registered": 13,
            "count":      16,
        })
    path = tmp_path / "train.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_preprocess_drops_casual_and_registered(tmp_path):
    """casual and registered columns must be removed from the output."""
    raw = _make_raw_csv(tmp_path)
    out = tmp_path / "processed.csv"
    preprocess(raw, out)

    result = pd.read_csv(out)
    assert "casual" not in result.columns
    assert "registered" not in result.columns


def test_preprocess_adds_time_features(tmp_path):
    """year, month, hour, dayofweek must be present in processed output."""
    raw = _make_raw_csv(tmp_path)
    out = tmp_path / "processed.csv"
    preprocess(raw, out)

    result = pd.read_csv(out)
    for col in ("year", "month", "hour", "dayofweek"):
        assert col in result.columns, f"Missing time feature: {col}"


def test_preprocess_writes_output_csv(tmp_path):
    """Output CSV must be created and non-empty."""
    raw = _make_raw_csv(tmp_path)
    out = tmp_path / "processed.csv"
    preprocess(raw, out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert len(pd.read_csv(out)) == 5


def test_preprocess_writes_feature_config(tmp_path, monkeypatch):
    """ARTIFACTS_DIR/feature_config.json must be written with expected keys."""
    import src.config as config_mod
    import src.data.preprocess as preprocess_mod

    # Redirect ARTIFACTS_DIR to tmp_path so we don't pollute the real workspace
    monkeypatch.setattr(preprocess_mod, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "ARTIFACTS_DIR", tmp_path)

    raw = _make_raw_csv(tmp_path)
    out = tmp_path / "processed.csv"
    preprocess(raw, out)

    config_path = tmp_path / "feature_config.json"
    assert config_path.exists(), "feature_config.json was not written"
    config = json.loads(config_path.read_text())
    assert "feature_columns" in config
    assert "target" in config
    assert len(config["feature_columns"]) > 0


def test_build_data_profile_structure():
    """build_data_profile should return the expected schema."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    profile = build_data_profile(frame)

    assert "row_count" in profile
    assert "column_count" in profile
    assert "columns" in profile
    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    # Numeric column must have descriptive stats
    assert "mean" in profile["columns"]["a"]
    # Non-numeric column must not error and must have dtype
    assert "dtype" in profile["columns"]["b"]
