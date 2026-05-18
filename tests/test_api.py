import json

import numpy as np
from fastapi.testclient import TestClient

import app.app as app_module
from src.features import get_feature_columns


class DummyModel:
    def predict(self, frame):
        return np.zeros(len(frame))


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(app_module, "load_feature_config", get_feature_columns)

    app = app_module.create_app()

    payload = {
        "datetime": "2011-01-01 00:00:00",
        "season": 1,
        "holiday": 0,
        "weather": 1,
        "temp": 9.84,
        "humidity": 81,
        "windspeed": 0.0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 1
        assert "prediction_ids" in body
        assert len(body["prediction_ids"]) == 1
        assert "request_id" in body


def test_predict_writes_inference_log(monkeypatch, tmp_path):
    log_path = tmp_path / "inference.jsonl"
    monkeypatch.setenv("INFERENCE_LOG_PATH", str(log_path))
    monkeypatch.setenv("INFERENCE_LOG_ENABLED", "true")
    monkeypatch.setattr(app_module, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(app_module, "load_feature_config", get_feature_columns)

    app = app_module.create_app()
    payload = {
        "datetime": "2011-01-01 00:00:00",
        "season": 1,
        "holiday": 0,
        "weather": 1,
        "temp": 9.84,
        "humidity": 81,
        "windspeed": 0.0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    assert log_path.exists()
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert events
    assert events[0]["event"] == "prediction"


def test_feedback_writes_actual_log(monkeypatch, tmp_path):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("INFERENCE_LOG_PATH", str(log_path))
    monkeypatch.setenv("INFERENCE_LOG_ENABLED", "true")
    monkeypatch.setattr(app_module, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(app_module, "load_feature_config", get_feature_columns)

    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/feedback", json={"prediction_id": "pred-1", "actual": 42}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recorded"] == 1

    events = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert events
    assert events[0]["event"] == "actual"
