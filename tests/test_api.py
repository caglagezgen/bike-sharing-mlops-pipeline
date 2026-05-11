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
