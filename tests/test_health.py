from fastapi.testclient import TestClient

import app.app as app_module


class DummyModel:
    def predict(self, frame):
        return [0] * len(frame)


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(app_module, "load_feature_config", lambda: [])

    app = app_module.create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"


def test_ready_endpoint_when_model_loaded(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(app_module, "load_feature_config", lambda: [])

    app = app_module.create_app()
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


def test_ready_endpoint_when_model_missing(monkeypatch):
    def _raise_missing(_):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(app_module, "load_model", _raise_missing)
    monkeypatch.setattr(app_module, "load_feature_config", lambda: [])

    app = app_module.create_app()
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
