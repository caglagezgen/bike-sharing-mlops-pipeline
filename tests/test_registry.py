from src.models.registry import ModelRegistry


def test_register_and_promote(tmp_path):
    registry_path = tmp_path / "model_registry.json"
    registry = ModelRegistry(registry_path=registry_path)

    registry.register(
        version="0.1.0",
        model_path="artifacts/model/model.joblib",
        metrics={"val_rmsle": 0.5},
        params={"n_estimators": 100},
        feature_columns=["temp"],
        dataset_meta={"raw_sha256": "abc"},
        stage="staging",
    )

    assert registry.current("staging") == "0.1.0"
    registry.promote("0.1.0", stage="production")
    assert registry.current("production") == "0.1.0"

    record = registry.get("0.1.0")
    assert record["version"] == "0.1.0"
