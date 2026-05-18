from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from monitoring.inference_log import log_actual, log_prediction
from src.config import ARTIFACTS_DIR, MODEL_DIR
from src.features import build_feature_frame, get_feature_columns
from src.models.registry import ModelRegistry
from src.models.version import ModelVersionManager

# Raw input fields that must be present in every prediction request.
# These are the original Kaggle columns before any feature engineering.
REQUIRED_INPUTS = [
    "datetime",
    "season",
    "holiday",
    "weather",
    "temp",
    "humidity",
    "windspeed",
]


class PredictionRecord(BaseModel):
    datetime: str = Field(..., description="Timestamp for the observation")
    season: int
    holiday: int
    weather: int
    temp: float
    humidity: float
    windspeed: float


class PredictionRequest(BaseModel):
    records: list[PredictionRecord] | None = None


PredictionBody = PredictionRequest | PredictionRecord | list[PredictionRecord]


class FeedbackRecord(BaseModel):
    prediction_id: str = Field(..., description="Prediction identifier")
    actual: float = Field(..., description="Observed ground-truth value")


class FeedbackRequest(BaseModel):
    records: list[FeedbackRecord] | None = None


FeedbackBody = FeedbackRequest | FeedbackRecord | list[FeedbackRecord]


def load_feature_config() -> list[str]:
    # Read the feature list written by preprocess.py so the API always uses
    # exactly the same columns the model was trained on.
    config_path = ARTIFACTS_DIR / "feature_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config.get("feature_columns", get_feature_columns())
    return get_feature_columns()


def load_model(model_path: Path):
    return joblib.load(model_path)


def resolve_model_info() -> dict:
    # Prefer explicit registry stages; fall back to version history.
    version = "unknown"
    stage = "unknown"
    try:
        registry = ModelRegistry()
        prod = registry.current("production")
        if prod:
            return {"version": prod, "stage": "production"}
        staging = registry.current("staging")
        if staging:
            return {"version": staging, "stage": "staging"}
    except Exception:
        pass

    try:
        version = ModelVersionManager().current
    except Exception:
        pass
    return {"version": version, "stage": stage}


def record_to_dict(record: PredictionRecord) -> dict:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record.dict()


def normalize_records(payload: PredictionBody) -> list[PredictionRecord]:
    # Accept three payload shapes: {"records": [...]}, [...], or a single object.
    # All are normalised to a flat list so the prediction logic is uniform.
    if isinstance(payload, PredictionRequest):
        if not payload.records:
            raise HTTPException(status_code=400, detail="records cannot be empty")
        return payload.records
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="records cannot be empty")
        return payload
    return [payload]


def normalize_feedback(payload: FeedbackBody) -> list[FeedbackRecord]:
    if isinstance(payload, FeedbackRequest):
        if not payload.records:
            raise HTTPException(status_code=400, detail="records cannot be empty")
        return payload.records
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="records cannot be empty")
        return payload
    return [payload]


def create_app() -> FastAPI:
    app = FastAPI(title="Bike Sharing Demand API", version="1.0.0")
    model_path = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

    # Default empty form values — keeps inputs populated after submission.
    _empty_form = dict(
        datetime="", season="1", weather="1", holiday="0",
        temp="", humidity="", windspeed="",
    )

    @app.on_event("startup")
    def _load_resources() -> None:
        # Load the model once at startup and cache it on app.state.
        # If the model file is missing the service still starts. /health remains
        # green for liveness while /ready returns 503 until the model is present.
        try:
            app.state.model = load_model(model_path)
        except FileNotFoundError:
            app.state.model = None
        app.state.feature_columns = load_feature_config()

    @app.get("/", response_class=HTMLResponse)
    def form_get(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "form": _empty_form, "prediction": None, "error": None},
        )

    @app.post("/", response_class=HTMLResponse)
    async def form_post(
        request: Request,
        datetime: str    = Form(...),
        season:   str    = Form(...),
        holiday:  str    = Form(...),
        weather:  str    = Form(...),
        temp:     float  = Form(...),
        humidity: float  = Form(...),
        windspeed: float = Form(...),
    ):
        # Re-populate the form with submitted values so the user sees their inputs.
        form_data = dict(
            datetime=datetime, season=season, holiday=holiday,
            weather=weather, temp=str(temp), humidity=str(humidity),
            windspeed=str(windspeed),
        )
        model = getattr(app.state, "model", None)
        if model is None:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "form": form_data,
                 "prediction": None, "error": "Model is not loaded yet."},
            )
        try:
            # datetime-local inputs produce "2011-01-01T08:00"; normalise to space + seconds.
            dt_str = datetime.replace("T", " ")
            if len(dt_str) == 16:
                dt_str += ":00"
            record = PredictionRecord(
                datetime=dt_str,
                season=int(season), holiday=int(holiday), weather=int(weather),
                temp=temp, humidity=humidity, windspeed=windspeed,
            )
            frame = pd.DataFrame([record_to_dict(record)])
            features = build_feature_frame(frame)
            feature_columns = getattr(app.state, "feature_columns", get_feature_columns())
            features = features[feature_columns]
            # Reverse the log1p transform the model was trained with.
            pred_log = model.predict(features)[0]
            pred = max(0, int(round(float(np.expm1(pred_log)))))
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "form": form_data,
                 "prediction": pred, "error": None},
            )
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "form": form_data,
                 "prediction": None, "error": str(exc)},
            )

    @app.get("/health")
    def health() -> dict:
        model = getattr(app.state, "model", None)
        return {"status": "ok", "model_loaded": model is not None}

    @app.get("/ready")
    def ready() -> dict:
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")
        return {"status": "ready"}

    @app.post("/predict")
    def predict(payload: PredictionBody = Body(...)) -> dict:
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        records = normalize_records(payload)
        frame = pd.DataFrame([record_to_dict(record) for record in records])
        missing = [col for col in REQUIRED_INPUTS if col not in frame.columns]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing required fields: {missing}"
            )

        try:
            features = build_feature_frame(frame)
            feature_columns = getattr(app.state, "feature_columns", get_feature_columns())
            features = features[feature_columns]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Model outputs log1p(count); reverse the transform and clip negatives.
        preds_log = model.predict(features)
        preds = np.expm1(preds_log)
        preds = np.maximum(preds, 0)

        request_id = str(uuid4())
        model_info = resolve_model_info()
        prediction_ids: list[str] = []
        preds_list = preds.tolist()
        for record, pred in zip(records, preds_list):
            prediction_id = str(uuid4())
            prediction_ids.append(prediction_id)
            try:
                log_prediction(
                    record=record_to_dict(record),
                    prediction=float(pred),
                    prediction_id=prediction_id,
                    request_id=request_id,
                    model_version=model_info["version"],
                    model_stage=model_info["stage"],
                    source="api",
                )
            except Exception:
                pass

        return {
            "predictions": preds_list,
            "prediction_ids": prediction_ids,
            "request_id": request_id,
            "model_version": model_info["version"],
            "model_stage": model_info["stage"],
        }

    @app.post("/feedback")
    def feedback(payload: FeedbackBody = Body(...)) -> dict:
        records = normalize_feedback(payload)
        request_id = str(uuid4())
        recorded = 0
        for record in records:
            try:
                log_actual(
                    prediction_id=record.prediction_id,
                    actual=float(record.actual),
                    request_id=request_id,
                    source="feedback",
                )
                recorded += 1
            except Exception:
                continue
        return {"status": "ok", "recorded": recorded, "request_id": request_id}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
