"""
Performance monitoring utilities.

Consumes JSONL inference logs and computes metrics when ground truth exists.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _within_window(ts: str, start: datetime | None, end: datetime | None) -> bool:
    if not start and not end:
        return True
    moment = _parse_ts(ts)
    if start and moment < start:
        return False
    if end and moment > end:
        return False
    return True


def load_events(path: Path, start: datetime | None = None, end: datetime | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            ts = event.get("timestamp")
            if ts and _within_window(ts, start, end):
                events.append(event)
    return events


def _split_events(events: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[str, dict]]:
    predictions: dict[str, dict] = {}
    actuals: dict[str, dict] = {}
    for event in events:
        event_type = event.get("event")
        prediction_id = event.get("prediction_id")
        if not prediction_id:
            continue
        if event_type == "prediction":
            predictions[prediction_id] = event
        elif event_type == "actual":
            actuals[prediction_id] = event
    return predictions, actuals


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmsle = float(
        np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    )
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "rmsle": round(rmsle, 6),
        "mape": round(mape, 6),
    }


def build_performance_report(
    log_path: Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    events = load_events(log_path, start=start, end=end)
    predictions, actuals = _split_events(events)

    matched: list[tuple[float, float]] = []
    for prediction_id, pred_event in predictions.items():
        actual_event = actuals.get(prediction_id)
        if not actual_event:
            continue
        pred = float(pred_event.get("prediction", 0.0))
        actual = float(actual_event.get("actual", 0.0))
        matched.append((actual, pred))

    y_true = np.array([m[0] for m in matched], dtype=float)
    y_pred = np.array([m[1] for m in matched], dtype=float)

    metrics = _compute_metrics(y_true, y_pred) if matched else {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "window": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "counts": {
            "events": len(events),
            "predictions": len(predictions),
            "actuals": len(actuals),
            "matched": len(matched),
        },
        "metrics": metrics,
    }
