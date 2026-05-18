"""
Inference logging helpers.

Writes JSONL events to a local log file for downstream performance monitoring.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR

DEFAULT_LOG_PATH = ARTIFACTS_DIR / "inference_logs.jsonl"


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _logging_enabled() -> bool:
    value = os.getenv("INFERENCE_LOG_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no"}


def get_log_path() -> Path:
    env = os.getenv("INFERENCE_LOG_PATH")
    if env:
        return Path(env)
    return DEFAULT_LOG_PATH


def log_event(event: dict[str, Any]) -> None:
    if not _logging_enabled():
        return
    path = get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def log_prediction(
    *,
    record: dict[str, Any],
    prediction: float,
    prediction_id: str,
    request_id: str,
    model_version: str,
    model_stage: str,
    source: str = "api",
) -> None:
    log_event(
        {
            "event": "prediction",
            "timestamp": _iso_now(),
            "prediction_id": prediction_id,
            "request_id": request_id,
            "model_version": model_version,
            "model_stage": model_stage,
            "source": source,
            "features": record,
            "prediction": prediction,
        }
    )


def log_actual(
    *,
    prediction_id: str,
    actual: float,
    request_id: str,
    source: str = "feedback",
) -> None:
    log_event(
        {
            "event": "actual",
            "timestamp": _iso_now(),
            "prediction_id": prediction_id,
            "request_id": request_id,
            "source": source,
            "actual": actual,
        }
    )
