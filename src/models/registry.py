"""
Model Registry

Stores model candidates and stage promotions in a simple JSON registry.
Stages are intended for deployment workflows (e.g., staging -> production).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR

REGISTRY_FILE = ARTIFACTS_DIR / "model_registry.json"
STAGES = {"staging", "production"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_FILE) -> None:
        self._path = registry_path
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {
            "models": {},
            "stages": {"staging": None, "production": None},
            "promotions": [],
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def register(
        self,
        *,
        version: str,
        model_path: str,
        metrics: dict[str, Any],
        params: dict[str, Any],
        feature_columns: list[str],
        dataset_meta: dict[str, Any] | None = None,
        stage: str = "staging",
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}")

        entry: dict[str, Any] = {
            "version": version,
            "registered_at": _now(),
            "model_path": model_path,
            "metrics": metrics,
            "params": params,
            "feature_columns": feature_columns,
        }
        if dataset_meta:
            entry["dataset"] = {
                "raw_sha256": dataset_meta.get("raw_sha256"),
                "raw_rows": dataset_meta.get("raw_rows"),
                "processed_sha256": dataset_meta.get("processed_sha256"),
                "processed_rows": dataset_meta.get("processed_rows"),
                "source": dataset_meta.get("source"),
            }

        self._data["models"][version] = entry
        self._data["stages"][stage] = version
        self._save()
        return entry

    def promote(self, version: str, stage: str = "production") -> None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}")
        if version not in self._data["models"]:
            raise KeyError(f"Version {version!r} not found in registry")
        self._data["stages"][stage] = version
        self._data["promotions"].append(
            {"version": version, "stage": stage, "promoted_at": _now()}
        )
        self._save()

    def current(self, stage: str = "production") -> str | None:
        return self._data.get("stages", {}).get(stage)

    def get(self, version: str) -> dict[str, Any]:
        if version not in self._data["models"]:
            raise KeyError(f"Version {version!r} not found in registry")
        return self._data["models"][version]

    def history(self) -> list[dict[str, Any]]:
        return sorted(
            self._data["models"].values(),
            key=lambda v: v.get("registered_at", ""),
        )
