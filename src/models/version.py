"""
Model Version Manager

Handles semantic versioning for models:
- Auto-increments version numbers (major.minor.patch)
- Tracks version history with timestamps and metadata
- Compares two versions to identify changes
- Maintains artifacts/model_versions.json for audit trail

MLOps Best Practice: Every model deployment should have a unique, immutable version.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR

VERSION_FILE = ARTIFACTS_DIR / "model_versions.json"

_BUMP_POSITIONS = {"major": 0, "minor": 1, "patch": 2}


def _parse(version: str) -> list[int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version!r}. Expected major.minor.patch")
    return [int(p) for p in parts]


def _format(parts: list[int]) -> str:
    return ".".join(str(p) for p in parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelVersionManager:
    def __init__(self, version_file: Path = VERSION_FILE) -> None:
        self._path = version_file
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {"current": "0.0.0", "versions": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> str:
        return self._data["current"]

    def next_version(self, bump: str = "patch") -> str:
        """Return the next version string without registering it."""
        if bump not in _BUMP_POSITIONS:
            raise ValueError(f"bump must be one of {list(_BUMP_POSITIONS)}")
        parts = _parse(self._data["current"])
        pos = _BUMP_POSITIONS[bump]
        parts[pos] += 1
        for i in range(pos + 1, 3):
            parts[i] = 0
        return _format(parts)

    def register(
        self,
        metrics: dict[str, Any],
        params: dict[str, Any],
        feature_columns: list[str],
        dataset_meta: dict[str, Any] | None = None,
        bump: str = "patch",
    ) -> str:
        """Register a new model version and return the version string."""
        version = self.next_version(bump)
        entry: dict[str, Any] = {
            "version": version,
            "registered_at": _now(),
            "metrics": metrics,
            "params": params,
            "feature_columns": feature_columns,
        }
        if dataset_meta:
            entry["dataset"] = {
                # raw_sha256 identifies the original Kaggle file.
                # processed_sha256 is the true training-data fingerprint — it changes
                # whenever preprocessing logic changes (e.g. new Winsorization bounds,
                # dropped columns) even if the raw file is identical.
                "raw_sha256": dataset_meta.get("raw_sha256"),
                "raw_rows": dataset_meta.get("raw_rows"),
                "processed_sha256": dataset_meta.get("processed_sha256"),
                "processed_rows": dataset_meta.get("processed_rows"),
                "source": dataset_meta.get("source"),
            }
        self._data["versions"][version] = entry
        self._data["current"] = version
        self._save()
        print(f"[version] Registered model version {version}")
        return version

    def history(self) -> list[dict]:
        """Return all versions sorted oldest → newest."""
        return sorted(
            self._data["versions"].values(),
            key=lambda v: v["registered_at"],
        )

    def get(self, version: str) -> dict:
        """Retrieve metadata for a specific version."""
        if version not in self._data["versions"]:
            available = list(self._data["versions"])
            raise KeyError(f"Version {version!r} not found. Available: {available}")
        return self._data["versions"][version]

    def diff(self, v1: str, v2: str) -> dict:
        """
        Model Version Diff Viewer

        Compare two model versions to see what changed:
        - Performance metrics
        - Hyperparameters
        - Data version
        - Feature changes
        - Training details
        """
        a = self.get(v1)
        b = self.get(v2)

        def _metric_diff(key: str) -> dict | None:
            va = a["metrics"].get(key)
            vb = b["metrics"].get(key)
            if va is None and vb is None:
                return None
            delta = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = round(vb - va, 6)
            return {"before": va, "after": vb, "delta": delta}

        metric_keys = set(a["metrics"]) | set(b["metrics"])
        metrics_diff = {k: _metric_diff(k) for k in metric_keys if _metric_diff(k)}

        param_keys = set(a["params"]) | set(b["params"])
        params_diff = {
            k: {"before": a["params"].get(k), "after": b["params"].get(k)}
            for k in param_keys
            if a["params"].get(k) != b["params"].get(k)
        }

        features_added = [f for f in b["feature_columns"] if f not in a["feature_columns"]]
        features_removed = [f for f in a["feature_columns"] if f not in b["feature_columns"]]

        dataset_diff: dict = {}
        if a.get("dataset") or b.get("dataset"):
            da = a.get("dataset", {})
            db = b.get("dataset", {})
            for key in {"raw_sha256", "raw_rows", "source"}:
                if da.get(key) != db.get(key):
                    dataset_diff[key] = {"before": da.get(key), "after": db.get(key)}

        return {
            "from_version": v1,
            "to_version": v2,
            "registered_at": {"before": a["registered_at"], "after": b["registered_at"]},
            "metrics": metrics_diff,
            "params": params_diff,
            "features": {"added": features_added, "removed": features_removed},
            "dataset": dataset_diff,
        }

    def print_diff(self, v1: str, v2: str) -> None:
        d = self.diff(v1, v2)
        print(f"\n{'='*60}")
        print(f"  Model Diff:  {d['from_version']}  →  {d['to_version']}")
        print(f"{'='*60}")

        if d["metrics"]:
            print("\nMetrics:")
            for k, v in d["metrics"].items():
                delta_str = f"  (Δ {v['delta']:+.4f})" if v["delta"] is not None else ""
                print(f"  {k}: {v['before']} → {v['after']}{delta_str}")
        else:
            print("\nMetrics: no change")

        if d["params"]:
            print("\nHyperparameters:")
            for k, v in d["params"].items():
                print(f"  {k}: {v['before']} → {v['after']}")
        else:
            print("\nHyperparameters: no change")

        feat = d["features"]
        if feat["added"] or feat["removed"]:
            print("\nFeatures:")
            for f in feat["added"]:
                print(f"  + {f}")
            for f in feat["removed"]:
                print(f"  - {f}")
        else:
            print("\nFeatures: no change")

        if d["dataset"]:
            print("\nDataset:")
            for k, v in d["dataset"].items():
                print(f"  {k}: {v['before']} → {v['after']}")
        else:
            print("\nDataset: no change")
        print()

    def print_history(self) -> None:
        versions = self.history()
        if not versions:
            print("No versions registered yet.")
            return
        print(f"\n{'Version':<10} {'Registered At':<32} {'val_rmsle':<12} {'rows':<8}")
        print("-" * 65)
        for v in versions:
            rmsle = v["metrics"].get("val_rmsle", "—")
            rows = v["metrics"].get("val_rows", "—")
            rmsle_str = f"{rmsle:.5f}" if isinstance(rmsle, float) else str(rmsle)
            marker = " ← current" if v["version"] == self.current else ""
            print(f"{v['version']:<10} {v['registered_at']:<32} {rmsle_str:<12} {rows}{marker}")
        print()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model version management")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("history", help="Print version history")

    diff_p = sub.add_parser("diff", help="Diff two versions")
    diff_p.add_argument("v1", help="First version (e.g. 1.0.0)")
    diff_p.add_argument("v2", help="Second version (e.g. 1.0.1)")

    sub.add_parser("current", help="Print current version")

    args = parser.parse_args()
    mgr = ModelVersionManager()

    if args.cmd == "history":
        mgr.print_history()
    elif args.cmd == "diff":
        mgr.print_diff(args.v1, args.v2)
    elif args.cmd == "current":
        print(mgr.current)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
