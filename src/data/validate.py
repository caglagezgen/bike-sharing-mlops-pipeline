"""
CSV quality validation for bike-sharing data.

Checks performed:
  1. File exists and is non-empty
  2. CSV is parseable
  3. Required columns present (schema check)
  4. Minimum row count >= MIN_ROWS
  5. `datetime` column is fully parseable
  6. No column is entirely null
  7. Numeric range checks (season, holiday, weather, humidity, temp, windspeed, count)
  8. `count` has no negative values

Raises DataQualityError on any failure.
Writes artifacts/validation_report.json on every run (pass or fail).

Usage:
    python -m src.data.validate --path data/raw/train.csv
    python -m src.data.validate --path data/bronze/train.csv --min-rows 100
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402

# Minimal columns required for the model to train and serve predictions.
# `casual`, `registered`, `workingday`, `atemp` are present in the full
# Kaggle download but are dropped during preprocessing and not required here,
# so that the same validator also works on lean bronze data drops.
REQUIRED_COLUMNS = {
    "datetime", "season", "holiday", "weather",
    "temp", "humidity", "windspeed", "count",
}

# The full Kaggle train split has 10,886 rows. Accept anything >= 100 so the
# check is useful for smaller bronze data drops as well as the full download.
MIN_ROWS = 100

# Domain-constrained value bounds (inclusive).
# These are not statistical outlier thresholds — they encode hard constraints
# defined by the Kaggle dataset specification and real-world sensor limits.
RANGE_CHECKS: list[tuple[str, float, float]] = [
    ("season",    1,    4),    # 1=spring, 2=summer, 3=fall, 4=winter
    ("holiday",   0,    1),    # binary flag
    ("weather",   1,    4),    # 1=clear → 4=heavy rain/snow
    ("humidity",  0,  100),    # percentage
    ("temp",    -50,   60),    # Celsius; D.C. range is roughly -10..40
    ("windspeed", 0,  100),    # km/h
    ("count",     0, 1000),    # Kaggle observed max is ~977
]


class DataQualityError(RuntimeError):
    """Raised when one or more data quality checks fail."""

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


def validate(path: Path, min_rows: int = MIN_ROWS) -> dict:
    """
    Run all quality checks on the CSV at *path*.

    Returns a validation report dict on success.
    Raises DataQualityError if any check fails.
    Always writes artifacts/validation_report.json.
    """
    report: dict = {
        "path": str(path),
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "passed": False,
        "checks": {},
        "errors": [],
    }

    # ── 1. File exists and is non-empty ───────────────────────────────────────
    if not path.exists():
        report["errors"].append(f"File not found: {path}")
        _save_and_raise(report)

    size = path.stat().st_size
    if size == 0:
        report["errors"].append(f"File is empty (0 bytes): {path}")
        _save_and_raise(report)

    report["checks"]["file_exists"] = {"passed": True, "size_bytes": size}

    # ── 2. Parseable CSV ──────────────────────────────────────────────────────
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        report["errors"].append(f"CSV parse error: {exc}")
        _save_and_raise(report)

    report["checks"]["parseable"] = {"passed": True, "row_count": len(frame)}

    # ── 3. Schema check ───────────────────────────────────────────────────────
    missing_cols = sorted(REQUIRED_COLUMNS - set(frame.columns))
    report["checks"]["schema"] = {
        "passed": len(missing_cols) == 0,
        "missing_columns": missing_cols,
    }
    if missing_cols:
        report["errors"].append(f"Missing required columns: {missing_cols}")

    # ── 4. Minimum row count ──────────────────────────────────────────────────
    row_count = len(frame)
    rows_ok = row_count >= min_rows
    report["checks"]["min_rows"] = {
        "passed": rows_ok,
        "row_count": row_count,
        "min_required": min_rows,
    }
    if not rows_ok:
        report["errors"].append(
            f"Only {row_count} rows; minimum required is {min_rows}."
        )

    # ── 5. datetime parseable ─────────────────────────────────────────────────
    if "datetime" in frame.columns:
        parsed = pd.to_datetime(frame["datetime"], errors="coerce")
        unparseable = int(parsed.isna().sum())
        dt_ok = unparseable == 0
        report["checks"]["datetime_parseable"] = {
            "passed": dt_ok,
            "unparseable_count": unparseable,
        }
        if not dt_ok:
            report["errors"].append(
                f"{unparseable} rows have unparseable datetime values."
            )

    # ── 6. No fully-null columns ──────────────────────────────────────────────
    null_cols = [c for c in frame.columns if frame[c].isna().all()]
    report["checks"]["no_null_columns"] = {
        "passed": len(null_cols) == 0,
        "null_columns": null_cols,
    }
    if null_cols:
        report["errors"].append(f"Columns are entirely null: {null_cols}")

    # ── 7. Numeric range checks ───────────────────────────────────────────────
    violations: list[str] = []
    for col, lo, hi in RANGE_CHECKS:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        out_of_range = int(((series < lo) | (series > hi)).sum())
        if out_of_range > 0:
            violations.append(f"{col}: {out_of_range} value(s) outside [{lo}, {hi}]")
    report["checks"]["numeric_ranges"] = {
        "passed": len(violations) == 0,
        "violations": violations,
    }
    if violations:
        report["errors"].append(f"Numeric range violations: {violations}")

    # ── 8. count non-negative ─────────────────────────────────────────────────
    if "count" in frame.columns:
        neg = int(
            (pd.to_numeric(frame["count"], errors="coerce").fillna(0) < 0).sum()
        )
        report["checks"]["count_non_negative"] = {"passed": neg == 0, "negative_count": neg}
        if neg > 0:
            report["errors"].append(f"`count` has {neg} negative value(s).")

    # ── Final verdict ─────────────────────────────────────────────────────────
    report["passed"] = len(report["errors"]) == 0
    _save_report(report)

    if not report["passed"]:
        raise DataQualityError(
            f"Data quality validation failed ({len(report['errors'])} error(s)): "
            + "; ".join(report["errors"]),
            report=report,
        )

    print(
        f"[validate] PASSED — {row_count} rows, "
        f"all {len(REQUIRED_COLUMNS)} required columns present, "
        "range checks clear."
    )
    return report


def _save_report(report: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[validate] Report written to {out}")


def _save_and_raise(report: dict) -> None:
    _save_report(report)
    raise DataQualityError(
        "Data quality validation failed: " + "; ".join(report["errors"]),
        report=report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a bike-sharing CSV file")
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to the CSV file to validate (e.g. data/raw/train.csv)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_ROWS,
        help=f"Minimum acceptable row count (default: {MIN_ROWS})",
    )
    args = parser.parse_args()

    try:
        validate(args.path, min_rows=args.min_rows)
    except DataQualityError as exc:
        print(f"[validate] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
