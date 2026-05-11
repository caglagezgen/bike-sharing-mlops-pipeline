import csv
from pathlib import Path

import pytest

from src.data.validate import MIN_ROWS, DataQualityError, validate


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _valid_row(**overrides) -> dict:
    """Return a minimal valid row; override any field via kwargs."""
    base = {
        "datetime": "2011-01-01 00:00:00",
        "season": 1,
        "holiday": 0,
        "weather": 1,
        "temp": 9.84,
        "humidity": 81,
        "windspeed": 0.0,
        "count": 13,
    }
    base.update(overrides)
    return base


def test_validate_passes_on_valid_csv(tmp_path):
    """A well-formed CSV with all required columns and valid values passes."""
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [_valid_row() for _ in range(MIN_ROWS)])
    report = validate(csv_path)
    assert report["passed"] is True
    assert report["errors"] == []


def test_validate_fails_on_missing_column(tmp_path):
    """Missing a required column raises DataQualityError."""
    csv_path = tmp_path / "train.csv"
    rows = [{k: v for k, v in _valid_row().items() if k != "count"} for _ in range(MIN_ROWS)]
    _write_csv(csv_path, rows)
    with pytest.raises(DataQualityError) as exc_info:
        validate(csv_path)
    assert "count" in str(exc_info.value)


def test_validate_fails_on_too_few_rows(tmp_path):
    """A CSV with fewer rows than MIN_ROWS raises DataQualityError."""
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [_valid_row()])  # only 1 row
    with pytest.raises(DataQualityError) as exc_info:
        validate(csv_path)
    assert "rows" in str(exc_info.value)


def test_validate_fails_on_negative_count(tmp_path):
    """Negative values in `count` raise DataQualityError."""
    csv_path = tmp_path / "train.csv"
    rows = [_valid_row() for _ in range(MIN_ROWS)]
    rows[0]["count"] = -5
    _write_csv(csv_path, rows)
    with pytest.raises(DataQualityError) as exc_info:
        validate(csv_path)
    assert "negative" in str(exc_info.value)


def test_validate_fails_on_range_violation(tmp_path):
    """A season value outside [1,4] raises DataQualityError."""
    csv_path = tmp_path / "train.csv"
    rows = [_valid_row() for _ in range(MIN_ROWS)]
    rows[0]["season"] = 99
    _write_csv(csv_path, rows)
    with pytest.raises(DataQualityError) as exc_info:
        validate(csv_path)
    assert "season" in str(exc_info.value)


def test_validate_fails_on_unparseable_datetime(tmp_path):
    """Rows with unparseable datetime values raise DataQualityError."""
    csv_path = tmp_path / "train.csv"
    rows = [_valid_row() for _ in range(MIN_ROWS)]
    rows[0]["datetime"] = "not-a-date"
    _write_csv(csv_path, rows)
    with pytest.raises(DataQualityError) as exc_info:
        validate(csv_path)
    assert "unparseable datetime" in str(exc_info.value)
