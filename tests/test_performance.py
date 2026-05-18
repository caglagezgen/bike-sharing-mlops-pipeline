import json
from datetime import datetime, timezone

from monitoring.performance import build_performance_report


def _write_events(path, events):
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def test_build_performance_report(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    events = [
        {
            "event": "prediction",
            "timestamp": "2026-05-18T00:00:00Z",
            "prediction_id": "p1",
            "prediction": 10.0,
        },
        {
            "event": "actual",
            "timestamp": "2026-05-18T01:00:00Z",
            "prediction_id": "p1",
            "actual": 12.0,
        },
        {
            "event": "prediction",
            "timestamp": "2026-05-18T02:00:00Z",
            "prediction_id": "p2",
            "prediction": 5.0,
        },
        {
            "event": "actual",
            "timestamp": "2026-05-18T03:00:00Z",
            "prediction_id": "p2",
            "actual": 5.0,
        },
    ]
    _write_events(log_path, events)

    report = build_performance_report(log_path)

    assert report["counts"]["predictions"] == 2
    assert report["counts"]["actuals"] == 2
    assert report["counts"]["matched"] == 2
    assert "mae" in report["metrics"]
    assert report["metrics"]["mae"] >= 0


def test_performance_report_respects_window(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    events = [
        {
            "event": "prediction",
            "timestamp": "2026-05-18T00:00:00Z",
            "prediction_id": "p1",
            "prediction": 10.0,
        },
        {
            "event": "actual",
            "timestamp": "2026-05-18T00:10:00Z",
            "prediction_id": "p1",
            "actual": 10.0,
        },
        {
            "event": "prediction",
            "timestamp": "2026-05-19T00:00:00Z",
            "prediction_id": "p2",
            "prediction": 7.0,
        },
    ]
    _write_events(log_path, events)

    start = datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 18, 23, 59, tzinfo=timezone.utc)

    report = build_performance_report(log_path, start=start, end=end)
    assert report["counts"]["predictions"] == 1
    assert report["counts"]["actuals"] == 1
