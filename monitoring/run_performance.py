"""
Performance monitoring entry point.

Reads inference logs, matches predictions with ground truth, and writes a
performance report JSON.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from monitoring.performance import build_performance_report


def _parse_window(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("artifacts/inference_logs.jsonl"),
        help="Path to inference JSONL log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/performance_report.json"),
        help="Path to write the performance report JSON",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Window start (ISO-8601, e.g. 2026-05-18T00:00:00Z)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Window end (ISO-8601, e.g. 2026-05-19T00:00:00Z)",
    )
    args = parser.parse_args()

    report = build_performance_report(
        log_path=args.log_path,
        start=_parse_window(args.start),
        end=_parse_window(args.end),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[performance] Report written to {args.output}")


if __name__ == "__main__":
    main()
