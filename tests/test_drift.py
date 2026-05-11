"""
Tests for monitoring/drift.py and monitoring/run_drift.py
Covers: ks_drift, chi2_drift, psi, run_report, and the --fail-on-drift CLI path.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import chi2_drift, ks_drift, psi, run_report

# ---------------------------------------------------------------------------
# ks_drift
# ---------------------------------------------------------------------------

def test_ks_no_drift_for_identical_distributions():
    """KS test should not flag drift when reference == current."""
    data = pd.Series(np.random.default_rng(0).normal(0, 1, 500))
    result = ks_drift(data, data.copy())
    assert result["drift_detected"] is False
    assert 0 <= result["p_value"] <= 1


def test_ks_drift_detected_for_shifted_distribution():
    """KS test must detect a large mean shift between reference and current."""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(10, 1, 500))   # clearly different mean
    result = ks_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["p_value"] < 0.05


def test_ks_result_has_required_keys():
    data = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = ks_drift(data, data)
    assert {"statistic", "p_value", "drift_detected"} == result.keys()


# ---------------------------------------------------------------------------
# chi2_drift
# ---------------------------------------------------------------------------

def test_chi2_no_drift_for_identical_categories():
    """Chi-squared test should not flag drift for identical category distributions."""
    cats = pd.Series([1, 2, 3, 1, 2, 3, 1, 1] * 20)
    result = chi2_drift(cats, cats.copy())
    assert result["drift_detected"] is False


def test_chi2_drift_for_completely_different_categories():
    """Chi-squared test must detect drift when category proportions are inverted."""
    ref = pd.Series([1] * 90 + [2] * 10)   # 90% cat 1
    cur = pd.Series([1] * 10 + [2] * 90)   # 10% cat 1 — very different
    result = chi2_drift(ref, cur)
    assert result["drift_detected"] is True


def test_chi2_empty_expected_returns_no_drift():
    """chi2_drift must not raise when reference is all one value and current matches."""
    ref = pd.Series([1, 1, 1, 1])
    cur = pd.Series([1, 1, 1, 1])
    result = chi2_drift(ref, cur)
    assert isinstance(result["drift_detected"], bool)


# ---------------------------------------------------------------------------
# psi
# ---------------------------------------------------------------------------

def test_psi_zero_for_identical_distributions():
    """PSI must be (near) zero for identical distributions."""
    data = pd.Series(np.random.default_rng(0).uniform(0, 1, 1000))
    score = psi(data, data.copy())
    assert score == pytest.approx(0.0, abs=0.01)


def test_psi_large_for_very_different_distributions():
    """PSI must exceed the 0.2 'significant drift' threshold for very different data."""
    rng = np.random.default_rng(7)
    ref = pd.Series(rng.uniform(0, 1, 1000))
    cur = pd.Series(rng.uniform(5, 6, 1000))   # completely separate range
    score = psi(ref, cur)
    assert score > 0.2, f"Expected PSI > 0.2 for clearly different dists, got {score}"


# ---------------------------------------------------------------------------
# run_report
# ---------------------------------------------------------------------------

def test_run_report_structure():
    """run_report must return a dict with 'numerical' and 'categorical' keys."""
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({
        "temp":    rng.normal(20, 5, 200),
        "humidity": rng.normal(60, 10, 200),
        "season":  np.tile([1, 2, 3, 4], 50),
    })
    cur = pd.DataFrame({
        "temp":    rng.normal(20, 5, 200),
        "humidity": rng.normal(60, 10, 200),
        "season":  np.tile([1, 2, 3, 4], 50),
    })

    report = run_report(
        ref, cur,
        num_cols=["temp", "humidity"],
        cat_cols=["season"],
    )

    assert "numerical" in report
    assert "categorical" in report
    assert "temp" in report["numerical"]
    assert "humidity" in report["numerical"]
    assert "season" in report["categorical"]
    # Each numerical entry must have ks + psi
    assert "ks" in report["numerical"]["temp"]
    assert "psi" in report["numerical"]["temp"]


def test_run_report_skips_missing_columns():
    """run_report must not raise when a column in num_cols is absent from data."""
    ref = pd.DataFrame({"temp": [10.0, 20.0, 30.0]})
    cur = pd.DataFrame({"temp": [10.0, 20.0, 30.0]})

    # "humidity" is not in either df — should be silently skipped
    report = run_report(ref, cur, num_cols=["temp", "humidity"])
    assert "temp" in report["numerical"]
    assert "humidity" not in report["numerical"]


# ---------------------------------------------------------------------------
# run_drift.py CLI — --fail-on-drift exit-code path
# ---------------------------------------------------------------------------

def _write_csv(path, data: dict):
    pd.DataFrame(data).to_csv(path, index=False)


def test_run_drift_exits_0_when_no_drift(tmp_path):
    """run_drift.py must exit 0 with --fail-on-drift when distributions match."""
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    rng = np.random.default_rng(0)
    common = {
        "temp": rng.normal(20, 1, 300).tolist(),
        "humidity": rng.normal(60, 2, 300).tolist(),
        "windspeed": rng.normal(10, 1, 300).tolist(),
        "year": [2011] * 300,
        "month": (np.arange(300) % 12 + 1).tolist(),
        "hour": (np.arange(300) % 24).tolist(),
        "dayofweek": (np.arange(300) % 7).tolist(),
        "season": (np.arange(300) % 4 + 1).tolist(),
        "holiday": [0] * 300,
        "weather": [1] * 300,
    }
    _write_csv(ref, common)
    _write_csv(cur, common)   # identical → no drift

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference", str(ref),
            "--current",   str(cur),
            "--output",    str(tmp_path / "report.json"),
            "--fail-on-drift",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit 0 (no drift), got {result.returncode}.\n{result.stdout}"
    )


def test_run_drift_exits_1_when_drift_detected(tmp_path):
    """run_drift.py must exit 1 with --fail-on-drift when clear drift is present."""
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    rng = np.random.default_rng(42)

    n = 400
    ref_data = {
        "temp": rng.normal(10, 1, n).tolist(),      # cold
        "humidity": rng.normal(40, 2, n).tolist(),
        "windspeed": rng.normal(5, 1, n).tolist(),
        "year": [2011] * n,
        "month": (np.arange(n) % 12 + 1).tolist(),
        "hour": (np.arange(n) % 24).tolist(),
        "dayofweek": (np.arange(n) % 7).tolist(),
        "season": [1] * n,
        "holiday": [0] * n,
        "weather": [1] * n,
    }
    cur_data = {
        "temp": rng.normal(35, 1, n).tolist(),      # very hot — clear drift
        "humidity": rng.normal(90, 2, n).tolist(),  # very humid — clear drift
        "windspeed": rng.normal(5, 1, n).tolist(),
        "year": [2012] * n,
        "month": (np.arange(n) % 12 + 1).tolist(),
        "hour": (np.arange(n) % 24).tolist(),
        "dayofweek": (np.arange(n) % 7).tolist(),
        "season": [3] * n,
        "holiday": [0] * n,
        "weather": [1] * n,
    }
    _write_csv(ref, ref_data)
    _write_csv(cur, cur_data)

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference", str(ref),
            "--current",   str(cur),
            "--output",    str(tmp_path / "report.json"),
            "--fail-on-drift",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Expected exit 1 (drift detected), got {result.returncode}.\n{result.stdout}"
    )


def test_run_drift_exits_0_without_fail_flag_even_on_drift(tmp_path):
    """Without --fail-on-drift the script must exit 0 regardless of drift."""
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    rng = np.random.default_rng(1)
    n = 200
    _write_csv(ref, {
        "temp": rng.normal(10, 1, n).tolist(),
        "humidity": rng.normal(40, 2, n).tolist(),
        "windspeed": rng.normal(5, 1, n).tolist(),
        "year": [2011] * n, "month": [1] * n, "hour": [0] * n,
        "dayofweek": [0] * n, "season": [1] * n, "holiday": [0] * n, "weather": [1] * n,
    })
    _write_csv(cur, {
        "temp": rng.normal(35, 1, n).tolist(),
        "humidity": rng.normal(90, 2, n).tolist(),
        "windspeed": rng.normal(5, 1, n).tolist(),
        "year": [2012] * n, "month": [7] * n, "hour": [0] * n,
        "dayofweek": [0] * n, "season": [3] * n, "holiday": [0] * n, "weather": [1] * n,
    })

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference", str(ref),
            "--current",   str(cur),
            "--output",    str(tmp_path / "report.json"),
            # no --fail-on-drift
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit 0 (no fail flag), got {result.returncode}.\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# run_drift.py CLI — new threshold logic (--psi-threshold / --ks-threshold /
# --min-drift-ratio) and file-not-found exit codes
# ---------------------------------------------------------------------------

def _make_drifted_csvs(tmp_path, n: int = 300) -> tuple:
    """Return (ref_path, cur_path) where all numerical columns are heavily drifted."""
    rng = np.random.default_rng(99)
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    _write_csv(ref, {
        "temp":      rng.normal(10, 1, n).tolist(),
        "humidity":  rng.normal(40, 2, n).tolist(),
        "windspeed": rng.normal(5,  1, n).tolist(),
        "year":      [2011] * n,
        "month":     (np.arange(n) % 12 + 1).tolist(),
        "hour":      (np.arange(n) % 24).tolist(),
        "dayofweek": (np.arange(n) % 7).tolist(),
        "season":    [1] * n,
        "holiday":   [0] * n,
        "weather":   [1] * n,
    })
    _write_csv(cur, {
        "temp":      rng.normal(35, 1, n).tolist(),   # heavy drift
        "humidity":  rng.normal(90, 2, n).tolist(),   # heavy drift
        "windspeed": rng.normal(50, 1, n).tolist(),   # heavy drift
        "year":      [2012] * n,
        "month":     (np.arange(n) % 12 + 1).tolist(),
        "hour":      (np.arange(n) % 24).tolist(),
        "dayofweek": (np.arange(n) % 7).tolist(),
        "season":    [3] * n,
        "holiday":   [0] * n,
        "weather":   [1] * n,
    })
    return ref, cur


def test_run_drift_exits_2_when_reference_missing(tmp_path):
    """run_drift.py must exit 2 when --reference file does not exist."""
    cur = tmp_path / "cur.csv"
    _write_csv(cur, {"temp": [1.0, 2.0]})

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference", str(tmp_path / "does_not_exist.csv"),
            "--current",   str(cur),
            "--output",    str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"Expected exit 2 (missing reference), got {result.returncode}.\n{result.stdout}"
    )


def test_run_drift_exits_2_when_current_missing(tmp_path):
    """run_drift.py must exit 2 when --current file does not exist."""
    ref = tmp_path / "ref.csv"
    _write_csv(ref, {"temp": [1.0, 2.0]})

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference", str(ref),
            "--current",   str(tmp_path / "does_not_exist.csv"),
            "--output",    str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"Expected exit 2 (missing current), got {result.returncode}.\n{result.stdout}"
    )


def test_run_drift_exit_0_when_ratio_below_min_drift_ratio(tmp_path):
    """
    With --min-drift-ratio 1.0 (all features must drift) and only some drifting,
    exit must be 0 even with --fail-on-drift.
    """
    ref, cur = _make_drifted_csvs(tmp_path)

    result = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference",      str(ref),
            "--current",        str(cur),
            "--output",         str(tmp_path / "report.json"),
            "--min-drift-ratio", "1.0",   # requires 100% of features to drift
            "--fail-on-drift",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit 0 (ratio < 1.0 threshold), got {result.returncode}.\n{result.stdout}"
    )


def test_run_drift_report_contains_threshold_block(tmp_path):
    """drift_report.json must include thresholds, drift_ratio and drift_triggered."""
    ref, cur = _make_drifted_csvs(tmp_path)
    out = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference",       str(ref),
            "--current",         str(cur),
            "--output",          str(out),
            "--psi-threshold",   "0.15",
            "--ks-threshold",    "0.10",
            "--min-drift-ratio", "0.25",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    import json as _json
    report = _json.loads(out.read_text())
    summary = report["summary"]

    assert "drift_ratio" in summary
    assert "drift_triggered" in summary
    assert "thresholds" in summary
    assert summary["thresholds"]["psi"] == 0.15
    assert summary["thresholds"]["ks_p_value"] == 0.10
    assert summary["thresholds"]["min_drift_ratio"] == 0.25
    assert isinstance(summary["drift_ratio"], float)
    assert isinstance(summary["drift_triggered"], bool)


def test_run_drift_custom_thresholds_control_trigger(tmp_path):
    """
    With a very low --min-drift-ratio (0.01) even a single drifted feature
    triggers CT (exit 1); with a high ratio (0.99) the same data does not.
    """
    ref, cur = _make_drifted_csvs(tmp_path)

    # Low threshold → should trigger
    low = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference",       str(ref),
            "--current",         str(cur),
            "--output",          str(tmp_path / "r1.json"),
            "--min-drift-ratio", "0.01",
            "--fail-on-drift",
        ],
        capture_output=True, text=True,
    )
    # High threshold → should NOT trigger
    high = subprocess.run(
        [
            sys.executable, "monitoring/run_drift.py",
            "--reference",       str(ref),
            "--current",         str(cur),
            "--output",          str(tmp_path / "r2.json"),
            "--min-drift-ratio", "0.99",
            "--fail-on-drift",
        ],
        capture_output=True, text=True,
    )

    assert low.returncode == 1,  f"Low ratio should trigger exit 1, got {low.returncode}"
    assert high.returncode == 0, f"High ratio should not trigger, got {high.returncode}"


# ---------------------------------------------------------------------------
# run_drift.py — direct import tests (counted toward coverage)
# These call main() via monkeypatched sys.argv so pytest-cov can trace them.
# ---------------------------------------------------------------------------

from monitoring.run_drift import main as drift_main  # noqa: E402


def _run_main(argv: list[str]):
    """
    Call drift_main() with a patched sys.argv.
    Returns the SystemExit code, or 0 if main() returned normally.
    """
    import sys as _sys
    orig = _sys.argv
    _sys.argv = ["run_drift.py"] + argv
    try:
        drift_main()
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        _sys.argv = orig


def test_main_exit2_on_missing_reference(tmp_path):
    """main() must exit 2 when reference path does not exist."""
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"temp": [1.0]}).to_csv(cur, index=False)
    code = _run_main([
        "--reference", str(tmp_path / "nope.csv"),
        "--current",   str(cur),
        "--output",    str(tmp_path / "out.json"),
    ])
    assert code == 2


def test_main_exit2_on_missing_current(tmp_path):
    """main() must exit 2 when current path does not exist."""
    ref = tmp_path / "ref.csv"
    pd.DataFrame({"temp": [1.0]}).to_csv(ref, index=False)
    code = _run_main([
        "--reference", str(ref),
        "--current",   str(tmp_path / "nope.csv"),
        "--output",    str(tmp_path / "out.json"),
    ])
    assert code == 2


def test_main_exit0_no_drift(tmp_path):
    """main() returns 0 when data is identical and --fail-on-drift is set."""
    # build identical CSVs inline
    rng = np.random.default_rng(7)
    n = 200
    data = {
        "temp":      rng.normal(20, 1, n).tolist(),
        "humidity":  rng.normal(60, 2, n).tolist(),
        "windspeed": rng.normal(10, 1, n).tolist(),
        "year":      [2011] * n,
        "month":     (np.arange(n) % 12 + 1).tolist(),
        "hour":      (np.arange(n) % 24).tolist(),
        "dayofweek": (np.arange(n) % 7).tolist(),
        "season":    (np.arange(n) % 4 + 1).tolist(),
        "holiday":   [0] * n,
        "weather":   [1] * n,
    }
    ref_p = tmp_path / "ref.csv"
    cur_p = tmp_path / "cur.csv"
    pd.DataFrame(data).to_csv(ref_p, index=False)
    pd.DataFrame(data).to_csv(cur_p, index=False)  # identical → no drift

    code = _run_main([
        "--reference",       str(ref_p),
        "--current",         str(cur_p),
        "--output",          str(tmp_path / "out.json"),
        "--fail-on-drift",
    ])
    assert code == 0


def test_main_exit1_with_heavy_drift_low_threshold(tmp_path):
    """main() exits 1 when drift ratio exceeds min-drift-ratio with --fail-on-drift."""
    ref_p, cur_p = _make_drifted_csvs(tmp_path)

    code = _run_main([
        "--reference",       str(ref_p),
        "--current",         str(cur_p),
        "--output",          str(tmp_path / "out.json"),
        "--min-drift-ratio", "0.01",  # very sensitive
        "--fail-on-drift",
    ])
    assert code == 1


def test_main_report_written_with_correct_threshold_fields(tmp_path):
    """main() writes drift_report.json with thresholds, drift_ratio, drift_triggered."""
    import json as _json
    ref_p, cur_p = _make_drifted_csvs(tmp_path)
    out_p = tmp_path / "report.json"

    _run_main([
        "--reference",       str(ref_p),
        "--current",         str(cur_p),
        "--output",          str(out_p),
        "--psi-threshold",   "0.10",
        "--ks-threshold",    "0.05",
        "--min-drift-ratio", "0.20",
    ])

    assert out_p.exists()
    summary = _json.loads(out_p.read_text())["summary"]
    assert "drift_ratio" in summary
    assert "drift_triggered" in summary
    assert summary["thresholds"]["psi"] == 0.10
    assert summary["thresholds"]["min_drift_ratio"] == 0.20


def test_main_exit0_without_fail_on_drift_flag(tmp_path):
    """main() exits 0 even with drift when --fail-on-drift is absent."""
    ref_p, cur_p = _make_drifted_csvs(tmp_path)

    code = _run_main([
        "--reference",       str(ref_p),
        "--current",         str(cur_p),
        "--output",          str(tmp_path / "out.json"),
        "--min-drift-ratio", "0.01",  # all features would trigger
        # no --fail-on-drift
    ])
    assert code == 0

