"""
regression.py — Regression detection and CI gate logic.

Ported from eval_suite.py:316-362 and extended with:
  - Absolute threshold checks (min pass rate)
  - Per-pipeline comparison
  - CI-ready exit code support
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRACKED_METRICS = [
    "pass_rate", "localization_accuracy", "fix_rate",
    "approval_rate", "patch_correctness_avg",
]


def detect_regressions(
    previous: dict, current: dict, threshold: float = 0.05,
) -> dict:
    """Compare current summary against previous. Flag any metric that dropped
    by more than *threshold* (default 5 percentage points).

    Parameters
    ----------
    previous : dict
        Previous run's summary (from build_summary).
    current : dict
        Current run's summary.
    threshold : float
        Minimum drop to flag as regression.

    Returns
    -------
    dict
        With 'regressions' and 'improvements' lists.
    """
    regressions: list[dict] = []
    improvements: list[dict] = []

    for metric in _TRACKED_METRICS:
        prev_val = previous.get(metric, 0.0)
        curr_val = current.get(metric, 0.0)
        delta = curr_val - prev_val
        entry = {
            "metric": metric,
            "previous": prev_val,
            "current": curr_val,
            "delta": round(delta, 4),
        }
        if delta < -threshold:
            regressions.append(entry)
        elif delta > threshold:
            improvements.append(entry)

    return {"regressions": regressions, "improvements": improvements}


def check_regression_gate(
    current_report: Any,
    previous_report: Any | None = None,
    min_pass_rate: float = 0.75,
    max_regression: float = 0.05,
) -> tuple[bool, str]:
    """Check if the eval run passes the regression gate.

    Parameters
    ----------
    current_report : EvalRunReport
        Current eval run report.
    previous_report : EvalRunReport or None
        Previous eval run report for relative comparison.
    min_pass_rate : float
        Minimum absolute pass rate required (0-1).
    max_regression : float
        Maximum allowed regression vs previous (0-1).

    Returns
    -------
    tuple[bool, str]
        (passed, reason). Use exit code 0/1 for CI.
    """
    reasons: list[str] = []

    # Check absolute threshold per pipeline
    for pipeline, summary in current_report.summary.items():
        pr = summary.get("pass_rate", 0)
        if pr < min_pass_rate:
            reasons.append(
                f"{pipeline} pass_rate {pr:.0%} < {min_pass_rate:.0%} minimum"
            )

    # Check relative regression vs previous
    if previous_report and previous_report.summary:
        for pipeline in current_report.summary:
            curr_summary = current_report.summary[pipeline]
            prev_summary = previous_report.summary.get(pipeline)
            if not prev_summary:
                continue

            regression = detect_regressions(prev_summary, curr_summary, max_regression)
            for reg in regression.get("regressions", []):
                reasons.append(
                    f"{pipeline}.{reg['metric']} regressed: "
                    f"{reg['previous']:.0%} → {reg['current']:.0%} "
                    f"(Δ{reg['delta']:.0%})"
                )

    if reasons:
        return False, "GATE FAILED: " + "; ".join(reasons)

    return True, "All gates passed"


def load_previous_report(results_dir: Path | str) -> Any | None:
    """Load the most recent eval report from results directory.

    Returns None if no previous report exists.
    """
    results_dir = Path(results_dir)
    latest = results_dir / "latest.json"

    if not latest.exists():
        return None

    try:
        with open(latest) as f:
            data = json.load(f)
        # Return as a simple namespace object with .summary and .comparison
        return _ReportProxy(data)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load previous report: %s", e)
        return None


class _ReportProxy:
    """Lightweight proxy to access report data as attributes."""

    def __init__(self, data: dict):
        self._data = data
        self.summary = data.get("summary", {})
        self.comparison = data.get("comparison", {})
        self.run_id = data.get("run_id", "")
        self.timestamp = data.get("timestamp", 0)
        self.total_bugs = data.get("total_bugs", 0)
