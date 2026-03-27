"""
eval_suite.py — Evaluation framework that replays historical bugs through the
AI Deploy Agent pipeline and measures performance.

Loads test bugs from DATA_DIR/{repo}/eval_bugs.json, runs each through
run_ticket, and scores localization accuracy, root-cause matching,
fix generation, and review outcomes.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-case timeout in seconds (autoplan eng review: critical gap)
EVAL_CASE_TIMEOUT = int(os.environ.get("EVAL_CASE_TIMEOUT", "600"))

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


# ---------------------------------------------------------------------------
# Bug loader
# ---------------------------------------------------------------------------

def load_eval_bugs(repo: str) -> list[dict]:
    """Load eval bugs from DATA_DIR/{repo}/eval_bugs.json.

    Falls back to the bundled sample_eval_bugs.json shipped with the agent
    package when the repo-specific file does not exist.
    """
    repo_file = DATA_DIR / repo / "eval_bugs.json"
    if repo_file.exists():
        with open(repo_file) as f:
            return json.load(f)

    # Fallback: bundled sample
    sample = Path(__file__).parent / "sample_eval_bugs.json"
    if sample.exists():
        with open(sample) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No eval bugs found at {repo_file} and no sample_eval_bugs.json bundled."
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_localization_hit(result: dict, expected_files: list[str]) -> bool:
    """Did the agent find at least one of the expected files?"""
    localization = result.get("localization") or {}
    found_files = [f.lower() for f in localization.get("fault_files", [])]
    for expected in expected_files:
        expected_lower = expected.lower()
        for found in found_files:
            if expected_lower in found or found.endswith(expected_lower):
                return True
    return False


def _score_root_cause_match(result: dict, expected_root_cause: str) -> bool:
    """Does the localization hypothesis mention the expected root-cause keywords?"""
    localization = result.get("localization") or {}
    hypothesis = (localization.get("root_cause_hypothesis") or "").lower()
    if not hypothesis or not expected_root_cause:
        return False

    keywords = expected_root_cause.lower().split()
    # Require at least 40% of keywords to appear in hypothesis
    matches = sum(1 for kw in keywords if kw in hypothesis)
    threshold = max(1, len(keywords) * 0.4)
    return matches >= threshold


def _score_fix_generated(result: dict) -> bool:
    """Were patches produced?"""
    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    return len(patches) > 0


def _score_review_approved(result: dict) -> bool:
    """Did the reviewer approve?"""
    review = result.get("review") or {}
    return review.get("verdict", "").upper() == "APPROVE"


def _get_review_confidence(result: dict) -> float:
    """Extract the review confidence score."""
    review = result.get("review") or {}
    return float(review.get("confidence", 0.0))


def score_single_run(result: dict, bug: dict) -> dict:
    """Score a single pipeline run against its ground truth bug entry."""
    return {
        "ticket_id": bug["ticket_id"],
        "title": bug.get("title", ""),
        "localization_hit": _score_localization_hit(result, bug.get("expected_files", [])),
        "root_cause_match": _score_root_cause_match(result, bug.get("expected_root_cause", "")),
        "fix_generated": _score_fix_generated(result),
        "review_approved": _score_review_approved(result),
        "confidence": _get_review_confidence(result),
        "pipeline_status": result.get("status", "unknown"),
        "error": result.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------

def _run_single_case(work_order: dict) -> dict:
    """Run a single eval case inside a thread (for timeout support)."""
    from agent.pipeline import run_ticket
    from agent.trace import RunTrace

    trace = RunTrace(job_id=work_order["ticket_id"])
    result = run_ticket(work_order, trace=trace)
    result["_trace"] = trace.to_report()
    return result


def run_eval(repo: str, repo_path: str = "") -> dict:
    """Run all eval bugs through the pipeline and produce a summary report.

    Each case runs with a per-case timeout (EVAL_CASE_TIMEOUT env var,
    default 600s). Crashes and timeouts are captured per-case without
    aborting the suite. Each case gets its own RunTrace for debugging.

    Parameters
    ----------
    repo : str
        Repository slug (used to locate eval_bugs.json and as repo_name).
    repo_path : str, optional
        Filesystem path to the repository. If empty, uses DATA_DIR/{repo}.

    Returns
    -------
    dict with keys: repo, started_at, finished_at, total, scores, summary.
    """
    bugs = load_eval_bugs(repo)
    effective_repo_path = repo_path or str(DATA_DIR / repo)

    started_at = time.time()
    scores: list[dict] = []

    for i, bug in enumerate(bugs):
        logger.info("Eval [%d/%d] running ticket %s: %s", i + 1, len(bugs), bug["ticket_id"], bug.get("title", ""))

        work_order = {
            "ticket_id": bug["ticket_id"],
            "title": bug.get("title", ""),
            "description": bug.get("description", ""),
            "repo_name": repo,
            "repo_path": effective_repo_path,
            "priority": bug.get("priority", "medium"),
            "comments": bug.get("comments", []),
        }

        case_start = time.time()
        result: dict
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_single_case, work_order)
                result = future.result(timeout=EVAL_CASE_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.error("Eval case %s timed out after %ds", bug["ticket_id"], EVAL_CASE_TIMEOUT)
            result = {"status": "failed", "error": f"Timeout after {EVAL_CASE_TIMEOUT}s"}
        except Exception as e:
            logger.exception("Eval run failed for ticket %s", bug["ticket_id"])
            result = {"status": "failed", "error": str(e), "_traceback": traceback.format_exc()}

        case_duration = round(time.time() - case_start, 2)

        score = score_single_run(result, bug)
        score["duration_seconds"] = case_duration
        score["trace"] = result.get("_trace")
        scores.append(score)
        logger.info(
            "Eval [%d/%d] %s (%.1fs) — loc_hit=%s root_match=%s fix=%s approved=%s conf=%.2f",
            i + 1, len(bugs), bug["ticket_id"], case_duration,
            score["localization_hit"], score["root_cause_match"],
            score["fix_generated"], score["review_approved"],
            score["confidence"],
        )

    finished_at = time.time()
    summary = _build_summary(scores, started_at, finished_at)

    report = {
        "repo": repo,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - started_at, 2),
        "total": len(scores),
        "scores": scores,
        "summary": summary,
    }

    # Regression tracking: compare with previous results before overwriting
    results_dir = DATA_DIR / repo
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "eval_results.json"
    history_file = results_dir / "eval_history.json"

    previous = load_latest_results(repo)
    if previous and previous.get("summary"):
        prev_summary = previous["summary"]
        regression = _detect_regressions(prev_summary, summary)
        report["regression"] = regression
        if regression.get("regressions"):
            logger.warning("REGRESSIONS DETECTED: %s", regression["regressions"])
        else:
            logger.info("No regressions vs previous run")

    # Persist current results
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Eval results written to %s", results_file)

    # Append to history for trend tracking
    _append_to_history(history_file, summary, started_at)

    return report


def _build_summary(scores: list[dict], started_at: float, finished_at: float) -> dict:
    """Compute aggregate metrics from individual scores."""
    total = len(scores)
    if total == 0:
        return {
            "pass_rate": 0.0,
            "localization_accuracy": 0.0,
            "root_cause_accuracy": 0.0,
            "fix_rate": 0.0,
            "approval_rate": 0.0,
            "avg_confidence": 0.0,
            "failures": [],
        }

    loc_hits = sum(1 for s in scores if s["localization_hit"])
    root_matches = sum(1 for s in scores if s["root_cause_match"])
    fixes = sum(1 for s in scores if s["fix_generated"])
    approvals = sum(1 for s in scores if s["review_approved"])
    avg_conf = sum(s["confidence"] for s in scores) / total

    # A "pass" = localization hit + fix generated + review approved
    passes = sum(
        1 for s in scores
        if s["localization_hit"] and s["fix_generated"] and s["review_approved"]
    )

    # Categorized failures
    failures: list[dict] = []
    for s in scores:
        reasons = []
        if not s["localization_hit"]:
            reasons.append("localization_miss")
        if not s["root_cause_match"]:
            reasons.append("root_cause_miss")
        if not s["fix_generated"]:
            reasons.append("no_fix")
        if not s["review_approved"]:
            reasons.append("not_approved")
        if s.get("error"):
            reasons.append("error")
        if reasons:
            failures.append({
                "ticket_id": s["ticket_id"],
                "reasons": reasons,
                "error": s.get("error", ""),
            })

    return {
        "pass_rate": round(passes / total, 4),
        "localization_accuracy": round(loc_hits / total, 4),
        "root_cause_accuracy": round(root_matches / total, 4),
        "fix_rate": round(fixes / total, 4),
        "approval_rate": round(approvals / total, 4),
        "avg_confidence": round(avg_conf, 4),
        "failures": failures,
    }


def load_latest_results(repo: str) -> dict | None:
    """Load the most recent eval results from disk, or None."""
    results_file = DATA_DIR / repo / "eval_results.json"
    if not results_file.exists():
        return None
    try:
        with open(results_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load eval results: %s", e)
        return None


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

_TRACKED_METRICS = ["pass_rate", "localization_accuracy", "fix_rate", "approval_rate"]


def _detect_regressions(
    previous: dict, current: dict, threshold: float = 0.05,
) -> dict:
    """Compare current summary against previous. Flag any metric that dropped
    by more than *threshold* (default 5 percentage points).

    Returns dict with 'regressions' list and 'improvements' list.
    """
    regressions: list[dict] = []
    improvements: list[dict] = []

    for metric in _TRACKED_METRICS:
        prev_val = previous.get(metric, 0.0)
        curr_val = current.get(metric, 0.0)
        delta = curr_val - prev_val
        entry = {"metric": metric, "previous": prev_val, "current": curr_val, "delta": round(delta, 4)}
        if delta < -threshold:
            regressions.append(entry)
        elif delta > threshold:
            improvements.append(entry)

    return {"regressions": regressions, "improvements": improvements}


def _append_to_history(history_file: Path, summary: dict, timestamp: float) -> None:
    """Append a summary snapshot to the eval history file for trend tracking."""
    entry = {
        "timestamp": timestamp,
        **{k: summary.get(k, 0.0) for k in _TRACKED_METRICS},
        "avg_confidence": summary.get("avg_confidence", 0.0),
    }
    history: list[dict] = []
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(entry)
    # Keep last 100 runs
    history = history[-100:]
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        logger.warning("Failed to write eval history: %s", e)
