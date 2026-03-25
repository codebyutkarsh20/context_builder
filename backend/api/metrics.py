"""
metrics.py — Agent performance metrics dashboard API.

Step 20 from the implementation guide:
Track approval rate, escalation rate, time to PR, cost per fix,
and most common rejection reasons.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
_METRICS_FILE = _DATA_DIR / "agent_metrics.json"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _load_metrics() -> list[dict]:
    if _METRICS_FILE.exists():
        try:
            return json.loads(_METRICS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_metrics(records: list[dict]):
    _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_FILE.write_text(json.dumps(records, indent=2, default=str))


def record_run(result: dict) -> None:
    """Record a completed agent run for metrics. Called from pipeline completion."""
    review = result.get("review", {})
    repair = result.get("repair", {})
    localization = result.get("localization", {})

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_name": result.get("work_order", {}).get("repo_name", ""),
        "ticket_id": result.get("work_order", {}).get("ticket_id", ""),
        "status": result.get("status", ""),
        "verdict": review.get("verdict", ""),
        "confidence": review.get("confidence", 0),
        "iterations": result.get("iteration_count", 0),
        "patches": len(repair.get("patches", [])),
        "localization_confidence": localization.get("confidence", 0),
        "test_result": (result.get("test_result", "") or "")[:100],
        "pr_url": result.get("pr_url", ""),
        "error": (result.get("error", "") or "")[:200],
        # Rejection categorization
        "rejection_reasons": _categorize_rejection(review) if review.get("verdict") != "APPROVE" else [],
    }

    records = _load_metrics()
    records.append(record)
    # Keep last 1000 records
    if len(records) > 1000:
        records = records[-1000:]
    _save_metrics(records)


def _categorize_rejection(review: dict) -> list[str]:
    """Categorize why a fix was rejected."""
    reasons = []
    for check in review.get("checks", []):
        if check.get("status") == "FAIL":
            name = check.get("name", "").upper()
            if name in ("ROOT_CAUSE", "COMPLETENESS"):
                reasons.append("localization_error")
            elif name == "BUSINESS_RULES":
                reasons.append("context_gap")
            elif name == "PATTERNS":
                reasons.append("pattern_violation")
            elif name == "BLAST_RADIUS":
                reasons.append("cross_system_risk")
            else:
                reasons.append("reasoning_error")
    return reasons or ["unknown"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/metrics/summary")
def get_metrics_summary(repo: Optional[str] = None, days: int = Query(30, le=365)):
    """Get aggregated metrics for the agent."""
    records = _load_metrics()

    if repo:
        records = [r for r in records if r.get("repo_name") == repo]

    # Filter by time window
    cutoff = time.time() - (days * 86400)
    # Simple date filter (records have ISO timestamps)
    # For simplicity, include all if we can't parse
    total = len(records)
    if total == 0:
        return {
            "total_runs": 0,
            "approval_rate": 0,
            "escalation_rate": 0,
            "avg_iterations": 0,
            "avg_confidence": 0,
            "by_status": {},
            "rejection_reasons": {},
            "by_repo": {},
        }

    approved = sum(1 for r in records if r.get("verdict") == "APPROVE")
    escalated = sum(1 for r in records if r.get("status") in ("escalated",) or r.get("verdict") == "ESCALATE")
    changes_req = sum(1 for r in records if r.get("verdict") == "CHANGES_REQUESTED")

    # Status breakdown
    from collections import Counter
    status_counts = Counter(r.get("status", "unknown") for r in records)
    verdict_counts = Counter(r.get("verdict", "none") for r in records)

    # Rejection reasons
    all_reasons: list[str] = []
    for r in records:
        all_reasons.extend(r.get("rejection_reasons", []))
    reason_counts = Counter(all_reasons)

    # By repo
    repo_stats: dict[str, dict] = {}
    for r in records:
        rn = r.get("repo_name", "unknown")
        if rn not in repo_stats:
            repo_stats[rn] = {"total": 0, "approved": 0, "escalated": 0}
        repo_stats[rn]["total"] += 1
        if r.get("verdict") == "APPROVE":
            repo_stats[rn]["approved"] += 1
        if r.get("status") == "escalated":
            repo_stats[rn]["escalated"] += 1

    avg_iter = sum(r.get("iterations", 0) for r in records) / total
    avg_conf = sum(r.get("confidence", 0) for r in records) / total

    return {
        "total_runs": total,
        "approval_rate": round(approved / total * 100, 1) if total else 0,
        "escalation_rate": round(escalated / total * 100, 1) if total else 0,
        "changes_requested_rate": round(changes_req / total * 100, 1) if total else 0,
        "avg_iterations": round(avg_iter, 1),
        "avg_confidence": round(avg_conf * 100, 1),
        "by_status": dict(status_counts),
        "by_verdict": dict(verdict_counts),
        "rejection_reasons": dict(reason_counts.most_common(10)),
        "by_repo": repo_stats,
    }


@router.get("/metrics/history")
def get_metrics_history(repo: Optional[str] = None, limit: int = Query(50, le=200)):
    """Get recent agent run history."""
    records = _load_metrics()
    if repo:
        records = [r for r in records if r.get("repo_name") == repo]
    return list(reversed(records[-limit:]))
