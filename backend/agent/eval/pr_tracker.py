"""
pr_tracker.py — GitHub PR review tracking for the 80% approval metric.

Tracks PRs created by eval runs, polls GitHub API for review outcomes,
and computes the approval rate.

Uses `gh api` CLI to avoid adding PyGithub dependency.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_FILE = Path("eval/pr_tracking.json")


@dataclass
class PRReviewStatus:
    """Tracked PR with review status."""
    pr_url: str
    ticket_id: str
    pipeline: str
    eval_run_id: str
    state: str = "open"  # open, closed, merged
    review_decision: str | None = None  # APPROVED, CHANGES_REQUESTED, REVIEW_REQUIRED
    review_count: int = 0
    comments: int = 0
    last_checked: float = 0.0


class PRTracker:
    """Track GitHub PR review outcomes for eval runs.

    Stores tracking state in a JSON file and polls GitHub API via `gh` CLI.
    """

    def __init__(self, state_file: Path | str = DEFAULT_TRACKING_FILE):
        self.state_file = Path(state_file)
        self._entries: list[PRReviewStatus] = []
        self._load()

    def _load(self) -> None:
        """Load tracking state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    raw = json.load(f)
                self._entries = [PRReviewStatus(**e) for e in raw]
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.warning("Failed to load PR tracking state: %s", e)
                self._entries = []

    def _save(self) -> None:
        """Persist tracking state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    def register_pr(
        self, pr_url: str, ticket_id: str, pipeline: str, eval_run_id: str = ""
    ) -> None:
        """Register a PR for tracking.

        Skips if the PR is already tracked.
        """
        if any(e.pr_url == pr_url for e in self._entries):
            logger.debug("PR already tracked: %s", pr_url)
            return

        self._entries.append(PRReviewStatus(
            pr_url=pr_url,
            ticket_id=ticket_id,
            pipeline=pipeline,
            eval_run_id=eval_run_id,
        ))
        self._save()
        logger.info("Registered PR for tracking: %s", pr_url)

    def poll_all(self) -> list[PRReviewStatus]:
        """Poll GitHub API for all tracked PRs.

        Updates review status in-place and persists to disk.
        """
        import time
        updated = 0

        for entry in self._entries:
            try:
                self._poll_single(entry)
                updated += 1
            except Exception as e:
                logger.warning("Failed to poll %s: %s", entry.pr_url, e)
            entry.last_checked = time.time()

        if updated:
            self._save()
            logger.info("Polled %d PRs", updated)

        return list(self._entries)

    def _poll_single(self, entry: PRReviewStatus) -> None:
        """Poll a single PR via gh api."""
        parsed = _parse_pr_url(entry.pr_url)
        if not parsed:
            return

        owner, repo, number = parsed

        # Get PR state
        pr_data = _gh_api(f"repos/{owner}/{repo}/pulls/{number}")
        if pr_data:
            entry.state = pr_data.get("state", "open")
            if pr_data.get("merged"):
                entry.state = "merged"
            entry.comments = pr_data.get("comments", 0)

        # Get reviews
        reviews = _gh_api(f"repos/{owner}/{repo}/pulls/{number}/reviews")
        if isinstance(reviews, list):
            entry.review_count = len(reviews)
            # Latest non-PENDING review determines the decision
            for review in reversed(reviews):
                state = review.get("state", "")
                if state in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
                    entry.review_decision = state
                    break

    def compute_approval_rate(self) -> dict:
        """Compute the 80% target metric.

        Returns
        -------
        dict
            total_prs, reviewed, approved, approval_rate, target_met.
        """
        total = len(self._entries)
        reviewed = [e for e in self._entries if e.review_decision is not None]
        approved = [e for e in reviewed if e.review_decision == "APPROVED"]

        rate = len(approved) / len(reviewed) if reviewed else 0.0

        return {
            "total_prs": total,
            "reviewed": len(reviewed),
            "approved": len(approved),
            "changes_requested": sum(1 for e in reviewed if e.review_decision == "CHANGES_REQUESTED"),
            "approval_rate": round(rate, 4),
            "target_met": rate >= 0.8,
            "by_pipeline": self._rate_by_pipeline(),
        }

    def _rate_by_pipeline(self) -> dict:
        """Compute approval rate per pipeline."""
        pipelines: dict[str, dict] = {}
        for e in self._entries:
            if e.pipeline not in pipelines:
                pipelines[e.pipeline] = {"total": 0, "reviewed": 0, "approved": 0}
            p = pipelines[e.pipeline]
            p["total"] += 1
            if e.review_decision:
                p["reviewed"] += 1
                if e.review_decision == "APPROVED":
                    p["approved"] += 1

        for p in pipelines.values():
            p["approval_rate"] = round(p["approved"] / p["reviewed"], 4) if p["reviewed"] else 0.0

        return pipelines

    def list_tracked(self) -> list[dict]:
        """Return all tracked PRs as dicts."""
        return [asdict(e) for e in self._entries]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pr_url(url: str) -> tuple[str, str, str] | None:
    """Extract (owner, repo, number) from a GitHub PR URL."""
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def _gh_api(endpoint: str) -> Any:
    """Call GitHub API via gh CLI. Returns parsed JSON or None."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint, "--jq", "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        logger.debug("gh api %s failed: %s", endpoint, e)
    return None
