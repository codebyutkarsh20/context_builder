"""
runner.py — A/B eval runner that orchestrates both pipelines on a bug dataset.

Supports:
  - A/B mode: run each bug on fixed + ReAct pipelines
  - Single pipeline mode: run on one only
  - Sentinel mode: run 5 fast bugs for quick regression checks
  - Per-case timeout with crash isolation
  - Graph-less execution (exploration tools work on raw repos)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from .dataset import load_eval_dataset, EvalBug
from .repo_manager import RepoManager
from .scoring import score_case, build_summary

logger = logging.getLogger(__name__)

EVAL_CASE_TIMEOUT = int(os.environ.get("EVAL_CASE_TIMEOUT", "600"))
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", "eval/results"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalCaseResult:
    """Result of running one bug through one pipeline."""
    ticket_id: str
    pipeline: str
    score: dict
    trace_report: dict | None = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalRunReport:
    """Complete report for one eval run."""
    run_id: str
    timestamp: float
    dataset_path: str
    total_bugs: int
    pipelines: list[str]
    results: list[EvalCaseResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    graph_less: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """A/B eval runner for the AI Deploy Agent.

    Runs each bug through one or both pipelines, scores results,
    builds aggregate summaries, and persists reports.

    Graph-less by default: eval runs do NOT require Neo4j. Both pipelines
    fall back to exploration tools (grep, read_file, etc.) on raw repos.
    """

    def __init__(
        self,
        dataset_path: Path | str = Path("eval/bugs.json"),
        pipelines: list[str] | None = None,
        timeout_per_case: int = EVAL_CASE_TIMEOUT,
        create_prs: bool = False,
        results_dir: Path | str = RESULTS_DIR,
        repo_cache_dir: Path | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.pipelines = pipelines or ["fixed", "react"]
        self.timeout_per_case = timeout_per_case
        self.dry_run = not create_prs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.repo_manager = RepoManager(cache_dir=repo_cache_dir)

    def run(
        self,
        bug_filter: str | None = None,
        sentinel: bool = False,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> EvalRunReport:
        """Run the eval suite.

        Parameters
        ----------
        bug_filter : str or None
            If set, run only the bug with this ticket_id.
        sentinel : bool
            If True, run only the first 5 bugs (fast regression check).
        progress_cb : callable or None
            Called with (ticket_id, current_index, total) after each bug.

        Returns
        -------
        EvalRunReport
            Complete report with per-case results and aggregate summaries.
        """
        bugs = load_eval_dataset(self.dataset_path)

        if bug_filter:
            bugs = [b for b in bugs if b["ticket_id"] == bug_filter]
            if not bugs:
                raise ValueError(f"No bug found with ticket_id={bug_filter}")

        if sentinel:
            bugs = bugs[:5]
            logger.info("Sentinel mode: running first %d bugs", len(bugs))

        run_id = str(uuid.uuid4())[:8]
        report = EvalRunReport(
            run_id=run_id,
            timestamp=time.time(),
            dataset_path=str(self.dataset_path),
            total_bugs=len(bugs),
            pipelines=list(self.pipelines),
            graph_less=True,
        )

        logger.info(
            "Starting eval run %s: %d bugs × %d pipeline(s) = %d cases",
            run_id, len(bugs), len(self.pipelines), len(bugs) * len(self.pipelines),
        )

        for i, bug in enumerate(bugs):
            logger.info("=" * 60)
            logger.info("Bug %d/%d: %s — %s", i + 1, len(bugs), bug["ticket_id"], bug.get("title", "")[:60])

            # Ensure repo is cloned and at correct SHA
            try:
                repo_path = self.repo_manager.ensure_repo(bug)
            except Exception as e:
                logger.error("Clone failed for %s: %s", bug["ticket_id"], e)
                for pipeline in self.pipelines:
                    report.results.append(EvalCaseResult(
                        ticket_id=bug["ticket_id"],
                        pipeline=pipeline,
                        score=_error_score(bug, pipeline, str(e)),
                        error=str(e),
                    ))
                continue

            # Run each pipeline
            for pipeline in self.pipelines:
                case_result = self._run_single_case(bug, repo_path, pipeline)
                report.results.append(case_result)

            if progress_cb:
                progress_cb(bug["ticket_id"], i + 1, len(bugs))

        # Build summaries
        all_scores = [r.score for r in report.results]
        report.summary = {
            pipeline: build_summary(all_scores, pipeline=pipeline)
            for pipeline in self.pipelines
        }

        # A/B comparison
        if len(self.pipelines) == 2:
            report.comparison = self._build_comparison(report)

        # Persist
        self._persist_report(report)

        logger.info("Eval run %s complete: %d results", run_id, len(report.results))
        return report

    def _run_single_case(
        self, bug: dict, repo_path: Path, pipeline: str
    ) -> EvalCaseResult:
        """Run one bug through one pipeline with timeout isolation."""
        from agent.trace import RunTrace

        trace = RunTrace(job_id=f"{bug['ticket_id']}_{pipeline}", enabled=True)
        work_order = _bug_to_work_order(bug, repo_path)

        case_start = time.time()
        result: dict

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._invoke_pipeline, pipeline, work_order, trace
                )
                result = future.result(timeout=self.timeout_per_case)
        except concurrent.futures.TimeoutError:
            logger.error(
                "%s/%s timed out after %ds", bug["ticket_id"], pipeline, self.timeout_per_case
            )
            result = {"status": "failed", "error": f"Timeout after {self.timeout_per_case}s"}
        except Exception as e:
            logger.exception("%s/%s crashed", bug["ticket_id"], pipeline)
            result = {"status": "failed", "error": str(e), "_traceback": traceback.format_exc()}

        duration = round(time.time() - case_start, 2)

        # Score
        score = score_case(result, bug, pipeline)
        score["duration_seconds"] = duration

        # Extract cost from result or trace
        cost = result.get("cost_usd", 0.0) or 0.0
        if not cost:
            trace_report = trace.to_report()
            cost = trace_report.get("summary", {}).get("total_cost_usd", 0.0)

        score["cost_usd"] = cost

        logger.info(
            "%s/%s (%.0fs, $%.2f) — loc=%s fix=%s approved=%s pass=%s",
            bug["ticket_id"], pipeline, duration, cost,
            score["localization_hit"], score["fix_generated"],
            score["review_approved"], score["full_pass"],
        )

        return EvalCaseResult(
            ticket_id=bug["ticket_id"],
            pipeline=pipeline,
            score=score,
            trace_report=trace.to_report(),
            duration_seconds=duration,
            cost_usd=cost,
            error=result.get("error", ""),
        )

    def _invoke_pipeline(self, pipeline: str, work_order: dict, trace: Any) -> dict:
        """Call the appropriate pipeline entry point."""
        if pipeline == "react":
            from agent.react_pipeline import run_ticket_react
            return run_ticket_react(work_order, trace=trace, dry_run=self.dry_run)
        else:
            from agent.pipeline import run_ticket
            return run_ticket(work_order, trace=trace, dry_run=self.dry_run)

    def _build_comparison(self, report: EvalRunReport) -> dict:
        """Build A/B comparison between two pipelines."""
        p1, p2 = self.pipelines[0], self.pipelines[1]
        s1 = report.summary.get(p1, {})
        s2 = report.summary.get(p2, {})

        comparison_metrics = [
            "pass_rate", "localization_accuracy", "fix_rate",
            "approval_rate", "patch_correctness_avg", "avg_cost_usd",
            "avg_duration_seconds",
        ]

        deltas = {}
        for metric in comparison_metrics:
            v1 = s1.get(metric, 0.0)
            v2 = s2.get(metric, 0.0)
            deltas[metric] = {
                p1: v1,
                p2: v2,
                "delta": round(v2 - v1, 4),
            }

        # Per-bug comparison
        per_bug = []
        bug_ids = list(dict.fromkeys(r.ticket_id for r in report.results))
        for tid in bug_ids:
            r1 = next((r for r in report.results if r.ticket_id == tid and r.pipeline == p1), None)
            r2 = next((r for r in report.results if r.ticket_id == tid and r.pipeline == p2), None)
            per_bug.append({
                "ticket_id": tid,
                p1: {
                    "pass": r1.score.get("full_pass", False) if r1 else False,
                    "cost": r1.cost_usd if r1 else 0,
                    "duration": r1.duration_seconds if r1 else 0,
                },
                p2: {
                    "pass": r2.score.get("full_pass", False) if r2 else False,
                    "cost": r2.cost_usd if r2 else 0,
                    "duration": r2.duration_seconds if r2 else 0,
                },
                "winner": _pick_winner(r1, r2, p1, p2),
            })

        # Overall winner
        pass1 = s1.get("pass_rate", 0)
        pass2 = s2.get("pass_rate", 0)
        if pass1 > pass2:
            overall_winner = p1
        elif pass2 > pass1:
            overall_winner = p2
        else:
            # Tiebreak on cost
            cost1 = s1.get("avg_cost_usd", 0)
            cost2 = s2.get("avg_cost_usd", 0)
            overall_winner = p1 if cost1 <= cost2 else p2

        return {
            "pipelines": [p1, p2],
            "deltas": deltas,
            "per_bug": per_bug,
            "overall_winner": overall_winner,
        }

    def _persist_report(self, report: EvalRunReport) -> None:
        """Save report to eval/results/."""
        # Timestamped report
        report_file = self.results_dir / f"report_{report.run_id}_{int(report.timestamp)}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Report written to %s", report_file)

        # Latest symlink (overwrite)
        latest_file = self.results_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Append to history
        self._append_history(report)

    def _append_history(self, report: EvalRunReport) -> None:
        """Append summary to eval_history.json for trend tracking."""
        history_file = self.results_dir / "eval_history.json"
        history: list[dict] = []

        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
            except (json.JSONDecodeError, OSError):
                history = []

        entry = {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "total_bugs": report.total_bugs,
            "pipelines": report.pipelines,
        }
        for pipeline, summary in report.summary.items():
            entry[f"{pipeline}_pass_rate"] = summary.get("pass_rate", 0)
            entry[f"{pipeline}_avg_cost"] = summary.get("avg_cost_usd", 0)

        history.append(entry)
        history = history[-100:]  # Keep last 100 runs

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bug_to_work_order(bug: dict, repo_path: Path) -> dict:
    """Convert EvalBug to pipeline work_order format."""
    return {
        "ticket_id": bug["ticket_id"],
        "title": bug.get("title", ""),
        "description": bug.get("description", ""),
        "repo_name": bug.get("repo_name", bug["ticket_id"].lower()),
        "repo_path": str(repo_path),
        "priority": bug.get("priority", "medium"),
        "comments": bug.get("comments", []),
    }


def _error_score(bug: dict, pipeline: str, error: str) -> dict:
    """Generate a failure score for a bug that couldn't be run."""
    return {
        "ticket_id": bug["ticket_id"],
        "title": bug.get("title", ""),
        "pipeline": pipeline,
        "localization_hit": False,
        "root_cause_match": False,
        "fix_generated": False,
        "review_approved": False,
        "confidence": 0.0,
        "patch_correctness": 0.0,
        "multi_file_complete": False,
        "test_pass": False,
        "patch_hits_target": False,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "tool_call_count": 0,
        "status": "failed",
        "error": error,
        "full_pass": False,
    }


def _pick_winner(
    r1: EvalCaseResult | None, r2: EvalCaseResult | None, p1: str, p2: str
) -> str:
    """Pick winner for a single bug comparison."""
    if r1 is None and r2 is None:
        return "tie"
    if r1 is None:
        return p2
    if r2 is None:
        return p1

    pass1 = r1.score.get("full_pass", False)
    pass2 = r2.score.get("full_pass", False)

    if pass1 and not pass2:
        return p1
    if pass2 and not pass1:
        return p2
    if not pass1 and not pass2:
        return "neither"

    # Both passed — tiebreak on cost
    if r1.cost_usd < r2.cost_usd:
        return f"{p1} (cheaper)"
    elif r2.cost_usd < r1.cost_usd:
        return f"{p2} (cheaper)"
    return "tie"
