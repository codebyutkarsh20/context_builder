"""
eval/runner.py — Eval suite runner with timeout isolation and trace persistence.

Runs bugs through the ReAct pipeline, scores results, persists traces per-run.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.eval.dataset import load_eval_dataset
from agent.eval.scoring import score_case, build_summary

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
    trace_summary: dict | None = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop full trace from serialized report (too large for aggregate JSON)
        d.pop("trace_report", None)
        return d


@dataclass
class EvalRunReport:
    """Complete eval run report."""
    run_id: str
    timestamp: float
    dataset_path: str
    total_bugs: int
    pipelines: list[str]
    results: list[EvalCaseResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    comparison: dict | None = None
    graph_less: bool = False
    commit_sha: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "total_bugs": self.total_bugs,
            "pipelines": self.pipelines,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "comparison": self.comparison,
            "graph_less": self.graph_less,
            "commit_sha": self.commit_sha,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Run eval bugs through the ReAct pipeline."""

    def __init__(
        self,
        dataset_path: str | Path = "eval/bugs.json",
        pipelines: list[str] | None = None,
        timeout_per_case: int = EVAL_CASE_TIMEOUT,
        create_prs: bool = False,
        results_dir: Path | str = RESULTS_DIR,
        repo_cache_dir: Path | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.pipelines = pipelines or ["react"]
        self.timeout_per_case = timeout_per_case
        self.dry_run = not create_prs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        from agent.eval.repo_manager import RepoManager
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

        # Capture commit SHA for cross-run comparison
        commit_sha = ""
        try:
            commit_sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            pass

        report = EvalRunReport(
            run_id=run_id,
            timestamp=time.time(),
            dataset_path=str(self.dataset_path),
            total_bugs=len(bugs),
            pipelines=list(self.pipelines),
            graph_less=True,
            commit_sha=commit_sha,
        )

        logger.info(
            "Starting eval run %s (commit %s): %d bugs × %d pipeline(s) = %d cases",
            run_id, commit_sha or "?", len(bugs), len(self.pipelines), len(bugs) * len(self.pipelines),
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
                case_result = self._run_single_case(bug, repo_path, pipeline, run_id)
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
        self, bug: dict, repo_path: Path, pipeline: str, run_id: str = ""
    ) -> EvalCaseResult:
        """Run one bug through one pipeline with timeout isolation.

        Uses ThreadPoolExecutor WITHOUT context manager to avoid shutdown(wait=True)
        blocking on timeout. On timeout, shutdown(wait=False, cancel_futures=True)
        abandons the worker immediately.
        """
        from agent.trace import RunTrace

        trace = RunTrace(job_id=f"{bug['ticket_id']}_{pipeline}", enabled=True)
        work_order = _bug_to_work_order(bug, repo_path)

        case_start = time.time()
        result: dict

        # Do NOT use `with executor:` — its __exit__ calls shutdown(wait=True)
        # which blocks until the worker finishes, defeating the timeout.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self._invoke_pipeline, pipeline, work_order, trace
            )
            result = future.result(timeout=self.timeout_per_case)
        except concurrent.futures.TimeoutError:
            logger.error(
                "%s/%s timed out after %ds", bug["ticket_id"], pipeline, self.timeout_per_case
            )
            result = {"status": "failed", "error": f"Timeout after {self.timeout_per_case}s"}
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.exception("%s/%s crashed", bug["ticket_id"], pipeline)
            result = {"status": "failed", "error": str(e), "_traceback": traceback.format_exc()}
        else:
            executor.shutdown(wait=False)

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

        trace_report = trace.to_report()

        # Save individual trace JSON — scoped by run_id to prevent clobbering
        trace_dir = self.results_dir / "traces" / (run_id or "unknown")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{bug['ticket_id']}_{pipeline}.json"
        try:
            trace_path.write_text(json.dumps(trace_report, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save trace: %s", e)

        # Build compact trace summary for the eval report
        trace_summary = {
            "tool_calls": trace_report.get("summary", {}).get("total_tool_calls", 0),
            "llm_calls": trace_report.get("summary", {}).get("total_llm_calls", 0),
            "total_tokens": trace_report.get("summary", {}).get("total_tokens", 0),
            "cost_usd": trace_report.get("summary", {}).get("total_cost_usd", 0.0),
            "phase_breakdown": trace_report.get("phase_breakdown", {}),
            "wasted_calls": trace_report.get("wasted_calls", {}),
            "run_outcome": trace_report.get("run_outcome", {}),
            "context_timeline": trace_report.get("context_timeline", []),
            "guardrail_events": len(trace_report.get("guardrail_events", [])),
            "trace_path": str(trace_path),
        }

        return EvalCaseResult(
            ticket_id=bug["ticket_id"],
            pipeline=pipeline,
            score=score,
            trace_report=trace_report,
            duration_seconds=duration,
            cost_usd=cost,
            error=result.get("error", ""),
            trace_summary=trace_summary,
        )

    def _invoke_pipeline(self, pipeline: str, work_order: dict, trace: Any) -> dict:
        """Call the ReAct pipeline entry point."""
        from agent.react_pipeline import run_ticket_react
        return run_ticket_react(work_order, trace=trace, dry_run=self.dry_run)

    def _build_comparison(self, report: EvalRunReport) -> dict:
        """Build A/B comparison between two pipelines."""
        p1, p2 = self.pipelines[0], self.pipelines[1]
        s1 = report.summary.get(p1, {})
        s2 = report.summary.get(p2, {})

        comparison_metrics = [
            "pass_rate", "localization_accuracy", "fix_rate",
            "avg_tool_calls", "avg_cost_usd",
        ]
        deltas = {}
        for metric in comparison_metrics:
            v1 = s1.get(metric, 0)
            v2 = s2.get(metric, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                deltas[metric] = {p1: v1, p2: v2, "delta": round(v2 - v1, 4)}

        # Per-bug comparison
        per_bug: dict[str, dict] = {}
        for r in report.results:
            if r.ticket_id not in per_bug:
                per_bug[r.ticket_id] = {}
            per_bug[r.ticket_id][r.pipeline] = {
                "pass": r.score.get("full_pass", False),
                "cost": r.cost_usd,
                "tool_calls": r.score.get("tool_call_count", 0),
            }

        # Overall winner by pass rate
        pass1 = s1.get("pass_rate", 0)
        pass2 = s2.get("pass_rate", 0)
        if pass1 > pass2:
            overall_winner = p1
        elif pass2 > pass1:
            overall_winner = p2
        else:
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
            "commit_sha": report.commit_sha,
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
    """Convert eval bug dict to pipeline work_order."""
    repo_name = (
        bug.get("repo_name")
        or bug.get("repo")
        or bug["ticket_id"].lower()
    )
    return {
        "ticket_id": bug["ticket_id"],
        "title": bug.get("title", bug["ticket_id"]),
        "description": bug.get("description", ""),
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "priority": bug.get("priority", "medium"),
        "comments": bug.get("comments", []),
    }


def _pick_winner(r1: EvalCaseResult, r2: EvalCaseResult, p1: str, p2: str) -> str:
    """Pick a winner between two pipeline results for the same bug."""
    pass1 = r1.score.get("full_pass", False)
    pass2 = r2.score.get("full_pass", False)

    if pass1 and not pass2:
        return p1
    if pass2 and not pass1:
        return p2
    if pass1 and pass2:
        # Both pass — cheaper wins
        return f"{p2} (cheaper)" if r2.cost_usd < r1.cost_usd else f"{p1} (cheaper)"
    return "neither"


def _error_score(bug: dict, pipeline: str, error_msg: str) -> dict:
    """Build an error score dict for cases that fail before the pipeline runs."""
    return {
        "ticket_id": bug["ticket_id"],
        "pipeline": pipeline,
        "localization_hit": False,
        "root_cause_match": False,
        "fix_generated": False,
        "review_approved": False,
        "full_pass": False,
        "patch_hits_target": False,
        "error": error_msg,
    }
