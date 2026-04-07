"""
eval/runner.py — Eval suite runner with timeout isolation and trace persistence.

Runs bugs through the ReAct pipeline, scores results, persists traces per-run.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
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
        build_graph: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.pipelines = pipelines or ["react"]
        self.timeout_per_case = timeout_per_case
        self.dry_run = not create_prs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.build_graph = build_graph
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
            graph_less=not self.build_graph,
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

            # Set up isolated virtualenv for external (SWE-bench) repos
            if not bug.get("local_repo_path"):
                try:
                    self.repo_manager.setup_venv(repo_path, bug)
                except Exception as e:
                    logger.warning(
                        "Venv setup failed for %s (%s) — tests may use system Python",
                        bug["ticket_id"], e,
                    )

            # Build knowledge graph (if requested)
            if self.build_graph:
                repo_name = bug.get("repo_name") or bug["ticket_id"].lower().replace("-", "_")
                try:
                    from agent.eval.graph_builder import build_eval_graph, DATA_DIR
                    logger.info("Building graph for %s...", repo_name)
                    build_eval_graph(repo_name, repo_path, data_dir=DATA_DIR)
                    logger.info("Graph ready for %s", repo_name)
                except Exception as e:
                    logger.warning(
                        "Graph build failed for %s (%s) — continuing without graph",
                        repo_name, e,
                    )

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
        """Run one bug through one pipeline with process-level timeout isolation.

        Each case runs in a child process that can be killed on timeout.
        The child writes its result + trace to a temp file. If the process
        exceeds the timeout, it is killed (SIGKILL) and the case is marked failed.
        """
        work_order = _bug_to_work_order(bug, repo_path)
        case_start = time.time()

        # Temp files for IPC — child writes result + trace, parent reads
        result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="eval_result_")
        os.close(result_fd)

        proc = multiprocessing.Process(
            target=_run_case_in_child,
            args=(pipeline, work_order, result_path),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=self.timeout_per_case)

        if proc.is_alive():
            logger.error(
                "%s/%s timed out after %ds — killing process",
                bug["ticket_id"], pipeline, self.timeout_per_case,
            )
            proc.kill()
            proc.join(timeout=5)
            result = {"status": "failed", "error": f"Timeout after {self.timeout_per_case}s (killed)"}
            trace_report = {}
        else:
            # Read result from temp file
            try:
                with open(result_path) as f:
                    child_output = json.load(f)
                result = child_output.get("result", {"status": "failed", "error": "No result from child"})
                trace_report = child_output.get("trace_report", {})
            except Exception as e:
                logger.error("Failed to read child result: %s", e)
                result = {"status": "failed", "error": f"Child process error: {e}"}
                trace_report = {}

        # Clean up temp file
        try:
            os.unlink(result_path)
        except OSError:
            pass

        duration = round(time.time() - case_start, 2)

        # Score
        score = score_case(result, bug, pipeline)
        score["duration_seconds"] = duration

        # Extract cost
        cost = result.get("cost_usd", 0.0) or 0.0
        if not cost and trace_report:
            cost = trace_report.get("summary", {}).get("total_cost_usd", 0.0)
        score["cost_usd"] = cost

        logger.info(
            "%s/%s (%.0fs, $%.2f) — loc=%s fix=%s approved=%s pass=%s",
            bug["ticket_id"], pipeline, duration, cost,
            score["localization_hit"], score["fix_generated"],
            score["review_approved"], score["full_pass"],
        )

        # Save individual trace JSON — scoped by run_id
        trace_dir = self.results_dir / "traces" / (run_id or "unknown")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{bug['ticket_id']}_{pipeline}.json"
        if trace_report:
            try:
                trace_path.write_text(json.dumps(trace_report, indent=2, default=str))
            except Exception as e:
                logger.warning("Failed to save trace: %s", e)

        # Build compact trace summary
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
            trace_report=trace_report or None,
            duration_seconds=duration,
            cost_usd=cost,
            error=result.get("error", ""),
            trace_summary=trace_summary,
        )

    @staticmethod
    def _invoke_pipeline(pipeline: str, work_order: dict, trace: Any) -> dict:
        """Call the ReAct pipeline entry point.

        ``react``    — full v3.0 pipeline (BRT + Scout + Leiden + speculative review)
        ``react_v2`` — v2.0 baseline (no BRT, no Scout, no Leiden, no spec review)
                       Useful for A/B comparison: run --pipeline react react_v2
        """
        from agent.react_pipeline import run_ticket_react
        # react_v2 = disable new features to simulate v2.0 baseline
        disable_new = (pipeline == "react_v2")
        return run_ticket_react(
            work_order, trace=trace, dry_run=True,
            disable_brt=disable_new,
            disable_scout=disable_new,
        )

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

def _run_case_in_child(pipeline: str, work_order: dict, result_path: str) -> None:
    """Entry point for child process. Runs pipeline, writes result + trace to file.

    This runs in a separate process so it can be killed on timeout.
    All output goes to a JSON file that the parent reads.
    """
    # Child process must load .env since it doesn't inherit parent's dotenv state
    try:
        from dotenv import load_dotenv
        # Always try to load from the project root
        root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if root_env.exists():
            load_dotenv(root_env)
        else:
            load_dotenv()  # Fallback
    except ImportError:
        pass  # dotenv not installed, rely on OS env

    from agent.trace import RunTrace

    trace = RunTrace(
        job_id=f"{work_order.get('ticket_id', 'unknown')}_{pipeline}",
        enabled=True,
    )

    try:
        result = EvalRunner._invoke_pipeline(pipeline, work_order, trace)
    except Exception as e:
        result = {
            "status": "failed",
            "error": str(e),
            "_traceback": traceback.format_exc(),
        }

    trace_report = trace.to_report()

    # Write result + trace to the temp file for the parent to read
    try:
        with open(result_path, "w") as f:
            json.dump({"result": result, "trace_report": trace_report}, f, default=str)
    except Exception:
        pass  # Parent will detect missing/corrupt file


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
