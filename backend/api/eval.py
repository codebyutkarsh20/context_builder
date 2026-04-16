"""
eval.py — API endpoints for triggering and viewing evaluation suite runs.

Updated to use the new unified eval package (agent.eval) with A/B support.
Falls back to legacy eval_suite.py for backward compatibility.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["eval"])
logger = logging.getLogger(__name__)

# In-memory store for eval jobs (one per repo, keeps latest only)
_eval_jobs: dict[str, dict] = {}
_eval_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/eval/{repo}/run")
def run_eval_endpoint(
    repo: str,
    repo_path: str = "",
    pipeline: str = Query("both", description="Pipeline: 'fixed', 'react', or 'both'"),
    dataset: str = Query("eval/bugs.json", description="Path to bugs.json"),
    sentinel: bool = Query(False, description="Run only first 5 bugs"),
) -> dict:
    """Trigger an evaluation run in the background.

    Supports A/B comparison via pipeline='both'.
    """
    job_id = str(uuid.uuid4())

    with _eval_jobs_lock:
        existing = _eval_jobs.get(repo)
        if existing and existing.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Eval already running for repo '{repo}' (job_id={existing['job_id']})",
            )

        _eval_jobs[repo] = {
            "job_id": job_id,
            "repo": repo,
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": "",
            "pipeline": pipeline,
        }

    thread = threading.Thread(
        target=_run_eval_background,
        args=(repo, repo_path, job_id, pipeline, dataset, sentinel),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "repo": repo, "status": "running", "pipeline": pipeline}


@router.get("/eval/{repo}/results")
def get_eval_results(repo: str) -> dict:
    """Return the latest evaluation results for a repository."""
    with _eval_jobs_lock:
        job = _eval_jobs.get(repo)

    if job and job.get("result"):
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "result": job["result"],
        }

    if job and job.get("status") == "running":
        return {
            "job_id": job["job_id"],
            "status": "running",
            "result": None,
        }

    # Fall back to disk (new eval package)
    from agent.eval.regression import load_previous_report

    report = load_previous_report("eval/results")
    if report:
        return {
            "job_id": report.run_id or None,
            "status": "done",
            "result": report._data,
        }

    # Legacy fallback
    try:
        from agent.eval_suite import load_latest_results
        persisted = load_latest_results(repo)
        if persisted:
            return {"job_id": None, "status": "done", "result": persisted}
    except ImportError:
        pass

    raise HTTPException(status_code=404, detail=f"No eval results found for repo '{repo}'")


# ---------------------------------------------------------------------------
# Trace browsing — historical eval traces
# ---------------------------------------------------------------------------

@router.get("/traces")
def list_trace_runs() -> list[dict]:
    """List all eval runs that have traces, newest first."""
    from pathlib import Path
    import json

    traces_dir = Path("eval/results/traces")
    if not traces_dir.exists():
        return []

    runs = []
    for run_dir in sorted(traces_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        bug_files = list(run_dir.glob("*.json"))
        # Read first trace for summary metadata
        summary = {}
        if bug_files:
            try:
                with open(bug_files[0]) as f:
                    first = json.load(f)
                summary = {
                    "started_at": first.get("started_at", ""),
                    "total_duration_ms": first.get("total_duration_ms", 0),
                }
            except Exception:
                pass
        runs.append({
            "run_id": run_dir.name,
            "bug_count": len(bug_files),
            "bugs": [f.stem.replace("_react", "") for f in sorted(bug_files)],
            **summary,
        })
    return runs


@router.get("/traces/{run_id}")
def list_trace_bugs(run_id: str) -> list[dict]:
    """List bugs in a specific trace run with summary metrics."""
    from pathlib import Path
    import json

    trace_dir = Path("eval/results/traces") / run_id
    if not trace_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trace run '{run_id}' not found")

    bugs = []
    for tf in sorted(trace_dir.glob("*.json")):
        try:
            with open(tf) as f:
                t = json.load(f)
            outcome = {}
            for e in t.get("events", []):
                if e.get("event_type") == "run_outcome":
                    outcome = e.get("data", {})
                    break
            bugs.append({
                "bug_id": tf.stem.replace("_react", ""),
                "outcome": outcome.get("outcome", "unknown"),
                "tool_calls": outcome.get("tool_call_count", 0),
                "cost_usd": outcome.get("cost_usd", 0),
                "elapsed_s": outcome.get("elapsed_seconds", 0),
                "submitted": outcome.get("submitted", False),
                "tests_passed": outcome.get("tests_passed", False),
                "review_verdict": outcome.get("review_verdict", ""),
            })
        except Exception:
            bugs.append({"bug_id": tf.stem, "outcome": "error"})
    return bugs


@router.get("/traces/{run_id}/{bug_id}")
def get_trace_detail(run_id: str, bug_id: str) -> dict:
    """Get full trace for a specific bug in a run."""
    from pathlib import Path
    import json

    trace_file = Path("eval/results/traces") / run_id / f"{bug_id}_react.json"
    if not trace_file.exists():
        # Try without _react suffix
        trace_file = Path("eval/results/traces") / run_id / f"{bug_id}.json"
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace '{bug_id}' not found in run '{run_id}'")

    with open(trace_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _run_eval_background(
    repo: str, repo_path: str, job_id: str,
    pipeline: str, dataset: str, sentinel: bool,
) -> None:
    """Run the eval suite in a background thread using the new eval package."""
    try:
        from pathlib import Path
        from agent.eval.runner import EvalRunner

        pipelines = ["fixed", "react"] if pipeline == "both" else [pipeline]

        runner = EvalRunner(
            dataset_path=Path(dataset),
            pipelines=pipelines,
        )
        report = runner.run(sentinel=sentinel)

        with _eval_jobs_lock:
            _eval_jobs[repo].update({
                "status": "done",
                "finished_at": time.time(),
                "result": report.to_dict(),
            })

        # Log summary per pipeline
        for p, summary in report.summary.items():
            logger.info(
                "Eval %s/%s: %d bugs, pass_rate=%.0f%%",
                repo, p, summary.get("total", 0),
                summary.get("pass_rate", 0) * 100,
            )

    except Exception as e:
        logger.exception("Eval run failed for repo %s", repo)
        with _eval_jobs_lock:
            _eval_jobs[repo].update({
                "status": "failed",
                "finished_at": time.time(),
                "error": str(e),
            })
