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
