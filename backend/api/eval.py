"""
eval.py — API endpoints for triggering and viewing evaluation suite runs.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["eval"])
logger = logging.getLogger(__name__)

# In-memory store for eval jobs (one per repo, keeps latest only)
_eval_jobs: dict[str, dict] = {}
_eval_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/eval/{repo}/run")
def run_eval_endpoint(repo: str, repo_path: str = "") -> dict:
    """Trigger an evaluation run for a repository in the background.

    Query params:
        repo_path: optional filesystem path to the repo.
    """
    job_id = str(uuid.uuid4())

    with _eval_jobs_lock:
        # Check if an eval is already running for this repo
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
        }

    thread = threading.Thread(
        target=_run_eval_background, args=(repo, repo_path, job_id), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "repo": repo, "status": "running"}


@router.get("/eval/{repo}/results")
def get_eval_results(repo: str) -> dict:
    """Return the latest evaluation results for a repository.

    Checks the in-memory job store first, then falls back to persisted results
    on disk.
    """
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

    # Fall back to disk
    from agent.eval_suite import load_latest_results

    persisted = load_latest_results(repo)
    if persisted:
        return {
            "job_id": None,
            "status": "done",
            "result": persisted,
        }

    raise HTTPException(status_code=404, detail=f"No eval results found for repo '{repo}'")


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _run_eval_background(repo: str, repo_path: str, job_id: str) -> None:
    """Run the eval suite in a background thread."""
    try:
        from agent.eval_suite import run_eval

        report = run_eval(repo, repo_path=repo_path)

        with _eval_jobs_lock:
            _eval_jobs[repo].update({
                "status": "done",
                "finished_at": time.time(),
                "result": report,
            })

        logger.info(
            "Eval run complete for %s: %d bugs, pass_rate=%.1f%%",
            repo,
            report.get("total", 0),
            report.get("summary", {}).get("pass_rate", 0) * 100,
        )

    except Exception as e:
        logger.exception("Eval run failed for repo %s", repo)
        with _eval_jobs_lock:
            _eval_jobs[repo].update({
                "status": "failed",
                "finished_at": time.time(),
                "error": str(e),
            })
