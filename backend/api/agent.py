"""
agent.py — API endpoints for controlling the AI Deploy Agent pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)

# In-memory job store with cleanup
_agent_jobs: dict[str, dict] = {}
_agent_jobs_lock = threading.Lock()
_MAX_AGENT_JOBS = 50


_STALE_RUNNING_SECONDS = 1800  # 30 minutes — running jobs older than this are considered stale


def _cleanup_old_jobs() -> None:
    """Remove oldest completed/stale jobs when store exceeds max size.

    Must be called while holding _agent_jobs_lock.

    Cleans up:
    1. Completed jobs (done / failed / escalated) — oldest first.
    2. Stale running jobs (started > 30 min ago) — guards against threads that
       crashed without updating their status, which would otherwise pin entries
       in memory forever.
    """
    if len(_agent_jobs) <= _MAX_AGENT_JOBS:
        return

    now = time.time()

    # Collect terminal jobs (safe to remove)
    removable = [
        (jid, job) for jid, job in _agent_jobs.items()
        if job.get("status") in ("done", "failed", "escalated")
    ]
    # Also collect stale running jobs (thread likely dead)
    removable += [
        (jid, job) for jid, job in _agent_jobs.items()
        if job.get("status") == "running"
        and (now - job.get("_created", now)) > _STALE_RUNNING_SECONDS
    ]

    # Sort by creation time (oldest first) and deduplicate
    seen: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for item in sorted(removable, key=lambda x: x[1].get("_created", 0)):
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)

    to_remove = len(_agent_jobs) - _MAX_AGENT_JOBS
    for jid, _ in unique[:to_remove]:
        del _agent_jobs[jid]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RunTicketRequest(BaseModel):
    ticket_id: str = ""
    title: str = ""
    description: str = ""
    repo_name: str = ""
    repo_path: str = ""
    priority: str = "medium"
    comments: list[str] = Field(default_factory=list)


class AgentJobStatus(BaseModel):
    job_id: str
    status: str
    stage: str = ""
    iteration_count: int = 0
    result: dict | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/agent/run")
def run_agent(req: RunTicketRequest) -> dict:
    """Run a bug ticket through the full agent pipeline (async)."""
    job_id = str(uuid.uuid4())

    work_order = {
        "ticket_id": req.ticket_id or f"MANUAL-{job_id[:8]}",
        "title": req.title,
        "description": req.description,
        "repo_name": req.repo_name,
        "repo_path": req.repo_path,
        "priority": req.priority,
        "comments": req.comments,
    }

    with _agent_jobs_lock:
        _cleanup_old_jobs()
        _agent_jobs[job_id] = {
            "status": "pending",
            "stage": "Queued",
            "iteration_count": 0,
            "result": None,
            "error": "",
            "_created": time.time(),
        }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, work_order), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@router.get("/agent/status/{job_id}")
def get_agent_status(job_id: str) -> AgentJobStatus:
    """Check the status of an agent pipeline run."""
    with _agent_jobs_lock:
        if job_id not in _agent_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = {k: v for k, v in _agent_jobs[job_id].items() if not k.startswith("_")}
    return AgentJobStatus(job_id=job_id, **job)


@router.get("/agent/jobs")
def list_agent_jobs() -> list[dict]:
    """List all agent jobs, newest first."""
    with _agent_jobs_lock:
        items = sorted(
            _agent_jobs.items(),
            key=lambda x: x[1].get("_created", 0),
            reverse=True,
        )
        jobs = []
        for jid, job in items:
            entry = {k: v for k, v in job.items() if not k.startswith("_")}
            entry["job_id"] = jid
            jobs.append(entry)
    return jobs


@router.get("/agent/tickets")
def list_mock_tickets() -> list[dict]:
    """List available mock bug tickets for testing."""
    from agent.intake.mock_jira import load_tickets
    return load_tickets()


@router.post("/agent/run-mock/{ticket_id}")
def run_mock_ticket(ticket_id: str) -> dict:
    """Run a specific mock ticket through the pipeline."""
    from agent.intake.mock_jira import get_ticket

    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Mock ticket '{ticket_id}' not found")

    job_id = str(uuid.uuid4())
    with _agent_jobs_lock:
        _cleanup_old_jobs()
        _agent_jobs[job_id] = {
            "status": "pending",
            "stage": "Queued",
            "iteration_count": 0,
            "result": None,
            "error": "",
            "_created": time.time(),
        }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, ticket), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "pending", "ticket_id": ticket_id}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _make_progress_callback(job_id: str):
    """Create a callback that updates the job store with live pipeline state."""
    def callback(state: dict) -> None:
        stage = state.get("status", "running")
        # Convert PipelineStatus enum to string if needed
        if hasattr(stage, "value"):
            stage = stage.value

        # Build partial result from whatever's available so far
        result = {}
        if state.get("intent"):
            result["intent"] = state["intent"]
        if state.get("localization"):
            result["localization"] = state["localization"]
        if state.get("repair"):
            result["repair"] = state["repair"]
        if state.get("review"):
            result["review"] = state["review"]
        if state.get("pr_url"):
            result["pr_url"] = state["pr_url"]
        if state.get("context_nodes"):
            result["context_nodes"] = state["context_nodes"]
        if state.get("test_result"):
            result["test_result"] = state["test_result"]

        with _agent_jobs_lock:
            _agent_jobs[job_id]["stage"] = stage
            _agent_jobs[job_id]["iteration_count"] = state.get("iteration_count", 0)
            if result:
                _agent_jobs[job_id]["result"] = result

    return callback


def _run_pipeline(job_id: str, work_order: dict) -> None:
    """Run the LangGraph pipeline in a background thread."""
    try:
        with _agent_jobs_lock:
            _agent_jobs[job_id]["status"] = "running"
            _agent_jobs[job_id]["stage"] = "Starting pipeline"

        from agent.pipeline import run_ticket

        progress_cb = _make_progress_callback(job_id)
        result = run_ticket(work_order, progress_cb=progress_cb)

        # Determine final status from pipeline outcome
        pipeline_status = result.get("status", "done")
        if hasattr(pipeline_status, "value"):
            pipeline_status = pipeline_status.value

        final_status = "done"
        if pipeline_status == "escalated":
            final_status = "escalated"
        elif pipeline_status == "failed":
            final_status = "failed"

        with _agent_jobs_lock:
            _agent_jobs[job_id].update({
                "status": final_status,
                "stage": pipeline_status,
                "iteration_count": result.get("iteration_count", 0),
                "result": {
                    "intent": result.get("intent"),
                    "localization": result.get("localization"),
                    "repair": result.get("repair"),
                    "review": result.get("review"),
                    "pr_url": result.get("pr_url", ""),
                    "context_nodes": result.get("context_nodes", 0),
                    "test_result": result.get("test_result", ""),
                    "sandbox_path": result.get("sandbox_path", ""),
                    "branch_name": result.get("branch_name", ""),
                    "patches_applied": result.get("patches_applied", 0),
                    "repo_name": result.get("work_order", {}).get("repo_name", ""),
                    "repo_path": result.get("work_order", {}).get("repo_path", ""),
                },
                "error": result.get("error", ""),
            })

        logger.info(
            "Agent pipeline complete for job %s: status=%s, verdict=%s, iterations=%d",
            job_id,
            final_status,
            result.get("review", {}).get("verdict", "?"),
            result.get("iteration_count", 0),
        )

    except Exception as e:
        logger.exception("Agent pipeline failed for job %s", job_id)
        with _agent_jobs_lock:
            _agent_jobs[job_id].update({
                "status": "failed",
                "stage": "Error",
                "error": str(e),
            })
