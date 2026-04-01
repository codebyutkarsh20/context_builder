"""
agent.py — API endpoints for controlling the AI Deploy Agent pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# ---------------------------------------------------------------------------
# SQLite-backed job persistence
# ---------------------------------------------------------------------------

_DB_PATH = Path(os.environ.get("AGENT_DB_PATH", "data/agent_jobs.db"))


def _init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                stage TEXT,
                iteration_count INTEGER,
                result TEXT,
                error TEXT,
                debug INTEGER,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                pr_url TEXT,
                verdict TEXT NOT NULL,
                comment TEXT,
                bug_category TEXT,
                fault_files TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def _save_job(job: dict) -> None:
    """Persist a completed job to SQLite."""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_jobs
                (job_id, status, stage, iteration_count, result, error, debug, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                job.get("job_id"),
                job.get("status"),
                job.get("stage"),
                job.get("iteration_count", 0),
                json.dumps(job.get("result")),
                job.get("error"),
                1 if job.get("debug") else 0,
            ))
            conn.commit()
    except Exception as e:
        logger.warning("Failed to persist job: %s", e)


def _load_jobs_from_db() -> dict:
    """Load completed jobs from SQLite on startup."""
    if not _DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("SELECT * FROM agent_jobs").fetchall()
        jobs = {}
        for row in rows:
            job_id, status, stage, iteration_count, result_json, error, debug, _ = row
            jobs[job_id] = {
                "job_id": job_id,
                "status": status,
                "stage": stage or "",
                "iteration_count": iteration_count or 0,
                "result": json.loads(result_json) if result_json else None,
                "error": error or "",
                "debug": bool(debug),
            }
        return jobs
    except Exception as e:
        logger.warning("Failed to load jobs from DB: %s", e)
        return {}


# In-memory job store with cleanup
# Completed jobs are also persisted to SQLite so they survive restarts.
# Active/running jobs are memory-only (they'll restart anyway).
_init_db()
_agent_jobs: dict[str, dict] = _load_jobs_from_db()
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
    debug: bool = False  # Enable tracing/observability
    dry_run: bool = False  # Run pipeline but skip PR creation + feature flags

    @model_validator(mode="after")
    def require_ticket_or_description(self) -> "RunTicketRequest":
        if not self.ticket_id and not self.description:
            raise ValueError(
                "Provide either 'ticket_id' (to run an existing ticket) "
                "or 'description' (to describe the bug inline)."
            )
        return self


class AgentJobStatus(BaseModel):
    job_id: str
    status: str
    stage: str = ""
    iteration_count: int = 0
    result: dict | None = None
    error: str = ""
    debug: bool = False


class FeedbackRequest(BaseModel):
    job_id: str
    verdict: str          # "approved" | "rejected" | "modified"
    pr_url: str | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool
    feedback_id: int


class FeedbackStats(BaseModel):
    total: int
    approved: int
    rejected: int
    modified: int
    approval_rate: float
    by_category: dict[str, dict[str, int]]
    by_fault_file: dict[str, dict[str, int]]


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
            "debug": req.debug,
            "_created": time.time(),
        }

    thread = threading.Thread(
        target=_run_pipeline, args=(job_id, work_order, req.debug, req.dry_run), daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "pending", "debug": req.debug, "dry_run": req.dry_run}


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


_SAMPLE_TICKETS_PATH = Path(__file__).parent.parent / "agent" / "intake" / "sample_tickets.json"
# Resolves to: backend/agent/intake/sample_tickets.json


@router.get("/agent/tickets")
def list_mock_tickets() -> list[dict]:
    """List available mock bug tickets for testing."""
    from agent.intake.mock_jira import load_tickets
    return load_tickets(_SAMPLE_TICKETS_PATH)


@router.post("/agent/run-mock/{ticket_id}")
def run_mock_ticket(ticket_id: str, debug: bool = Query(False)) -> dict:
    """Run a specific mock ticket through the pipeline."""
    from agent.intake.mock_jira import get_ticket

    ticket = get_ticket(ticket_id, _SAMPLE_TICKETS_PATH)
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
            "debug": debug,
            "_created": time.time(),
        }

    thread = threading.Thread(
        target=_run_pipeline, args=(job_id, ticket, debug), daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "pending", "ticket_id": ticket_id, "debug": debug}


# ---------------------------------------------------------------------------
# Human feedback endpoints
# ---------------------------------------------------------------------------

@router.post("/agent/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """
    Record a human's review decision on a pipeline PR.

    Call this after a human approves, rejects, or modifies a PR created by the agent.
    Used to track accuracy over time and improve future runs.
    """
    if req.verdict not in ("approved", "rejected", "modified"):
        raise HTTPException(400, detail="verdict must be 'approved', 'rejected', or 'modified'")

    # Look up the job to get bug_category and fault_files
    job = _agent_jobs.get(req.job_id)
    bug_category = None
    fault_files_json = "[]"

    if job:
        result = job.get("result") or {}
        localization = result.get("localization") or {}
        fault_files_json = json.dumps(localization.get("fault_files", []))
        # bug_category is stored in work_order inside the job
        # It may not be directly in the job status — that's ok

    try:
        with sqlite3.connect(_DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO human_feedback (job_id, pr_url, verdict, comment, bug_category, fault_files)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (req.job_id, req.pr_url, req.verdict, req.comment, bug_category, fault_files_json))
            conn.commit()
            feedback_id = cursor.lastrowid

        logger.info("Human feedback recorded: job=%s verdict=%s", req.job_id, req.verdict)
        return FeedbackResponse(ok=True, feedback_id=feedback_id)
    except Exception as e:
        logger.error("Failed to store feedback: %s", e)
        raise HTTPException(500, detail=str(e))


@router.get("/agent/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """
    Aggregate statistics on human review verdicts.

    Use to track: approval rate over time, which bug categories succeed,
    which files have the most rejections.
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("""
                SELECT verdict, bug_category, fault_files, created_at
                FROM human_feedback
                ORDER BY created_at DESC
                LIMIT 500
            """).fetchall()
    except Exception:
        rows = []

    total = len(rows)
    approved = sum(1 for r in rows if r[0] == "approved")
    rejected = sum(1 for r in rows if r[0] == "rejected")
    modified = sum(1 for r in rows if r[0] == "modified")

    # By category
    by_category: dict[str, dict[str, int]] = {}
    for verdict, category, _, _ in rows:
        if category:
            cat = by_category.setdefault(category, {"approved": 0, "rejected": 0, "modified": 0})
            cat[verdict] = cat.get(verdict, 0) + 1

    # By fault file (top 20)
    file_counts: dict[str, dict[str, int]] = {}
    for verdict, _, fault_files_json, _ in rows:
        try:
            files = json.loads(fault_files_json or "[]")
        except Exception:
            files = []
        for f in files[:3]:  # Don't over-index on large fault sets
            fname = os.path.basename(f)
            fc = file_counts.setdefault(fname, {"approved": 0, "rejected": 0, "modified": 0})
            fc[verdict] = fc.get(verdict, 0) + 1

    # Sort by total activity, keep top 20
    top_files = dict(
        sorted(file_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]
    )

    return FeedbackStats(
        total=total,
        approved=approved,
        rejected=rejected,
        modified=modified,
        approval_rate=approved / total if total > 0 else 0.0,
        by_category=by_category,
        by_fault_file=top_files,
    )


@router.get("/agent/feedback")
async def list_feedback(limit: int = 50):
    """List recent human feedback entries."""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("""
                SELECT id, job_id, pr_url, verdict, comment, bug_category, created_at
                FROM human_feedback
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [
            {
                "id": r[0], "job_id": r[1], "pr_url": r[2],
                "verdict": r[3], "comment": r[4],
                "bug_category": r[5], "created_at": r[6],
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# SSE trace streaming
# ---------------------------------------------------------------------------

@router.get("/agent/trace/{job_id}")
def stream_trace(job_id: str):
    """Stream pipeline trace events via Server-Sent Events (SSE)."""
    with _agent_jobs_lock:
        job = _agent_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        trace = job.get("_trace")
        if not trace:
            raise HTTPException(status_code=404, detail="Tracing not enabled for this job (pass debug=true)")

    def event_generator():
        import asyncio
        max_sse_end = time.monotonic() + 3600  # 1-hour max lifetime
        q = trace.subscribe()
        try:
            # Catch up: send all existing events and track last index sent
            existing = trace.events_since(0)
            last_sent_idx = -1
            for evt in existing:
                yield f"data: {json.dumps(evt, default=str)}\n\n"
                last_sent_idx = evt.get("index", last_sent_idx)

            # Stream new events — deduplicate by index
            while True:
                if time.monotonic() > max_sse_end:
                    logger.info("SSE connection reached 1-hour max lifetime, closing (job %s)", job_id)
                    break
                try:
                    evt = q.get(timeout=30)
                    if evt is None:  # sentinel = trace complete
                        yield f"event: done\ndata: {{}}\n\n"
                        break
                    # Skip events already sent during catchup
                    if evt.index <= last_sent_idx:
                        continue
                    last_sent_idx = evt.index
                    try:
                        yield f"data: {json.dumps(evt.to_dict(), default=str)}\n\n"
                    except (GeneratorExit, asyncio.CancelledError):
                        logger.info("SSE client disconnected (job %s)", job_id)
                        return
                except queue.Empty:
                    try:
                        yield f": keepalive\n\n"
                    except (GeneratorExit, asyncio.CancelledError):
                        logger.info("SSE client disconnected during keepalive (job %s)", job_id)
                        return
        except (GeneratorExit, asyncio.CancelledError):
            logger.info("SSE generator cancelled (job %s)", job_id)
        finally:
            trace.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/trace/{job_id}/report")
def get_trace_report(job_id: str) -> dict:
    """Get the full trace report for a completed job."""
    # Try in-memory trace first
    with _agent_jobs_lock:
        job = _agent_jobs.get(job_id)
        if job:
            trace = job.get("_trace")
            if trace:
                return trace.to_report()

    # Try on-disk report
    report_path = DATA_DIR / "traces" / f"{job_id}.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    raise HTTPException(status_code=404, detail="Trace report not found")


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


def _run_pipeline(job_id: str, work_order: dict, debug: bool = False, dry_run: bool = False) -> None:
    """Run the LangGraph pipeline in a background thread."""
    trace = None
    try:
        with _agent_jobs_lock:
            _agent_jobs[job_id]["status"] = "running"
            _agent_jobs[job_id]["stage"] = "Starting pipeline"

        # Always create trace for token tracking and observability
        from agent.trace import RunTrace
        trace = RunTrace(job_id=job_id, enabled=True)
        with _agent_jobs_lock:
            _agent_jobs[job_id]["_trace"] = trace

        from agent.react_pipeline import run_ticket_react

        progress_cb = _make_progress_callback(job_id)
        result = run_ticket_react(work_order, progress_cb=progress_cb, trace=trace, dry_run=dry_run)

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
            completed_job = {**_agent_jobs[job_id], "job_id": job_id}

        # Persist terminal job to SQLite
        _save_job(completed_job)

        # Save trace report to disk
        if trace:
            try:
                trace.save_report(DATA_DIR / "traces" / f"{job_id}.json")
            except Exception as te:
                logger.warning("Failed to save trace report: %s", te)

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
            failed_job = {**_agent_jobs[job_id], "job_id": job_id}

        # Persist failed job to SQLite
        _save_job(failed_job)

        if trace:
            trace.emit("error", "pipeline", {"message": f"Pipeline crashed: {e}"})
            trace.complete()
