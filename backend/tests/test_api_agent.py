"""
Tests for the Agent API endpoints — Phases 1.2, 2.2, 4.2

Verifies:
  - Thread safety of job store
  - repo_path acceptance
  - /api/agent/jobs endpoint
  - Job lifecycle (submit → poll → terminal)
  - Error handling (404, concurrent access)
"""

import sys
import threading
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://localhost:8000"


def api(method, path, **kwargs):
    return getattr(requests, method)(f"{BASE}{path}", timeout=10, **kwargs)


# ── Job Submission ───────────────────────────────────────────────────

class TestJobSubmission:
    """POST /api/agent/run creates jobs correctly."""

    def test_basic_submission(self):
        resp = api("post", "/api/agent/run", json={
            "title": "Test bug",
            "description": "Something is broken",
            "repo_name": "crest-be",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_submission_with_repo_path(self):
        resp = api("post", "/api/agent/run", json={
            "title": "Test with path",
            "description": "Test repo_path passthrough",
            "repo_name": "crest-be",
            "repo_path": "/Users/utkarshpatidar/work/crest-work/crest-be",
        })
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_submission_with_all_fields(self):
        resp = api("post", "/api/agent/run", json={
            "ticket_id": "PROJ-999",
            "title": "Full field test",
            "description": "All fields populated",
            "repo_name": "crest-be",
            "repo_path": "/some/path",
            "priority": "critical",
            "comments": ["urgent", "prod down"],
        })
        assert resp.status_code == 200

    def test_empty_submission_gets_auto_id(self):
        resp = api("post", "/api/agent/run", json={
            "title": "No ticket ID",
            "description": "Auto-generate ticket ID",
            "repo_name": "crest-be",
        })
        data = resp.json()
        assert data["job_id"]  # Should have auto-generated ID

    def test_minimal_submission(self):
        """Even a nearly empty request succeeds (fields have defaults)."""
        resp = api("post", "/api/agent/run", json={})
        assert resp.status_code == 200


# ── Job Status ───────────────────────────────────────────────────────

class TestJobStatus:
    """GET /api/agent/status/{job_id} returns correct data."""

    def test_valid_job_returns_status(self):
        submit = api("post", "/api/agent/run", json={
            "title": "Status test",
            "description": "Check status",
            "repo_name": "crest-be",
        })
        job_id = submit.json()["job_id"]

        resp = api("get", f"/api/agent/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "done", "failed", "escalated")
        assert "stage" in data
        assert "iteration_count" in data

    def test_nonexistent_job_returns_404(self):
        resp = api("get", "/api/agent/status/nonexistent-id-12345")
        assert resp.status_code == 404

    def test_invalid_job_id_format(self):
        resp = api("get", "/api/agent/status/!!!invalid!!!")
        assert resp.status_code == 404


# ── Jobs List ────────────────────────────────────────────────────────

class TestJobsList:
    """GET /api/agent/jobs returns all jobs."""

    def test_returns_list(self):
        resp = api("get", "/api/agent/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_jobs_have_required_fields(self):
        # Submit a job first
        api("post", "/api/agent/run", json={
            "title": "Fields test",
            "description": "Check fields",
            "repo_name": "test",
        })
        time.sleep(0.5)

        resp = api("get", "/api/agent/jobs")
        jobs = resp.json()
        assert len(jobs) > 0

        job = jobs[0]
        assert "job_id" in job
        assert "status" in job
        assert "stage" in job

    def test_submitted_job_appears_in_list(self):
        submit = api("post", "/api/agent/run", json={
            "title": "Appear in list",
            "description": "Should show up",
            "repo_name": "test",
        })
        job_id = submit.json()["job_id"]
        time.sleep(0.5)

        jobs = api("get", "/api/agent/jobs").json()
        job_ids = [j["job_id"] for j in jobs]
        assert job_id in job_ids

    def test_newest_first_ordering(self):
        """Jobs should be ordered newest first."""
        id1 = api("post", "/api/agent/run", json={"title": "First", "repo_name": "t"}).json()["job_id"]
        time.sleep(0.1)
        id2 = api("post", "/api/agent/run", json={"title": "Second", "repo_name": "t"}).json()["job_id"]
        time.sleep(0.5)

        jobs = api("get", "/api/agent/jobs").json()
        job_ids = [j["job_id"] for j in jobs]
        # id2 should appear before id1
        assert job_ids.index(id2) < job_ids.index(id1)


# ── Thread Safety ────────────────────────────────────────────────────

class TestThreadSafety:
    """Concurrent job operations don't crash."""

    def test_concurrent_submissions(self):
        """Submit 5 jobs simultaneously."""
        results = [None] * 5
        errors = [None] * 5

        def submit(idx):
            try:
                resp = api("post", "/api/agent/run", json={
                    "title": f"Concurrent {idx}",
                    "description": f"Thread safety test {idx}",
                    "repo_name": "test",
                })
                results[idx] = resp.json()
            except Exception as e:
                errors[idx] = str(e)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(e is None for e in errors), f"Errors: {errors}"
        assert all(r is not None for r in results), "All submissions returned"

        # All job IDs should be unique
        ids = [r["job_id"] for r in results]
        assert len(set(ids)) == 5

    def test_concurrent_status_and_submit(self):
        """Mix status checks and submissions concurrently."""
        submit_resp = api("post", "/api/agent/run", json={
            "title": "Anchor job",
            "repo_name": "test",
        })
        anchor_id = submit_resp.json()["job_id"]
        errors = []

        def poll():
            try:
                api("get", f"/api/agent/status/{anchor_id}")
            except Exception as e:
                errors.append(str(e))

        def submit():
            try:
                api("post", "/api/agent/run", json={"title": "bg", "repo_name": "t"})
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=poll))
            threads.append(threading.Thread(target=submit))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent errors: {errors}"


# ── Mock Tickets ─────────────────────────────────────────────────────

class TestMockTickets:
    """GET /api/agent/tickets endpoint."""

    def test_returns_list(self):
        resp = api("get", "/api/agent/tickets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_nonexistent_mock_ticket_returns_404(self):
        resp = api("post", "/api/agent/run-mock/NONEXISTENT-999")
        assert resp.status_code == 404
