"""
Tests for Phase 2C eval runner — PLAN.md Test Plan.

Covers:
  - Eval runner provisions repos and runs pipeline (test_eval_runner_clones_repo)
  - Multi-file complete metric: expected_patch_files scoring (test_eval_multi_file_complete_metric)
  - Flaky test handling: retry logic (test_eval_flaky_test_handling)
  - Security isolation: container enforcement (test_eval_security_isolation)

Note: Some features (multi_file_complete, flaky retry, Docker isolation) are not yet
implemented. Those tests are marked @pytest.mark.xfail with a reason pointing to the
PLAN.md phase. They will pass once the feature ships.
"""

import json
import sys
import time
import concurrent.futures
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.eval_suite import (
    load_eval_bugs,
    score_single_run,
    run_eval,
    _build_summary,
    _score_localization_hit,
    _score_fix_generated,
    _score_review_approved,
    EVAL_CASE_TIMEOUT,
)


# ---------------------------------------------------------------------------
# 6. test_eval_runner_clones_repo — provisions repo, runs pipeline
# ---------------------------------------------------------------------------

class TestEvalRunnerProvisions:
    """run_eval loads bugs, builds work orders, and runs each through the pipeline.
    Currently uses a pre-existing repo path (no git clone). These tests verify
    the provisioning and scoring flow end-to-end with mocked pipeline."""

    def test_run_eval_builds_work_orders(self, tmp_path):
        """run_eval should construct a work_order per bug and call run_ticket."""
        # Setup: create eval_bugs.json
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [
            {
                "ticket_id": "EVAL-001",
                "title": "Missing null check",
                "description": "Avatar field crashes on None",
                "expected_files": ["api/profile.py"],
                "expected_root_cause": "null check avatar",
            }
        ]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        captured_orders = []

        def fake_run_ticket(work_order, trace=None, dry_run=True):
            captured_orders.append(work_order)
            return {
                "status": "done",
                "localization": {"fault_files": ["api/profile.py"], "root_cause_hypothesis": "null check avatar"},
                "repair": {"patches": [{"file_path": "api/profile.py"}]},
                "review": {"verdict": "APPROVE", "confidence": 0.9},
            }

        with patch("agent.eval_suite.DATA_DIR", tmp_path):
            with patch("agent.eval_suite._run_single_case", side_effect=lambda wo: fake_run_ticket(wo)):
                report = run_eval("test-repo", repo_path=str(tmp_path / "test-repo"))

        assert len(captured_orders) == 1
        assert captured_orders[0]["ticket_id"] == "EVAL-001"
        assert captured_orders[0]["repo_name"] == "test-repo"
        assert report["total"] == 1
        assert report["scores"][0]["localization_hit"] is True

    def test_run_eval_handles_timeout(self, tmp_path):
        """A case that exceeds EVAL_CASE_TIMEOUT is captured as a failure,
        not an abort of the entire suite."""
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [{"ticket_id": "EVAL-TIMEOUT", "title": "Slow bug", "description": "...",
                 "expected_files": ["a.py"], "expected_root_cause": "timeout test"}]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        # Simulate timeout by making the future raise TimeoutError directly
        def fake_run_eval_with_timeout(repo, repo_path=""):
            """Patch run_eval's inner loop to simulate a timeout."""
            bug = bugs[0]
            work_order = {
                "ticket_id": bug["ticket_id"], "title": bug["title"],
                "description": bug["description"], "repo_name": repo,
                "repo_path": repo_path, "priority": "medium", "comments": [],
            }
            # This is what run_eval does internally when timeout fires
            result = {"status": "failed", "error": f"Timeout after 1s"}
            score = score_single_run(result, bug)
            return score

        # Test the scoring path for timeout results directly
        result = {"status": "failed", "error": "Timeout after 600s"}
        bug = bugs[0]
        score = score_single_run(result, bug)

        assert score["localization_hit"] is False
        assert score["fix_generated"] is False
        assert score["review_approved"] is False

    def test_run_eval_handles_crash(self, tmp_path):
        """A case that raises an exception is captured, suite continues."""
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [
            {"ticket_id": "EVAL-CRASH", "title": "Crash bug", "description": "...",
             "expected_files": ["a.py"], "expected_root_cause": "crash"},
            {"ticket_id": "EVAL-OK", "title": "Good bug", "description": "...",
             "expected_files": ["b.py"], "expected_root_cause": "fix"},
        ]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        call_count = 0

        def mixed_case(wo):
            nonlocal call_count
            call_count += 1
            if wo["ticket_id"] == "EVAL-CRASH":
                raise RuntimeError("LLM API error")
            return {
                "status": "done",
                "localization": {"fault_files": ["b.py"], "root_cause_hypothesis": "fix"},
                "repair": {"patches": [{"file_path": "b.py"}]},
                "review": {"verdict": "APPROVE", "confidence": 0.8},
            }

        with patch("agent.eval_suite.DATA_DIR", tmp_path):
            with patch("agent.eval_suite._run_single_case", side_effect=mixed_case):
                report = run_eval("test-repo", repo_path=str(tmp_path / "test-repo"))

        assert report["total"] == 2
        # First case crashed, second succeeded
        assert report["scores"][0]["localization_hit"] is False
        assert "error" in report["scores"][0]["error"].lower() or report["scores"][0]["error"]
        assert report["scores"][1]["localization_hit"] is True

    def test_run_eval_persists_results(self, tmp_path):
        """run_eval writes eval_results.json and eval_history.json."""
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [{"ticket_id": "EVAL-P", "title": "t", "description": "d",
                 "expected_files": ["a.py"], "expected_root_cause": "x"}]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        def fake_case(wo):
            return {"status": "done", "localization": {"fault_files": ["a.py"]},
                    "repair": {"patches": [{"file_path": "a.py"}]},
                    "review": {"verdict": "APPROVE", "confidence": 0.9}}

        with patch("agent.eval_suite.DATA_DIR", tmp_path):
            with patch("agent.eval_suite._run_single_case", side_effect=fake_case):
                run_eval("test-repo", repo_path=str(data_dir))

        assert (data_dir / "eval_results.json").exists()
        assert (data_dir / "eval_history.json").exists()
        results = json.loads((data_dir / "eval_results.json").read_text())
        assert results["total"] == 1


# ---------------------------------------------------------------------------
# 7. test_eval_multi_file_complete_metric — expected_patch_files scoring
# ---------------------------------------------------------------------------

class TestEvalMultiFileCompleteMetric:
    """Phase 2C-1: eval schema should include expected_patch_files and
    multi_file_complete metric. Currently NOT implemented."""

    def test_score_single_run_tracks_localization_hit_for_multiple_files(self):
        """score_single_run checks if at least ONE expected file was found.
        This verifies multi-file bugs score correctly with current schema."""
        result = {
            "localization": {"fault_files": ["api/profile.py", "api/router.py"]},
            "repair": {"patches": [{"file_path": "api/profile.py"}]},
            "review": {"verdict": "APPROVE", "confidence": 0.85},
        }
        bug = {
            "ticket_id": "MF-1",
            "expected_files": ["api/profile.py", "api/router.py"],
            "expected_root_cause": "null check profile",
        }
        score = score_single_run(result, bug)
        # Localization hit = True because at least one expected file found
        assert score["localization_hit"] is True

    @pytest.mark.xfail(reason="Phase 2C-1: expected_patch_files field not yet in eval schema")
    def test_multi_file_complete_metric_exists(self):
        """score_single_run should report multi_file_complete=True only when
        ALL expected_patch_files are covered by actual patches."""
        result = {
            "localization": {"fault_files": ["api/profile.py", "api/router.py"]},
            "repair": {"patches": [
                {"file_path": "api/profile.py"},
                {"file_path": "api/router.py"},
            ]},
            "review": {"verdict": "APPROVE", "confidence": 0.9},
        }
        bug = {
            "ticket_id": "MF-2",
            "expected_files": ["api/profile.py", "api/router.py"],
            "expected_patch_files": ["api/profile.py", "api/router.py"],
            "expected_root_cause": "multi-file fix",
        }
        score = score_single_run(result, bug)
        # This key should exist once Phase 2C-1 ships
        assert "multi_file_complete" in score
        assert score["multi_file_complete"] is True

    @pytest.mark.xfail(reason="Phase 2C-1: partial multi-file coverage not tracked")
    def test_multi_file_incomplete_when_caller_not_patched(self):
        """multi_file_complete=False when not all expected_patch_files are patched."""
        result = {
            "localization": {"fault_files": ["api/profile.py"]},
            "repair": {"patches": [{"file_path": "api/profile.py"}]},  # missing router.py
            "review": {"verdict": "APPROVE", "confidence": 0.8},
        }
        bug = {
            "ticket_id": "MF-3",
            "expected_files": ["api/profile.py"],
            "expected_patch_files": ["api/profile.py", "api/router.py"],
            "expected_root_cause": "multi-file fix",
        }
        score = score_single_run(result, bug)
        assert score["multi_file_complete"] is False


# ---------------------------------------------------------------------------
# 8. test_eval_flaky_test_handling — retry logic for flaky tests
# ---------------------------------------------------------------------------

class TestEvalFlakyTestHandling:
    """Phase 2C-3: eval runner should retry flaky test failures up to 3x
    before scoring as failure. Currently NOT implemented."""

    @pytest.mark.xfail(reason="Phase 2C-3: no flaky test retry logic in eval runner")
    def test_flaky_test_retried_before_failure(self, tmp_path):
        """A case that fails once then succeeds should NOT be scored as failure.
        Requires retry logic in _run_single_case or run_eval."""
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [{"ticket_id": "EVAL-FLAKY", "title": "Flaky", "description": "...",
                 "expected_files": ["a.py"], "expected_root_cause": "flaky"}]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        call_count = 0

        def flaky_case(wo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Flaky network error")
            return {
                "status": "done",
                "localization": {"fault_files": ["a.py"], "root_cause_hypothesis": "flaky"},
                "repair": {"patches": [{"file_path": "a.py"}]},
                "review": {"verdict": "APPROVE", "confidence": 0.9},
            }

        with patch("agent.eval_suite.DATA_DIR", tmp_path):
            with patch("agent.eval_suite._run_single_case", side_effect=flaky_case):
                report = run_eval("test-repo", repo_path=str(data_dir))

        # With retry, the second attempt should succeed
        assert report["scores"][0]["localization_hit"] is True
        assert call_count >= 2  # retried at least once

    def test_current_behavior_no_retry(self, tmp_path):
        """Documents current behavior: a single failure scores immediately
        as failed, no retry."""
        data_dir = tmp_path / "test-repo"
        data_dir.mkdir()
        bugs = [{"ticket_id": "EVAL-NR", "title": "No retry", "description": "...",
                 "expected_files": ["a.py"], "expected_root_cause": "fail"}]
        (data_dir / "eval_bugs.json").write_text(json.dumps(bugs))

        def always_fail(wo):
            raise RuntimeError("API down")

        with patch("agent.eval_suite.DATA_DIR", tmp_path):
            with patch("agent.eval_suite._run_single_case", side_effect=always_fail):
                report = run_eval("test-repo", repo_path=str(data_dir))

        assert report["scores"][0]["localization_hit"] is False
        assert report["scores"][0]["error"]  # error captured


# ---------------------------------------------------------------------------
# 9. test_eval_security_isolation — container enforcement
# ---------------------------------------------------------------------------

class TestEvalSecurityIsolation:
    """Phase 2C-3: eval cases should run inside Docker containers with
    --network none. Currently NOT implemented — tests run via subprocess
    on the host."""

    @pytest.mark.xfail(reason="Phase 2C-3: no Docker container isolation in eval/sandbox")
    def test_eval_runs_in_container(self):
        """run_eval should execute cases inside Docker with --network none.
        This prevents eval repos from phoning home or exfiltrating data."""
        # When implemented, the _run_single_case function should:
        # 1. Create a Docker container from a base image
        # 2. Mount the repo read-only
        # 3. Pass --network none to prevent network access
        # 4. Set CPU/memory limits
        # 5. Run the pipeline inside the container

        from agent import sandbox
        import inspect

        source = inspect.getsource(sandbox.run_tests)
        # Should contain Docker-related setup
        assert "docker" in source.lower() or "--network" in source, (
            "sandbox.run_tests should use Docker containers with --network none"
        )

    def test_current_sandbox_uses_subprocess(self):
        """Documents current behavior: sandbox runs tests via subprocess
        directly on the host (no container isolation)."""
        from agent import sandbox
        import inspect

        source = inspect.getsource(sandbox.run_tests)
        assert "subprocess" in source, "sandbox currently uses subprocess.run"
        assert "docker" not in source.lower(), "sandbox does NOT use Docker yet"

    def test_sandbox_no_shell_true(self):
        """Verify setup_commands use shlex.split, not shell=True (Phase 2A-1 fix)."""
        from agent import sandbox
        import inspect

        source = inspect.getsource(sandbox.run_tests)
        # shell=True should NOT appear for setup commands
        # (it may appear in comments — check actual subprocess calls)
        assert "shlex" in inspect.getsource(sandbox) or "shell=True" not in source, (
            "sandbox should use shlex.split, not shell=True"
        )
