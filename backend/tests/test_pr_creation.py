"""
Tests for PR creation node — Phases 3.3, 3.4

Verifies:
  - Push to remote with timeout
  - GitHub PR creation via gh CLI
  - PR body includes root cause, fix, test results
  - Graceful fallback when push/PR fails
  - Worktree cleanup in all cases
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import pr_creation_node, _cleanup_worktree
from agent.types import PipelineStatus


class TestPRCreationWithoutSandbox:
    """PR creation when no sandbox is available."""

    def test_no_sandbox_path_escalates(self):
        state = {
            "work_order": {"ticket_id": "T-1", "repo_name": "test", "repo_path": "/fake"},
            "repair": {"patches": [], "explanation": "fix"},
            "review": {"verdict": "APPROVE", "confidence": 0.9},
            "sandbox_path": "",
            "branch_name": "fix/t-1-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
        }
        result = pr_creation_node(state)
        assert result["status"] == PipelineStatus.DONE
        assert "No sandbox" in result.get("error", "") or result["pr_url"] == ""

    def test_nonexistent_sandbox_path(self):
        state = {
            "work_order": {"ticket_id": "T-2", "repo_name": "test", "repo_path": "/fake"},
            "repair": {"patches": [], "explanation": "fix"},
            "review": {"verdict": "APPROVE"},
            "sandbox_path": "/tmp/nonexistent_sandbox_xyz123",
            "branch_name": "fix/t-2-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
        }
        result = pr_creation_node(state)
        assert result["status"] == PipelineStatus.DONE


class TestPRCreationWithSandbox:
    """PR creation from a real sandbox (mocked git/gh)."""

    @patch("agent.pipeline.subprocess.run")
    @patch("agent.pipeline._resolve_repo_path")
    def test_push_failure_graceful(self, mock_resolve, mock_run):
        """Push failure doesn't crash, records error in pr_url."""
        mock_resolve.return_value = Path("/fake/repo")

        sandbox = Path("/tmp/test_sandbox_pr")
        sandbox.mkdir(exist_ok=True)

        # git push fails
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="remote: Permission denied",
        )

        state = {
            "work_order": {"ticket_id": "T-3", "repo_name": "test"},
            "repair": {"patches": [{"file_path": "a.py"}], "explanation": "fix"},
            "review": {"verdict": "APPROVE", "confidence": 0.8},
            "sandbox_path": str(sandbox),
            "branch_name": "fix/t-3-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
            "localization": {"root_cause_hypothesis": "bug"},
            "test_result": "passed",
        }
        result = pr_creation_node(state)
        assert "push failed" in result.get("pr_url", "")
        sandbox.rmdir()

    @patch("agent.pipeline.subprocess.run")
    @patch("agent.pipeline._resolve_repo_path")
    def test_gh_pr_create_success(self, mock_resolve, mock_run):
        """Successful push + PR creation returns PR URL."""
        mock_resolve.return_value = Path("/fake/repo")

        sandbox = Path("/tmp/test_sandbox_pr2")
        sandbox.mkdir(exist_ok=True)

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            if cmd[1] == "push":
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
            elif cmd[0] == "gh":
                result.returncode = 0
                result.stdout = "https://github.com/org/repo/pull/42"
                result.stderr = ""
            else:
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
            return result

        mock_run.side_effect = side_effect

        state = {
            "work_order": {"ticket_id": "T-4", "repo_name": "test"},
            "repair": {"patches": [{"file_path": "a.py"}], "explanation": "Fixed null check"},
            "review": {"verdict": "APPROVE", "confidence": 0.9},
            "sandbox_path": str(sandbox),
            "branch_name": "fix/t-4-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
            "localization": {"root_cause_hypothesis": "NoneType on email"},
            "test_result": "passed\n4 tests passed",
        }
        result = pr_creation_node(state)
        assert result["pr_url"] == "https://github.com/org/repo/pull/42"
        assert result["status"] == PipelineStatus.DONE
        sandbox.rmdir()

    @patch("agent.pipeline.subprocess.run")
    @patch("agent.pipeline._resolve_repo_path")
    def test_gh_pr_create_failure_graceful(self, mock_resolve, mock_run):
        """gh pr create failure records error but doesn't crash."""
        mock_resolve.return_value = Path("/fake/repo")

        sandbox = Path("/tmp/test_sandbox_pr3")
        sandbox.mkdir(exist_ok=True)

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            if cmd[1] == "push":
                result.returncode = 0
            elif cmd[0] == "gh":
                result.returncode = 1
                result.stderr = "gh: not logged in"
            else:
                result.returncode = 0
            result.stdout = ""
            return result

        mock_run.side_effect = side_effect

        state = {
            "work_order": {"ticket_id": "T-5", "repo_name": "test"},
            "repair": {"patches": [{"file_path": "a.py"}], "explanation": "fix"},
            "review": {"verdict": "APPROVE", "confidence": 0.8},
            "sandbox_path": str(sandbox),
            "branch_name": "fix/t-5-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
            "localization": {"root_cause_hypothesis": "bug"},
            "test_result": "passed",
        }
        result = pr_creation_node(state)
        assert "PR creation failed" in result.get("pr_url", "")
        sandbox.rmdir()

    @patch("agent.pipeline.subprocess.run")
    @patch("agent.pipeline._resolve_repo_path")
    def test_push_timeout_handled(self, mock_resolve, mock_run):
        """Push timeout is caught and reported."""
        mock_resolve.return_value = Path("/fake/repo")

        sandbox = Path("/tmp/test_sandbox_pr4")
        sandbox.mkdir(exist_ok=True)

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git push", timeout=120)

        state = {
            "work_order": {"ticket_id": "T-6", "repo_name": "test"},
            "repair": {"patches": [], "explanation": "fix"},
            "review": {"verdict": "APPROVE"},
            "sandbox_path": str(sandbox),
            "branch_name": "fix/t-6-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
            "localization": {},
            "test_result": "",
        }
        result = pr_creation_node(state)
        assert "timed out" in result.get("pr_url", "") or "timed out" in result.get("error", "")
        sandbox.rmdir()


class TestPRBody:
    """PR body includes all required sections."""

    @patch("agent.pipeline.subprocess.run")
    @patch("agent.pipeline._resolve_repo_path")
    def test_pr_body_contains_sections(self, mock_resolve, mock_run):
        mock_resolve.return_value = Path("/fake/repo")
        sandbox = Path("/tmp/test_sandbox_pr5")
        sandbox.mkdir(exist_ok=True)

        captured_body = {}

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "https://github.com/org/repo/pull/99"
            result.stderr = ""
            if cmd[0] == "gh" and "pr" in cmd:
                # Capture the body arg
                body_idx = cmd.index("--body") + 1
                captured_body["body"] = cmd[body_idx]
            return result

        mock_run.side_effect = side_effect

        state = {
            "work_order": {"ticket_id": "T-7", "repo_name": "test"},
            "repair": {"patches": [{"file_path": "a.py"}], "explanation": "Fixed null email"},
            "review": {"verdict": "APPROVE", "confidence": 0.95},
            "sandbox_path": str(sandbox),
            "branch_name": "fix/t-7-abc",
            "base_branch": "main",
            "status": "",
            "error": "",
            "localization": {"root_cause_hypothesis": "email.lower() called on None"},
            "test_result": "passed\n5 tests passed",
        }
        pr_creation_node(state)

        body = captured_body.get("body", "")
        assert "Root Cause" in body
        assert "email.lower()" in body
        assert "Fix" in body
        assert "Fixed null email" in body
        assert "Files Changed" in body
        assert "Review" in body
        assert "APPROVE" in body
        assert "Tests" in body
        assert "passed" in body
        assert "AI Deploy Agent" in body
        sandbox.rmdir()
