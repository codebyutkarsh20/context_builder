"""
Tests for dry_run mode in the agent pipeline.

Verifies:
  - dry_run completes without creating a PR
  - Worktree cleanup still happens in dry_run
  - _enrich_from_fix is NOT called in dry_run
  - Feature flag creation is NOT called in dry_run
  - git push is NOT called in dry_run
  - API dry_run flag threads to pipeline
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import pr_creation_node
from agent.types import PipelineStatus


@pytest.fixture
def dry_run_state(tmp_path):
    """Build a minimal state dict for dry_run PR creation tests.

    Uses tmp_path to ensure the sandbox directory actually exists on disk,
    so we get past the sandbox existence check in pr_creation_node.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return {
        "work_order": {
            "ticket_id": "DRY-1",
            "repo_name": "test-repo",
            "repo_path": str(tmp_path),
        },
        "repair": {
            "patches": [{"file_path": "src/app.py", "patched_code": "fixed"}],
            "explanation": "Fixed the bug",
        },
        "review": {"verdict": "APPROVE", "confidence": 0.95},
        "localization": {"root_cause_hypothesis": "Off-by-one error"},
        "sandbox_path": str(sandbox),
        "branch_name": "fix/dry-1-abc",
        "base_branch": "main",
        "test_result": "passed\n2 tests passed",
        "status": PipelineStatus.PR_CREATING,
        "error": "",
        "pr_url": "",
        "dry_run": True,
    }


class TestDryRunPRCreation:
    """Test dry_run mode skips external side effects."""

    @patch("agent.pipeline._cleanup_worktree")
    @patch("agent.pipeline._report_progress")
    @patch("agent.pipeline._get_trace", return_value=None)
    def test_dry_run_completes_without_pr(
        self, mock_trace, mock_progress, mock_cleanup, dry_run_state
    ):
        """In dry_run, pr_creation_node should return without pushing or creating PR."""
        result = pr_creation_node(dry_run_state)

        assert "(dry-run" in result.get("pr_url", "")
        assert result["status"] == PipelineStatus.DONE

    @patch("agent.pipeline._cleanup_worktree")
    @patch("agent.pipeline._report_progress")
    @patch("agent.pipeline._get_trace", return_value=None)
    def test_dry_run_cleans_up_worktree(
        self, mock_trace, mock_progress, mock_cleanup, dry_run_state
    ):
        """Worktree cleanup must run even in dry_run mode (via finally block)."""
        pr_creation_node(dry_run_state)

        mock_cleanup.assert_called_once()

    @patch("agent.pipeline._enrich_from_fix")
    @patch("agent.pipeline._cleanup_worktree")
    @patch("agent.pipeline._report_progress")
    @patch("agent.pipeline._get_trace", return_value=None)
    def test_dry_run_does_not_enrich_fix_history(
        self, mock_trace, mock_progress, mock_cleanup, mock_enrich, dry_run_state
    ):
        """_enrich_from_fix must NOT be called in dry_run (corrupts fix_history.json)."""
        pr_creation_node(dry_run_state)

        mock_enrich.assert_not_called()

    @patch("agent.pipeline._create_feature_flag")
    @patch("agent.pipeline._cleanup_worktree")
    @patch("agent.pipeline._report_progress")
    @patch("agent.pipeline._get_trace", return_value=None)
    def test_dry_run_does_not_create_feature_flag(
        self, mock_trace, mock_progress, mock_cleanup, mock_flag, dry_run_state
    ):
        """Feature flag creation must be skipped in dry_run (prevents orphaned flags)."""
        pr_creation_node(dry_run_state)

        mock_flag.assert_not_called()

    @patch("subprocess.run")
    @patch("agent.pipeline._cleanup_worktree")
    @patch("agent.pipeline._report_progress")
    @patch("agent.pipeline._get_trace", return_value=None)
    def test_dry_run_does_not_push_to_remote(
        self, mock_trace, mock_progress, mock_cleanup, mock_subprocess, dry_run_state
    ):
        """git push must not be called in dry_run mode."""
        pr_creation_node(dry_run_state)

        # subprocess.run should not be called at all (no push, no gh pr create)
        mock_subprocess.assert_not_called()


class TestDryRunAPIIntegration:
    """Test dry_run flag in API request model."""

    def test_run_ticket_request_has_dry_run_field(self):
        """RunTicketRequest should accept dry_run parameter."""
        from api.agent import RunTicketRequest

        req = RunTicketRequest(
            description="test bug",
            dry_run=True,
        )
        assert req.dry_run is True

    def test_run_ticket_request_dry_run_defaults_false(self):
        """dry_run should default to False."""
        from api.agent import RunTicketRequest

        req = RunTicketRequest(description="test bug")
        assert req.dry_run is False


class TestDryRunInRunTicket:
    """Test dry_run parameter in run_ticket function signature."""

    def test_run_ticket_accepts_dry_run(self):
        """run_ticket should accept dry_run kwarg without error."""
        import inspect
        from agent.pipeline import run_ticket

        sig = inspect.signature(run_ticket)
        assert "dry_run" in sig.parameters
        assert sig.parameters["dry_run"].default is False
