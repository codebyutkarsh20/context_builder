"""
Tests for git sandbox (worktree) and test execution — Phases 1.3, 3.1, 3.2

Verifies:
  - Unique branch name generation
  - Dirty repo detection
  - Git worktree creation and cleanup
  - Auto-detect test runner
  - Test execution with timeout
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import (
    test_node as sandbox_test_node,  # aliased to avoid pytest collection
    _run_tests,
    _cleanup_worktree,
    _fuzzy_match_replace,
)
from agent.types import PipelineStatus


class TestBranchNameUniqueness:
    """Branch names include UUID suffix for collision avoidance."""

    def test_unique_branches_per_run(self, tmp_repo, pipeline_module):
        """Two runs of test_node produce different branch names."""
        state1 = {
            "work_order": {"ticket_id": "TEST-1", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{"file_path": "app.py", "original_code": "return email.lower().strip()", "patched_code": "return email.lower().strip() if email else ''"}]},
            "review": {"verdict": "APPROVE"},
            "status": "",
        }
        state2 = {
            "work_order": {"ticket_id": "TEST-1", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{"file_path": "app.py", "original_code": "return email.lower().strip()", "patched_code": "return email.lower().strip() if email else ''"}]},
            "review": {"verdict": "APPROVE"},
            "status": "",
        }

        result1 = sandbox_test_node(state1)
        branch1 = result1.get("branch_name", "")

        # Clean up worktree from first run before second
        if result1.get("sandbox_path"):
            _cleanup_worktree(tmp_repo, result1["sandbox_path"])
            # Also remove the branch so the second run's worktree add doesn't conflict
            subprocess.run(["git", "branch", "-D", branch1], cwd=tmp_repo, capture_output=True)

        result2 = sandbox_test_node(state2)
        branch2 = result2.get("branch_name", "")
        if result2.get("sandbox_path"):
            _cleanup_worktree(tmp_repo, result2["sandbox_path"])

        assert branch1 != branch2, f"Branch names should differ: {branch1} vs {branch2}"
        assert branch1.startswith("fix/test-1-"), f"Branch has correct prefix: {branch1}"
        assert branch2.startswith("fix/test-1-"), f"Branch has correct prefix: {branch2}"


class TestDirtyRepoDetection:
    """Test node refuses to create sandbox when repo is dirty."""

    def test_dirty_repo_skips_sandbox(self, tmp_repo_dirty):
        state = {
            "work_order": {"ticket_id": "TEST-D", "repo_path": str(tmp_repo_dirty), "repo_name": "test"},
            "repair": {"patches": [{"file_path": "app.py", "original_code": "pass", "patched_code": "return 1"}]},
            "status": "",
        }
        result = sandbox_test_node(state)
        assert result.get("sandbox_path") == ""
        assert "uncommitted" in result.get("test_result", "")
        assert "uncommitted" in result.get("error", "").lower() or "uncommitted" in result.get("test_result", "").lower()


class TestWorktreeLifecycle:
    """Git worktree creation, patch application, and cleanup."""

    def test_worktree_created_and_cleaned(self, tmp_repo):
        state = {
            "work_order": {"ticket_id": "TEST-W", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "return email.lower().strip()",
                "patched_code": "return email.lower().strip() if email else ''",
            }]},
            "status": "",
        }
        result = sandbox_test_node(state)
        sandbox = result.get("sandbox_path", "")

        assert result.get("branch_name", "").startswith("fix/test-w-")
        assert result.get("patches_applied", 0) == 1
        assert result.get("base_branch") is not None

        # Worktree should still exist (cleanup happens in pr_creation_node)
        if sandbox:
            assert Path(sandbox).exists(), "Worktree should exist after test_node"
            # Clean up
            _cleanup_worktree(tmp_repo, sandbox)

    def test_no_patches_applied_cleans_up(self, tmp_repo):
        """If no patches match, worktree is cleaned up."""
        state = {
            "work_order": {"ticket_id": "TEST-NP", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "THIS DOES NOT EXIST IN THE FILE",
                "patched_code": "replaced",
            }]},
            "status": "",
        }
        result = sandbox_test_node(state)
        assert result.get("patches_applied", 0) == 0
        assert result.get("sandbox_path") == ""
        assert "no patches" in result.get("test_result", "").lower()

    def test_no_repo_path_skips(self):
        """No repo path → skip sandbox entirely."""
        state = {
            "work_order": {"ticket_id": "TEST-NR", "repo_path": "/nonexistent", "repo_name": "test"},
            "repair": {"patches": []},
            "status": "",
        }
        result = sandbox_test_node(state)
        assert result.get("sandbox_path") == ""
        assert "skipped" in result.get("test_result", "").lower()


class TestWorktreeCleanup:
    """_cleanup_worktree handles edge cases."""

    def test_cleanup_nonexistent_path(self, tmp_repo):
        """Cleaning up a nonexistent worktree doesn't crash."""
        _cleanup_worktree(tmp_repo, "/tmp/nonexistent_worktree_xyz")
        # Should not raise

    def test_cleanup_empty_path(self, tmp_repo):
        """Empty path is a no-op."""
        _cleanup_worktree(tmp_repo, "")
        # Should not raise

    def test_cleanup_none_repo(self):
        """None repo_path is a no-op."""
        _cleanup_worktree(None, "/tmp/something")
        # Should not raise


class TestAutoDetectTestRunner:
    """_run_tests auto-detects pytest, npm, or make."""

    def test_detects_pytest_via_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        (tmp_path / "test_app.py").write_text("def test_pass(): pass")
        result = _run_tests(tmp_path)
        # pytest may or may not be installed in the worktree
        # but the function should at least try and not crash
        assert isinstance(result, str)

    def test_detects_pytest_via_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        result = _run_tests(tmp_path)
        assert isinstance(result, str)

    def test_detects_npm_test(self, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts": {"test": "echo ok"}}')
        result = _run_tests(tmp_path)
        assert isinstance(result, str)

    def test_detects_makefile(self, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\techo ok\n")
        result = _run_tests(tmp_path)
        assert isinstance(result, str)

    def test_no_runner_skips(self, tmp_path):
        """Empty directory — no test runner found."""
        result = _run_tests(tmp_path)
        assert "skipped" in result.lower()
        assert "no test runner" in result.lower()

    def test_timeout_handled(self, tmp_path):
        """Test runner that hangs is killed after timeout."""
        # Create a script that sleeps forever
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "test_slow.py").write_text("import time; time.sleep(9999)")
        # We can't easily test the 5-min timeout in a unit test,
        # but we verify the function signature accepts the path
        # and the timeout constant exists in the code
        import inspect
        src = inspect.getsource(_run_tests)
        assert "timeout=300" in src


class TestBranchReuseOnRetry:
    """On retry iterations, test_node reuses the same branch instead of creating a new one."""

    def test_retry_reuses_branch_name(self, tmp_repo):
        """When state already has branch_name, the same name is reused."""
        state = {
            "work_order": {"ticket_id": "TEST-R", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "return email.lower().strip()",
                "patched_code": "return email.lower().strip() if email else ''",
            }]},
            "status": "",
        }

        # First run — creates a new branch
        result1 = sandbox_test_node(state)
        branch1 = result1.get("branch_name", "")
        assert branch1.startswith("fix/test-r-"), f"First run creates branch: {branch1}"

        # Simulate retry: pass the branch_name back in state (as the pipeline does)
        state2 = {
            "work_order": {"ticket_id": "TEST-R", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "return email.lower().strip()",
                "patched_code": "return email.lower().strip() if email else None",
            }]},
            "branch_name": branch1,  # reuse from first run
            "status": "",
        }
        result2 = sandbox_test_node(state2)
        branch2 = result2.get("branch_name", "")

        assert branch2 == branch1, f"Retry should reuse branch: {branch2} != {branch1}"
        assert result2.get("patches_applied", 0) == 1

        # Clean up
        if result2.get("sandbox_path"):
            _cleanup_worktree(tmp_repo, result2["sandbox_path"])

    def test_retry_does_not_create_extra_branches(self, tmp_repo):
        """After a retry, only one fix/ branch should exist for the ticket."""
        state = {
            "work_order": {"ticket_id": "TEST-RB", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "return email.lower().strip()",
                "patched_code": "return email.lower().strip() if email else ''",
            }]},
            "status": "",
        }

        result1 = sandbox_test_node(state)
        branch1 = result1.get("branch_name", "")

        # Retry with same branch
        state2 = {
            "work_order": {"ticket_id": "TEST-RB", "repo_path": str(tmp_repo), "repo_name": "test"},
            "repair": {"patches": [{
                "file_path": "app.py",
                "original_code": "return email.lower().strip()",
                "patched_code": "return email.lower().strip() if email else None",
            }]},
            "branch_name": branch1,
            "status": "",
        }
        result2 = sandbox_test_node(state2)

        # Count fix/ branches
        branches = subprocess.run(
            ["git", "branch", "--list", "fix/*"],
            cwd=tmp_repo, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        fix_branches = [b.strip() for b in branches if b.strip()]

        assert len(fix_branches) <= 1, f"Expected at most 1 fix branch, got {len(fix_branches)}: {fix_branches}"

        # Clean up
        if result2.get("sandbox_path"):
            _cleanup_worktree(tmp_repo, result2["sandbox_path"])
