"""
Tests for pipeline routing logic — Phases 1.3, 3.2

Verifies the legacy pipeline's routing functions (should_iterate, etc.)
which are still used as utility logic. The fixed LangGraph pipeline itself
is retired — ReAct (react_pipeline.py) is the only supported runtime.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import (
    should_iterate,
    MAX_ITERATIONS,
    multi_file_coordinator_node,
)
from agent.types import PipelineStatus



# ── Review Loop ──────────────────────────────────────────────────────

class TestReviewLoop:
    """Review routing: test vs retry vs escalate."""

    def test_approve_routes_to_test(self):
        state = {"review": {"verdict": "APPROVE"}, "iteration_count": 1}
        assert should_iterate(state) == "test"

    def test_changes_requested_routes_to_retry(self):
        state = {"review": {"verdict": "CHANGES_REQUESTED"}, "iteration_count": 1}
        assert should_iterate(state) == "retry_fix"

    def test_escalate_verdict_routes_to_escalate(self):
        state = {"review": {"verdict": "ESCALATE"}, "iteration_count": 1}
        assert should_iterate(state) == "escalate"

    def test_max_iterations_escalates(self):
        state = {"review": {"verdict": "CHANGES_REQUESTED"}, "iteration_count": MAX_ITERATIONS}
        assert should_iterate(state) == "escalate"

    def test_beyond_max_iterations_escalates(self):
        state = {"review": {"verdict": "CHANGES_REQUESTED"}, "iteration_count": MAX_ITERATIONS + 1}
        assert should_iterate(state) == "escalate"

    def test_first_iteration_allows_retry(self):
        state = {"review": {"verdict": "CHANGES_REQUESTED"}, "iteration_count": 1}
        assert should_iterate(state) == "retry_fix"

    def test_second_iteration_allows_retry(self):
        state = {"review": {"verdict": "CHANGES_REQUESTED"}, "iteration_count": 2}
        assert should_iterate(state) == "retry_fix"

    def test_missing_verdict_escalates(self):
        state = {"review": {}, "iteration_count": 1}
        assert should_iterate(state) == "escalate"

    def test_empty_state_escalates(self):
        assert should_iterate({}) == "escalate"

    def test_unknown_verdict_retries_if_under_max(self):
        state = {"review": {"verdict": "UNKNOWN"}, "iteration_count": 1}
        assert should_iterate(state) == "retry_fix"

    def test_approve_at_max_iterations_still_tests(self):
        """APPROVE should go to test regardless of iteration count."""
        state = {"review": {"verdict": "APPROVE"}, "iteration_count": MAX_ITERATIONS}
        assert should_iterate(state) == "test"


# TestGraphStructure removed — fixed LangGraph pipeline is retired.
# ReAct pipeline (react_pipeline.py) is the only supported runtime.


# ── PipelineStatus Enum ──────────────────────────────────────────────

class TestPipelineStatus:
    """PipelineStatus enum has all required values."""

    def test_pending(self):
        assert PipelineStatus.PENDING.value == "pending"

    def test_intake(self):
        assert PipelineStatus.INTAKE.value == "intake"

    def test_context(self):
        assert PipelineStatus.CONTEXT.value == "context_assembly"

    def test_localizing(self):
        assert PipelineStatus.LOCALIZING.value == "localizing"

    def test_reading_source(self):
        assert PipelineStatus.READING_SOURCE.value == "reading_source"

    def test_repairing(self):
        assert PipelineStatus.REPAIRING.value == "repairing"

    def test_reviewing(self):
        assert PipelineStatus.REVIEWING.value == "reviewing"

    def test_testing(self):
        assert PipelineStatus.TESTING.value == "testing"

    def test_pr_creating(self):
        assert PipelineStatus.PR_CREATING.value == "pr_creating"

    def test_done(self):
        assert PipelineStatus.DONE.value == "done"

    def test_escalated(self):
        assert PipelineStatus.ESCALATED.value == "escalated"

    def test_failed(self):
        assert PipelineStatus.FAILED.value == "failed"


# ── Multi-File Coordinator ────────────────────────────────────────────

class TestMultiFileCoordinator:
    """multi_file_coordinator_node: early-return paths that don't require LLM calls."""

    def _base_state(self, **overrides):
        state = {
            "repair": {"patches": [{"file_path": "service.py", "original_code": "x", "patched_code": "y"}], "explanation": "fix"},
            "caller_files": [],
            "work_order": {},
            "localization": {},
        }
        state.update(overrides)
        return state

    def test_no_caller_files_returns_unchanged(self):
        """Empty caller_files → coordinator is a no-op."""
        state = self._base_state(caller_files=[])
        result = multi_file_coordinator_node(state)
        assert result["repair"] == state["repair"]

    def test_no_patches_returns_unchanged(self):
        """Empty repair (no patches) → coordinator is a no-op."""
        state = self._base_state(repair={}, caller_files=["router.py"])
        result = multi_file_coordinator_node(state)
        assert result.get("repair", {}) == {}

    def test_all_callers_already_patched_returns_unchanged(self):
        """All caller files have patches → coordinator is a no-op, no LLM call."""
        state = self._base_state(
            repair={
                "patches": [
                    {"file_path": "service.py", "original_code": "x", "patched_code": "y"},
                    {"file_path": "router.py",  "original_code": "a", "patched_code": "b"},
                ],
                "explanation": "renamed fn",
            },
            caller_files=["router.py"],  # already in patches
        )
        result = multi_file_coordinator_node(state)
        assert len(result["repair"]["patches"]) == 2  # no additions

    def test_multiple_callers_all_patched_returns_unchanged(self):
        """Multiple callers, all covered — still a no-op."""
        state = self._base_state(
            repair={
                "patches": [
                    {"file_path": "service.py", "original_code": "x", "patched_code": "y"},
                    {"file_path": "router.py",  "original_code": "a", "patched_code": "b"},
                    {"file_path": "views.py",   "original_code": "c", "patched_code": "d"},
                ],
                "explanation": "fix",
            },
            caller_files=["router.py", "views.py"],
        )
        result = multi_file_coordinator_node(state)
        assert len(result["repair"]["patches"]) == 3
