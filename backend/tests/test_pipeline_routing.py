"""
Tests for pipeline routing logic — Phases 1.3, 3.2

Verifies the LangGraph state machine routing:
  - Confidence gate (localization → read_source / escalate)
  - Review loop (review → test / retry_fix / escalate)
  - Max iteration enforcement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import (
    should_read_source_or_escalate,
    should_iterate,
    MIN_CONFIDENCE_TO_REPAIR,
    MAX_ITERATIONS,
    build_agent_graph,
)
from agent.types import PipelineStatus


# ── Confidence Gate ──────────────────────────────────────────────────

class TestConfidenceGate:
    """Localization confidence gate routing."""

    def test_high_confidence_proceeds(self):
        state = {"localization": {"confidence": 0.9, "fault_files": ["app.py"]}}
        assert should_read_source_or_escalate(state) == "read_source"

    def test_threshold_confidence_proceeds(self):
        state = {"localization": {"confidence": MIN_CONFIDENCE_TO_REPAIR, "fault_files": ["app.py"]}}
        assert should_read_source_or_escalate(state) == "read_source"

    def test_low_confidence_escalates(self):
        state = {"localization": {"confidence": 0.1, "fault_files": ["app.py"]}}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_zero_confidence_escalates(self):
        state = {"localization": {"confidence": 0.0, "fault_files": ["app.py"]}}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_no_fault_files_escalates(self):
        state = {"localization": {"confidence": 0.9, "fault_files": []}}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_missing_localization_escalates(self):
        state = {"localization": {}}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_empty_state_escalates(self):
        state = {}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_just_below_threshold_escalates(self):
        state = {"localization": {"confidence": MIN_CONFIDENCE_TO_REPAIR - 0.01, "fault_files": ["f.py"]}}
        assert should_read_source_or_escalate(state) == "escalate"

    def test_just_above_threshold_proceeds(self):
        state = {"localization": {"confidence": MIN_CONFIDENCE_TO_REPAIR + 0.01, "fault_files": ["f.py"]}}
        assert should_read_source_or_escalate(state) == "read_source"


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


# ── Graph Structure ──────────────────────────────────────────────────

class TestGraphStructure:
    """The LangGraph state machine has the correct nodes and edges."""

    def test_graph_compiles(self):
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        graph = build_agent_graph()
        # LangGraph compiled graph stores nodes internally
        if hasattr(graph, 'nodes'):
            node_names = set(graph.nodes.keys()) if isinstance(graph.nodes, dict) else set()
            for expected in ["intake", "context_assembly", "localization",
                             "read_source", "repair", "review", "test",
                             "create_pr", "escalate"]:
                assert expected in node_names, f"Missing node: {expected}"


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
