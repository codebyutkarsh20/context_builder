"""
Tests for Pydantic models and type definitions — Phase 3

Verifies:
  - AgentState has all required fields
  - PipelineStatus enum completeness
  - Structured output models validate correctly
  - WorkOrder model accepts repo_path
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.types import (
    AgentState,
    IntentAnalysis,
    LocalizationResult,
    Patch,
    PipelineStatus,
    RepairResult,
    ReviewCheck,
    ReviewResult,
    WorkOrder,
)


# ── AgentState ───────────────────────────────────────────────────────

class TestAgentState:
    """AgentState TypedDict has all required fields."""

    def test_has_core_fields(self):
        annotations = AgentState.__annotations__
        for field in ["work_order", "intent", "context", "context_nodes",
                      "source_code", "localization", "repair", "review",
                      "iteration_count", "status", "error", "pr_url"]:
            assert field in annotations, f"Missing field: {field}"

    def test_has_new_hardening_fields(self):
        annotations = AgentState.__annotations__
        for field in ["test_result", "sandbox_path", "branch_name",
                      "base_branch", "patches_applied"]:
            assert field in annotations, f"Missing new field: {field}"

    def test_is_total_false(self):
        """All fields are optional (total=False)."""
        # TypedDict with total=False allows partial dicts
        state: AgentState = {"status": "pending"}  # type: ignore
        assert state["status"] == "pending"


# ── PipelineStatus ───────────────────────────────────────────────────

class TestPipelineStatusEnum:
    """PipelineStatus covers every pipeline stage."""

    def test_all_stages_present(self):
        expected = {
            "pending", "intake", "context_assembly", "exploring", "localizing",
            "reading_source", "repairing", "reviewing", "testing",
            "pr_creating", "done", "escalated", "failed",
        }
        actual = {s.value for s in PipelineStatus}
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_testing_stage_exists(self):
        assert PipelineStatus.TESTING.value == "testing"

    def test_is_string_enum(self):
        assert isinstance(PipelineStatus.PENDING, str)
        assert PipelineStatus.PENDING == "pending"


# ── WorkOrder ────────────────────────────────────────────────────────

class TestWorkOrder:
    """WorkOrder model accepts all fields including repo_path."""

    def test_minimal_creation(self):
        wo = WorkOrder(ticket_id="T-1", title="Bug", description="Broken", repo_name="repo")
        assert wo.ticket_id == "T-1"
        assert wo.repo_path == ""
        assert wo.priority == "medium"

    def test_with_repo_path(self):
        wo = WorkOrder(
            ticket_id="T-2",
            title="Bug",
            description="Broken",
            repo_name="repo",
            repo_path="/path/to/repo",
        )
        assert wo.repo_path == "/path/to/repo"

    def test_with_all_fields(self):
        wo = WorkOrder(
            ticket_id="T-3",
            title="Critical bug",
            description="Server crash",
            repo_name="my-app",
            repo_path="/home/user/my-app",
            priority="critical",
            affected_component="auth",
            reproduction_steps="1. Login\n2. Click profile",
            comments=["urgent", "affects all users"],
        )
        assert wo.priority == "critical"
        assert wo.affected_component == "auth"
        assert len(wo.comments) == 2


# ── IntentAnalysis ───────────────────────────────────────────────────

class TestIntentAnalysis:
    """IntentAnalysis structured output model."""

    def test_creation_with_defaults(self):
        ia = IntentAnalysis(
            expected_behavior="Should work",
            actual_behavior="Doesn't work",
        )
        assert ia.fix_type == "bug_fix"
        assert ia.severity == "medium"
        assert ia.likely_affected_modules == []

    def test_full_creation(self):
        ia = IntentAnalysis(
            expected_behavior="200 OK",
            actual_behavior="500 error",
            likely_affected_modules=["auth.py", "users.py"],
            likely_affected_functions=["get_user", "validate"],
            fix_type="bug_fix",
            severity="critical",
        )
        assert len(ia.likely_affected_modules) == 2


# ── LocalizationResult ───────────────────────────────────────────────

class TestLocalizationResult:
    """LocalizationResult structured output model."""

    def test_default_confidence_zero(self):
        lr = LocalizationResult()
        assert lr.confidence == 0.0
        assert lr.fault_files == []

    def test_with_data(self):
        lr = LocalizationResult(
            fault_files=["app.py"],
            fault_functions=["get_user"],
            root_cause_hypothesis="NoneType error",
            confidence=0.85,
            evidence=["traceback shows line 42"],
        )
        assert lr.confidence == 0.85
        assert len(lr.evidence) == 1


# ── Patch / RepairResult ─────────────────────────────────────────────

class TestPatch:
    """Patch model for code changes."""

    def test_creation(self):
        p = Patch(
            file_path="app.py",
            original_code="return x",
            patched_code="return x if x else ''",
            explanation="Handle None case",
        )
        assert p.file_path == "app.py"

    def test_empty_code_allowed(self):
        p = Patch(file_path="app.py")
        assert p.original_code == ""
        assert p.patched_code == ""


class TestRepairResult:
    """RepairResult with patches list."""

    def test_empty_repair(self):
        rr = RepairResult()
        assert rr.patches == []
        assert rr.explanation == ""

    def test_with_patches(self):
        rr = RepairResult(
            patches=[
                Patch(file_path="a.py", original_code="x", patched_code="y"),
                Patch(file_path="b.py", original_code="1", patched_code="2"),
            ],
            explanation="Fixed the bug",
            tests_added=["test_fix"],
        )
        assert len(rr.patches) == 2


# ── ReviewResult ─────────────────────────────────────────────────────

class TestReviewResult:
    """ReviewResult with checks list."""

    def test_defaults(self):
        rr = ReviewResult()
        assert rr.verdict == "PENDING"
        assert rr.confidence == 0.0

    def test_approve(self):
        rr = ReviewResult(
            verdict="APPROVE",
            confidence=0.9,
            checks=[
                ReviewCheck(name="ROOT_CAUSE", status="PASS", comment="Fix is correct"),
            ],
        )
        assert rr.verdict == "APPROVE"
        assert len(rr.checks) == 1

    def test_changes_requested(self):
        rr = ReviewResult(
            verdict="CHANGES_REQUESTED",
            confidence=0.4,
            feedback="Need to handle edge case",
        )
        assert rr.feedback != ""
