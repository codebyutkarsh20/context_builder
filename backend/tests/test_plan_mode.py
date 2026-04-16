"""
test_plan_mode.py — Tests for the produce_plan tool and plan-mode guardrail.

Plan mode is an autonomous self-commitment device: the agent declares a
structured plan (root_cause + target_files + approach + success_criteria +
risk) before any sandbox edits. The guardrail enforces "no create_sandbox
without a plan."
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.react_tools import (
    produce_plan,
    get_current_plan,
    get_plan_history,
    reset_plan_state,
    set_react_context,
)
from agent.react_guardrails import GuardrailState, check_tool_call, update_from_tool_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_plan():
    reset_plan_state()
    yield
    reset_plan_state()


# ---------------------------------------------------------------------------
# produce_plan — input validation
# ---------------------------------------------------------------------------

class TestProducePlanValidation:
    def test_valid_plan_recorded(self):
        result = produce_plan.invoke({
            "root_cause": "The retry decorator catches BaseException and swallows KeyboardInterrupt.",
            "target_files": ["app/retry.py"],
            "approach": "Narrow the except clause to Exception so KeyboardInterrupt propagates.",
            "success_criteria": "Ctrl-C terminates the worker within 1s.",
            "risk": "LOW",
        })
        assert result.startswith("OK:")
        plan = get_current_plan()
        assert plan is not None
        assert plan["risk"] == "LOW"
        assert plan["target_files"] == ["app/retry.py"]

    def test_missing_root_cause_rejected(self):
        result = produce_plan.invoke({
            "root_cause": "",
            "target_files": ["x.py"],
            "approach": "do something",
            "success_criteria": "tests pass",
        })
        assert result.startswith("ERROR")
        assert get_current_plan() is None

    def test_missing_target_files_rejected(self):
        result = produce_plan.invoke({
            "root_cause": "bug exists",
            "target_files": [],
            "approach": "do something",
            "success_criteria": "tests pass",
        })
        assert result.startswith("ERROR")

    def test_invalid_risk_value_rejected(self):
        result = produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["x.py"],
            "approach": "y",
            "success_criteria": "z",
            "risk": "MEDIUM-HIGH",  # invalid
        })
        assert result.startswith("ERROR")
        assert "risk must be one of" in result

    def test_high_risk_requires_rollback(self):
        result = produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["x.py", "y.py", "z.py", "a.py", "b.py", "c.py"],
            "approach": "big rewrite",
            "success_criteria": "tests pass",
            "risk": "HIGH",
            "rollback": "",  # empty
        })
        assert result.startswith("ERROR")
        assert "rollback" in result.lower()

    def test_high_risk_with_rollback_accepted(self):
        result = produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["x.py", "y.py", "z.py", "a.py", "b.py", "c.py"],
            "approach": "big rewrite",
            "success_criteria": "tests pass",
            "risk": "HIGH",
            "rollback": "git revert HEAD if anything breaks in production",
        })
        assert result.startswith("OK:")

    def test_target_files_must_be_list_of_strings(self):
        result = produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["valid.py", "  "],  # whitespace-only entry
            "approach": "y",
            "success_criteria": "z",
        })
        assert result.startswith("ERROR")


# ---------------------------------------------------------------------------
# Plan revisions
# ---------------------------------------------------------------------------

class TestPlanRevision:
    def test_first_call_no_revision_marker(self):
        result = produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["a.py"],
            "approach": "y",
            "success_criteria": "z",
        })
        # First call has no "revision #N" marker (the help text mentions the
        # word "revision" but not the numbered marker).
        assert "revision #" not in result

    def test_second_call_marked_as_revision(self):
        produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["a.py"],
            "approach": "y",
            "success_criteria": "z",
        })
        result = produce_plan.invoke({
            "root_cause": "actually it's something else",
            "target_files": ["b.py"],
            "approach": "different approach",
            "success_criteria": "z2",
        })
        assert "revision #2" in result

    def test_history_preserves_all_revisions(self):
        produce_plan.invoke({
            "root_cause": "first guess",
            "target_files": ["a.py"],
            "approach": "approach 1",
            "success_criteria": "criteria",
        })
        produce_plan.invoke({
            "root_cause": "second guess",
            "target_files": ["b.py"],
            "approach": "approach 2",
            "success_criteria": "criteria",
        })
        history = get_plan_history()
        assert len(history) == 2
        assert history[0]["root_cause"] == "first guess"
        assert history[1]["root_cause"] == "second guess"

    def test_current_plan_is_latest(self):
        produce_plan.invoke({
            "root_cause": "first",
            "target_files": ["a.py"],
            "approach": "x",
            "success_criteria": "y",
        })
        produce_plan.invoke({
            "root_cause": "latest",
            "target_files": ["b.py"],
            "approach": "x2",
            "success_criteria": "y2",
        })
        assert get_current_plan()["root_cause"] == "latest"


# ---------------------------------------------------------------------------
# Guardrail integration
# ---------------------------------------------------------------------------

class TestPlanGuardrail:
    def test_create_sandbox_allowed_without_plan(self):
        gs = GuardrailState()
        # Plan-gate removed in v4 — create_sandbox is no longer blocked
        err = check_tool_call("create_sandbox", {}, gs)
        assert err is None

    def test_create_sandbox_allowed_after_plan(self):
        gs = GuardrailState()
        # Simulate a successful produce_plan call
        update_from_tool_result(
            "produce_plan",
            {},
            "OK: Plan recorded. risk=LOW, target_files=1.",
            gs,
        )
        assert gs.plan_produced
        err = check_tool_call("create_sandbox", {}, gs)
        # Plan exists → sandbox creation should pass plan-gate (other gates may
        # still apply, but plan-gate specifically should not block)
        if err is not None:
            assert "produce_plan" not in err

    def test_failed_produce_plan_does_not_set_flag(self):
        gs = GuardrailState()
        update_from_tool_result(
            "produce_plan",
            {},
            "ERROR: Plan is incomplete.",
            gs,
        )
        assert not gs.plan_produced
        # Plan-gate removed in v4 — create_sandbox allowed regardless
        err = check_tool_call("create_sandbox", {}, gs)
        assert err is None

    def test_plan_revision_tracked(self):
        gs = GuardrailState()
        update_from_tool_result(
            "produce_plan", {}, "OK: Plan recorded. risk=LOW", gs,
        )
        update_from_tool_result(
            "produce_plan", {}, "OK: Plan recorded (revision #2). risk=LOW", gs,
        )
        update_from_tool_result(
            "produce_plan", {}, "OK: Plan recorded (revision #3). risk=LOW", gs,
        )
        assert gs.plan_revision_count == 2  # 2 revisions after the first

    def test_produce_plan_itself_not_gated(self):
        """produce_plan must be callable when no plan exists yet."""
        gs = GuardrailState()
        err = check_tool_call("produce_plan", {}, gs)
        assert err is None


# ---------------------------------------------------------------------------
# State reset between runs
# ---------------------------------------------------------------------------

class TestPlanStateReset:
    def test_set_react_context_resets_plan(self, tmp_path):
        produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["a.py"],
            "approach": "y",
            "success_criteria": "z",
        })
        assert get_current_plan() is not None

        # New run starts — context reset clears plan
        set_react_context("repo2", str(tmp_path), fix_type="bug_fix")
        assert get_current_plan() is None
        assert get_plan_history() == []

    def test_reset_plan_state_helper(self):
        produce_plan.invoke({
            "root_cause": "x",
            "target_files": ["a.py"],
            "approach": "y",
            "success_criteria": "z",
        })
        reset_plan_state()
        assert get_current_plan() is None
        assert get_plan_history() == []

    def test_reset_when_nothing_to_reset(self):
        """reset_plan_state on a clean TLS doesn't crash."""
        reset_plan_state()
        reset_plan_state()
        assert get_current_plan() is None
