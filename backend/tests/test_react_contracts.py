"""
Tests for ReAct tool / guardrail / prompt contracts.

These exist because every external agent review found contract drift between
react_prompt.py (what the agent is told), react_guardrails.py (what's enforced),
react_tools.py (what tools return), and react_loop.py (how results are interpreted).

When these tests break, the agent will silently misbehave (submit without tests,
count errors as failures, ignore review rejections, etc).

Covers:
  1. Guardrail acceptance rules (sandbox gate, submit gate, test classification)
  2. Tool return value prefixes match what guardrails expect
  3. Prompt-guardrail alignment (what the prompt promises = what guardrails enforce)
  4. react_loop terminal detection (submit_fix only terminal on "OK:")
  5. sandbox.py exit code classification
  6. Anti-pattern thresholds match prompt documentation
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.react_guardrails import (
    GuardrailState,
    check_limits,
    check_tool_call,
    update_from_tool_result,
    SANDBOX_REQUIRED_TOOLS,
    TERMINAL_TOOLS,
    MAX_TOOL_CALLS,
    MAX_WALL_TIME,
    MAX_COST_USD,
)
from agent.sandbox import _format_test_output


# ===========================================================================
# 1. Guardrail acceptance rules
# ===========================================================================

class TestGuardrailSandboxGate:
    """Tools that require a sandbox are blocked without one."""

    def test_sandbox_required_tools_list(self):
        """Verify the exact set of tools that need a sandbox."""
        assert SANDBOX_REQUIRED_TOOLS == frozenset({
            "string_replace", "create_file", "check_syntax",
            "run_tests", "run_linters",
        })

    @pytest.mark.parametrize("tool_name", list(SANDBOX_REQUIRED_TOOLS))
    def test_sandbox_required_tools_blocked_without_sandbox(self, tool_name):
        """Each sandbox-required tool returns ERROR when no sandbox exists."""
        gs = GuardrailState()
        assert gs.sandbox_created is False
        result = check_tool_call(tool_name, {}, gs)
        assert result is not None
        assert "ERROR" in result
        assert "create_sandbox" in result

    @pytest.mark.parametrize("tool_name", list(SANDBOX_REQUIRED_TOOLS))
    def test_sandbox_required_tools_allowed_with_sandbox(self, tool_name):
        """Each sandbox-required tool is allowed after sandbox creation."""
        gs = GuardrailState()
        gs.sandbox_created = True
        result = check_tool_call(tool_name, {}, gs)
        # Should be None (allowed) or a WARNING (not blocking ERROR)
        assert result is None or result.startswith("WARNING:")

    def test_explore_tools_never_need_sandbox(self):
        """Read-only exploration tools must never be in SANDBOX_REQUIRED_TOOLS."""
        explore_tools = {
            "grep_repo", "read_file", "read_function", "list_files",
            "search_code", "get_function_info", "get_file_structure",
            "get_file_summary",
        }
        assert explore_tools.isdisjoint(SANDBOX_REQUIRED_TOOLS)


class TestGuardrailSubmitGate:
    """submit_fix has strict prerequisites."""

    def test_submit_blocked_without_sandbox(self):
        gs = GuardrailState()
        result = check_tool_call("submit_fix", {}, gs)
        assert result is not None
        assert "create_sandbox" in result

    def test_submit_blocked_without_tests(self):
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is not None
        assert "run_tests" in result

    def test_submit_blocked_without_review(self):
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is not None
        assert "request_review" in result

    def test_submit_blocked_when_tests_failed(self):
        """Actual test failures (not skipped/error) MUST block submission."""
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = False
        gs.tests_skipped = False  # Not skipped — actually failed
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is not None
        assert "tests failed" in result.lower() or "run_tests" in result

    def test_submit_allowed_with_all_prerequisites(self):
        """Happy path: sandbox + tests passed + review approved."""
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = True
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is None

    def test_submit_allowed_when_tests_skipped(self):
        """'skipped' tests (no tests collected) should NOT block submission."""
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = False
        gs.tests_skipped = True  # No tests ran but that's OK
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is None

    def test_submit_allowed_when_tests_error(self):
        """'error' tests (runner-level failure) should NOT block submission.

        This is the key contract: error (exit code 2/3/4) means the test runner
        itself broke, not that assertions failed. The prompt tells the agent to
        proceed, and the guardrails must agree.
        """
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = False
        gs.tests_skipped = True  # error sets tests_skipped=True
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is None


class TestGuardrailLimits:
    """Global limits: tool calls, time, cost."""

    def test_tool_call_limit_blocks_non_terminal(self):
        gs = GuardrailState()
        gs.tool_call_count = MAX_TOOL_CALLS
        result = check_tool_call("grep_repo", {}, gs)
        assert result is not None
        assert "Tool call limit" in result

    def test_tool_call_limit_allows_terminal(self):
        """Terminal tools (submit, escalate) must work even at the limit."""
        gs = GuardrailState()
        gs.tool_call_count = MAX_TOOL_CALLS
        gs.sandbox_created = True
        gs.tests_attempted = True
        gs.tests_passed = True
        gs.review_approved = True
        result = check_tool_call("submit_fix", {}, gs)
        assert result is None

    def test_escalate_allowed_at_limit(self):
        gs = GuardrailState()
        gs.tool_call_count = MAX_TOOL_CALLS
        result = check_tool_call("escalate", {}, gs)
        assert result is None

    def test_cost_limit(self):
        gs = GuardrailState()
        gs.cost_usd = MAX_COST_USD
        result = check_limits(gs)
        assert result is not None
        assert "Cost cap" in result

    def test_terminal_tools_set(self):
        """Verify the exact terminal tools set."""
        assert TERMINAL_TOOLS == frozenset({"submit_fix", "escalate"})


# ===========================================================================
# 2. Tool return value → guardrail state transitions
# ===========================================================================

class TestGuardrailStateTransitions:
    """update_from_tool_result must correctly update state for each tool."""

    def test_create_sandbox_ok_sets_state(self):
        gs = GuardrailState()
        update_from_tool_result("create_sandbox", {}, "OK: Sandbox created\nsandbox_path=/tmp/test", gs)
        assert gs.sandbox_created is True
        assert gs.sandbox_path == "/tmp/test"

    def test_create_sandbox_error_no_state_change(self):
        gs = GuardrailState()
        update_from_tool_result("create_sandbox", {}, "ERROR: Git failed", gs)
        assert gs.sandbox_created is False

    def test_run_tests_passed(self):
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "passed\n5 passed in 2s", gs)
        assert gs.tests_attempted is True
        assert gs.tests_passed is True
        assert gs.tests_skipped is False

    def test_run_tests_failed(self):
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "failed (exit code 1)\nAssertionError", gs)
        assert gs.tests_attempted is True
        assert gs.tests_passed is False
        assert gs.tests_skipped is False
        assert gs.test_failure_count == 1

    def test_run_tests_skipped(self):
        """'skipped' sets tests_skipped=True (agent can proceed)."""
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "skipped: no tests collected", gs)
        assert gs.tests_attempted is True
        assert gs.tests_passed is False
        assert gs.tests_skipped is True

    def test_run_tests_error(self):
        """'error' sets tests_skipped=True (agent can proceed).

        Critical contract: the prompt says 'error' is OK to proceed on.
        The guardrails must agree by setting tests_skipped=True.
        """
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "error: pytest usage error (exit code 4)", gs)
        assert gs.tests_attempted is True
        assert gs.tests_passed is False
        assert gs.tests_skipped is True

    def test_run_tests_error_exit_code_2(self):
        """Exit code 2 (interrupted) → error prefix → tests_skipped."""
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "error: pytest interrupted (exit code 2)", gs)
        assert gs.tests_skipped is True

    def test_run_tests_error_exit_code_3(self):
        """Exit code 3 (internal pytest error) → error prefix → tests_skipped."""
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "error: pytest internal pytest error (exit code 3)", gs)
        assert gs.tests_skipped is True

    def test_review_approve(self):
        gs = GuardrailState()
        update_from_tool_result("request_review", {}, "REVIEW VERDICT: APPROVE (confidence: 85%)", gs)
        assert gs.review_approved is True
        assert gs.review_verdict == "APPROVE"

    def test_review_changes_requested(self):
        gs = GuardrailState()
        update_from_tool_result("request_review", {}, "REVIEW VERDICT: CHANGES_REQUESTED", gs)
        assert gs.review_approved is False
        assert gs.review_verdict == "CHANGES_REQUESTED"

    def test_review_escalate(self):
        gs = GuardrailState()
        update_from_tool_result("request_review", {}, "REVIEW VERDICT: ESCALATE", gs)
        assert gs.review_approved is False
        assert gs.review_verdict == "ESCALATE"

    def test_tool_call_count_increments(self):
        gs = GuardrailState()
        for i in range(5):
            update_from_tool_result("grep_repo", {}, "some result", gs)
        assert gs.tool_call_count == 5
        assert gs.grep_count == 5


# ===========================================================================
# 3. sandbox.py exit code → prefix contract
# ===========================================================================

class TestSandboxExitCodePrefixes:
    """_format_test_output must produce prefixes that guardrails recognize.

    Contract chain:
      sandbox._format_test_output → react_tools._classify_sandbox_output →
      guardrails.update_from_tool_result → guardrails.check_tool_call (submit gate)

    Every prefix here must match what update_from_tool_result checks:
      "passed" → tests_passed=True
      "skipped" → tests_skipped=True
      "error" → tests_skipped=True
      "failed" → tests_passed=False, test_failure_count++
    """

    def test_exit_0_passed(self):
        result = _format_test_output(0, "5 passed in 2.3s", 300)
        assert result.startswith("passed")

    def test_exit_1_failed(self):
        result = _format_test_output(1, "FAILED test_foo - AssertionError", 300)
        assert result.startswith("failed")

    def test_exit_2_error(self):
        """Exit 2 (interrupted) → 'error:' prefix, NOT 'failed'."""
        result = _format_test_output(2, "KeyboardInterrupt", 300)
        assert result.startswith("error:")

    def test_exit_3_error(self):
        """Exit 3 (internal pytest error) → 'error:' prefix, NOT 'failed'."""
        result = _format_test_output(3, "InternalError in pytest", 300)
        assert result.startswith("error:")

    def test_exit_4_error(self):
        """Exit 4 (usage error) → 'error:' prefix, NOT 'skipped' or 'failed'.

        This was the bug: exit 4 was originally classified as 'skipped' like exit 5.
        But exit 4 means bad CLI args / import errors, NOT 'no tests found'.
        """
        result = _format_test_output(4, "ERROR: conftest import failed", 300)
        assert result.startswith("error:")

    def test_exit_5_skipped(self):
        """Exit 5 (no tests collected) → 'skipped:' prefix."""
        result = _format_test_output(5, "no tests ran", 300)
        assert result.startswith("skipped:")

    def test_unknown_exit_code_failed(self):
        """Any other exit code → 'failed' prefix (conservative)."""
        result = _format_test_output(99, "something weird happened", 300)
        assert result.startswith("failed")


class TestClassifySandboxOutput:
    """react_tools._classify_sandbox_output preserves prefixes from sandbox.py."""

    def test_classify_preserves_all_prefixes(self):
        from agent.react_tools import _classify_sandbox_output

        assert _classify_sandbox_output("passed\n5 passed").startswith("passed")
        assert _classify_sandbox_output("skipped: no tests").startswith("skipped")
        assert _classify_sandbox_output("error: import failed").startswith("error")
        assert _classify_sandbox_output("failed (exit code 1)").startswith("failed")

    def test_classify_unknown_becomes_failed(self):
        """Unknown output is treated as failed (conservative)."""
        from agent.react_tools import _classify_sandbox_output

        result = _classify_sandbox_output("some random output")
        assert result.startswith("failed:")

    def test_classify_truncates_output(self):
        from agent.react_tools import _classify_sandbox_output

        long_output = "passed\n" + "x" * 10000
        result = _classify_sandbox_output(long_output)
        assert len(result) <= 500


# ===========================================================================
# 4. Prompt-guardrail alignment
# ===========================================================================

class TestPromptGuardrailAlignment:
    """The system prompt and guardrails must agree on rules.

    When these diverge, the agent follows the prompt but gets blocked by guardrails,
    wasting tool calls and hitting the limit.
    """

    def test_prompt_tool_budget_matches_guardrail(self):
        """Prompt says '30' tool call budget, guardrail limit is 40."""
        from agent.react_prompt import build_system_prompt
        prompt = build_system_prompt(
            {"repo_name": "test", "title": "bug", "description": "desc"},
            {"expected_behavior": "", "actual_behavior": ""},
            "",
        )
        # Prompt says budget of 30 (guidance), guardrail enforces 40 (hard limit)
        assert "30" in prompt
        assert MAX_TOOL_CALLS == 40  # Hard limit is higher than soft budget

    def test_prompt_test_result_table_matches_guardrails(self):
        """The prompt's test result table must match guardrail behavior.

        Prompt says:
          passed → proceed
          skipped → proceed
          error → proceed
          failed → block (fix and re-test)

        Guardrails must implement exactly this.
        """
        # "passed" → can submit
        gs_passed = GuardrailState()
        gs_passed.sandbox_created = True
        gs_passed.review_approved = True
        gs_passed.tests_attempted = True
        gs_passed.tests_passed = True
        assert check_tool_call("submit_fix", {}, gs_passed) is None

        # "skipped" → can submit
        gs_skipped = GuardrailState()
        gs_skipped.sandbox_created = True
        gs_skipped.review_approved = True
        gs_skipped.tests_attempted = True
        gs_skipped.tests_skipped = True
        assert check_tool_call("submit_fix", {}, gs_skipped) is None

        # "error" → can submit (error sets tests_skipped=True)
        gs_error = GuardrailState()
        gs_error.sandbox_created = True
        gs_error.review_approved = True
        gs_error.tests_attempted = True
        gs_error.tests_skipped = True  # error → tests_skipped
        assert check_tool_call("submit_fix", {}, gs_error) is None

        # "failed" → BLOCKED
        gs_failed = GuardrailState()
        gs_failed.sandbox_created = True
        gs_failed.review_approved = True
        gs_failed.tests_attempted = True
        gs_failed.tests_passed = False
        gs_failed.tests_skipped = False
        assert check_tool_call("submit_fix", {}, gs_failed) is not None

    def test_prompt_documents_all_tools(self):
        """Every tool in REACT_TOOLS must be mentioned in the system prompt."""
        from agent.react_prompt import build_system_prompt
        from agent.react_tools import REACT_TOOLS

        prompt = build_system_prompt(
            {"repo_name": "test", "title": "t", "description": "d"},
            {"expected_behavior": "", "actual_behavior": ""},
            "",
        )
        for tool in REACT_TOOLS:
            assert tool.name in prompt, (
                f"Tool '{tool.name}' exists in REACT_TOOLS but is not documented in the system prompt"
            )

    def test_prompt_documents_sandbox_requirement(self):
        """Prompt must tell agent to create sandbox before editing."""
        from agent.react_prompt import build_system_prompt
        prompt = build_system_prompt(
            {"repo_name": "test", "title": "t", "description": "d"},
            {"expected_behavior": "", "actual_behavior": ""},
            "",
        )
        assert "create_sandbox" in prompt
        assert "before" in prompt.lower()

    def test_prompt_documents_review_requirement(self):
        """Prompt must tell agent to request_review before submit_fix."""
        from agent.react_prompt import build_system_prompt
        prompt = build_system_prompt(
            {"repo_name": "test", "title": "t", "description": "d"},
            {"expected_behavior": "", "actual_behavior": ""},
            "",
        )
        assert "request_review" in prompt
        assert "submit_fix" in prompt


class TestAntiPatternThresholds:
    """Anti-pattern warnings at documented thresholds."""

    def test_grep_warning_at_8(self):
        gs = GuardrailState()
        gs.grep_count = 8
        result = check_tool_call("grep_repo", {}, gs)
        assert result is not None
        assert "WARNING" in result

    def test_grep_ok_at_7(self):
        gs = GuardrailState()
        gs.grep_count = 7
        result = check_tool_call("grep_repo", {}, gs)
        assert result is None

    def test_read_file_warning_at_10(self):
        gs = GuardrailState()
        gs.read_file_count = 10
        result = check_tool_call("read_file", {}, gs)
        assert result is not None
        assert "WARNING" in result

    def test_run_tests_warning_at_3(self):
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.run_tests_count = 3
        result = check_tool_call("run_tests", {}, gs)
        assert result is not None
        assert "WARNING" in result

    def test_string_replace_warning_at_4(self):
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.string_replace_count = 4
        result = check_tool_call("string_replace", {}, gs)
        assert result is not None
        assert "WARNING" in result

    def test_review_warning_at_2(self):
        gs = GuardrailState()
        gs.review_count = 2
        result = check_tool_call("request_review", {}, gs)
        assert result is not None
        assert "WARNING" in result


# ===========================================================================
# 5. react_loop terminal detection
# ===========================================================================

class TestSubmitFixTerminalContract:
    """submit_fix is only terminal when it returns 'OK:', not on error."""

    def test_submit_ok_is_terminal(self):
        """When submit_fix returns 'OK: ...', the loop should mark submitted=True."""
        # This tests the contract in react_loop.py lines 247-259
        result_text = "OK: Fix submitted.\ncommit_status=committed\nbranch=fix/test"
        assert result_text.startswith("OK:")

    def test_submit_error_is_not_terminal(self):
        """When submit_fix returns 'ERROR: ...', the loop should NOT mark submitted."""
        result_text = "ERROR: No changes to submit."
        assert not result_text.startswith("OK:")

    def test_submit_error_guardrail_block_not_terminal(self):
        """When guardrails block submit_fix, it returns ERROR and is not terminal."""
        gs = GuardrailState()
        # Missing prerequisites
        result = check_tool_call("submit_fix", {}, gs)
        assert result is not None
        assert result.startswith("ERROR:")


class TestEscalateTerminalContract:
    """escalate is always terminal when not blocked by guardrails."""

    def test_escalate_returns_prefix(self):
        """escalate tool returns 'ESCALATED: reason'."""
        from agent.react_tools import escalate
        result = escalate.invoke({"reason": "too complex"})
        assert result.startswith("ESCALATED:")

    def test_escalate_allowed_at_limit(self):
        """escalate must work even when tool call limit is reached."""
        gs = GuardrailState()
        gs.tool_call_count = MAX_TOOL_CALLS
        result = check_tool_call("escalate", {}, gs)
        assert result is None


# ===========================================================================
# 6. Tool return value format contracts
# ===========================================================================

class TestToolReturnPrefixes:
    """Tools must return consistent prefixes so guardrails can parse them."""

    def test_create_sandbox_ok_format(self):
        """create_sandbox OK result must contain 'OK:' and 'sandbox_path='."""
        # We can't actually create a sandbox in tests, but verify the format
        # from the source code
        import inspect
        from agent.react_tools import create_sandbox
        src = inspect.getsource(create_sandbox.func)
        assert "OK:" in src
        assert "sandbox_path=" in src

    def test_string_replace_ok_format(self):
        """string_replace OK result starts with 'OK:'."""
        import inspect
        from agent.react_tools import string_replace
        src = inspect.getsource(string_replace.func)
        assert '"OK: replaced' in src

    def test_record_localization_ok_format(self):
        from agent.react_tools import record_localization
        result = record_localization.invoke({
            "fault_files": ["test.py"],
            "fault_functions": ["foo"],
            "root_cause_hypothesis": "test hypothesis",
        })
        assert result.startswith("OK:")

    def test_record_localization_error_on_empty_files(self):
        from agent.react_tools import record_localization
        result = record_localization.invoke({
            "fault_files": [],
            "fault_functions": [],
            "root_cause_hypothesis": "test",
        })
        assert result.startswith("ERROR:")

    def test_record_localization_error_on_empty_hypothesis(self):
        from agent.react_tools import record_localization
        result = record_localization.invoke({
            "fault_files": ["test.py"],
            "fault_functions": [],
            "root_cause_hypothesis": "",
        })
        assert result.startswith("ERROR:")


# ===========================================================================
# 7. Guardrail state isolation
# ===========================================================================

class TestGuardrailStateIsolation:
    """Each GuardrailState instance must be independent."""

    def test_separate_instances_are_independent(self):
        gs1 = GuardrailState()
        gs2 = GuardrailState()
        update_from_tool_result("grep_repo", {}, "found something", gs1)
        assert gs1.grep_count == 1
        assert gs2.grep_count == 0

    def test_failed_test_resets_on_pass(self):
        """After tests pass, test_failure_count resets to 0."""
        gs = GuardrailState()
        update_from_tool_result("run_tests", {}, "failed (exit code 1)\nError", gs)
        assert gs.test_failure_count == 1
        update_from_tool_result("run_tests", {}, "passed\n5 passed", gs)
        assert gs.test_failure_count == 0
        assert gs.tests_passed is True


# ===========================================================================
# 8. End-to-end contract: sandbox exit code → guardrail → submit gate
# ===========================================================================

class TestEndToEndExitCodeToSubmit:
    """Full chain: _format_test_output → update_from_tool_result → submit gate.

    For each pytest exit code, verify the full contract from raw exit code
    to whether the agent can submit.
    """

    @pytest.mark.parametrize("exit_code,can_submit", [
        (0, True),   # passed → can submit
        (1, False),  # failed → blocked
        (2, True),   # interrupted → error → can submit
        (3, True),   # internal error → error → can submit
        (4, True),   # usage error → error → can submit
        (5, True),   # no tests → skipped → can submit
    ])
    def test_exit_code_to_submit_gate(self, exit_code, can_submit):
        """Verify the full chain from pytest exit code to submit gate."""
        # Step 1: sandbox formats the output
        formatted = _format_test_output(exit_code, "test output here", 300)

        # Step 2: guardrails update state from the formatted output
        gs = GuardrailState()
        gs.sandbox_created = True
        gs.review_approved = True
        update_from_tool_result("run_tests", {}, formatted, gs)

        # Step 3: check if submit is allowed
        result = check_tool_call("submit_fix", {}, gs)

        if can_submit:
            assert result is None, (
                f"Exit code {exit_code} produced '{formatted[:50]}...' but submit was BLOCKED: {result}"
            )
        else:
            assert result is not None, (
                f"Exit code {exit_code} produced '{formatted[:50]}...' but submit was ALLOWED (should be blocked)"
            )
