"""
test_explore_subagent.py — Tests for the Haiku-backed read-only explore subagent.

The explore subagent is called by the main agent via the `delegate_explore`
tool. It runs its own ReAct loop with read-only tools (grep, read_file, ...)
and returns a focused FINDINGS report.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.explore_subagent import (
    EXPLORE_MAX_TOOL_CALLS,
    EXPLORE_SUBAGENT_TOOLS,
    delegate_explore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(text: str = "", tool_calls: list = None, usage_in: int = 0, usage_out: int = 0):
    """Build a fake AIMessage-like response."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = {"input_tokens": usage_in, "output_tokens": usage_out} if usage_in or usage_out else {}
    return msg


def _set_repo_context(repo_name: str = "test_repo", repo_path: str = "/tmp/test_repo"):
    from agent.explore_tools import set_context
    set_context(repo_name, repo_path)


# ---------------------------------------------------------------------------
# delegate_explore — input validation
# ---------------------------------------------------------------------------

class TestDelegateExploreValidation:
    def test_empty_question_rejected(self):
        result = delegate_explore.invoke({"question": ""})
        assert result.startswith("ERROR")

    def test_whitespace_only_question_rejected(self):
        result = delegate_explore.invoke({"question": "   "})
        assert result.startswith("ERROR")

    def test_no_repo_context_rejected(self):
        # Clear any previous repo context
        from agent.explore_tools import set_context
        set_context("", "")  # falsy values
        result = delegate_explore.invoke({"question": "find process_payment"})
        assert result.startswith("ERROR")
        assert "repo context" in result.lower()


# ---------------------------------------------------------------------------
# Subagent loop — happy path with mocked LLM
# ---------------------------------------------------------------------------

class TestSubagentLoop:
    def test_subagent_returns_finding_after_no_tool_calls(self, tmp_path):
        """When the subagent's first response has no tool_calls, return its text."""
        _set_repo_context("repo_x", str(tmp_path))

        responses = [
            _llm_response(
                text="FINDINGS: process_payment is at payments.py:42",
                tool_calls=[],
                usage_in=200, usage_out=50,
            ),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses
        mock_llm.bind_tools.return_value = mock_llm

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            result = delegate_explore.invoke({"question": "find process_payment"})

        assert "EXPLORE SUBAGENT REPORT" in result
        assert "process_payment is at payments.py:42" in result
        assert "0 tool calls" in result

    def test_subagent_executes_tool_calls_then_returns(self, tmp_path):
        _set_repo_context("repo_y", str(tmp_path))

        # First response: subagent calls grep_repo
        # Second response: subagent produces FINDINGS with no tool calls
        responses = [
            _llm_response(
                text="Let me grep for it.",
                tool_calls=[{
                    "id": "tc1",
                    "name": "grep_repo",
                    "args": {"pattern": "process_payment"},
                }],
                usage_in=300, usage_out=40,
            ),
            _llm_response(
                text="FINDINGS: found in payments.py:42",
                tool_calls=[],
                usage_in=600, usage_out=80,
            ),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses
        mock_llm.bind_tools.return_value = mock_llm

        # Mock the grep_repo tool to avoid real filesystem access
        mock_grep = MagicMock()
        mock_grep.name = "grep_repo"
        mock_grep.invoke.return_value = "payments.py:42: def process_payment():"

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ), patch(
            "agent.explore_subagent._build_subagent_tools",
            return_value=[mock_grep],
        ):
            result = delegate_explore.invoke({"question": "find process_payment"})

        assert "FINDINGS" in result
        assert "1 tool calls" in result
        # grep_repo was actually called
        mock_grep.invoke.assert_called_once_with({"pattern": "process_payment"})

    def test_subagent_budget_exhaustion_forces_summary(self, tmp_path):
        """When tool budget is hit, subagent gets a forced 'stop and summarize' turn."""
        _set_repo_context("repo_z", str(tmp_path))

        # Subagent keeps calling grep — we never let it stop on its own
        looping_response = _llm_response(
            text="searching",
            tool_calls=[{"id": f"tc{i}", "name": "grep_repo", "args": {"pattern": "x"}}
                        for i in range(EXPLORE_MAX_TOOL_CALLS + 5)],
            usage_in=100, usage_out=10,
        )
        # Final summary after budget exhaustion
        final_summary = _llm_response(
            text="FINDINGS: budget hit; partial results found in foo.py",
            tool_calls=[],
            usage_in=50, usage_out=30,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [looping_response, final_summary]
        mock_llm.bind_tools.return_value = mock_llm

        mock_grep = MagicMock()
        mock_grep.name = "grep_repo"
        mock_grep.invoke.return_value = "match"

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ), patch(
            "agent.explore_subagent._build_subagent_tools",
            return_value=[mock_grep],
        ):
            result = delegate_explore.invoke({"question": "find x"})

        assert "budget hit" in result.lower() or "exhausted" in result.lower()

    def test_unknown_tool_name_returns_error_in_subagent(self, tmp_path):
        """Subagent calling an unknown tool gets a clear error string back."""
        _set_repo_context("repo_q", str(tmp_path))

        responses = [
            _llm_response(
                text="trying to call something",
                tool_calls=[{"id": "tc1", "name": "nonexistent_tool", "args": {}}],
                usage_in=100, usage_out=20,
            ),
            _llm_response(
                text="FINDINGS: I made an error.",
                tool_calls=[],
                usage_in=50, usage_out=20,
            ),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses
        mock_llm.bind_tools.return_value = mock_llm

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ), patch(
            "agent.explore_subagent._build_subagent_tools",
            return_value=[],  # No tools available
        ):
            result = delegate_explore.invoke({"question": "do something"})

        # Subagent recovered + produced a FINDINGS line — the tool error
        # was handled internally without crashing the delegate.
        assert "FINDINGS" in result

    def test_llm_exception_returns_error(self, tmp_path):
        _set_repo_context("repo_err", str(tmp_path))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        mock_llm.bind_tools.return_value = mock_llm

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ), patch(
            "agent.explore_subagent._build_subagent_tools",
            return_value=[],
        ):
            result = delegate_explore.invoke({"question": "find anything"})

        # Should not crash — returns a clear error report
        assert "ERROR" in result or "failed" in result.lower()


# ---------------------------------------------------------------------------
# Tool-result truncation (subagent context safety)
# ---------------------------------------------------------------------------

class TestSubagentSafety:
    def test_large_tool_result_is_truncated(self, tmp_path):
        """Tool results > 6000 chars get truncated so the subagent doesn't OOM its own context."""
        _set_repo_context("repo_big", str(tmp_path))

        big_result = "x" * 100_000

        responses = [
            _llm_response(
                text="grepping",
                tool_calls=[{"id": "tc1", "name": "grep_repo", "args": {"pattern": "y"}}],
                usage_in=100, usage_out=10,
            ),
            _llm_response(
                text="FINDINGS: too much output",
                tool_calls=[],
                usage_in=50, usage_out=10,
            ),
        ]

        mock_llm = MagicMock()
        # Capture the messages passed to the SECOND invoke (after tool result is appended)
        captured: list = []

        def capture_invoke(messages):
            captured.append(messages)
            return responses[len(captured) - 1]

        mock_llm.invoke.side_effect = capture_invoke
        mock_llm.bind_tools.return_value = mock_llm

        mock_grep = MagicMock()
        mock_grep.name = "grep_repo"
        mock_grep.invoke.return_value = big_result

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ), patch(
            "agent.explore_subagent._build_subagent_tools",
            return_value=[mock_grep],
        ):
            delegate_explore.invoke({"question": "find y"})

        # Find the ToolMessage in the second invocation
        from langchain_core.messages import ToolMessage
        second_messages = captured[1]
        tool_msgs = [m for m in second_messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        # Truncated content should be much smaller than the original
        truncated = str(tool_msgs[-1].content)
        assert len(truncated) < 7000  # 6000 cap + some marker text


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_delegate_explore_in_tool_collection(self):
        names = [t.name for t in EXPLORE_SUBAGENT_TOOLS]
        assert "delegate_explore" in names

    def test_delegate_explore_tool_has_docstring(self):
        """LangChain tools require a docstring for the model to know when to use them."""
        assert delegate_explore.description
        assert len(delegate_explore.description) > 100  # non-trivial guidance

    def test_delegate_explore_added_to_main_agent_tool_set(self):
        """react_loop combines explore_tools + EXPLORE_SUBAGENT_TOOLS + REACT_TOOLS."""
        import agent.react_loop as react_loop
        import inspect
        src = inspect.getsource(react_loop.react_loop)
        # The wiring should reference the subagent tool collection
        assert "EXPLORE_SUBAGENT_TOOLS" in src
