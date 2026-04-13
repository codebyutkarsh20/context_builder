"""
test_forked_subagent.py — Tests for cache-preserving subagent forks.

The forked subagent infrastructure (ported from Claude Code's
utils/forkedAgent.ts) lets subordinate LLM calls (verifier, summarizer,
etc.) inherit the parent's prompt-cached prefix instead of paying full
price for a fresh call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from agent.forked_subagent import (
    CacheSafeParams,
    clear_cache_safe_params,
    get_last_cache_safe_params,
    run_forked_subagent,
    save_cache_safe_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_params():
    clear_cache_safe_params()
    yield
    clear_cache_safe_params()


# ---------------------------------------------------------------------------
# CacheSafeParams + module-level slot
# ---------------------------------------------------------------------------

class TestParamsSlot:
    def test_slot_starts_empty(self):
        assert get_last_cache_safe_params() is None

    def test_save_and_retrieve(self):
        p = CacheSafeParams(
            system_prompt="You are an AI.",
            messages=[HumanMessage(content="hi")],
            model="claude-sonnet-4-6",
        )
        save_cache_safe_params(p)
        retrieved = get_last_cache_safe_params()
        assert retrieved is p
        assert retrieved.system_prompt == "You are an AI."
        assert len(retrieved.messages) == 1
        assert retrieved.model == "claude-sonnet-4-6"

    def test_save_none_clears(self):
        save_cache_safe_params(CacheSafeParams(system_prompt="x"))
        save_cache_safe_params(None)
        assert get_last_cache_safe_params() is None

    def test_clear_helper(self):
        save_cache_safe_params(CacheSafeParams(system_prompt="x"))
        clear_cache_safe_params()
        assert get_last_cache_safe_params() is None

    def test_clear_when_already_empty(self):
        clear_cache_safe_params()
        clear_cache_safe_params()
        assert get_last_cache_safe_params() is None

    def test_overwrite_replaces_previous(self):
        save_cache_safe_params(CacheSafeParams(system_prompt="first"))
        save_cache_safe_params(CacheSafeParams(system_prompt="second"))
        assert get_last_cache_safe_params().system_prompt == "second"

    def test_thread_local_isolation(self):
        """Each thread sees its own slot — concurrent runs don't trample."""
        import threading

        results: dict = {}

        def worker(tid: int):
            save_cache_safe_params(CacheSafeParams(system_prompt=f"thread{tid}"))
            # Sleep briefly to let other threads also write
            import time
            time.sleep(0.01)
            params = get_last_cache_safe_params()
            results[tid] = params.system_prompt if params else None

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should see ITS OWN value, not another thread's
        for tid in range(5):
            assert results[tid] == f"thread{tid}"


# ---------------------------------------------------------------------------
# run_forked_subagent — happy path with a mocked LLM
# ---------------------------------------------------------------------------

class _Verdict(BaseModel):
    verdict: str = "APPROVE"
    reason: str = ""


class TestRunForkedSubagent:
    def test_fallback_when_no_parent_params(self):
        """Without saved params, the helper falls back to a fresh call."""
        mock_response = MagicMock()
        mock_response.content = "Hello from the LLM"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agent.forked_subagent.ChatAnthropic", return_value=mock_llm) \
                if False else patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            result = run_forked_subagent("Just say hi")

        assert result["cached"] is False
        assert "Hello" in result["response_text"]
        assert result["error"] is None

    def test_uses_parent_system_prompt_and_messages(self):
        """When parent params exist, the LLM call sees the parent's prompt."""
        save_cache_safe_params(CacheSafeParams(
            system_prompt="PARENT_SYS_PROMPT_MARKER",
            messages=[
                HumanMessage(content="parent task"),
                AIMessage(content="parent reply"),
            ],
            model="claude-sonnet-4-6",
        ))

        mock_response = MagicMock()
        mock_response.content = "subagent response"
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_token_details": {"cache_read": 90},
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        constructor = MagicMock(return_value=mock_llm)

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", constructor,
        ):
            result = run_forked_subagent("Now do the subagent task")

        assert result["cached"] is True
        assert result["cache_read_tokens"] == 90
        assert result["response_text"] == "subagent response"

        # Verify the LLM was invoked with: SystemMessage + parent messages + new HumanMessage
        called_with = mock_llm.invoke.call_args[0][0]
        assert isinstance(called_with[0], SystemMessage)
        assert "PARENT_SYS_PROMPT_MARKER" in str(called_with[0].content)
        # Last message must be the new task
        assert isinstance(called_with[-1], HumanMessage)
        assert "subagent task" in str(called_with[-1].content)
        # Total: 1 system + 2 parent + 1 new task = 4 messages
        assert len(called_with) == 4

    def test_fallback_uses_provided_system_prompt(self):
        """When no parent params, the fallback_system_prompt is used."""
        mock_response = MagicMock()
        mock_response.content = "hi"
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            run_forked_subagent("task", fallback_system_prompt="FALLBACK_PROMPT")

        called_with = mock_llm.invoke.call_args[0][0]
        assert isinstance(called_with[0], SystemMessage)
        assert called_with[0].content == "FALLBACK_PROMPT"

    def test_uses_parent_model(self):
        """The subagent must use the SAME model as the parent for cache reuse."""
        save_cache_safe_params(CacheSafeParams(
            system_prompt="x",
            messages=[],
            model="claude-haiku-4-5-20251001",
        ))

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.usage_metadata = {}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        constructor = MagicMock(return_value=mock_llm)

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", constructor,
        ):
            run_forked_subagent("task")

        # Constructor called with model from parent params
        kwargs = constructor.call_args.kwargs
        assert kwargs.get("model") == "claude-haiku-4-5-20251001"

    def test_explicit_parent_params_override_module_slot(self):
        """parent_params kwarg takes precedence over the module-level slot."""
        save_cache_safe_params(CacheSafeParams(system_prompt="MODULE_LEVEL"))
        explicit = CacheSafeParams(system_prompt="EXPLICIT_PARAMS")

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.usage_metadata = {}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            run_forked_subagent("task", parent_params=explicit)

        called_with = mock_llm.invoke.call_args[0][0]
        assert called_with[0].content == "EXPLICIT_PARAMS"

    def test_llm_exception_returns_error_dict(self):
        save_cache_safe_params(CacheSafeParams(system_prompt="x"))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            result = run_forked_subagent("task")

        assert result["error"] is not None
        assert "API down" in result["error"]
        assert result["response_text"] == ""
        assert result["parsed"] is None

    def test_structured_output_with_schema(self):
        """When schema is provided, the parsed instance is returned."""
        save_cache_safe_params(CacheSafeParams(system_prompt="x"))

        verdict_instance = _Verdict(verdict="REJECT", reason="bad fix")

        mock_bound = MagicMock()
        mock_bound.invoke.return_value = verdict_instance
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_bound

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            result = run_forked_subagent("verify this", schema=_Verdict)

        assert result["parsed"] is verdict_instance
        assert result["parsed"].verdict == "REJECT"
        assert result["error"] is None
        # with_structured_output must have been invoked with the schema
        mock_llm.with_structured_output.assert_called_once_with(_Verdict)


# ---------------------------------------------------------------------------
# Cache-prefix integrity (the whole point of forking)
# ---------------------------------------------------------------------------

class TestCachePrefixIntegrity:
    """Sanity checks that the helper doesn't accidentally mutate the parent
    state in ways that would break the cache.
    """

    def test_parent_messages_not_mutated(self):
        parent_messages = [
            HumanMessage(content="m1"),
            AIMessage(content="r1"),
        ]
        save_cache_safe_params(CacheSafeParams(
            system_prompt="sys",
            messages=parent_messages,
            model="claude-sonnet-4-6",
        ))

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.usage_metadata = {}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            run_forked_subagent("task")

        # Parent message list reference unchanged (didn't get extended)
        assert len(parent_messages) == 2
        # And subsequent lookup still returns the original list
        assert get_last_cache_safe_params().messages is parent_messages

    def test_subagent_can_be_called_multiple_times(self):
        save_cache_safe_params(CacheSafeParams(
            system_prompt="sys",
            messages=[HumanMessage(content="m1")],
            model="claude-sonnet-4-6",
        ))

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.usage_metadata = {}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch.object(
            __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
            "ChatAnthropic", return_value=mock_llm,
        ):
            for _ in range(3):
                result = run_forked_subagent("task")
                assert result["cached"] is True
                assert result["error"] is None

        # Still cached after 3 calls
        assert get_last_cache_safe_params() is not None
