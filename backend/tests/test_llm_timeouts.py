"""
Tests for LLM timeout and retry behavior — Phases 1.4, 1.6

Verifies:
  - ChatAnthropic is constructed with timeout and max_retries
  - Error messages are truncated in retry prompts
  - _structured_call retries once on failure
"""

import sys
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import _structured_call


class TestStructuredCallConfig:
    """_structured_call configures the LLM client correctly."""

    def test_timeout_in_source(self):
        src = inspect.getsource(_structured_call)
        assert "timeout=120.0" in src

    def test_max_retries_in_source(self):
        src = inspect.getsource(_structured_call)
        assert "max_retries=2" in src

    def test_error_truncation_in_source(self):
        src = inspect.getsource(_structured_call)
        assert "[:300]" in src


class TestStructuredCallRetry:
    """_structured_call retries on failure with truncated error."""

    @patch("agent.pipeline.ChatAnthropic")
    def test_retries_on_first_failure(self, mock_cls):
        """First call fails, retry succeeds."""
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # First call raises, second succeeds
        expected = MagicMock()
        mock_structured.invoke.side_effect = [ValueError("parse error"), expected]

        result = _structured_call("model", 100, dict, "prompt", retries=1)
        assert result == expected
        assert mock_structured.invoke.call_count == 2

    @patch("agent.pipeline.ChatAnthropic")
    def test_raises_if_no_retries(self, mock_cls):
        """With retries=0, first failure raises immediately."""
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            _structured_call("model", 100, dict, "prompt", retries=0)

    @patch("agent.pipeline.ChatAnthropic")
    def test_retry_prompt_contains_truncated_error(self, mock_cls):
        """The retry prompt includes the (truncated) error message."""
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        long_error = "x" * 500
        mock_structured.invoke.side_effect = [ValueError(long_error), MagicMock()]

        _structured_call("model", 100, dict, "prompt", retries=1)

        # Check the retry prompt
        retry_call = mock_structured.invoke.call_args_list[1]
        retry_prompt = retry_call[0][0]
        # Error should be truncated to 300 chars
        assert "x" * 300 in retry_prompt
        assert "x" * 301 not in retry_prompt

    @patch("agent.pipeline.ChatAnthropic")
    def test_both_calls_fail_raises(self, mock_cls):
        """If both original and retry fail, the retry error is raised."""
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = [ValueError("first"), ValueError("second")]

        with pytest.raises(ValueError, match="second"):
            _structured_call("model", 100, dict, "prompt", retries=1)

    @patch("agent.pipeline.ChatAnthropic")
    def test_constructor_params(self, mock_cls):
        """ChatAnthropic is called with correct timeout and max_retries."""
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = MagicMock()

        _structured_call("claude-sonnet-4-6", 1000, dict, "test")

        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            timeout=120.0,
            max_retries=2,
        )
