"""Tests for _generate_function_summaries bug-fix.

Verifies that:
  1. A response with no .text block logs a warning and returns {}.
  2. A response with no .text block does NOT raise an exception.
  3. A valid response correctly parses and returns the expected dict.
  4. A valid response does NOT log a warning.
"""
import logging
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so we can import the module without real infrastructure.
# ---------------------------------------------------------------------------

# Stub out heavy dependencies before importing the module under test.
import sys

# anthropic stub
anthropics_stub = types.ModuleType("anthropic")
anthropics_stub.APIError = Exception
anthropics_stub.Anthropic = MagicMock
sys.modules.setdefault("anthropic", anthropics_stub)

# neo4j_client stub
backend_stub = types.ModuleType("backend")
db_stub = types.ModuleType("backend.db")
neo4j_stub = types.ModuleType("backend.db.neo4j_client")
neo4j_stub.neo4j_client = MagicMock()
neo4j_stub.run = MagicMock()
sys.modules.setdefault("backend", backend_stub)
sys.modules.setdefault("backend.db", db_stub)
sys.modules.setdefault("backend.db.neo4j_client", neo4j_stub)

# Patch the module-level neo4j_client import used inside summarizer
with patch.dict(
    "sys.modules",
    {
        "anthropic": anthropics_stub,
    },
):
    # Ensure the module can be imported even if neo4j_client is missing.
    pass


def _make_summarizer():
    """Return a CodeSummarizer instance with a mocked Anthropic client."""
    # We import here so stubs are already in place.
    with patch("backend.enricher.summarizer.neo4j_client", MagicMock()), \
         patch("backend.enricher.summarizer.anthropic", anthropics_stub):
        from backend.enricher.summarizer import CodeSummarizer  # noqa: PLC0415
        summarizer = CodeSummarizer.__new__(CodeSummarizer)
        summarizer.repo_name = "test-repo"
        summarizer._client = MagicMock()
        return summarizer


def _no_text_response():
    """Fake API response whose content blocks have NO .text attribute."""
    block = MagicMock(spec=[])          # spec=[] means hasattr(block, 'text') is False
    response = MagicMock()
    response.content = [block]
    return response


def _text_response(text: str):
    """Fake API response with a single text block."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


FUNCTIONS = [
    {"id": "id_foo", "name": "foo", "params": [], "return_type": None, "docstring": ""},
    {"id": "id_bar", "name": "bar", "params": ["x"], "return_type": "int", "docstring": ""},
]


class TestGenerateFunctionSummaries(unittest.TestCase):

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call(self, response):
        summarizer = _make_summarizer()
        with patch("backend.enricher.summarizer.neo4j_client", MagicMock()):
            summarizer._client.messages.create.return_value = response
            return summarizer._generate_function_summaries("some/file.py", FUNCTIONS)

    # ------------------------------------------------------------------
    # AC-1 & AC-2: no .text block → warning logged, empty dict returned
    # ------------------------------------------------------------------

    def test_no_text_block_returns_empty_dict(self):
        """AC-2: must return {} without raising when no text block present."""
        result = self._call(_no_text_response())
        self.assertEqual(result, {})

    def test_no_text_block_logs_warning(self):
        """AC-1: must log a WARNING when no text block is found."""
        with self.assertLogs("backend.enricher.summarizer", level=logging.WARNING) as cm:
            self._call(_no_text_response())
        # At least one WARNING record about missing text content
        warning_msgs = [r for r in cm.output if "WARNING" in r]
        self.assertTrue(
            warning_msgs,
            "Expected at least one WARNING log entry, got: " + str(cm.output),
        )
        # The warning should mention the file or 'text content'
        combined = " ".join(warning_msgs).lower()
        self.assertTrue(
            "text" in combined or "no text" in combined or "content" in combined,
            f"Warning message did not mention text content: {combined}",
        )

    # ------------------------------------------------------------------
    # AC-3: valid response → correct dict returned
    # ------------------------------------------------------------------

    def test_valid_response_parses_summaries(self):
        """AC-3: a well-formed text response produces the expected dict."""
        text = "foo: does foo things\nbar: computes bar"
        result = self._call(_text_response(text))
        self.assertEqual(result, {"id_foo": "does foo things", "id_bar": "computes bar"})

    # ------------------------------------------------------------------
    # AC-4: valid response → no warning logged
    # ------------------------------------------------------------------

    def test_valid_response_no_warning(self):
        """AC-4: no WARNING should be emitted when text is successfully extracted."""
        text = "foo: does foo things\nbar: computes bar"
        import logging as _logging
        with self.assertLogs("backend.enricher.summarizer", level=_logging.DEBUG) as cm:
            # Emit a dummy debug message so assertLogs doesn't fail when there
            # are zero log records at all.
            import backend.enricher.summarizer as _mod  # noqa: PLC0415
            _mod.logger.debug("sentinel")
            self._call(_text_response(text))
        warning_msgs = [r for r in cm.output if "WARNING" in r]
        self.assertEqual(
            warning_msgs,
            [],
            "No WARNING should be logged for a valid response, got: " + str(warning_msgs),
        )


if __name__ == "__main__":
    unittest.main()
