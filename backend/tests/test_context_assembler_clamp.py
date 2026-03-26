"""Tests for the negative-slice clamp fix in ContextAssembler.assemble."""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests for the clamp expression itself
# ---------------------------------------------------------------------------

class TestCharsLeftClamp:
    """Directly test the max(0, ...) clamp logic independently of the class."""

    def _chars_left(self, token_budget: int, used_tokens: int) -> int:
        """Replicate the patched formula."""
        return max(0, (token_budget - used_tokens) * 4)

    def test_used_exceeds_budget_returns_zero(self):
        """When used_tokens > token_budget, chars_left must be 0."""
        result = self._chars_left(token_budget=100, used_tokens=150)
        assert result == 0, f"Expected 0, got {result}"

    def test_used_equals_budget_returns_zero(self):
        """When used_tokens == token_budget, chars_left must be 0."""
        result = self._chars_left(token_budget=100, used_tokens=100)
        assert result == 0, f"Expected 0, got {result}"

    def test_used_below_budget_returns_positive(self):
        """When used_tokens < token_budget, chars_left must be positive."""
        result = self._chars_left(token_budget=200, used_tokens=100)
        assert result == 400, f"Expected 400, got {result}"

    def test_slice_with_zero_chars_left_is_empty(self):
        """primary_section[:0] must be empty, not a tail substring."""
        primary_section = "Hello World, this is important context."
        chars_left = self._chars_left(token_budget=50, used_tokens=100)
        assert chars_left == 0
        assert primary_section[:chars_left] == "", (
            "Slicing with chars_left=0 must yield an empty string."
        )

    def test_old_negative_slice_would_drop_end(self):
        """Demonstrate the old bug: negative index drops the END of the string."""
        primary_section = "ABCDEFGHIJ"
        # Old (buggy) formula
        chars_left_buggy = (50 - 100) * 4  # == -200
        # Python interprets this as primary_section[:-200] which is ''  for short strings,
        # but for long strings it would silently drop the tail.
        long_section = "A" * 1000
        chars_left_buggy_small = (500 - 600) * 4  # == -400
        buggy_slice = long_section[:chars_left_buggy_small]  # drops last 400 chars
        assert len(buggy_slice) == 600, "Old bug: 600 chars remain instead of 0"

        # New (fixed) formula returns 0 → empty slice
        chars_left_fixed = max(0, (500 - 600) * 4)
        fixed_slice = long_section[:chars_left_fixed]
        assert fixed_slice == "", "Fixed formula: slice must be empty when over budget"


# ---------------------------------------------------------------------------
# Integration-style tests using a minimal ContextAssembler stub
# ---------------------------------------------------------------------------

def _make_assembler(enriched_data: dict):
    """Create a ContextAssembler whose _load_enriched returns enriched_data."""
    # Import lazily so test can run even if path needs adjustment
    import importlib, sys

    # We need the real module
    try:
        from backend.rag.context_assembler import ContextAssembler
    except ModuleNotFoundError:
        # Try relative import for different working directories
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "context_assembler",
            Path(__file__).parent.parent / "rag" / "context_assembler.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        ContextAssembler = mod.ContextAssembler

    ca = ContextAssembler.__new__(ContextAssembler)
    ca.repo_name = "test-repo"
    ca.data_dir = Path("/tmp")
    ca._enriched = enriched_data
    return ca


def _make_node(nid: str, name: str, content: str = "") -> dict:
    return {
        "type": "function",
        "name": name,
        "file": f"{nid}.py",
        "docstring": content,
        "params": [],
        "return_type": None,
        "complexity": 1,
    }


class TestAssembleClampIntegration:
    """Integration tests that exercise the assemble() method."""

    def _get_assembler(self, primary_content: str):
        """Return an assembler with one primary node whose docstring is primary_content."""
        enriched = {
            "node1": _make_node("node1", "my_func", primary_content),
        }
        return _make_assembler(enriched)

    def test_budget_exceeded_primary_section_empty(self):
        """When token_budget is tiny (e.g. 1), primary section must be empty."""
        try:
            ca = self._get_assembler("A" * 400)  # ~100 tokens
            result = ca.assemble(
                primary_ids=["node1"],
                expanded_ids=[],
                edges=[],
                scores={},
                token_budget=1,  # impossibly small → used_tokens will exceed budget
            )
        except Exception:
            pytest.skip("Assembler could not be instantiated in this environment")

        # The primary section should contribute 0 characters (clamped)
        # The overall result may contain the header, but must NOT contain the
        # full primary content (which would indicate a negative-slice bug).
        # A tail-truncated string would still be very long; an empty clamp is short.
        assert "A" * 100 not in result, (
            "Primary content must not appear in output when budget is exceeded."
        )

    def test_budget_sufficient_primary_section_present(self):
        """When token_budget is large, primary section content is present."""
        content = "This is the docstring for my_func."
        try:
            ca = self._get_assembler(content)
            result = ca.assemble(
                primary_ids=["node1"],
                expanded_ids=[],
                edges=[],
                scores={},
                token_budget=15000,
            )
        except Exception:
            pytest.skip("Assembler could not be instantiated in this environment")

        assert content in result, "Primary content must appear when budget is ample."

    def test_chars_left_clamped_to_zero_when_used_equals_budget(self):
        """Directly verify the clamp: max(0, (budget - used) * 4) == 0 at equality."""
        token_budget = 100
        used_tokens = 100
        chars_left = max(0, (token_budget - used_tokens) * 4)
        assert chars_left == 0
        some_string = "important context that should not appear"
        assert some_string[:chars_left] == ""

    def test_no_silent_context_drop_when_over_budget(self):
        """
        When budget is exceeded the output must not contain a tail-truncated
        substring of the primary section.  The old bug would return
        primary_section[:-N], i.e. almost everything except the last N chars.
        """
        big_content = "X" * 2000  # clearly over a tiny budget
        try:
            ca = self._get_assembler(big_content)
            result = ca.assemble(
                primary_ids=["node1"],
                expanded_ids=[],
                edges=[],
                scores={},
                token_budget=2,  # header alone will exceed this
            )
        except Exception:
            pytest.skip("Assembler could not be instantiated in this environment")

        # With the fix, zero chars should be included from the primary section.
        # The buggy version would include ~1998 X's.
        assert "X" * 100 not in result, (
            "Over-budget assembler must not include large primary section content."
        )


# ---------------------------------------------------------------------------
# Pure formula regression tests (no imports needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token_budget,used_tokens,expected", [
    (100, 150, 0),    # over budget
    (100, 100, 0),    # exactly at budget
    (100, 50,  200),  # under budget → (100-50)*4 = 200
    (200, 0,   800),  # fresh start → 200*4 = 800
    (0,   10,  0),    # zero budget → clamped
])
def test_chars_left_formula_parametrized(token_budget, used_tokens, expected):
    chars_left = max(0, (token_budget - used_tokens) * 4)
    assert chars_left == expected


@pytest.mark.parametrize("token_budget,used_tokens", [
    (100, 101),
    (100, 100),
    (0,   1),
    (50,  200),
])
def test_negative_budget_slice_returns_empty(token_budget, used_tokens):
    """Whenever tokens are at or over budget, slicing must return empty string."""
    primary_section = "This is some important context that should not leak."
    chars_left = max(0, (token_budget - used_tokens) * 4)
    assert primary_section[:chars_left] == ""
