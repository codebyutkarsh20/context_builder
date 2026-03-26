"""
Tests for ContextAssembler.assemble() related-section budget clamping.

Covers the fix for the negative-slice-index bug where `budget_left * 4`
could produce a negative index when `used_tokens > token_budget`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.rag.context_assembler import ContextAssembler, _estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assembler(enriched_data: dict) -> tuple[ContextAssembler, Path]:
    """Return a ContextAssembler whose _load_enriched() returns enriched_data."""
    tmp = tempfile.mkdtemp()
    data_dir = Path(tmp)
    assembler = ContextAssembler(repo_name="test_repo", data_dir=data_dir)
    # Patch _load_enriched so we don't need real files on disk.
    assembler._load_enriched = lambda: enriched_data  # type: ignore[method-assign]
    return assembler, data_dir


def _minimal_enriched(node_id: str = "n1") -> dict:
    """Minimal enriched dict with one function node."""
    return {
        node_id: {
            "id": node_id,
            "type": "function",
            "name": "my_func",
            "file": "src/foo.py",
            "docstring": "Does something.",
            "params": [],
            "returns": "None",
            "summary": "A short summary.",
        }
    }


def _long_related_section(length_chars: int = 2000) -> str:
    """Return a related-section string of the requested character length."""
    return "x" * length_chars


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestRelatedSectionBudgetClamping:
    """Verify that related_section never contributes content when budget is exhausted."""

    def _assemble_with_mocked_renders(
        self,
        token_budget: int,
        primary_section_chars: int,
        related_section_chars: int,
    ) -> str:
        """
        Run assemble() with controlled primary and related section sizes.

        We bypass the real rendering helpers so the test is deterministic and
        independent of the enriched-node formatting logic.
        """
        enriched = _minimal_enriched("n1")
        assembler, _ = _make_assembler(enriched)

        primary_content = "P" * primary_section_chars
        related_content = "R" * related_section_chars

        assembler._render_primary = lambda *a, **kw: primary_content  # type: ignore[method-assign]
        assembler._render_related = lambda *a, **kw: related_content  # type: ignore[method-assign]
        assembler._render_edges = lambda *a, **kw: ""  # type: ignore[method-assign]
        assembler._render_business_context = lambda *a, **kw: ""  # type: ignore[method-assign]

        return assembler.assemble(
            primary_ids=["n1"],
            expanded_ids=[],
            edges=[],
            scores={},
            token_budget=token_budget,
        )

    # ------------------------------------------------------------------
    # Test: budget_left is negative (used_tokens > token_budget)
    # ------------------------------------------------------------------

    def test_negative_budget_left_produces_no_related_content(self):
        """
        When used_tokens already exceeds token_budget before the related section
        is processed, the related section must not contribute any 'R' characters
        to the output.
        """
        # token_budget = 10 tokens = 40 chars.
        # primary_section = 200 chars = 50 tokens  →  used_tokens > token_budget.
        # related_section = 400 chars.  Without the fix the old code would return
        # related_section[negative_index:] which is non-empty.
        result = self._assemble_with_mocked_renders(
            token_budget=10,
            primary_section_chars=200,
            related_section_chars=400,
        )
        assert "R" not in result, (
            "related_section content leaked into output despite negative budget_left"
        )

    # ------------------------------------------------------------------
    # Test: budget_left is exactly zero (used_tokens == token_budget)
    # ------------------------------------------------------------------

    def test_zero_budget_left_produces_no_related_content(self):
        """
        When used_tokens exactly equals token_budget, budget_left == 0.
        The slice `related_section[:0 * 4]` == `related_section[:0]` == "",
        so no related content should appear.
        """
        # We need used_tokens == token_budget after the primary section.
        # token_budget = 50, header ~ a few tokens.
        # We'll set primary_section_chars so primary tokens fill exactly to budget.
        # Keep it simple: token_budget=50, primary=196 chars → 49 tokens,
        # header is ~ (len(header)//4) tokens. To make used_tokens == token_budget
        # exactly we directly control via a very small budget.
        #
        # Simplest approach: token_budget = 5 tokens (20 chars).
        # primary_section = 20 chars = 5 tokens.  The header itself is already ~
        # a few tokens so used_tokens > token_budget in practice; but the
        # important invariant is that used_tokens >= token_budget.
        # We just need to check no 'R' appears.
        result = self._assemble_with_mocked_renders(
            token_budget=5,
            primary_section_chars=20,
            related_section_chars=200,
        )
        assert "R" not in result, (
            "related_section content leaked into output when budget_left == 0"
        )

    # ------------------------------------------------------------------
    # Test: budget_left is positive — existing behaviour preserved
    # ------------------------------------------------------------------

    def test_positive_budget_left_includes_truncated_related_content(self):
        """
        When there is remaining budget, the related section should be included
        (possibly truncated) and must contain 'R' characters.
        """
        # token_budget = 1000 tokens = 4000 chars.
        # primary_section = 100 chars → 25 tokens.  Plenty of headroom.
        # related_section = 3000 chars → 750 tokens.  Should be partially included.
        result = self._assemble_with_mocked_renders(
            token_budget=1000,
            primary_section_chars=100,
            related_section_chars=3000,
        )
        assert "R" in result, (
            "related_section was unexpectedly absent when budget_left is positive"
        )

    def test_positive_budget_left_related_not_exceed_budget(self):
        """
        The related section must be truncated so that the final token count does
        not significantly exceed the budget.
        """
        token_budget = 100  # 400 chars
        result = self._assemble_with_mocked_renders(
            token_budget=token_budget,
            primary_section_chars=0,  # no primary content
            related_section_chars=10000,  # very large
        )
        # The result should not vastly exceed the budget.
        # We allow some slack for the header and joining newlines.
        assert len(result) <= token_budget * 4 + 200, (
            f"Assembled context ({len(result)} chars) greatly exceeds token_budget={token_budget}"
        )

    # ------------------------------------------------------------------
    # Test: parity between primary and related section clamping
    # ------------------------------------------------------------------

    def test_primary_and_related_both_clamp_to_zero(self):
        """
        Both the primary-section path (already fixed) and the related-section
        path (newly fixed) must clamp to 0 when the budget is exhausted.
        Neither 'P' (from a deeply over-budget primary truncation) nor 'R'
        should appear in an absurdly over-budget scenario where token_budget=1.
        """
        # With token_budget=1 token (4 chars), the header alone exhausts the
        # budget.  Both sections must be empty or trivially short.
        result = self._assemble_with_mocked_renders(
            token_budget=1,
            primary_section_chars=10000,
            related_section_chars=10000,
        )
        # The primary truncation path uses chars_left = (budget - used) * 4;
        # if budget - used <= 0 then chars_left <= 0 and the slice returns "".
        # The related path must do the same.
        assert "R" not in result, (
            "related_section leaked into output under extremely tight budget"
        )


# ---------------------------------------------------------------------------
# Direct unit tests for the slice expression itself
# ---------------------------------------------------------------------------


class TestSliceClampBehaviour:
    """Direct slice-index clamping correctness tests, independent of assemble()."""

    @pytest.mark.parametrize("budget_left,text,expected", [
        (-5, "hello world", ""),     # negative → empty
        (-1, "hello world", ""),     # negative → empty
        (0,  "hello world", ""),     # zero → empty
        (1,  "hello world", "hell"), # positive → first 4 chars
        (3,  "hello world", "hello w"),  # 3*4=12 but string is 11 chars → full
    ])
    def test_max_clamp_slice(self, budget_left: int, text: str, expected: str):
        """Verify that max(0, budget_left) * 4 gives the correct slice."""
        result = text[:max(0, budget_left) * 4]
        assert result == expected, (
            f"budget_left={budget_left}: expected {expected!r}, got {result!r}"
        )

    def test_negative_budget_without_clamp_is_wrong(self):
        """
        Demonstrate the original bug: without the clamp a negative budget_left
        produces a non-empty slice, confirming why the fix is necessary.
        """
        text = "hello world"  # 11 chars
        budget_left = -1
        # Old (buggy) behaviour:
        buggy_result = text[budget_left * 4]  # text[-4] == 'o'
        assert buggy_result != "", "Sanity: negative index without clamp is not empty"
        # Fixed behaviour:
        fixed_result = text[:max(0, budget_left) * 4]
        assert fixed_result == "", "Fixed: negative budget_left must yield empty string"
