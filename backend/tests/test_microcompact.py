"""
test_microcompact.py — Tests for cache-friendly in-place microcompact.

Microcompact is a stateful, idempotent eviction strategy ported from Claude
Code's services/compact/microCompact.ts. The key correctness property is
**prefix stability**: once a tool result is replaced with a placeholder, the
placeholder must stay byte-identical across all subsequent iterations so the
Anthropic prompt cache survives.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.context_manager import (
    COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MicrocompactState,
    microcompact_in_place,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_round(tool_name: str, tool_call_id: str, content: str) -> list:
    """Build an AIMessage(tool_call) + ToolMessage(result) pair."""
    ai = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": tool_name, "args": {}}],
    )
    tm = ToolMessage(content=content, tool_call_id=tool_call_id)
    return [ai, tm]


def _build_conversation(rounds: list[tuple[str, str, str]]) -> list:
    """Build a conversation: system + task + N tool rounds."""
    msgs: list = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="task: fix the bug"),
    ]
    for tool_name, tcid, content in rounds:
        msgs.extend(_make_round(tool_name, tcid, content))
    return msgs


# ---------------------------------------------------------------------------
# Compactable / non-compactable tool whitelist
# ---------------------------------------------------------------------------

class TestCompactableTools:
    def test_read_file_is_compactable(self):
        assert "read_file" in COMPACTABLE_TOOLS

    def test_grep_repo_is_compactable(self):
        assert "grep_repo" in COMPACTABLE_TOOLS

    def test_run_tests_NOT_compactable(self):
        """Test results are critical state — must never be evicted."""
        assert "run_tests" not in COMPACTABLE_TOOLS

    def test_string_replace_NOT_compactable(self):
        """Edit results are critical state — must never be evicted."""
        assert "string_replace" not in COMPACTABLE_TOOLS

    def test_create_sandbox_NOT_compactable(self):
        assert "create_sandbox" not in COMPACTABLE_TOOLS

    def test_request_review_NOT_compactable(self):
        assert "request_review" not in COMPACTABLE_TOOLS

    def test_run_brt_NOT_compactable(self):
        assert "run_brt" not in COMPACTABLE_TOOLS


# ---------------------------------------------------------------------------
# Basic compaction behavior
# ---------------------------------------------------------------------------

class TestBasicCompaction:
    def test_no_compaction_below_keep_recent(self):
        rounds = [("read_file", f"tc{i}", f"file {i} content " * 100) for i in range(5)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state, keep_recent=12)
        assert state.count == 0
        # Messages identical (or at least equal-content) to input
        assert len(result) == len(msgs)
        for r, m in zip(result, msgs):
            assert str(r.content) == str(m.content)

    def test_oldest_compactable_replaced_when_aged_out(self):
        # 14 read_file rounds, keep_recent=10 → 4 should be evicted
        rounds = [("read_file", f"tc{i}", f"file {i} content " * 200) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state, keep_recent=10)

        # 4 oldest tool results compacted
        assert state.count == 4
        # First 4 ToolMessages should be placeholder strings
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 14
        for tm in tool_msgs[:4]:
            # Placeholder format: [tool_name: summary]
            assert str(tm.content).startswith("[read_file:")
            assert len(str(tm.content)) < 200  # Shorter than original
        # Last 10 retain their full content
        for tm in tool_msgs[4:]:
            assert "file" in str(tm.content)
            assert len(str(tm.content)) > 500  # Original size

    def test_non_compactable_tools_never_replaced(self):
        # Mix: 8 read_file + 8 run_tests + 8 read_file
        rounds = (
            [("read_file", f"r{i}", "x" * 1000) for i in range(8)]
            + [("run_tests", f"t{i}", "passed: 5 tests" * 50) for i in range(8)]
            + [("read_file", f"r2{i}", "y" * 1000) for i in range(8)]
        )
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state, keep_recent=10)

        # Of the 16 read_file, 6 should be evicted (16 compactable - 10 kept = 6)
        # All 8 run_tests should be untouched
        run_test_msgs = [
            m for m in result
            if isinstance(m, ToolMessage) and "passed" in str(m.content)
        ]
        # Each run_tests result preserved (no placeholder format)
        for tm in run_test_msgs:
            assert not str(tm.content).startswith("[")
            assert "passed: 5 tests" in str(tm.content)

    def test_keeps_system_and_human_messages(self):
        rounds = [("read_file", f"tc{i}", "x" * 1000) for i in range(15)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state, keep_recent=10)

        assert isinstance(result[0], SystemMessage)
        assert str(result[0].content) == "system prompt"
        assert isinstance(result[1], HumanMessage)
        assert str(result[1].content) == "task: fix the bug"


# ---------------------------------------------------------------------------
# CRITICAL: prefix stability across iterations
# ---------------------------------------------------------------------------

class TestPrefixStability:
    """The defining property of microcompact: same input prefix → same output
    prefix bytes, every iteration. Without this, the Anthropic prompt cache
    invalidates on every LLM call.
    """

    def test_second_call_produces_byte_identical_prefix(self):
        rounds = [("read_file", f"tc{i}", "x" * 1000) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        first = microcompact_in_place(msgs, state, keep_recent=10)

        # Simulate the loop: pass `first` back in (same messages, same state)
        second = microcompact_in_place(first, state, keep_recent=10)

        # Every message content must be identical — this is what gives us cache hits
        assert len(first) == len(second)
        for f, s in zip(first, second):
            assert str(f.content) == str(s.content)
            assert type(f) is type(s)

    def test_appending_new_round_keeps_already_compacted_stable(self):
        """When a new round is appended, the messages that were ALREADY
        compacted in the previous iteration must stay byte-identical (so the
        prompt cache prefix up to that point survives). New compactions may
        happen for the next-oldest message, but anything ALREADY compacted
        stays put.
        """
        rounds = [("read_file", f"tc{i}", "x" * 1000) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        first = microcompact_in_place(msgs, state, keep_recent=10)
        compacted_ids_after_first = set(state.compacted.keys())

        # Agent does another tool call → new round appended
        new_round = _make_round("read_file", "tc14", "y" * 1000)
        appended = list(first) + new_round

        second = microcompact_in_place(appended, state, keep_recent=10)

        # All previously-compacted IDs must still map to the SAME placeholder
        for tcid in compacted_ids_after_first:
            assert state.compacted[tcid] == \
                _find_placeholder_for_id(first, tcid), \
                f"Placeholder for {tcid} mutated between iterations"
            # And the message at the corresponding position in `second` must
            # still equal that placeholder
            second_msg = _find_msg_for_id(second, tcid)
            assert str(second_msg.content) == state.compacted[tcid]


def _find_placeholder_for_id(messages, tcid):
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id == tcid:
            return str(m.content)
    return None


def _find_msg_for_id(messages, tcid):
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id == tcid:
            return m
    return None

    def test_state_counter_does_not_double_count_revisits(self):
        rounds = [("read_file", f"tc{i}", "x" * 1000) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        microcompact_in_place(msgs, state, keep_recent=10)
        first_count = state.count

        # Run again with same messages
        microcompact_in_place(msgs, state, keep_recent=10)
        # Counter must NOT increase — those tool_call_ids were already compacted
        assert state.count == first_count

    def test_placeholder_restored_if_externally_modified(self):
        """Defensive: if the message list is rebuilt from scratch, we restore
        the cached placeholder so the prefix stays stable.
        """
        rounds = [("read_file", f"tc{i}", "x" * 1000) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        first = microcompact_in_place(msgs, state, keep_recent=10)

        # Externally rebuild the list with original (uncompacted) content for the
        # FIRST tool message — simulates a buggy caller
        rebuilt = list(msgs)  # fresh originals
        # Append the same recent tail from `first`
        # Actually just pass the originals — microcompact should restore placeholders
        second = microcompact_in_place(rebuilt, state, keep_recent=10)

        # Old indices should be the cached placeholders, not the original content
        compactable_idx = [i for i, m in enumerate(rebuilt) if isinstance(m, ToolMessage)]
        evicted = compactable_idx[:-10]  # 4 oldest
        for idx in evicted:
            tcid = rebuilt[idx].tool_call_id
            assert str(second[idx].content) == state.compacted[tcid]


# ---------------------------------------------------------------------------
# Token savings tracking
# ---------------------------------------------------------------------------

class TestTokensSaved:
    def test_tokens_saved_increases_after_first_compaction(self):
        rounds = [("read_file", f"tc{i}", "long content " * 500) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        microcompact_in_place(msgs, state, keep_recent=10)
        assert state.tokens_saved > 0

    def test_tokens_saved_stable_on_revisit(self):
        rounds = [("read_file", f"tc{i}", "long content " * 500) for i in range(14)]
        msgs = _build_conversation(rounds)
        state = MicrocompactState()
        microcompact_in_place(msgs, state, keep_recent=10)
        first_savings = state.tokens_saved
        microcompact_in_place(msgs, state, keep_recent=10)
        # Same messages, no new compactions → savings stays put
        assert state.tokens_saved == first_savings


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_messages(self):
        state = MicrocompactState()
        result = microcompact_in_place([], state)
        assert result == []
        assert state.count == 0

    def test_only_system_message(self):
        msgs = [SystemMessage(content="x")]
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state)
        assert result == msgs

    def test_tool_message_without_tool_call_id_skipped(self):
        msgs = [
            SystemMessage(content="s"),
            HumanMessage(content="h"),
            AIMessage(content="", tool_calls=[{"id": "tc0", "name": "read_file", "args": {}}]),
            ToolMessage(content="x" * 500, tool_call_id=""),  # missing id
        ]
        # Add 14 more compactable rounds so we cross the threshold
        for i in range(1, 14):
            msgs.extend(_make_round("read_file", f"tc{i}", "y" * 500))
        state = MicrocompactState()
        result = microcompact_in_place(msgs, state, keep_recent=10)
        # No crash; the empty-id tool message is skipped (still original content)
        # Find the empty-id message
        empty_msg = next((m for m in result if isinstance(m, ToolMessage) and m.tool_call_id == ""), None)
        assert empty_msg is not None
        # Its content should be unchanged (compaction skipped)
        assert "x" * 500 in str(empty_msg.content)

    def test_state_reset(self):
        state = MicrocompactState()
        state.compacted["tc1"] = "[read_file: foo]"
        state.tokens_saved = 1000
        state.count = 5
        state.reset()
        assert state.compacted == {}
        assert state.tokens_saved == 0
        assert state.count == 0
