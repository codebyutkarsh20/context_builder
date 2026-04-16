"""
context_manager.py — Context window management for the ReAct agent loop.

Three-layer strategy to keep token usage under control:
  Layer 1: Per-tool output caps (at execution time, zero cost)
  Layer 2a: Observation masking with sliding window (legacy — rebuilds prefix)
  Layer 2b: Microcompact (cache-friendly, in-place, idempotent — preferred)
  Layer 3: LLM summarization as safety net (triggered rarely, ~$0.002 via Haiku)

Research basis: SWE-Agent observation masking (arXiv 2508.21433) shows tool outputs
are ~84% of tokens in coding agents. Simple masking works as well as expensive
LLM summarization while avoiding trajectory elongation.

Microcompact (Layer 2b) is ported from Claude Code's services/compact/microCompact.ts:
once a tool result is "old enough" it's replaced with a placeholder ONCE and
never modified again. The prefix stays stable across iterations, preserving
the Anthropic prompt cache (~87% cost savings on cached prefix).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Microcompact configuration (Layer 2b)
# ---------------------------------------------------------------------------

# Tools whose results can be safely replaced with a one-line placeholder once
# they're no longer in the recent window. These are exploration / lookup tools
# whose value is "did you read this file? what did you find?" — the agent doesn't
# need to re-read the full content from history once it's old.
#
# NOT compactable (kept in full forever): test results, edits, sandbox creation,
# review verdicts, BRT runs, plan production, submit_fix — these are critical
# state the agent and verifier must reference.
COMPACTABLE_TOOLS = frozenset({
    "read_file",
    "read_function",
    "grep_repo",
    "list_files",
    "get_file_structure",
    "get_function_info",
    "get_file_summary",
    "get_callers",
    # run_shell: diagnostic output (pip list, which pytest, ...) becomes
    # stale within turns. Agent doesn't need to re-read 10-turn-old `pip list`.
    "run_shell",
    # write_brt: BRT diagnostic output (test code + reasoning) is ephemeral.
    # The agent only needs the latest BRT results, not historical ones.
    "write_brt",
})

MICROCOMPACT_KEEP_RECENT = 12  # Keep last N tool results in full

# Layer 2: Observation masking config
# With 160K context, we can keep more turns in full than the default 10.
# Recent turns give the agent memory of what it already tried.
OBSERVATION_WINDOW = 15  # Keep last 15 tool results in full (not 10)
MASK_TEMPLATE = "[{tool_name}: {summary}]"

# Layer 3: Summarization config
# 160K context window → trigger summarization at ~120K tokens (~480K chars)
# This gives 40K headroom for the LLM response + safety margin.
SUMMARIZATION_TRIGGER = 120_000  # tokens (not 80K — we have 160K available)
SUMMARIZATION_MODEL = "claude-haiku-4-5-20251001"
# Fallback ratio when real token counts aren't available yet.
# Conservative: 1 token ≈ 2 chars for code (Anthropic tokenizer averages ~1.3 chars/token
# for English, higher for code). Old value 0.25 was 4-5x too low.
TOKEN_ESTIMATE_RATIO = 0.50

# Layer 1: Per-tool output caps — now sourced from tool_metadata.py registry.
# Kept as fallback constant only; actual caps come from ToolMeta.max_output_chars.
DEFAULT_CAP = 4000


def cap_tool_output(tool_name: str, output: str) -> tuple[str, dict]:
    """Layer 1: Cap tool output at the per-tool limit (from tool_metadata registry).

    Returns (capped_output, truncation_info) where truncation_info is a dict:
      - truncated: bool
      - original_chars: int (only if truncated)
      - kept_chars: int (only if truncated)
      - lost_chars: int (only if truncated)
    """
    from agent.tool_metadata import get_output_cap
    cap = get_output_cap(tool_name)
    if len(output) <= cap:
        return output, {"truncated": False}
    lost = len(output) - cap
    return (
        output[:cap] + f"\n[... truncated, {lost} more chars]",
        {
            "truncated": True,
            "original_chars": len(output),
            "kept_chars": cap,
            "lost_chars": lost,
        },
    )


def count_tokens_approx(messages: list) -> int:
    """Fast approximate token count without calling a tokenizer."""
    total_chars = sum(len(str(m.content)) for m in messages)
    return int(total_chars * TOKEN_ESTIMATE_RATIO)


def mask_old_observations(
    messages: list,
    window_size: int = OBSERVATION_WINDOW,
) -> list:
    """Layer 2: Replace tool results older than the sliding window with compact placeholders.

    Preserves:
      - SystemMessage and HumanMessage (always)
      - All AIMessage content (reasoning + tool call names, only ~16% of tokens)
      - Recent ToolMessage content (last `window_size` tool results)

    Masks:
      - ToolMessage content older than `window_size` tool results
    """
    if len(messages) < 4:
        return messages

    # Find all ToolMessage indices
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]

    if len(tool_indices) <= window_size:
        return messages  # Nothing to mask

    # Indices to mask (everything except the last `window_size`)
    mask_set = set(tool_indices[:-window_size])

    new_messages = []
    for i, msg in enumerate(messages):
        if i in mask_set:
            tool_name = _extract_tool_name(messages, i)
            summary = _extract_summary(str(msg.content))
            placeholder = MASK_TEMPLATE.format(tool_name=tool_name, summary=summary)
            new_messages.append(
                ToolMessage(content=placeholder, tool_call_id=msg.tool_call_id)
            )
        else:
            new_messages.append(msg)

    return new_messages


class MicrocompactState:
    """Per-run state tracking which tool results have been microcompacted.

    Keeping this state ensures the same `tool_call_id` is replaced with the
    SAME placeholder every iteration — so the message prefix stays byte-stable
    across LLM calls and the Anthropic prompt cache survives.
    """

    def __init__(self) -> None:
        # Map tool_call_id → placeholder content (pre-rendered once, reused)
        self.compacted: dict[str, str] = {}
        # Total tokens saved across all compactions in this run
        self.tokens_saved: int = 0
        # Number of tool results compacted
        self.count: int = 0

    def reset(self) -> None:
        self.compacted.clear()
        self.tokens_saved = 0
        self.count = 0


def microcompact_in_place(
    messages: list,
    state: MicrocompactState,
    keep_recent: int = MICROCOMPACT_KEEP_RECENT,
    compactable_tools: frozenset[str] = COMPACTABLE_TOOLS,
) -> list:
    """Layer 2b: Cache-friendly in-place tool result eviction.

    For each compactable tool result older than the recent window:
      - If already compacted (in state.compacted), reuse the existing placeholder
        — message stays byte-identical to previous iteration.
      - Otherwise, render a placeholder ONCE, store it in state, replace the
        message content.
    Test results, edits, reviews, plans, BRTs, submit_fix — never compacted
    (those are critical state that the agent + verifier must reference).

    Differences from `mask_old_observations`:
      - Stateful (idempotent): same input always produces the same prefix bytes
      - Compactable-tool whitelist: only exploration/lookup tools are eligible
      - Cache-preserving: the prompt prefix doesn't drift between iterations
        as the window slides, so prompt cache hits remain near 100%

    Args:
        messages: Current message list.
        state: Per-run MicrocompactState carrying the compacted-id map.
        keep_recent: Number of compactable tool results to keep in full at the tail.
        compactable_tools: Tool names eligible for compaction.

    Returns:
        New message list with old compactable tool results replaced by
        placeholders. Already-compacted messages are reused verbatim.
    """
    if len(messages) < 4:
        return messages

    # Find compactable tool message indices (only Read/Grep/etc., not test results)
    compactable_indices: list[int] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = _extract_tool_name(messages, i)
        if tool_name in compactable_tools:
            compactable_indices.append(i)

    if len(compactable_indices) <= keep_recent:
        return messages  # Nothing aged out yet

    # Anything beyond the last `keep_recent` compactable results is eligible
    eligible = compactable_indices[:-keep_recent]

    # Build new message list — reuse existing references where possible to avoid
    # extra copies (and to make the prefix-stability guarantee explicit).
    new_messages = list(messages)
    newly_compacted = 0
    for idx in eligible:
        msg = messages[idx]
        tool_call_id = getattr(msg, "tool_call_id", "")
        if not tool_call_id:
            continue

        # Reuse placeholder if we already compacted this id — message stays
        # byte-identical to the previous iteration's prefix.
        cached = state.compacted.get(tool_call_id)
        if cached is not None:
            # The message in `messages` should already BE the cached placeholder
            # (from the previous iteration). If it's not (e.g. message list was
            # reconstructed externally), restore the cached placeholder to keep
            # the prefix stable.
            if str(msg.content) != cached:
                new_messages[idx] = ToolMessage(content=cached, tool_call_id=tool_call_id)
            continue

        # First compaction for this tool_call_id — render placeholder once
        original = str(msg.content)
        tool_name = _extract_tool_name(messages, idx)
        summary = _extract_summary(original)
        placeholder = MASK_TEMPLATE.format(tool_name=tool_name, summary=summary)
        state.compacted[tool_call_id] = placeholder
        state.tokens_saved += int(max(0, len(original) - len(placeholder)) * TOKEN_ESTIMATE_RATIO)
        state.count += 1
        newly_compacted += 1
        new_messages[idx] = ToolMessage(content=placeholder, tool_call_id=tool_call_id)

    if newly_compacted:
        logger.info(
            "Microcompact: replaced %d tool results (%d total compacted, ~%d tokens saved)",
            newly_compacted, state.count, state.tokens_saved,
        )
    return new_messages


def _extract_summary(content: str) -> str:
    """Extract a 1-line summary (50-150 chars) from tool output for masked history.
    Preserves key info: file names, match counts, pass/fail status."""
    if not content:
        return "empty"
    first_line = content.split("\n")[0].strip()
    # For structured outputs, the first line is usually the summary
    if first_line.startswith("==="):
        # File read: "=== path (lines X-Y of Z) ==="
        return first_line[:120]
    if first_line.startswith("Found"):
        # Grep: "Found N matches:"
        return first_line[:80]
    if first_line.startswith("OK:"):
        return first_line[:80]
    if first_line.startswith("ERROR:"):
        return first_line[:80]
    if first_line.startswith("passed") or first_line.startswith("failed") or first_line.startswith("skipped") or first_line.startswith("error:"):
        return first_line[:80]
    # Default: first 100 chars
    return first_line[:100] + ("..." if len(first_line) > 100 else "")


def maybe_summarize(messages: list, *, force: bool = False, real_token_count: int = 0) -> list:
    """Layer 3: If token count exceeds threshold, summarize older turns via Haiku.

    Returns the (possibly compressed) message list. Only triggers when
    observation masking alone isn't enough (very long runs).

    Args:
        force: If True, skip the threshold check and always summarize.
               Used for recovery from prompt-too-long errors.
        real_token_count: Actual input_tokens from last API response. When > 0,
               used instead of the char-based estimate for accurate triggering.
    """
    # Prefer real token count from API when available
    token_count = real_token_count if real_token_count > 0 else count_tokens_approx(messages)

    if not force and token_count < SUMMARIZATION_TRIGGER:
        return messages

    logger.info("Context manager: triggering summarization (approx %d tokens)", token_count)

    try:
        from langchain_anthropic import ChatAnthropic

        # Extract fixed messages
        system_msg = messages[0]
        task_msg = messages[1]

        # Build conversation text for summarizer
        conv_text = _format_for_summary(messages[2:])

        # Call Haiku to summarize
        llm = ChatAnthropic(model=SUMMARIZATION_MODEL, max_tokens=2000, timeout=60.0)
        summary_response = llm.invoke([
            SystemMessage(content=_SUMMARIZATION_PROMPT),
            HumanMessage(content=conv_text[:50000]),
        ])
        summary_text = summary_response.content

        # Keep last 5 rounds verbatim
        recent = _extract_recent_rounds(messages, n_rounds=5)

        # Reconstruct
        new_messages = [
            system_msg,
            task_msg,
            HumanMessage(content=f"[CONTEXT SUMMARY — earlier exploration was compacted]\n\n{summary_text}"),
            AIMessage(content="Understood. Continuing from where I left off."),
        ] + recent

        new_count = count_tokens_approx(new_messages)
        logger.info("Context manager: summarized %d → %d tokens", token_count, new_count)
        return new_messages

    except Exception as e:
        logger.warning("Context manager: summarization failed (%s), continuing without", e)
        return messages


def _extract_tool_name(messages: list, tool_msg_index: int) -> str:
    """Walk backwards from a ToolMessage to find its tool name."""
    tool_msg = messages[tool_msg_index]
    tool_call_id = getattr(tool_msg, "tool_call_id", "")

    for j in range(tool_msg_index - 1, -1, -1):
        msg = messages[j]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id") == tool_call_id:
                    return tc.get("name", "unknown")
            # This AIMessage has no matching ID — keep searching backwards
            continue
    return "unknown"


def _format_for_summary(messages: list) -> str:
    """Format messages into text for the summarizer."""
    parts = []
    for m in messages:
        content = str(m.content)[:1000]
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_names = [tc.get("name", "?") for tc in m.tool_calls]
            parts.append(f"[Agent called: {', '.join(tool_names)}]\n{content}")
        elif isinstance(m, ToolMessage):
            parts.append(f"[Tool result]: {content}")
        else:
            role = type(m).__name__.replace("Message", "")
            parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def _extract_recent_rounds(messages: list, n_rounds: int = 5) -> list:
    """Extract the last n_rounds of AI+Tool message pairs."""
    recent: list = []
    rounds_found = 0
    i = len(messages) - 1

    while i >= 2 and rounds_found < n_rounds:
        if isinstance(messages[i], ToolMessage):
            tool_msgs = [messages[i]]
            i -= 1
            while i >= 2 and isinstance(messages[i], ToolMessage):
                tool_msgs.insert(0, messages[i])
                i -= 1
            if i >= 2 and isinstance(messages[i], AIMessage):
                recent = [messages[i]] + tool_msgs + recent
                rounds_found += 1
                i -= 1
            else:
                recent = tool_msgs + recent
                break
        else:
            i -= 1

    return recent


_SUMMARIZATION_PROMPT = """Summarize this AI agent's bug-fixing progress. Preserve ALL of:

1. The bug being fixed (title + root cause)
2. Which files and functions are buggy
3. What edits were made (file path + what changed)
4. Test results (pass/fail)
5. Review feedback (if any)
6. What the agent planned to do next

Output format:

## Bug
[one sentence]

## Root Cause
[one sentence]

## Files Investigated
- [file]: [what was found]

## Edits Made
- [file]: [what changed]

## Test Results
[pass/fail + details]

## Review Status
[status + feedback]

## Next Steps
[what to do next]
"""
