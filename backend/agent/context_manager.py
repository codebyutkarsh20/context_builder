"""
context_manager.py — Context window management for the ReAct agent loop.

Three-layer strategy to keep token usage under control:
  Layer 1: Per-tool output caps (at execution time, zero cost)
  Layer 2: Observation masking with sliding window (every iteration, zero cost)
  Layer 3: LLM summarization as safety net (triggered rarely, ~$0.002 via Haiku)

Research basis: SWE-Agent observation masking (arXiv 2508.21433) shows tool outputs
are ~84% of tokens in coding agents. Simple masking works as well as expensive
LLM summarization while avoiding trajectory elongation.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# Layer 2: Observation masking config
OBSERVATION_WINDOW = 10  # Keep last N tool results in full
MASK_TEMPLATE = "[Tool result from {tool_name}: {char_count} chars — masked to save context]"

# Layer 3: Summarization config
SUMMARIZATION_TRIGGER = 80_000  # Approximate tokens before triggering summarization
SUMMARIZATION_MODEL = "claude-haiku-4-5-20251001"
TOKEN_ESTIMATE_RATIO = 0.25  # Rough chars-to-tokens for code

# Layer 1: Per-tool output caps (chars)
TOOL_OUTPUT_CAPS = {
    "read_file": 6000,
    "grep_repo": 4000,
    "read_function": 4000,
    "list_files": 2000,
    "search_code": 4000,
    "get_function_info": 2000,
    "get_file_summary": 3000,
    "get_file_structure": 3000,
    "run_tests": 3000,
    "request_review": 2000,
    "string_replace": 500,
    "check_syntax": 500,
    "create_file": 500,
    "create_sandbox": 500,
    "record_localization": 500,
    "submit_fix": 500,
    "escalate": 500,
}
DEFAULT_CAP = 4000


def cap_tool_output(tool_name: str, output: str) -> str:
    """Layer 1: Cap tool output at the per-tool limit."""
    cap = TOOL_OUTPUT_CAPS.get(tool_name, DEFAULT_CAP)
    if len(output) <= cap:
        return output
    return output[:cap] + f"\n[... truncated, {len(output) - cap} more chars]"


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
            original_len = len(str(msg.content))
            placeholder = MASK_TEMPLATE.format(
                tool_name=tool_name,
                char_count=original_len,
            )
            new_messages.append(
                ToolMessage(content=placeholder, tool_call_id=msg.tool_call_id)
            )
        else:
            new_messages.append(msg)

    return new_messages


def maybe_summarize(messages: list) -> list:
    """Layer 3: If token count exceeds threshold, summarize older turns via Haiku.

    Returns the (possibly compressed) message list. Only triggers when
    observation masking alone isn't enough (very long runs).
    """
    token_count = count_tokens_approx(messages)

    if token_count < SUMMARIZATION_TRIGGER:
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
            return msg.tool_calls[0].get("name", "unknown")
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
