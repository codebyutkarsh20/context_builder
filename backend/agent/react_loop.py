"""
react_loop.py — Core ReAct while-loop for the AI Deploy Agent.

Single agent loop where the LLM decides: explore → localize → edit → test → review → submit.
Mirrors the exploration_node pattern (pipeline.py:798-894) but with ALL tools available.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.context_manager import cap_tool_output, count_tokens_approx, mask_old_observations, maybe_summarize
from agent.trace import _PHASE_MAP
from agent.react_guardrails import (
    GuardrailState,
    check_limits,
    check_tool_call,
    update_from_tool_result,
    MAX_TOOL_CALLS,
    MAX_WALL_TIME,
)
from agent.react_tools import (
    REACT_TOOLS,
    get_sandbox_path,
    get_branch_name,
    get_base_branch,
    get_localization,
)

if TYPE_CHECKING:
    from agent.trace import RunTrace
    from agent.types import ReactAgentState

logger = logging.getLogger(__name__)

# Model pricing for cost tracking (USD per 1M tokens)
_MODEL_PRICING = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

REACT_MODEL = "claude-sonnet-4-6"


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate cost accounting for Anthropic prompt caching.

    Cache writes cost 1.25x base input price.
    Cache reads cost 0.1x base input price (90% discount).
    """
    pricing = _MODEL_PRICING.get(model, (3.0, 15.0))
    input_cost = input_tokens * pricing[0]
    cache_write_cost = cache_creation_tokens * pricing[0] * 1.25
    cache_read_cost = cache_read_tokens * pricing[0] * 0.1
    output_cost = output_tokens * pricing[1]
    return round((input_cost + cache_write_cost + cache_read_cost + output_cost) / 1_000_000, 6)


def react_loop(
    state: ReactAgentState,
    system_prompt: str,
    task_message: str,
    explore_tools: list,
    trace: RunTrace | None = None,
) -> ReactAgentState:
    """Run the core ReAct loop.

    The agent is given ALL tools (explore + edit + sandbox + review + submit)
    and decides what to do. The loop continues until the agent calls a terminal
    tool (submit_fix, escalate) or hits a guardrail limit.

    Args:
        state: ReactAgentState to populate with results.
        system_prompt: Full system prompt with context.
        task_message: Initial user message.
        explore_tools: Read-only exploration tools from explore_tools.py.
        trace: Optional RunTrace for observability.

    Returns:
        Updated state dict.
    """
    all_tools = explore_tools + REACT_TOOLS
    tool_map = {t.name: t for t in all_tools}

    llm = ChatAnthropic(
        model=REACT_MODEL,
        max_tokens=8000,
        timeout=120.0,
    ).bind_tools(all_tools)

    # Enable Anthropic prompt caching on the system prompt.
    # The system prompt + tool definitions are identical across all 30+ LLM calls
    # in a single bug fix. Caching saves ~87% on input tokens for the static prefix.
    # Cache read: 0.1x base price. Write: 1.25x (amortized over 30 calls).
    messages: list = [
        SystemMessage(
            content=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        ),
        HumanMessage(content=task_message),
    ]

    gs = GuardrailState()
    current_phase = "explore"  # Track phase transitions

    if trace:
        trace.stage_start("react_loop")

    logger.info("=== REACT LOOP: Starting agent with %d tools (160K context) ===", len(all_tools))

    while gs.tool_call_count < MAX_TOOL_CALLS:
        # Check time limit
        if gs.elapsed >= MAX_WALL_TIME:
            logger.warning("ReAct loop hit time limit after %d tool calls", gs.tool_call_count)
            state["error"] = "Time limit reached"
            break

        # Check cost cap
        limit_error = check_limits(gs)
        if limit_error:
            logger.warning("ReAct loop hit limit: %s", limit_error[:100])
            break

        # Emit llm_request BEFORE calling LLM — captures what the model sees
        context_tokens = count_tokens_approx(messages)
        if trace:
            # Capture full message contents for replay/debugging
            messages_snapshot = []
            for m in messages:
                role = type(m).__name__
                content = m.content
                if isinstance(content, list):
                    # SystemMessage with cache_control blocks — extract text
                    text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    content = "\n".join(text_parts) if text_parts else str(content)
                elif not isinstance(content, str):
                    content = str(content)
                messages_snapshot.append({"role": role, "content": content})

            trace.emit("llm_request", "react_loop", {
                "model": REACT_MODEL,
                "message_count": len(messages),
                "context_tokens": context_tokens,
                "context_pct": round(context_tokens * 100 / 160_000, 1),
                "tool_call_count": gs.tool_call_count,
                "cost_usd_so_far": round(gs.cost_usd, 6),
                "phase": current_phase,
                "messages": messages_snapshot,
            })

        # Call the LLM
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("ReAct LLM call failed: %s", e)
            state["error"] = f"LLM call failed: {e}"
            break

        # Track token usage (including Anthropic prompt caching)
        usage = getattr(response, "response_metadata", {}).get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        call_cost = _estimate_cost(REACT_MODEL, input_tokens, output_tokens, cache_creation, cache_read)
        gs.cost_usd += call_cost

        if trace:
            # Capture full response content for replay/debugging
            response_text = ""
            if isinstance(response.content, str):
                response_text = response.content
            elif isinstance(response.content, list):
                text_blocks = [b.get("text", "") for b in response.content if isinstance(b, dict) and b.get("type") == "text"]
                response_text = "\n".join(text_blocks)

            response_tool_calls = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    response_tool_calls.append({
                        "name": tc["name"],
                        "args": tc["args"],
                        "id": tc["id"],
                    })

            trace.emit("llm_response", "react_loop", {
                "model": REACT_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_tokens": cache_creation,
                "cache_read_tokens": cache_read,
                "cost_usd": call_cost,
                "cumulative_cost_usd": gs.cost_usd,
                "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
                "tool_call_count": gs.tool_call_count,
                "context_tokens": context_tokens,
                "response_text": response_text,
                "response_tool_calls": response_tool_calls,
            })

        # Log cache hit on first cached call
        if cache_read > 0 and gs.tool_call_count <= 2:
            logger.info("Prompt cache HIT: %d tokens cached (saving ~%.0f%% on prefix)",
                        cache_read, 90.0)

        messages.append(response)

        # No tool calls — agent is done (shouldn't happen without submit/escalate)
        if not response.tool_calls:
            logger.info("ReAct agent stopped without terminal tool after %d calls", gs.tool_call_count)
            if isinstance(response.content, str) and response.content:
                state["explanation"] = response.content[:500]
            break

        # Capture agent reasoning (text content before tool calls) — full text, no truncation
        agent_reasoning = ""
        if isinstance(response.content, str):
            agent_reasoning = response.content
        elif isinstance(response.content, list):
            text_blocks = [b.get("text", "") for b in response.content if isinstance(b, dict) and b.get("type") == "text"]
            agent_reasoning = " ".join(text_blocks)

        # Execute each tool call
        terminal = False
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            # Detect phase transition
            new_phase = _PHASE_MAP.get(tool_name, current_phase)
            if new_phase != current_phase:
                if trace:
                    trace.emit("state_transition", "react_loop", {
                        "from_phase": current_phase,
                        "to_phase": new_phase,
                        "trigger_tool": tool_name,
                        "at_call": gs.tool_call_count + 1,
                        "cost_usd_at_transition": round(gs.cost_usd, 6),
                    })
                current_phase = new_phase

            logger.info("ReAct tool call %d: %s(%s)",
                        gs.tool_call_count + 1, tool_name, str(tool_args)[:100])

            if trace:
                trace.emit("tool_call", "react_loop", {
                    "tool_name": tool_name,
                    "args": tool_args,
                    "call_number": gs.tool_call_count + 1,
                    "phase": current_phase,
                    "reasoning": agent_reasoning,
                })

            # Guardrail check BEFORE execution
            guardrail_error = check_tool_call(tool_name, tool_args, gs)

            if guardrail_error and guardrail_error.startswith("WARNING:"):
                # Warnings are advisory — let the tool proceed but inform the agent
                logger.info("Guardrail warning for %s: %s", tool_name, guardrail_error[:100])
                if trace:
                    trace.emit("guardrail_event", "react_loop", {
                        "tool_name": tool_name,
                        "action": "warn",
                        "message": guardrail_error[:200],
                        "call_number": gs.tool_call_count + 1,
                    })
                guardrail_error = None

            if guardrail_error:
                result_str = guardrail_error
                logger.info("Guardrail blocked %s: %s", tool_name, guardrail_error[:100])
                if trace:
                    trace.emit("guardrail_event", "react_loop", {
                        "tool_name": tool_name,
                        "action": "block",
                        "message": guardrail_error[:200],
                        "call_number": gs.tool_call_count + 1,
                    })
            else:
                # Execute the tool
                t = tool_map.get(tool_name)
                if t is None:
                    result_str = f"ERROR: Tool '{tool_name}' not found"
                else:
                    tool_t0 = time.monotonic()
                    try:
                        result_str = t.invoke(tool_args)
                    except Exception as te:
                        result_str = f"ERROR: Tool execution failed: {te}"
                    tool_duration = round((time.monotonic() - tool_t0) * 1000)
                    # Layer 1: Cap tool output size
                    result_str = cap_tool_output(tool_name, str(result_str))

                    if trace:
                        trace.emit("tool_result", "react_loop", {
                            "tool_name": tool_name,
                            "duration_ms": tool_duration,
                            "result_preview": str(result_str),
                            "phase": current_phase,
                        })

            # Update guardrail state
            update_from_tool_result(tool_name, tool_args, str(result_str), gs)

            # Check for terminal tools — only mark as terminal if NOT blocked
            # AND the tool actually succeeded (returned "OK:", not "ERROR:")
            if tool_name == "submit_fix" and guardrail_error is None:
                result_text = str(result_str)
                if result_text.startswith("OK:"):
                    state["submitted"] = True
                    state["explanation"] = tool_args.get("explanation", "")
                    terminal = True
                else:
                    logger.warning("submit_fix returned error: %s", result_text[:200])
                messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))
                if terminal:
                    break

            if tool_name == "escalate" and guardrail_error is None:
                state["escalated"] = True
                state["escalate_reason"] = tool_args.get("reason", "")
                terminal = True
                messages.append(ToolMessage(content=str(result_str), tool_call_id=tool_id))
                break

            # Update state from tool results
            if tool_name == "create_sandbox" and "OK:" in str(result_str):
                sandbox_path = get_sandbox_path()
                state["sandbox_path"] = str(sandbox_path or "")
                state["branch_name"] = get_branch_name()
                state["base_branch"] = get_base_branch()
                if sandbox_path:
                    from agent.explore_tools import set_context
                    repo_name = state.get("work_order", {}).get("repo_name", "")
                    set_context(repo_name, str(sandbox_path), None)

            if tool_name == "record_localization" and "OK:" in str(result_str):
                loc = get_localization()
                if loc:
                    state["localization"] = loc

            if tool_name == "run_tests":
                state["test_result"] = str(result_str)[:3000]

            if tool_name == "request_review":
                result_text = str(result_str)
                review_dict = {"verdict": "UNKNOWN", "confidence": 0.0}
                if "APPROVE" in result_text:
                    review_dict["verdict"] = "APPROVE"
                elif "CHANGES_REQUESTED" in result_text:
                    review_dict["verdict"] = "CHANGES_REQUESTED"
                elif "ESCALATE" in result_text:
                    review_dict["verdict"] = "ESCALATE"
                state["review"] = review_dict

            messages.append(ToolMessage(content=str(result_str), tool_call_id=tool_id))

        if terminal:
            break

        # Context window management (Layers 2 + 3)
        pre_mask_tokens = count_tokens_approx(messages)
        messages = mask_old_observations(messages)
        post_mask_tokens = count_tokens_approx(messages)
        if pre_mask_tokens != post_mask_tokens and trace:
            trace.emit("context_compaction", "react_loop", {
                "action": "observation_masking",
                "tokens_before": pre_mask_tokens,
                "tokens_after": post_mask_tokens,
                "tokens_saved": pre_mask_tokens - post_mask_tokens,
                "at_call": gs.tool_call_count,
            })

        if gs.tool_call_count > 0 and gs.tool_call_count % 20 == 0:
            pre_summarize = count_tokens_approx(messages)
            messages = maybe_summarize(messages)
            post_summarize = count_tokens_approx(messages)
            if pre_summarize != post_summarize and trace:
                trace.emit("context_compaction", "react_loop", {
                    "action": "summarization",
                    "tokens_before": pre_summarize,
                    "tokens_after": post_summarize,
                    "tokens_saved": pre_summarize - post_summarize,
                    "at_call": gs.tool_call_count,
                })

    # Finalize state
    state["tool_call_count"] = gs.tool_call_count
    state["cost_usd"] = round(gs.cost_usd, 6)
    state["messages"] = [
        {"role": type(m).__name__, "content": str(m.content)}
        for m in messages[-20:]
    ]

    if not state.get("submitted") and not state.get("escalated"):
        state["escalated"] = True
        state["escalate_reason"] = state.get("error", "Agent stopped without submitting or escalating")

    # Determine outcome
    if state.get("submitted"):
        outcome = "submitted"
    elif state.get("escalated"):
        outcome = "escalated"
    else:
        outcome = "timeout"

    logger.info(
        "ReAct loop done: %d tool calls, $%.4f cost, %.0fs elapsed, outcome=%s",
        gs.tool_call_count, gs.cost_usd, gs.elapsed, outcome,
    )

    # Emit run_outcome with full summary
    if trace:
        trace.emit("run_outcome", "react_loop", {
            "outcome": outcome,
            "tool_call_count": gs.tool_call_count,
            "cost_usd": round(gs.cost_usd, 6),
            "elapsed_seconds": round(gs.elapsed, 1),
            "final_phase": current_phase,
            "submitted": state.get("submitted", False),
            "escalated": state.get("escalated", False),
            "escalate_reason": state.get("escalate_reason", ""),
            "localization_found": bool(state.get("localization")),
            "sandbox_created": gs.sandbox_created,
            "tests_attempted": gs.tests_attempted,
            "tests_passed": gs.tests_passed,
            "tests_skipped": gs.tests_skipped,
            "review_approved": gs.review_approved,
            "review_verdict": gs.review_verdict,
            "test_failure_count": gs.test_failure_count,
            "guardrail_stats": {
                "grep_count": gs.grep_count,
                "read_file_count": gs.read_file_count,
                "run_tests_count": gs.run_tests_count,
                "string_replace_count": gs.string_replace_count,
                "review_count": gs.review_count,
            },
        })
        trace.stage_end("react_loop")

    return state
