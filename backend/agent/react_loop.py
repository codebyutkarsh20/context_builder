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

from agent.context_manager import cap_tool_output, mask_old_observations, maybe_summarize
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


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _MODEL_PRICING.get(model, (3.0, 15.0))
    return round((input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000, 6)


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

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_message),
    ]

    gs = GuardrailState()

    if trace:
        trace.stage_start("react_loop")

    logger.info("=== REACT LOOP: Starting agent with %d tools ===", len(all_tools))

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

        # Call the LLM
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("ReAct LLM call failed: %s", e)
            state["error"] = f"LLM call failed: {e}"
            break

        # Track token usage
        usage = getattr(response, "response_metadata", {}).get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        gs.cost_usd += _estimate_cost(REACT_MODEL, input_tokens, output_tokens)

        if trace:
            trace.emit("llm_response", "react_loop", {
                "model": REACT_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": _estimate_cost(REACT_MODEL, input_tokens, output_tokens),
                "cumulative_cost_usd": gs.cost_usd,
                "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
                "tool_call_count": gs.tool_call_count,
            })

        messages.append(response)

        # No tool calls — agent is done (shouldn't happen without submit/escalate)
        if not response.tool_calls:
            logger.info("ReAct agent stopped without terminal tool after %d calls", gs.tool_call_count)
            # Extract any final text as explanation
            if isinstance(response.content, str) and response.content:
                state["explanation"] = response.content[:500]
            break

        # Execute each tool call
        terminal = False
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            logger.info("ReAct tool call %d: %s(%s)",
                        gs.tool_call_count + 1, tool_name, str(tool_args)[:100])

            if trace:
                trace.emit("tool_call", "react_loop", {
                    "tool_name": tool_name,
                    "args": tool_args,
                    "call_number": gs.tool_call_count + 1,
                })

            # Guardrail check BEFORE execution
            guardrail_error = check_tool_call(tool_name, tool_args, gs)

            if guardrail_error and guardrail_error.startswith("WARNING:"):
                # Warnings are advisory — let the tool proceed but inform the agent
                logger.info("Guardrail warning for %s: %s", tool_name, guardrail_error[:100])
                # Still execute the tool
                guardrail_error = None

            if guardrail_error:
                result_str = guardrail_error
                logger.info("Guardrail blocked %s: %s", tool_name, guardrail_error[:100])
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
                            "result_preview": str(result_str)[:500],
                        })

            # Update guardrail state
            update_from_tool_result(tool_name, tool_args, str(result_str), gs)

            # Check for terminal tools — only mark as terminal if NOT blocked by guardrail
            if tool_name == "submit_fix" and guardrail_error is None:
                state["submitted"] = True
                state["explanation"] = tool_args.get("explanation", "")
                terminal = True
                messages.append(ToolMessage(content=str(result_str), tool_call_id=tool_id))
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
                # Redirect explore tools to read from sandbox instead of original repo
                # This ensures read_file, grep_repo etc. see the agent's edits
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
                # Parse review result for state
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
        messages = mask_old_observations(messages)
        if gs.tool_call_count > 0 and gs.tool_call_count % 20 == 0:
            messages = maybe_summarize(messages)

    # Finalize state
    state["tool_call_count"] = gs.tool_call_count
    state["cost_usd"] = round(gs.cost_usd, 6)
    state["messages"] = [
        {"role": type(m).__name__, "content": str(m.content)[:500]}
        for m in messages[-20:]  # Keep last 20 messages for debugging
    ]

    # If neither submitted nor escalated, force escalation
    if not state.get("submitted") and not state.get("escalated"):
        state["escalated"] = True
        state["escalate_reason"] = state.get("error", "Agent stopped without submitting or escalating")

    logger.info(
        "ReAct loop done: %d tool calls, $%.4f cost, %.0fs elapsed, submitted=%s, escalated=%s",
        gs.tool_call_count, gs.cost_usd, gs.elapsed,
        state.get("submitted"), state.get("escalated"),
    )

    if trace:
        trace.stage_end("react_loop")

    return state
