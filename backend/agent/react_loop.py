"""
react_loop.py — Core ReAct while-loop for the AI Deploy Agent.

Single agent loop where the LLM decides: explore → localize → edit → test → review → submit.
Mirrors the exploration_node pattern (pipeline.py:798-894) but with ALL tools available.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.context_manager import (
    MicrocompactState,
    cap_tool_output,
    count_tokens_approx,
    mask_old_observations,
    maybe_summarize,
    microcompact_in_place,
)
from agent.tool_metadata import is_concurrent_safe as _is_concurrent_safe
from agent.trace import _PHASE_MAP
from agent.react_guardrails import (
    GuardrailState,
    budget_for_difficulty,
    check_limits,
    check_tool_call,
    update_from_tool_result,
    MAX_TOOL_CALLS,
    MAX_WALL_TIME,
)

MAX_TOOL_CONCURRENCY = 6
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
    static_block: str,
    dynamic_block: str,
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
        static_block: Cacheable system prompt prefix (workflow, tools, rules, strategy).
            Identical across all bugs of the same fix_type. cache_control applied here.
        dynamic_block: Non-cached system prompt suffix (repo, ticket, intent, code map, BRTs).
            Changes every run — not worth caching.
        task_message: Initial user message.
        explore_tools: Read-only exploration tools from explore_tools.py.
        trace: Optional RunTrace for observability.

    Returns:
        Updated state dict.
    """
    all_tools = explore_tools + REACT_TOOLS
    tool_map = {t.name: t for t in all_tools}

    # Capture thread-local context from the CURRENT (main) thread so we can
    # propagate it into ThreadPoolExecutor worker threads.
    # threading.local() values are NOT inherited by new threads — each worker
    # starts with a blank TLS, causing "repo path not set" errors in explore_tools.
    # We store the context in a mutable dict so the sandbox-creation handler
    # can update the repo_path in place (switching from original → sandbox path)
    # and all subsequent worker threads automatically use the new value.
    from agent.explore_tools import _tls as _explore_tls, set_context as _explore_set_ctx
    from agent.react_tools import _tls as _react_tls_local, set_react_context as _react_set_ctx
    _thread_ctx: dict = {
        "repo_name": getattr(_explore_tls, "repo_name", ""),
        "repo_path": getattr(_explore_tls, "repo_path", None),
        "data_dir": getattr(_explore_tls, "data_dir", None),
        "fix_type": getattr(_react_tls_local, "fix_type", "bug_fix"),
    }

    llm = ChatAnthropic(
        model=REACT_MODEL,
        max_tokens=16000,
        timeout=120.0,
    ).bind_tools(all_tools, parallel_tool_calls=True)

    # Two-block prompt caching strategy:
    #   Block 1 (static_block): workflow, tools, rules, strategy — same for all bugs of the
    #     same fix_type. cache_control applied here. Cached across consecutive eval bugs
    #     that run within the 5-min Anthropic ephemeral window.
    #   Block 2 (dynamic_block): repo name, ticket, intent, code map, BRTs — changes per bug.
    #     No cache_control. Always fresh.
    # Within one run (30+ LLM calls): both blocks are identical → both cached after call 1.
    # Across eval bugs: block 1 cache hit saves ~3500 tokens × 0.9 per call.
    messages: list = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": static_block,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": dynamic_block,
                },
            ]
        ),
        HumanMessage(content=task_message),
    ]

    # Adaptive tool call budget: single-file bugs get 30 calls, multi-file get 45.
    # The difficulty is set by the eval runner (from the bug dataset) or can be
    # derived from intent.likely_affected_modules count for live runs.
    work_order = state.get("work_order", {})
    intent_for_budget = state.get("intent", {})
    difficulty = work_order.get("difficulty", "")
    if not difficulty:
        # Infer from intent: 1 module → single-file, 3+ → multi-file
        n_modules = len(intent_for_budget.get("likely_affected_modules", []))
        if n_modules >= 3:
            difficulty = "multi-file"
        elif n_modules <= 1:
            difficulty = "single-file"
    call_budget = budget_for_difficulty(difficulty)
    gs = GuardrailState(max_tool_calls=call_budget)
    # Per-run microcompact state — tracks which tool_call_ids have been
    # replaced with placeholders. Idempotent: same tool_call_id always maps
    # to the same placeholder string, so the message prefix stays byte-stable
    # across iterations and the Anthropic prompt cache survives.
    microcompact_state = MicrocompactState()
    current_phase = "explore"  # Track phase transitions

    if trace:
        trace.stage_start("react_loop")

    logger.info(
        "=== REACT LOOP: Starting agent with %d tools (160K context) | budget=%d calls (difficulty=%s) ===",
        len(all_tools), call_budget, difficulty or "unknown",
    )

    while gs.tool_call_count < gs.max_tool_calls:
        # Budget checkpoint every 10 calls
        if gs.tool_call_count > 0 and gs.tool_call_count % 10 == 0:
            logger.info(
                "BUDGET CHECK [%d/%d calls, $%.2f, %ds]: greps=%d reads=%d edits=%d sandbox=%s phase=%s",
                gs.tool_call_count, gs.max_tool_calls, gs.cost_usd, int(gs.elapsed),
                gs.grep_count, gs.read_file_count, gs.string_replace_count,
                gs.sandbox_created, current_phase,
            )

        # PHASE NUDGE: if past halfway with sandbox but 0 edits, force a fix attempt.
        # Threshold scales with the budget: half of max_tool_calls, min 15.
        _nudge_at = max(15, gs.max_tool_calls // 2)
        if (gs.tool_call_count == _nudge_at
                and gs.sandbox_created
                and gs.string_replace_count == 0):
            _remaining = gs.max_tool_calls - _nudge_at
            nudge = (
                f"SYSTEM: You are halfway through your budget ({_nudge_at}/{gs.max_tool_calls} calls) "
                "with a sandbox but ZERO edits. The function names in the bug description may not match the code. "
                "STOP GREPPING. Based on what you've read so far:\n"
                "1. Pick the most likely function to fix\n"
                "2. Call string_replace with your best fix attempt\n"
                "3. Even an imperfect fix you can iterate on beats more searching\n"
                f"You have {_remaining} calls left — use them for editing, testing, and submitting."
            )
            messages.append(HumanMessage(content=nudge))
            logger.info("PHASE NUDGE injected at call %d: forcing edit phase", _nudge_at)

        # FORCE ESCALATION: 75% of budget, sandbox exists, still 0 edits = stuck
        _escalate_at = max(20, int(gs.max_tool_calls * 0.75))
        if (gs.tool_call_count >= _escalate_at
                and gs.sandbox_created
                and gs.string_replace_count == 0):
            logger.warning(
                "Force escalation: %d calls with sandbox but 0 edits — agent is stuck",
                gs.tool_call_count,
            )
            state["escalated"] = True
            state["escalate_reason"] = (
                f"Agent explored for {gs.tool_call_count} calls with a sandbox but never attempted an edit. "
                "The bug description may not match the code's naming conventions. "
                "Human review needed to identify the correct fix location."
            )
            break

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

        # Call the LLM — with multi-stage recovery for context overflow
        response = None
        for _recovery_attempt in range(3):
            try:
                response = llm.invoke(messages)
                break
            except Exception as e:
                err_str = str(e).lower()
                is_prompt_too_long = "prompt is too long" in err_str or "413" in err_str or "context_length" in err_str
                is_output_limit = "max_tokens" in err_str or "output" in err_str

                if is_prompt_too_long and _recovery_attempt == 0:
                    # Recovery level 1: Force aggressive summarization
                    logger.warning("Prompt too long — forcing context summarization (attempt %d)", _recovery_attempt + 1)
                    messages = mask_old_observations(messages, window_size=5)  # Tighter window
                    messages = maybe_summarize(messages, force=True)
                    if trace:
                        trace.emit("context_compaction", "react_loop", {
                            "action": "recovery_summarization",
                            "reason": "prompt_too_long",
                            "tokens_after": count_tokens_approx(messages),
                        })
                    continue
                elif is_output_limit and _recovery_attempt < 2:
                    # Recovery: append resume message
                    logger.warning("Output limit hit — appending resume message (attempt %d)", _recovery_attempt + 1)
                    messages.append(HumanMessage(
                        content="Your output was truncated. Continue from exactly where you stopped. No apology needed."
                    ))
                    continue
                else:
                    logger.error("ReAct LLM call failed: %s", e)
                    state["error"] = f"LLM call failed: {e}"
                    break

        if response is None:
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

        # Execute tool calls — concurrent for read-only, serial for writes
        terminal = False

        # Partition tool calls into batches: consecutive concurrent-safe tools
        # run in parallel, others run serially.
        batches: list[list[dict]] = []
        for tc in response.tool_calls:
            safe = _is_concurrent_safe(tc["name"])
            if safe and batches and batches[-1][0].get("_concurrent"):
                batches[-1].append({**tc, "_concurrent": True})
            else:
                batches.append([{**tc, "_concurrent": safe}])

        for batch in batches:
            if terminal:
                break

            # --- Run a single batch (concurrent or serial) ---
            def _execute_one_tool(tc_entry: dict) -> tuple[dict, str]:
                """Execute a single tool call. Returns (tc_entry, result_str).

                When running inside a ThreadPoolExecutor worker, the thread-local
                storage (_tls) starts blank. We re-inject the captured context so
                explore_tools can find the repo_path.
                """
                # Re-inject thread-local context captured from the parent thread.
                # This is a no-op when called from the main thread (serial execution).
                _explore_set_ctx(
                    _thread_ctx["repo_name"],
                    _thread_ctx["repo_path"],
                    _thread_ctx["data_dir"],
                )
                tn = tc_entry["name"]
                ta = tc_entry["args"]
                t = tool_map.get(tn)
                if t is None:
                    return tc_entry, f"ERROR: Tool '{tn}' not found"
                tool_t0 = time.monotonic()
                try:
                    raw = t.invoke(ta)
                except Exception as te:
                    raw = f"ERROR: Tool execution failed: {te}"
                dur = round((time.monotonic() - tool_t0) * 1000)
                capped = cap_tool_output(tn, str(raw))
                if trace:
                    trace.emit("tool_result", "react_loop", {
                        "tool_name": tn, "duration_ms": dur,
                        "result_preview": str(capped), "phase": current_phase,
                    })
                return tc_entry, capped

            is_concurrent_batch = len(batch) > 1 and all(b.get("_concurrent") for b in batch)

            # Pre-execution: guardrail checks + phase tracking for all tools in batch
            checked: list[tuple[dict, str | None]] = []  # (tc, guardrail_error_or_None)
            for tc in batch:
                tool_name = tc["name"]
                tool_args = tc["args"]

                # Detect phase transition
                new_phase = _PHASE_MAP.get(tool_name, current_phase)
                if new_phase != current_phase:
                    if trace:
                        trace.emit("state_transition", "react_loop", {
                            "from_phase": current_phase, "to_phase": new_phase,
                            "trigger_tool": tool_name,
                            "at_call": gs.tool_call_count + 1,
                            "cost_usd_at_transition": round(gs.cost_usd, 6),
                        })
                    current_phase = new_phase

                logger.info("ReAct tool call %d: %s(%s)",
                            gs.tool_call_count + 1, tool_name, str(tool_args)[:100])

                if trace:
                    trace.emit("tool_call", "react_loop", {
                        "tool_name": tool_name, "args": tool_args,
                        "call_number": gs.tool_call_count + 1,
                        "phase": current_phase, "reasoning": agent_reasoning,
                    })

                guardrail_error = check_tool_call(tool_name, tool_args, gs)
                if guardrail_error and guardrail_error.startswith("WARNING:"):
                    logger.info("Guardrail warning for %s: %s", tool_name, guardrail_error[:100])
                    if trace:
                        trace.emit("guardrail_event", "react_loop", {
                            "tool_name": tool_name, "action": "warn",
                            "message": guardrail_error[:200],
                            "call_number": gs.tool_call_count + 1,
                        })
                    guardrail_error = None

                if guardrail_error:
                    logger.info("Guardrail blocked %s: %s", tool_name, guardrail_error[:100])
                    if trace:
                        trace.emit("guardrail_event", "react_loop", {
                            "tool_name": tool_name, "action": "block",
                            "message": guardrail_error[:200],
                            "call_number": gs.tool_call_count + 1,
                        })
                checked.append((tc, guardrail_error))

            # Execute: concurrent batch via ThreadPoolExecutor, serial one-by-one
            results: list[tuple[dict, str]] = []  # ordered (tc, result_str)

            if is_concurrent_batch:
                to_run = [(tc, ge) for tc, ge in checked if ge is None]
                blocked = [(tc, ge) for tc, ge in checked if ge is not None]
                results.extend((tc, ge) for tc, ge in blocked)

                if to_run:
                    with ThreadPoolExecutor(max_workers=min(len(to_run), MAX_TOOL_CONCURRENCY)) as pool:
                        futures = {pool.submit(_execute_one_tool, tc): tc for tc, _ in to_run}
                        for future in as_completed(futures):
                            tc_entry, result_str = future.result()
                            results.append((tc_entry, result_str))
                    # Re-order results to match original batch order
                    order = {tc["id"]: i for i, tc in enumerate(batch)}
                    results.sort(key=lambda x: order.get(x[0]["id"], 999))
            else:
                # Serial execution
                for tc, guardrail_error in checked:
                    if guardrail_error:
                        results.append((tc, guardrail_error))
                    else:
                        _, result_str = _execute_one_tool(tc)
                        results.append((tc, result_str))

            # Post-execution: update state, check terminal, append messages
            for tc, result_str in results:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                # Update guardrail state
                update_from_tool_result(tool_name, tool_args, str(result_str), gs)

                # Check for terminal tools
                if tool_name == "submit_fix":
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
                    continue

                if tool_name == "escalate":
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
                        repo_name = state.get("work_order", {}).get("repo_name", "")
                        _explore_set_ctx(repo_name, str(sandbox_path), None)
                        # Also update the captured context dict so future worker
                        # threads automatically use the sandbox path.
                        _thread_ctx["repo_name"] = repo_name
                        _thread_ctx["repo_path"] = sandbox_path

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

        # Context window management (Layers 2b + 3)
        # Microcompact: cache-friendly in-place tool result eviction.
        # Stable prefix → prompt cache survives across iterations.
        pre_mask_tokens = count_tokens_approx(messages)
        messages = microcompact_in_place(messages, microcompact_state)
        post_mask_tokens = count_tokens_approx(messages)
        if pre_mask_tokens != post_mask_tokens and trace:
            trace.emit("context_compaction", "react_loop", {
                "action": "microcompact",
                "tokens_before": pre_mask_tokens,
                "tokens_after": post_mask_tokens,
                "tokens_saved": pre_mask_tokens - post_mask_tokens,
                "compacted_count": microcompact_state.count,
                "cumulative_tokens_saved": microcompact_state.tokens_saved,
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

    # Save cache-safe params so post-loop subagents (verifier, summarizer) can
    # inherit our prompt-cached prefix instead of paying full price for a fresh
    # call. Mirrors Claude Code's saveCacheSafeParams pattern.
    try:
        from agent.forked_subagent import CacheSafeParams, save_cache_safe_params
        # Reconstruct a flat string version of the system prompt for cache match
        # (verifier uses ChatAnthropic.invoke with a SystemMessage(content=str)).
        save_cache_safe_params(CacheSafeParams(
            system_prompt=static_block + "\n\n" + dynamic_block,
            messages=list(messages[1:]),  # drop the system message — we re-add it
            model=REACT_MODEL,
            cache_control_block={"type": "ephemeral"},
        ))
    except Exception as e:
        logger.debug("Failed to save cache-safe params (non-fatal): %s", e)

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
