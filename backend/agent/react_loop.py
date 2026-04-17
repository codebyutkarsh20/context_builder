"""
react_loop.py — Core ReAct while-loop for the AI Deploy Agent.

Single agent loop where the LLM decides: explore → localize → edit → test → review → submit.
Mirrors the exploration_node pattern (pipeline.py:798-894) but with ALL tools available.
"""

from __future__ import annotations

import logging
import os
import re
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


def _build_status_refresh(gs, current_phase: str) -> str:
    """Build a short status block injected periodically so the agent sees its
    own progress and current sandbox state without re-reading files.

    Mirrors Claude Code's queryContext rebuild-per-turn pattern. Kept short
    (~300 tokens) so cache impact is bounded — the message is appended at the
    tail of the conversation, after the cached prefix.

    Returns empty string if there's nothing useful to report (e.g. agent
    hasn't created sandbox yet).
    """
    import subprocess

    parts: list[str] = ["[STATUS REFRESH — current state of your work]"]
    parts.append(f"Phase: {current_phase}")
    parts.append(f"Tool calls used: {gs.tool_call_count}/{gs.max_tool_calls}")

    if gs.sandbox_path:
        try:
            # Get diff stats (files + line counts) — short, high-signal
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=gs.sandbox_path,
                capture_output=True, text=True, timeout=5,
            )
            stat_lines = (result.stdout or "").strip().splitlines()
            if stat_lines:
                # Trim to first 6 lines (per-file stats) — last line is "N files, M insertions"
                shown = stat_lines[:6] + [stat_lines[-1]] if len(stat_lines) > 7 else stat_lines
                parts.append("\nFiles modified in sandbox:")
                for line in shown:
                    parts.append(f"  {line.strip()}")
            else:
                parts.append("Sandbox exists but no edits committed yet.")
        except Exception:
            pass

    # Test status
    if gs.tests_attempted:
        if gs.tests_passed:
            parts.append("\nLast test result: PASSED")
        elif gs.tests_skipped:
            parts.append("\nLast test result: SKIPPED (env issue, not your fix)")
        else:
            parts.append(f"\nLast test result: FAILED ({gs.test_failure_count} failures)")
    else:
        parts.append("\nNo tests attempted yet.")

    # Review status
    if gs.review_count > 0:
        parts.append(f"Review verdict: {gs.review_verdict or 'pending'} ({gs.review_count} requests)")

    # Anti-pattern hints — surface stuck-detection signals
    if gs.grep_count >= 8:
        parts.append(
            f"\n⚠ You've called grep_repo {gs.grep_count} times. "
            "Try delegate_explore() or read_function() instead."
        )
    if gs.tool_call_count >= int(gs.max_tool_calls * 0.6) and gs.string_replace_count == 0:
        parts.append(
            f"\n⚠ {gs.tool_call_count}/{gs.max_tool_calls} calls used and 0 edits made. "
            "Pick your best hypothesis and apply it now."
        )

    # Don't emit the block if it's just the boilerplate (no actionable info)
    if len(parts) <= 3:
        return ""
    return "\n".join(parts)


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
    # Add the explore subagent tool — main agent can delegate "find me X"
    # questions to a Haiku-backed read-only subagent, saving turns + cost.
    from agent.explore_subagent import EXPLORE_SUBAGENT_TOOLS
    # Add web_fetch + web_search — for looking up library docs / GitHub
    # issues / Stack Overflow answers when the agent encounters unfamiliar
    # APIs or error messages. Disabled by default (set ENABLE_WEB_TOOLS=1).
    from agent.web_tools import WEB_TOOLS, WEB_TOOLS_ENABLED
    web_tools = WEB_TOOLS if WEB_TOOLS_ENABLED else []
    all_tools = explore_tools + EXPLORE_SUBAGENT_TOOLS + web_tools + REACT_TOOLS
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

    # Two LLM instances — fast (no thinking) for exploration, thinking for
    # editing/recovery. v4 inverts the switch: start fast, switch to thinking
    # on first string_replace. Rationale:
    #   - Early turns are mechanical exploration (grep, read_file, list_files).
    #     Thinking adds cost without quality gain here.
    #   - Once the agent starts editing, decisions get hard: "is this the right
    #     fix?", "should I revert?", "what edge case am I missing?"
    #     Extended thinking helps during editing/recovery.
    #   - Once ON, thinking stays ON for the rest of the run.
    # Budget: 2048 tokens of thinking per turn, capped by the model.
    _THINKING_BUDGET = int(os.environ.get("REACT_THINKING_BUDGET", "2048"))
    _THINKING_ENABLED = os.environ.get("DISABLE_REACT_THINKING", "") not in ("1", "true", "True")

    def _build_llm(with_thinking: bool):
        kwargs: dict = {
            "model": REACT_MODEL,
            "max_tokens": 16000,
            "timeout": 180.0,  # 3 min per call — generous for large prompts with thinking
        }
        if with_thinking and _THINKING_ENABLED:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": _THINKING_BUDGET}
            # Thinking requires temperature=1 per Anthropic API contract
            kwargs["temperature"] = 1.0
        try:
            return ChatAnthropic(**kwargs).bind_tools(all_tools, parallel_tool_calls=True)
        except TypeError as e:
            # Older anthropic SDKs may not accept `thinking` kwarg. Fall back
            # to plain LLM rather than crashing the run. Logged once at startup.
            if with_thinking and "thinking" in str(e).lower():
                logger.warning(
                    "Thinking config rejected by SDK (%s) — falling back to plain LLM",
                    str(e)[:100],
                )
                kwargs.pop("thinking", None)
                kwargs.pop("temperature", None)
                return ChatAnthropic(**kwargs).bind_tools(all_tools, parallel_tool_calls=True)
            raise

    llm_thinking = _build_llm(with_thinking=True)
    llm_fast = _build_llm(with_thinking=False)
    # v4: Start with fast LLM (exploration is mechanical), switch to thinking
    # on first string_replace (editing/recovery benefits from thinking). Once ON, stays ON.
    llm = llm_fast

    # Prompt caching strategy — ORDER MATTERS for Anthropic's prefix cache.
    #
    # The cache matches the LONGEST byte-identical prefix. If any byte in the
    # prefix changes, everything after it is uncached. So we put content in
    # order of stability:
    #
    #   1. static_block (60 lines) — NEVER changes. Same for every bug.
    #      cache_control breakpoint here → cached across ALL bugs in an eval
    #      run (within the 5-min TTL window).
    #
    #   2. dynamic_block (~200 lines) — changes BETWEEN bugs but is STABLE
    #      within one bug run (30+ LLM calls, all identical).
    #      cache_control breakpoint here → cached for all calls within one bug.
    #
    #   3. task_message (2 lines) — identical within one bug (no breakpoint needed,
    #      it's part of the prefix that's already covered by #2).
    #
    #   4. conversation messages — GROW every call. Never cached.
    #
    # Two breakpoints = two cache tiers:
    #   - Tier 1 (static): hit rate ~100% across bugs in same eval batch
    #   - Tier 2 (static+dynamic): hit rate ~100% within one bug (30+ calls)
    #
    # Previous bug: only static had cache_control → only 4,931 tokens cached
    # out of 12K+ system prompt → 30% cache ratio. With both breakpoints →
    # 90%+ cache ratio within a bug run.
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
                    "cache_control": {"type": "ephemeral"},
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
    # Wire GuardrailState to thread-local so tools (write_brt, etc.) can
    # access gs.files_read and gs.file_cache for context-aware generation.
    from agent.react_tools import set_guardrail_state
    set_guardrail_state(gs)
    # Per-run microcompact state — tracks which tool_call_ids have been
    # replaced with placeholders. Idempotent: same tool_call_id always maps
    # to the same placeholder string, so the message prefix stays byte-stable
    # across iterations and the Anthropic prompt cache survives.
    microcompact_state = MicrocompactState()
    current_phase = "explore"  # Track phase transitions

    # Stuck-detection state (P3 — diminishing returns + auto-replan).
    # Tracks token-count growth per turn and edit/test counts at each check
    # point. If 3 consecutive checks show < 500 tokens added AND no new
    # edits or tests, the agent is in a non-productive loop — inject a
    # forced replan nudge instead of silently burning more budget.
    _stuck_history: list[dict] = []
    _replan_injected_at: set[int] = set()  # avoid re-injecting at the same call number

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

        # Phase nudge removed — was adding constraint without helping. If the agent
        # needs more exploration, let it explore. Force-escalation below is the only
        # hard stop, and it's been loosened to 90% of budget (was 75%).

        # Stuck detector v2 — tool-repetition-based (soft hint, not force).
        # Detects: same tool with identical args called 3x in a row, OR
        # same file grep'd repeatedly without reading it. Emits a GENTLE
        # suggestion — doesn't force a replan, doesn't escalate.
        # Design principle: agent is trusted; nudge is a helpful peer review.
        if gs.tool_call_count >= 6 and gs.tool_call_count % 3 == 0 \
                and gs.tool_call_count not in _replan_injected_at:
            recent = gs.tool_history[-6:] if len(gs.tool_history) >= 6 else []
            # Signal 1: same tool name 4+ times in last 6 calls
            hint = ""
            if recent:
                from collections import Counter
                tool_freq = Counter(recent)
                most_used, most_n = tool_freq.most_common(1)[0]
                if most_n >= 4 and most_used in ("grep_repo", "read_file", "list_files"):
                    hint = (
                        f"💡 Gentle reminder: you've called `{most_used}` {most_n} times in the last 6 turns. "
                        f"If the info you want isn't emerging, try:\n"
                        f"  - `delegate_explore(\"your question\")` — a Haiku subagent answers broad queries in one turn\n"
                        f"  - `read_function(file, function_name)` — full function instead of grep snippets\n"
                        f"  - `get_file_structure(file)` — function signatures + line numbers\n"
                        "You're not being cut off — just pointing out cheaper options."
                    )
            # Signal 2: sandbox created but no edits after 15+ calls
            if (not hint and gs.sandbox_created and gs.string_replace_count == 0
                    and gs.tool_call_count >= 15):
                hint = (
                    f"💡 Observation: you've done {gs.tool_call_count} exploration calls in the sandbox "
                    "but no edits yet. If you have a reasonable hypothesis, an imperfect fix you can "
                    "iterate on often beats more exploration. Your call though — keep exploring if needed."
                )

            if hint:
                messages.append(HumanMessage(content=hint))
                _replan_injected_at.add(gs.tool_call_count)
                logger.info("Soft hint injected at call %d", gs.tool_call_count)
                if trace:
                    trace.emit("auto_replan_nudge", "react_loop", {
                        "at_call": gs.tool_call_count,
                        "hint_type": "tool_repetition" if most_used in recent else "no_edits_yet",
                        "recent_tools": recent,
                    })

        # Force-escalation removed — was cutting off agents mid-exploration.
        # The budget limit alone (hit on `while` condition) is the only hard stop.
        # Agent is trusted to spend its full budget productively.

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
        # Use real input_tokens from last API call (accurate), fall back to estimate for first call
        context_tokens = gs.real_input_tokens if gs.real_input_tokens > 0 else count_tokens_approx(messages)
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

        # v4: Switch from fast-LLM to thinking-LLM once the first edit lands.
        # Before any edit, the agent is doing mechanical exploration (grep,
        # read_file) — fast LLM is sufficient.  Once editing starts, decisions
        # get hard (right fix? revert? edge cases?) — thinking helps.
        # Once ON, stays ON for the rest of the run.
        # v4 thinking triggers: first edit OR test failure OR verify_fix rejection
        _should_think = (
            gs.string_replace_count >= 1
            or (gs.tests_attempted and not gs.tests_passed and not gs.tests_skipped)
            or getattr(gs, "_verify_fix_called", False)
        )
        if llm is llm_fast and _should_think:
            llm = llm_thinking
            trigger = "edit" if gs.string_replace_count >= 1 else "test_failure" if gs.tests_attempted else "verify_fix"
            logger.info(
                "Switched to thinking LLM (trigger=%s) at call %d",
                trigger, gs.tool_call_count,
            )
            if trace:
                trace.emit("llm_mode_switch", "react_loop", {
                    "from": "fast",
                    "to": "thinking",
                    "at_call": gs.tool_call_count,
                    "reason": "first_edit_landed",
                })

        # ── Conversation-level prompt caching ────────────────────────────
        # Anthropic allows max 4 cache_control breakpoints per request.
        # We use: 1 on static_block, 1 on dynamic_block (in system msg),
        # and 1 on the last conversation message (moving breakpoint).
        # = 3 total, under the 4 limit.
        #
        # Each turn: remove the PREVIOUS conversation breakpoint, add one
        # to the NEW last message. This progressively caches more of the
        # conversation prefix.
        if len(messages) >= 3:
            # Step 1: Remove previous conversation-level cache_control
            # (skip messages[0] which is the system message)
            for msg in messages[1:-1]:
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict):
                            block.pop("cache_control", None)

            # Step 2: Add cache_control to the LAST message
            last_msg = messages[-1]
            if isinstance(last_msg, (ToolMessage, HumanMessage)):
                content = last_msg.content
                if isinstance(content, str):
                    last_msg.content = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list) and content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        last_block["cache_control"] = {"type": "ephemeral"}

        # Call the LLM — with multi-stage recovery for context overflow + transient errors
        response = None
        for _recovery_attempt in range(5):  # raised from 3 to 5 for transient retries
            try:
                response = llm.invoke(messages)
                break
            except Exception as e:
                err_str = str(e).lower()
                is_prompt_too_long = "prompt is too long" in err_str or "413" in err_str or "context_length" in err_str
                is_output_limit = "max_tokens" in err_str or "output" in err_str
                is_thinking_unsupported = "thinking" in err_str and "unexpected keyword" in err_str
                # Transient errors: timeouts, rate limits, dropped connections, 5xx
                is_transient = any(kw in err_str for kw in (
                    "timed out", "timeout", "long-requests", "connection",
                    "rate limit", "429", "overloaded", "529",
                    "internal server", "500", "502", "503", "bad gateway",
                    "request timed out", "interrupted",
                ))

                # Retry transient errors with exponential backoff BEFORE other recovery.
                # Agent was fine — network blip shouldn't kill the run.
                if is_transient and _recovery_attempt < 4:
                    backoff = min(60, 5 * (2 ** _recovery_attempt))  # 5s, 10s, 20s, 40s, 60s
                    logger.warning(
                        "Transient LLM error (attempt %d/5) — retrying in %ds: %s",
                        _recovery_attempt + 1, backoff, err_str[:150],
                    )
                    import time as _time
                    _time.sleep(backoff)
                    if trace:
                        trace.emit("llm_transient_retry", "react_loop", {
                            "attempt": _recovery_attempt + 1,
                            "backoff_sec": backoff,
                            "error": err_str[:200],
                        })
                    continue

                # Recovery: SDK rejects the `thinking` parameter (older anthropic
                # SDK or a server change). Permanently switch to the fast LLM
                # for the rest of the run so we don't loop forever.
                if is_thinking_unsupported and llm is llm_thinking:
                    logger.warning(
                        "SDK rejected thinking config — switching to fast LLM permanently",
                    )
                    llm = llm_fast
                    llm_thinking = llm_fast  # prevent re-switching back later
                    if trace:
                        trace.emit("llm_mode_switch", "react_loop", {
                            "from": "thinking",
                            "to": "fast",
                            "at_call": gs.tool_call_count,
                            "reason": "thinking_unsupported_by_sdk",
                        })
                    # Retry immediately with the new LLM — don't burn more attempts
                    try:
                        response = llm.invoke(messages)
                    except Exception as e2:
                        logger.error("Fast LLM also failed after thinking switch: %s", e2)
                        state["error"] = f"LLM call failed: {e2}"
                    break  # Exit recovery loop (either succeeded or failed definitively)

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

        # Track token usage from API response (real counts, not estimates)
        usage = getattr(response, "response_metadata", {}).get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        call_cost = _estimate_cost(REACT_MODEL, input_tokens, output_tokens, cache_creation, cache_read)
        gs.cost_usd += call_cost
        # Store real token counts — used for summarization triggers + stuck detection
        gs.real_input_tokens = input_tokens
        gs.real_output_tokens = output_tokens
        gs.cumulative_input_tokens += input_tokens
        gs.cumulative_output_tokens += output_tokens

        if trace:
            # Capture full response content for replay/debugging —
            # including extended thinking blocks (the agent's internal
            # reasoning before deciding on tool calls).
            response_text = ""
            thinking_text = ""
            if isinstance(response.content, str):
                response_text = response.content
            elif isinstance(response.content, list):
                text_blocks = []
                thinking_blocks = []
                for b in response.content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            text_blocks.append(b.get("text", ""))
                        elif b.get("type") == "thinking":
                            thinking_blocks.append(b.get("thinking", ""))
                response_text = "\n".join(text_blocks)
                thinking_text = "\n---\n".join(thinking_blocks)

            response_tool_calls = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    response_tool_calls.append({
                        "name": tc["name"],
                        "args": tc["args"],
                        "id": tc["id"],
                    })

            llm_event_data: dict = {
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
                "thinking_used": llm is llm_thinking,
            }
            # Include thinking text when present — this is the "decision
            # replay" data that lets you see WHY the agent chose its action.
            if thinking_text:
                llm_event_data["thinking_text"] = thinking_text[:8000]
            trace.emit("llm_response", "react_loop", llm_event_data)

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
                capped, truncation_info = cap_tool_output(tn, str(raw))
                if trace:
                    tool_result_data: dict = {
                        "tool_name": tn, "duration_ms": dur,
                        "result_preview": str(capped), "phase": current_phase,
                    }
                    # Log truncation so debugging can see what the agent missed
                    if truncation_info.get("truncated"):
                        tool_result_data["truncation"] = truncation_info
                    trace.emit("tool_result", "react_loop", tool_result_data)
                return tc_entry, capped

            is_concurrent_batch = len(batch) > 1 and all(b.get("_concurrent") for b in batch)

            # Pre-execution: guardrail checks + phase tracking for all tools in batch.
            # Pre-scan: if produce_plan is in this batch, temporarily mark plan as
            # produced so create_sandbox in the same batch isn't blocked by the gate.
            _batch_has_plan = any(tc["name"] == "produce_plan" for tc in batch)
            if _batch_has_plan and not gs.plan_produced:
                gs.plan_produced = True  # tentative — reverted if produce_plan fails
                gs._plan_tentative = True
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
                if guardrail_error and (guardrail_error.startswith("WARNING:") or guardrail_error.startswith("SUGGESTION:")):
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

            # Revert tentative plan flag if produce_plan was in batch but failed
            if getattr(gs, "_plan_tentative", False):
                plan_ok = any(
                    tc["name"] == "produce_plan" and str(rs).startswith("OK:")
                    for tc, rs in results
                )
                if not plan_ok:
                    gs.plan_produced = False
                gs._plan_tentative = False

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
                    # Parse confidence from text like "REVIEW VERDICT: APPROVE (confidence: 97%)"
                    conf_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", result_text, re.IGNORECASE)
                    if conf_match:
                        raw = float(conf_match.group(1))
                        conf = raw / 100.0 if raw > 1.0 else raw
                    else:
                        conf = 0.0
                    review_dict = {"verdict": "UNKNOWN", "confidence": conf}
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
            # Use real token count from last API response for accurate trigger
            pre_summarize = gs.real_input_tokens if gs.real_input_tokens > 0 else count_tokens_approx(messages)
            messages = maybe_summarize(messages, real_token_count=gs.real_input_tokens)
            post_summarize = count_tokens_approx(messages)
            if pre_summarize != post_summarize and trace:
                trace.emit("context_compaction", "react_loop", {
                    "action": "summarization",
                    "tokens_before": pre_summarize,
                    "tokens_after": post_summarize,
                    "tokens_saved": pre_summarize - post_summarize,
                    "at_call": gs.tool_call_count,
                })

        # Dynamic context refresh — every 5 tool calls, inject a short
        # status block so the agent sees its own progress without re-reading
        # files or re-running greps. Mirrors Claude Code's queryContext
        # rebuild-per-turn pattern, adapted for our cost-conscious budget.
        # Skip if no edit has happened yet (nothing has changed worth refreshing).
        _refresh_interval = int(os.environ.get("REACT_REFRESH_INTERVAL", "5"))
        if (gs.tool_call_count > 0
                and gs.tool_call_count % _refresh_interval == 0
                and gs.sandbox_created
                and gs.string_replace_count >= 1):
            refresh_block = _build_status_refresh(gs, current_phase)
            if refresh_block:
                messages.append(HumanMessage(content=refresh_block))
                if trace:
                    trace.emit("context_refresh", "react_loop", {
                        "at_call": gs.tool_call_count,
                        "phase": current_phase,
                        "chars": len(refresh_block),
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
