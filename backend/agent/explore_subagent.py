"""
explore_subagent.py — Haiku-backed read-only exploration subagent.

Ports Claude Code's exploreAgent pattern (tools/AgentTool/built-in/exploreAgent.ts).
The main agent can delegate "find me X" questions to this subagent which:
    - Runs on Haiku (faster + ~10x cheaper than Sonnet)
    - Has access to ONLY read-only tools (grep_repo, read_file, list_files, ...)
    - Cannot edit, create sandboxes, or call terminal tools
    - Returns a focused text report instead of dumping raw tool outputs into
      the main agent's context

This trades 1 main-agent tool call (delegate_explore) for N Haiku tool calls,
which is a net win when N > 3 because:
  - Haiku is ~10x cheaper than Sonnet
  - The main agent's context isn't polluted with intermediate tool results
  - The subagent can run multiple greps/reads in parallel

Use cases (from the prompt that the main agent receives):
- "Find all callers of `process_payment`"
- "What files import `auth.middleware`?"
- "How is the user-session lifecycle managed?"
- "Are there any TODO comments related to retries?"
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Cap on subagent tool calls — exploration shouldn't take forever
EXPLORE_MAX_TOOL_CALLS = 12
EXPLORE_TIMEOUT_SECONDS = 90
EXPLORE_MODEL = os.environ.get("EXPLORE_SUBAGENT_MODEL", "claude-haiku-4-5-20251001")


_EXPLORE_SYSTEM_PROMPT = """You are a file search specialist working as a subagent for a code-fixing AI.

=== READ-ONLY MODE — NO MODIFICATIONS ALLOWED ===
You are STRICTLY PROHIBITED from:
  - Creating, editing, or deleting any files
  - Running tests, creating sandboxes, or anything that changes state
  - Submitting fixes, requesting reviews, or escalating

Your role is EXCLUSIVELY to search and analyze existing code, then report
back to the calling agent.

=== YOUR STRENGTHS ===
- Rapidly finding files matching glob patterns
- Searching code with regex (grep)
- Reading specific functions by name
- Building up a focused picture across multiple sources

=== STRATEGY ===
1. **Parallelize aggressively.** Issue multiple grep/read calls in a single
   turn whenever the queries are independent. The caller is paying for your
   Haiku-speed turn-around.
2. **Stop when you can answer.** You have a budget of 12 tool calls but most
   questions resolve in 3-6. Don't keep exploring once you have a clear answer.
3. **Be specific in your report.** Quote line numbers, file paths, and the
   key code snippets. The caller will use your report to make decisions, so
   ambiguity wastes their next turn.

=== OUTPUT FORMAT ===
End with a "FINDINGS" section that:
  - Names the file(s) that answer the question
  - Quotes the relevant function/symbol names with file:line refs
  - States what the code DOES (one sentence per finding)
  - Flags any uncertainty ("could not find X" is more useful than guessing)

Example finding format:
    FINDINGS:
    - `process_payment` is defined at payments/service.py:142 and is called by:
        - api/checkout.py:78 (POST /checkout handler)
        - jobs/retry_failed.py:33 (background retry worker)
    - The function validates payment amount > 0 (line 145), but does NOT
      check requisition.approved_at — that gate lives in
      payments/preflight.py:67.
"""


def _build_subagent_tools(parent_repo_name: str, parent_repo_path: str) -> list:
    """Return the read-only tool set for the explore subagent.

    Uses the SAME tool implementations as the main agent (so the subagent
    sees the same files), but excludes anything that could mutate state.
    """
    from agent.explore_tools import ALL_TOOLS as _explore_tools, set_context as _set_explore_ctx
    # Re-set context so subagent threads inherit repo info via TLS
    _set_explore_ctx(parent_repo_name, parent_repo_path)
    return list(_explore_tools)


def _run_explore_subagent_loop(
    question: str,
    repo_name: str,
    repo_path: str,
    max_tool_calls: int = EXPLORE_MAX_TOOL_CALLS,
) -> dict:
    """Internal: run the subagent's ReAct loop and return its final report.

    Returns
    -------
    dict with keys:
        "report" : str — the subagent's final findings text
        "tool_calls" : int — how many tool calls it consumed
        "cost_usd" : float (best-effort; 0.0 if not reportable)
        "error" : optional error string
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage, ToolMessage,
        )
    except ImportError as e:
        return {"report": "", "tool_calls": 0, "cost_usd": 0.0, "error": str(e)}

    subagent_tools = _build_subagent_tools(repo_name, repo_path)
    tool_map = {t.name: t for t in subagent_tools}

    llm = ChatAnthropic(
        model=EXPLORE_MODEL,
        max_tokens=4000,
        timeout=EXPLORE_TIMEOUT_SECONDS,
    ).bind_tools(subagent_tools, parallel_tool_calls=True)

    messages: list = [
        SystemMessage(content=_EXPLORE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question from the main agent:\n\n{question}"),
    ]

    tool_call_count = 0
    cost_usd = 0.0
    last_text = ""

    for _ in range(max_tool_calls + 1):  # +1 to allow a final no-tools response
        try:
            response = llm.invoke(messages)
        except Exception as e:
            return {
                "report": last_text,
                "tool_calls": tool_call_count,
                "cost_usd": cost_usd,
                "error": f"LLM call failed: {e}",
            }

        # Track cost from usage_metadata if available
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            # Haiku 4.5 pricing: $1/M input, $5/M output (rough)
            in_t = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
            cost_usd += in_t * 1.0 / 1_000_000 + out_t * 5.0 / 1_000_000

        messages.append(response)

        # Capture the text content for the final report
        if response.content:
            last_text = (
                str(response.content)
                if not isinstance(response.content, list)
                else " ".join(
                    str(b.get("text", "")) for b in response.content if isinstance(b, dict)
                )
            )

        # No tool calls → subagent is done, return the text
        tool_calls = getattr(response, "tool_calls", []) or []
        if not tool_calls:
            return {
                "report": last_text or "(subagent returned no findings)",
                "tool_calls": tool_call_count,
                "cost_usd": round(cost_usd, 6),
                "error": None,
            }

        # Hard cap on tool calls — force a final summary turn
        if tool_call_count + len(tool_calls) > max_tool_calls:
            messages.append(HumanMessage(content=(
                "Tool call budget exhausted. Stop calling tools and produce your "
                "FINDINGS report based on what you have already gathered."
            )))
            try:
                final = llm.invoke(messages)
                report_text = (
                    str(final.content)
                    if not isinstance(final.content, list)
                    else " ".join(
                        str(b.get("text", "")) for b in final.content if isinstance(b, dict)
                    )
                )
                return {
                    "report": report_text or last_text or "(budget exhausted, no report)",
                    "tool_calls": tool_call_count,
                    "cost_usd": round(cost_usd, 6),
                    "error": "tool_call_budget_exhausted",
                }
            except Exception as e:
                return {
                    "report": last_text,
                    "tool_calls": tool_call_count,
                    "cost_usd": round(cost_usd, 6),
                    "error": f"final summary call failed: {e}",
                }

        # Execute the requested tools and append results
        for tc in tool_calls:
            tool_call_count += 1
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")
            tool_obj = tool_map.get(tool_name)
            if tool_obj is None:
                result_str = f"ERROR: Unknown tool '{tool_name}'."
            else:
                try:
                    result_str = str(tool_obj.invoke(tool_args))
                    # Cap individual results to avoid blowing the subagent's context
                    if len(result_str) > 6000:
                        result_str = result_str[:6000] + "\n[... truncated]"
                except Exception as e:
                    result_str = f"ERROR: tool '{tool_name}' raised {type(e).__name__}: {e}"
            messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

    # Loop exited without explicit termination — return whatever we have
    return {
        "report": last_text or "(subagent did not produce a final report)",
        "tool_calls": tool_call_count,
        "cost_usd": round(cost_usd, 6),
        "error": "loop_exited_without_termination",
    }


# ---------------------------------------------------------------------------
# Main-agent-facing tool
# ---------------------------------------------------------------------------

@tool
def delegate_explore(question: str) -> str:
    """Delegate a focused exploration question to a fast read-only subagent.

    Use this when you would otherwise burn 4+ tool calls grepping and reading
    files to answer a specific factual question about the codebase. The
    subagent runs Haiku (~10x cheaper than the main loop), has read-only
    tools, and returns a focused FINDINGS report instead of polluting your
    context with raw tool outputs.

    GOOD candidates for delegation:
      - "Find all callers of process_payment and what they pass as arguments"
      - "Where is the retry policy configured? What's the default backoff?"
      - "Are there any TODO comments mentioning 'session expiry'?"
      - "What does the test_settings.py file configure that's relevant to auth?"

    BAD candidates (do these yourself with read_file / read_function):
      - Reading a SPECIFIC file you already know the path of
      - Single grep with one regex (just call grep_repo directly)
      - Anything that requires editing or running tests

    Args:
        question: A specific, factual question. Be concrete about what you
            need to know. Vague questions get vague answers.

    Returns:
        The subagent's FINDINGS report (text), or an ERROR string if the
        subagent failed.
    """
    if not question or not question.strip():
        return "ERROR: question cannot be empty."

    # Resolve repo context from main-agent thread-local state
    from agent.explore_tools import _tls as _explore_tls
    repo_name = getattr(_explore_tls, "repo_name", "")
    repo_path = getattr(_explore_tls, "repo_path", None)
    if not repo_name or not repo_path:
        return (
            "ERROR: repo context not set — explore subagent cannot run. "
            "This usually means the tool is being called outside an active agent loop."
        )

    logger.info("delegate_explore: spawning subagent for question: %s", question[:120])
    result = _run_explore_subagent_loop(
        question=question,
        repo_name=repo_name,
        repo_path=str(repo_path),
        max_tool_calls=EXPLORE_MAX_TOOL_CALLS,
    )

    if result.get("error") and not result.get("report"):
        return f"ERROR: explore subagent failed — {result['error']}"

    err_note = ""
    if result.get("error"):
        err_note = f"\n\n[note: subagent ended with: {result['error']}]"

    return (
        f"=== EXPLORE SUBAGENT REPORT ===\n"
        f"Question: {question[:200]}\n"
        f"Subagent used {result['tool_calls']} tool calls "
        f"(${result['cost_usd']:.4f})\n\n"
        f"{result['report']}{err_note}"
    )


# Tool collection for registration in react_loop.py
EXPLORE_SUBAGENT_TOOLS = [delegate_explore]
