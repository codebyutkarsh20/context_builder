"""
pipeline.py — LangGraph state machine for the AI Deploy Agent.

Flow: Intake → Context Assembly → Localization → [Confidence Gate]
      → Read Source → Repair → Review → [Dev Loop] → Test → PR
"""

from __future__ import annotations

import ast
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from agent.trace import RunTrace

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from agent.feature_flags import create_flag as _create_feature_flag, set_pr_url as _set_flag_pr_url
from agent.patch_utils import (
    fuzzy_match_replace as _fuzzy_match_replace,
    check_syntax as _check_syntax,
    deduplicate_patches as _deduplicate_patches,
    pick_best_patch_per_file as _pick_best_patch_per_file,
)
from agent.sandbox import (
    run_tests as _run_tests,
    cleanup_worktree as _cleanup_worktree,
    append_test_business_context as _append_test_business_context,
)
from agent.types import (
    AgentState,
    IntentAnalysis,
    LocalizationResult,
    Patch,
    PipelineStatus,
    RepairResult,
    ReviewResult,
)

from graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
INTAKE_MODEL = "claude-haiku-4-5-20251001"  # Intake is a structured translation task — Haiku is sufficient
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# Thread-local storage for per-run progress callback + trace
_thread_local = threading.local()


def _get_trace():
    """Return the active RunTrace for this thread, or None."""
    return getattr(_thread_local, "trace", None)


def _emit_trace(event_type: str, data: dict | None = None):
    """Emit a trace event if tracing is active."""
    trace = _get_trace()
    if trace:
        stage = getattr(_thread_local, "current_stage", "unknown")
        trace.emit(event_type, stage, data)

# Binary extensions — skip these in read_source_node
_BINARY_EXTENSIONS = frozenset({
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.o', '.a', '.dylib', '.pyd',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
    '.jar', '.class', '.war',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.wav',
    '.db', '.sqlite', '.sqlite3',
    '.DS_Store',
})

# Secrets pattern — redact before sending to LLM
_SECRETS_RE = re.compile(
    r'(?i)((?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|'
    r'secret[_-]?key|password|passwd|private[_-]?key|credentials)'
    r'\s*[=:]\s*["\']?)[A-Za-z0-9+/=_\-]{16,}["\']?'
)

_ADDITIONAL_SECRET_PATTERNS = [
    re.compile(r'AKIA[A-Z0-9]{16}'),
    re.compile(r'(?:Bearer|token)\s+[A-Za-z0-9\-._~+/]+=*', re.I),
    re.compile(r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+'),
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'ghp_[A-Za-z0-9]{36}'),
]


def _redact_secrets(text: str) -> str:
    """Redact potential secrets/tokens from source code before sending to LLM."""
    text = _SECRETS_RE.sub(r'\1***REDACTED***', text)
    for pat in _ADDITIONAL_SECRET_PATTERNS:
        text = pat.sub('***REDACTED***', text)
    return text


def _report_progress(state: AgentState) -> None:
    """Push current stage + partial results to the API job store."""
    cb = getattr(_thread_local, "progress_callback", None)
    if cb:
        try:
            cb(state)
        except Exception as e:
            logger.debug("Progress callback error: %s", e)


def _structured_call(model: str, max_tokens: int, schema: type, prompt: str, retries: int = 1):
    """Call LLM with structured output (tool use). Returns a Pydantic model instance."""
    # Finding #11: Log approximate token usage to monitor context window utilization
    approx_tokens = len(prompt) // 4
    logger.info("LLM call: model=%s schema=%s ~%d input tokens", model, schema.__name__, approx_tokens)

    _emit_trace("llm_request", {
        "model": model,
        "schema": schema.__name__,
        "max_tokens": max_tokens,
        "prompt_tokens_approx": approx_tokens,
        "prompt_preview": prompt[:2000],
        "prompt_full": prompt,
    })

    llm = ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120.0, max_retries=2)
    structured = llm.with_structured_output(schema)

    t0 = time.monotonic()
    try:
        result = structured.invoke(prompt)
        duration_ms = round((time.monotonic() - t0) * 1000)
        _emit_trace("llm_response", {
            "model": model,
            "schema": schema.__name__,
            "duration_ms": duration_ms,
            "output": result.model_dump() if hasattr(result, "model_dump") else str(result),
        })
        return result
    except Exception as first_err:
        duration_ms = round((time.monotonic() - t0) * 1000)
        _emit_trace("error", {
            "message": f"Structured call failed: {str(first_err)[:500]}",
            "model": model,
            "schema": schema.__name__,
            "duration_ms": duration_ms,
        })
        if retries <= 0:
            raise
        logger.warning("Structured output failed (%s), retrying", first_err)
        error_msg = str(first_err)[:1000]
        retry_prompt = (
            f"Your previous response failed: {error_msg}\n"
            "Please try again. Respond with the exact structured data requested.\n\n"
            + prompt
        )
        t1 = time.monotonic()
        result = structured.invoke(retry_prompt)
        duration_ms = round((time.monotonic() - t1) * 1000)
        _emit_trace("llm_response", {
            "model": model,
            "schema": schema.__name__,
            "duration_ms": duration_ms,
            "retry": True,
            "output": result.model_dump() if hasattr(result, "model_dump") else str(result),
        })
        return result


def _resolve_repo_path(work_order: dict) -> Path | None:
    """Resolve the actual filesystem path for a repo."""
    # 1. Explicit path in work order
    if work_order.get("repo_path"):
        p = Path(work_order["repo_path"])
        if p.exists():
            return p

    # 2. Check graph.json stats for stored path
    repo_name = work_order.get("repo_name", "")
    stats_path = DATA_DIR / repo_name / "graph.json"
    if stats_path.exists():
        try:
            data = json.loads(stats_path.read_text())
            stored_path = data.get("stats", {}).get("repo_path", "")
            if stored_path:
                p = Path(stored_path)
                if p.exists():
                    return p
        except Exception:
            pass

    # 3. Check REPOS_BASE_DIR
    repos_base = os.environ.get("REPOS_BASE_DIR", "")
    if repos_base:
        p = Path(repos_base) / repo_name
        if p.exists():
            return p

    return None



# _fuzzy_match_replace, _run_tests, _cleanup_worktree, _append_test_business_context,
# _check_syntax, _deduplicate_patches, _pick_best_patch_per_file
# → extracted to agent/patch_utils.py and agent/sandbox.py


# ---------------------------------------------------------------------------
# Intake helpers
# ---------------------------------------------------------------------------

def _extract_stack_trace_hints(text: str) -> list[dict]:
    """Extract file:line:function hints from stack traces in ticket text."""
    hints = []

    # Python stack trace lines: File "path/to/file.py", line 123, in function_name
    python_pattern = re.compile(
        r'File ["\']([^"\']+\.py)["\'],\s+line\s+(\d+),\s+in\s+(\w+)'
    )
    for m in python_pattern.finditer(text):
        hints.append({"file": m.group(1), "line": int(m.group(2)), "function": m.group(3)})

    # Java/generic: at package.Class.method(File.java:123)
    java_pattern = re.compile(r'at\s+([\w.]+)\((\w+\.java):(\d+)\)')
    for m in java_pattern.finditer(text):
        hints.append({"file": m.group(2), "line": int(m.group(3)), "function": m.group(1).split('.')[-1]})

    # Simple file:line patterns: /path/to/file.py:123
    file_line_pattern = re.compile(r'([\w/.-]+\.(?:py|js|ts|go|rb|java)):(\d+)')
    for m in file_line_pattern.finditer(text):
        hints.append({"file": m.group(1), "line": int(m.group(2)), "function": None})

    # Deduplicate by (file, line)
    seen: set[tuple[str, int]] = set()
    deduped = []
    for h in hints:
        key = (h["file"], h["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    return deduped[:10]  # Cap at 10 hints


def _classify_bug_category(title: str, description: str) -> str:
    """
    Classify bug into Category A (can auto-fix), B (might work), C (skip).

    Category A: Off-by-one, null check, string format, wrong variable, missing import,
                wrong argument, logic inversion, missing return
    Category B: Business logic, API contract, missing error handling, config issues
    Category C: Race conditions, performance (N+1, slow query), multi-service,
                data migration, environment-specific, UI/visual, architecture
    """
    text = f"{title} {description}".lower()

    # Category C signals — skip these
    c_signals = [
        "race condition", "concurrency", "deadlock", "performance", "slow", "timeout",
        "n+1", "memory leak", "migration", "database migration", "schema change",
        "multi-service", "event", "kafka", "rabbitmq", "queue", "environment",
        "works locally", "works in dev", "only in prod", "ui", "visual", "layout",
        "animation", "css", "architecture", "redesign", "refactor",
    ]
    if any(s in text for s in c_signals):
        return "C"

    # Category A signals — high confidence
    a_signals = [
        "traceback", "exception", "error:", "typeerror", "attributeerror",
        "none", "null", "undefined", "missing", "import", "not found",
        "wrong value", "incorrect value", "returns wrong", "should return",
        "off by one", "index out", "keyerror", "valueerror",
    ]
    a_count = sum(1 for s in a_signals if s in text)
    if a_count >= 2:
        return "A"

    return "B"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _extract_repro_steps(description: str) -> list[str]:
    """Extract reproduction steps from ticket description."""
    import re
    steps = []

    # Look for "Steps to Reproduce", "How to reproduce", "Reproduction steps" sections
    section_patterns = [
        r'(?:steps?\s+to\s+reproduce|how\s+to\s+reproduce|reproduction\s+steps?|repro\s+steps?)\s*:?\s*\n([\s\S]+?)(?:\n\n|\n#{1,3}|\Z)',
        r'(?:to\s+reproduce|reproduce)\s*:?\s*\n([\s\S]+?)(?:\n\n|\n#{1,3}|\Z)',
    ]

    for pattern in section_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            section = match.group(1)
            # Extract numbered/bulleted items
            items = re.findall(r'(?:^|\n)\s*(?:\d+\.|[-*•])\s*(.+)', section)
            if items:
                steps = [item.strip() for item in items[:10]]
                break

    # Fallback: look for numbered list anywhere in description
    if not steps:
        numbered = re.findall(r'(?:^|\n)\s*(\d+)\.\s+(.+)', description)
        if len(numbered) >= 2:  # At least 2 numbered items looks like steps
            steps = [item for _, item in numbered[:10]]

    return steps


def intake_node(state: AgentState) -> AgentState:
    """Stage 1: Translate bug ticket into technical spec via structured output."""
    _thread_local.current_stage = "intake"
    trace = _get_trace()
    if trace:
        trace.stage_start("intake")
    logger.info("=== INTAKE: Translating bug ticket intent ===")
    state["status"] = PipelineStatus.INTAKE
    _report_progress(state)

    work_order = state.get("work_order", {})

    prompt = f"""Translate this bug ticket into a technical specification.

Ticket: {work_order.get('title', '')}
Description: {work_order.get('description', '')}
Priority: {work_order.get('priority', 'unknown')}
Component: {work_order.get('affected_component', 'unknown')}
Comments: {'; '.join(work_order.get('comments', []))}

Include acceptance_criteria: 2-4 testable assertions derived from the bug description
that prove the fix works. These must come from the SPEC (what the user reported),
not from guessing the implementation. Example: "calling set_pr_url with a nonexistent
flag name should log a warning message"."""

    try:
        result = _structured_call(INTAKE_MODEL, 1000, IntentAnalysis, prompt)
        state["intent"] = result.model_dump()
    except Exception as e:
        logger.error("Intent translation failed: %s", e)
        state["intent"] = {
            "expected_behavior": work_order.get("title", ""),
            "actual_behavior": work_order.get("description", ""),
            "likely_affected_modules": [],
            "likely_affected_functions": [],
            "fix_type": "bug_fix",
            "severity": work_order.get("priority", "medium"),
        }

    # Extract stack trace hints from the ticket description
    description = work_order.get("description", "")
    stack_hints = _extract_stack_trace_hints(description)
    if stack_hints:
        work_order["stack_trace_hints"] = stack_hints
        state["work_order"] = work_order
        hint_summary = "; ".join(
            f"{h['file']} line {h['line']}" + (f" in {h['function']}" if h["function"] else "")
            for h in stack_hints
        )
        logger.info("Stack trace hints extracted (%d): %s", len(stack_hints), hint_summary)
        # Surface hints in intent so localization can use them as high-confidence signals
        intent = state.get("intent", {})
        existing_notes = intent.get("notes", "")
        stack_note = "Stack trace points to: " + "; ".join(
            f"{h['file']} line {h['line']}" + (f" in {h['function']} — start here" if h["function"] else "")
            for h in stack_hints
        )
        intent["notes"] = (existing_notes + "\n" + stack_note).strip() if existing_notes else stack_note
        state["intent"] = intent

    # Extract reproduction steps from the ticket description
    repro_steps = _extract_repro_steps(description)
    if repro_steps:
        work_order["repro_steps"] = repro_steps
        state["work_order"] = work_order
        logger.info("Reproduction steps extracted (%d steps)", len(repro_steps))

    # Classify the bug to flag automation confidence
    ticket_id = work_order.get("ticket_id", "UNKNOWN")
    title = work_order.get("title", "")
    bug_category = _classify_bug_category(title, description)
    work_order["bug_category"] = bug_category
    state["work_order"] = work_order
    logger.info("Bug category: %s for ticket %s", bug_category, ticket_id)
    if bug_category == "C":
        intent = state.get("intent", {})
        existing_notes = intent.get("notes", "")
        cat_c_note = (
            "WARNING: Bug category C detected — likely involves concurrency, performance, "
            "multi-service coordination, or environment-specific issues. Auto-fix success rate "
            "is low; human review strongly recommended."
        )
        intent["notes"] = (existing_notes + "\n" + cat_c_note).strip() if existing_notes else cat_c_note
        state["intent"] = intent

    state["iteration_count"] = 0
    if trace:
        trace.stage_end("intake")
    return state



def _build_kickstart_context(
    repo_name: str,
    repo_path: str | None,
    intent: dict,
    data_dir: Path,
) -> str:
    """Build orientation context for exploration from graph + vector + failure signals.

    Non-prescriptive: orients the agent on the terrain without telling it where the bug is.
    All sections are best-effort — any individual signal failing is silently skipped.
    """
    sections: list[str] = []
    hint_files = [f for f in intent.get("likely_affected_modules", [])[:5] if f]
    hint_functions = [f for f in intent.get("likely_affected_functions", [])[:5] if f]
    bug_query = " ".join(filter(None, [
        intent.get("actual_behavior", ""),
        intent.get("expected_behavior", ""),
    ]))

    # 1. Vector search — semantically similar code to the bug description
    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo_name, data_dir)
        results = embedder.query(bug_query, n_results=5)
        if results:
            lines = []
            for r in results:
                meta = r.get("metadata", {})
                name = meta.get("name") or r.get("id", "?")
                file_ = meta.get("file", "")
                doc = (meta.get("docstring") or "")[:80]
                lines.append(f"  - {file_}::{name}" + (f" — {doc}" if doc else ""))
            sections.append(
                "SEMANTICALLY SIMILAR CODE (vector search on bug description):\n" + "\n".join(lines)
            )
    except Exception:
        pass

    # 2. Graph neighbors of hint area — Neo4j first, fallback to graph.json
    try:
        neighbors: list[str] = []
        queried_via_neo4j = False

        if neo4j_client.is_connected() and (hint_files or hint_functions):
            try:
                rows = neo4j_client.run(
                    "MATCH (a)-[r:CALLS|IMPORTS]->(b) "
                    "WHERE (a.path IN $files OR b.path IN $files "
                    "       OR a.name IN $funcs OR b.name IN $funcs) "
                    "RETURN a.path AS src_file, a.name AS src_name, "
                    "       b.path AS tgt_file, b.name AS tgt_name, type(r) AS rel "
                    "LIMIT 15",
                    {"files": hint_files, "funcs": hint_functions},
                )
                if rows:
                    queried_via_neo4j = True
                    for row in rows:
                        src = f"{row.get('src_file', '?')}::{row.get('src_name', '?')}"
                        tgt = f"{row.get('tgt_file', '?')}::{row.get('tgt_name', '?')}"
                        neighbors.append(f"  - {src} —[{row.get('rel','CALLS')}]→ {tgt}")
            except Exception:
                pass

        if not queried_via_neo4j:
            graph_data, _ = _load_graph_data(repo_name)
            for f in _find_callers_from_graph(graph_data, hint_files, hint_functions)[:8]:
                neighbors.append(f"  - {f}")

        if neighbors:
            sections.append(
                "GRAPH NEIGHBORS of hint area (callers / callees):\n" + "\n".join(neighbors[:12])
            )
    except Exception:
        pass

    # 3. Past failures scoped to hint files
    try:
        failure_lines: list[str] = []
        if neo4j_client.is_connected() and hint_files:
            for file_path in hint_files[:3]:
                rows = neo4j_client.run(
                    "MATCH (fr:FailureRecord)-[:RESULTED_IN_CHANGE]->(n) "
                    "WHERE (n:Function OR n:File) "
                    "  AND n.path ENDS WITH $file AND fr.repo = $repo "
                    "RETURN fr.message AS message, fr.date AS date, fr.issue_ref AS ref "
                    "ORDER BY fr.date DESC LIMIT 3",
                    {"file": file_path, "repo": repo_name},
                )
                for row in rows:
                    ref = f" ({row['ref']})" if row.get("ref") else ""
                    failure_lines.append(
                        f"  - [{row.get('date', '?')}]{ref} {row.get('message', '')[:120]}"
                    )
        if failure_lines:
            sections.append("PAST FAILURES in hint area:\n" + "\n".join(failure_lines))
    except Exception:
        pass

    # 4. Business rules linked to hint files/functions
    try:
        rules_path = data_dir / repo_name / "business_rules.json"
        if rules_path.exists():
            all_rules = json.loads(rules_path.read_text())
            relevant = [
                r for r in all_rules
                if any(
                    h in r.get("file", "") or h in r.get("function_id", "")
                    for h in hint_files + hint_functions
                )
            ][:8]
            if relevant:
                lines = [
                    f"  - [{r.get('severity', '?').upper()}] {r.get('description', '')[:120]}"
                    for r in relevant
                ]
                sections.append(
                    "BUSINESS RULES (linked to hint area — keep in mind while exploring):\n"
                    + "\n".join(lines)
                )
    except Exception:
        pass

    # 5. PageRank hotspots — most central functions in this repo (orientation map)
    try:
        graph_data, _ = _load_graph_data(repo_name)
        hotspots = sorted(
            [n for n in graph_data.get("nodes", [])
             if n.get("type") == "function" and n.get("pagerank", 0) > 0],
            key=lambda n: n.get("pagerank", 0),
            reverse=True,
        )[:8]
        if hotspots:
            lines = [
                f"  - {h.get('id', '?')} (rank: {h.get('pagerank', 0):.3f})"
                for h in hotspots
            ]
            sections.append(
                "REPO HOTSPOTS (most central functions — not necessarily the bug):\n"
                + "\n".join(lines)
            )
    except Exception:
        pass

    # 6. Git recency for hint files
    try:
        if repo_path and hint_files:
            recent: list[str] = []
            for f in hint_files:
                score = _get_file_recency_score(repo_path, f)
                if score >= 2.0:
                    label = "last week" if score >= 3.0 else "last month"
                    recent.append(f"  - {f} (changed in {label})")
            if recent:
                sections.append("RECENTLY CHANGED (higher bug probability):\n" + "\n".join(recent))
    except Exception:
        pass

    if not sections:
        return ""
    return (
        "\n\nORIENTATION (starting map — explore freely, don't be constrained by this):\n\n"
        + "\n\n".join(sections)
    )


def exploration_node(state: AgentState) -> AgentState:
    """Stage 2: Agentic exploration — agent uses tools to find the bug itself.

    Kick-started with graph + vector + failure signals as orientation context,
    then explores freely with grep/read/search tools.
    """
    _thread_local.current_stage = "exploration"
    trace = _get_trace()
    if trace:
        trace.stage_start("exploration")
    logger.info("=== EXPLORATION: Agent actively exploring the codebase ===")
    state["status"] = PipelineStatus.EXPLORING
    _report_progress(state)

    work_order = state.get("work_order", {})
    intent = state.get("intent", {})
    repo_name = work_order.get("repo_name", "")
    repo_path = _resolve_repo_path(work_order)

    if not repo_path:
        logger.warning("No repo_path in work order — exploration requires a local repo path")
        state["localization"] = {
            "fault_files": [],
            "fault_functions": intent.get("likely_affected_functions", []),
            "fault_classes": [],
            "root_cause_hypothesis": "Could not explore: no repo_path provided.",
            "confidence": 0.0,
            "evidence": [],
        }
        return state

    # Set per-run context for tools
    from agent.explore_tools import set_context, ALL_TOOLS
    set_context(repo_name, repo_path, DATA_DIR)

    # Build orientation context from graph + vector + failure signals
    kickstart_context = _build_kickstart_context(
        repo_name, str(repo_path), intent, DATA_DIR
    )

    system_prompt = f"""You are debugging a production bug in repo `{repo_name}` at `{repo_path}`.

You have tools: grep_repo, read_file, read_function, list_files, search_code, get_function_info, get_file_structure.

A successful exploration ends with you writing a summary that contains:
- The exact fault file path(s) and function name(s)
- A root cause hypothesis explaining WHY the bug occurs
- The relevant source code of the buggy function(s) and their callers

Stop exploring as soon as you have enough evidence. Do not read files you don't need.
{kickstart_context}
"""

    user_message = f"""Bug: {work_order.get('title', '')}
{work_order.get('description', '')}
Component: {work_order.get('affected_component', 'unknown')}

Likely location: {intent.get('likely_affected_modules', [])} / {intent.get('likely_affected_functions', [])}
Actual behavior: {intent.get('actual_behavior', '')}
Expected behavior: {intent.get('expected_behavior', '')}

Find the bug and write your findings summary."""

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        timeout=120.0,
    ).bind_tools(ALL_TOOLS)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    exploration_log = []
    source_code = {}
    MAX_TOOL_CALLS = 30

    tool_call_count = 0
    _exploration_deadline = time.monotonic() + 600  # 10-minute hard cap
    while tool_call_count < MAX_TOOL_CALLS:
        if time.monotonic() > _exploration_deadline:
            logger.warning("Exploration hit 10-minute wall-clock time limit after %d tool calls", tool_call_count)
            break
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("Exploration LLM call failed: %s", e)
            break

        messages.append(response)

        # No more tool calls — agent is done exploring
        if not response.tool_calls:
            logger.info("Exploration complete after %d tool calls", tool_call_count)
            break

        # Execute each tool call
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]
            tool_call_count += 1

            logger.info("Exploration tool call %d/%d: %s(%s)",
                        tool_call_count, MAX_TOOL_CALLS, tool_name, str(tool_args)[:100])

            _emit_trace("tool_call", {
                "tool_name": tool_name,
                "args": tool_args,
                "call_number": tool_call_count,
                "max_calls": MAX_TOOL_CALLS,
            })

            # Find and invoke the tool
            result_str = f"Tool '{tool_name}' not found"
            tool_t0 = time.monotonic()
            for t in ALL_TOOLS:
                if t.name == tool_name:
                    try:
                        result_str = t.invoke(tool_args)
                    except Exception as te:
                        result_str = f"Tool error: {te}"
                    break
            tool_duration = round((time.monotonic() - tool_t0) * 1000)

            _emit_trace("tool_result", {
                "tool_name": tool_name,
                "duration_ms": tool_duration,
                "result_preview": str(result_str)[:500],
                "result_full": str(result_str),
            })

            # Log the tool call
            exploration_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result_preview": str(result_str)[:200],
            })

            # If the tool read file content, store it in source_code
            if tool_name in ("read_file", "read_function") and "ERROR" not in str(result_str):
                file_path = tool_args.get("file_path", "")
                if file_path and file_path not in source_code:
                    source_code[file_path] = result_str

            # If the agent made a direct edit via string_replace, record the patch
            if tool_name == "string_replace" and "OK:" in str(result_str):
                file_path = tool_args.get("file_path", "")
                old_str = tool_args.get("old_string", "")
                new_str = tool_args.get("new_string", "")
                if file_path and old_str and new_str:
                    exploration_log.append({
                        "tool": "patch_recorded",
                        "file": file_path,
                        "note": "Agent applied string_replace directly during exploration",
                    })

            messages.append(ToolMessage(content=str(result_str), tool_call_id=tool_id))

        if tool_call_count >= MAX_TOOL_CALLS:
            logger.warning("Exploration hit %d tool call limit", MAX_TOOL_CALLS)
            break

    # Extract agent's final summary from last non-tool-call message
    final_summary = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            final_summary = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Dual-signal localization: merge agent findings with embedding-based file retrieval
    # (Agentless-style: LLM picks files + semantic search picks files → union → higher recall)
    embedding_files: list[str] = []
    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo_name, DATA_DIR)
        info = embedder.collection_info()
        if info.get("count", 0) > 0:
            query = f"{intent.get('actual_behavior', '')} {intent.get('expected_behavior', '')}"
            emb_results = embedder.query(text=query, n_results=5)
            for r in emb_results:
                fpath = r.get("metadata", {}).get("file", "")
                if fpath and fpath not in embedding_files:
                    embedding_files.append(fpath)
            logger.info("Embedding dual-signal: found %d candidate files", len(embedding_files))
    except Exception as emb_err:
        logger.debug("Embedding dual-signal unavailable: %s", emb_err)

    # Parse fault locations from the summary using a structured call
    if final_summary or source_code:
        try:
            embedding_hint = ""
            if embedding_files:
                embedding_hint = f"\nSEMANTIC SEARCH also suggests these files as relevant:\n{embedding_files}"

            parse_prompt = f"""Based on this exploration summary, extract the fault location.

EXPLORATION SUMMARY:
{final_summary[:3000]}

FILES READ DURING EXPLORATION:
{list(source_code.keys())}
{embedding_hint}

Extract the most likely fault location.
IMPORTANT: fault_files must contain ONLY source files where the bug lives — NOT test files,
config files, or documentation. Test files (test_*.py, conftest.py) are never fault locations."""
            loc = _structured_call("claude-sonnet-4-6", 800, LocalizationResult, parse_prompt)

            # Merge: ensure embedding-suggested files appear in fault_files if relevant
            merged_fault_files = list(loc.fault_files)
            for ef in embedding_files:
                if ef not in merged_fault_files and len(merged_fault_files) < 5:
                    # Only add if not already covered
                    merged_fault_files.append(ef)

            # Filter out test/config files from fault_files — they're never the fault location
            _noise_patterns = (
                "test_", "conftest", "/tests/", "/test/", "/__pycache__/",
                ".json", ".md", ".yml", ".yaml", ".txt", ".cfg", ".ini",
                "setup.py", "setup.cfg", "pyproject.toml",
            )
            clean_fault_files = [
                f for f in merged_fault_files
                if not any(p in f for p in _noise_patterns)
            ]
            # Also filter embedding files added to the merge
            if not clean_fault_files:
                # Fallback: keep LLM-selected files (pre-merge), also filtered
                clean_fault_files = [
                    f for f in loc.fault_files
                    if not any(p in f for p in _noise_patterns)
                ]
            if not clean_fault_files:
                clean_fault_files = list(loc.fault_files)[:3]

            loc_dict = loc.model_dump()
            loc_dict["fault_files"] = clean_fault_files
            state["localization"] = loc_dict
            logger.info("Exploration localization: confidence=%.2f files=%s (pre-filter: %s)",
                        loc.confidence, clean_fault_files, merged_fault_files)
        except Exception as e:
            logger.warning("Could not parse localization from exploration: %s", e)
            state["localization"] = {
                "fault_files": list(source_code.keys())[:3] or embedding_files[:3],
                "fault_functions": intent.get("likely_affected_functions", []),
                "fault_classes": [],
                "root_cause_hypothesis": final_summary[:500] if final_summary else "See exploration log",
                "confidence": 0.5,
                "evidence": [f"Explored {tool_call_count} code locations"],
            }

    state["context"] = final_summary
    state["source_code"] = source_code
    state["exploration_log"] = exploration_log
    state["context_nodes"] = tool_call_count

    logger.info("Exploration done: %d tool calls, %d files read, summary_len=%d",
                tool_call_count, len(source_code), len(final_summary))
    if trace:
        trace.stage_end("exploration")
    return state


def _get_file_recency_score(repo_path: str | None, file_path: str) -> float:
    """Return a recency multiplier for localization: recently changed files are 3x more likely to contain bugs."""
    if not repo_path:
        return 1.0
    try:
        import subprocess
        from datetime import datetime, timezone, timedelta
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%ai", "-1", "--", file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse ISO date like "2026-03-15 14:23:01 +0000"
            date_str = result.stdout.strip().split('\n')[0]
            # Try to parse
            try:
                last_modified = datetime.fromisoformat(date_str.replace(' +0000', '+00:00').replace(' -0000', '+00:00'))
                age_days = (datetime.now(timezone.utc) - last_modified).days
                if age_days <= 7:
                    return 3.0   # Changed in last week — very likely culprit
                elif age_days <= 14:
                    return 2.5
                elif age_days <= 30:
                    return 2.0   # Changed in last month — elevated suspicion
            except Exception:
                pass
    except Exception:
        pass
    return 1.0



def _read_file_safe(file_path: Path, max_lines: int = 500, focus_lines: list[int] | None = None) -> str | None:
    """Read a file safely, skipping binary and truncating long files.

    If focus_lines is given (line numbers of interest), read a window around them
    instead of blindly taking the first max_lines.
    """
    if file_path.suffix.lower() in _BINARY_EXTENSIONS:
        return None
    try:
        content = file_path.read_text()
        content = _redact_secrets(content)
        lines = content.split('\n')

        if len(lines) <= max_lines:
            return content

        # If we have focus lines, build windows around them
        if focus_lines:
            margin = max_lines // (len(focus_lines) + 1)
            margin = max(margin, 80)  # at least 80 lines per window
            selected: set[int] = set()
            for fl in focus_lines:
                start = max(0, fl - margin)
                end = min(len(lines), fl + margin)
                selected.update(range(start, end))
            # Always include the first 30 lines (imports/class def)
            selected.update(range(0, min(30, len(lines))))
            ordered = sorted(selected)
            # Build content with gap markers
            parts = []
            prev = -2
            for idx in ordered:
                if idx > prev + 1:
                    parts.append(f"\n# ... (lines {prev + 2}-{idx} omitted) ...\n")
                parts.append(f"{lines[idx]}")
                prev = idx
            if ordered[-1] < len(lines) - 1:
                parts.append(f"\n# ... ({len(lines) - ordered[-1] - 1} more lines)")
            return '\n'.join(parts)

        # Fallback: first max_lines
        return '\n'.join(lines[:max_lines]) + f"\n# ... truncated ({len(lines) - max_lines} more lines)"
    except (UnicodeDecodeError, Exception):
        return None


def _strip_gap_markers(content: str) -> str:
    """Remove gap-marker lines inserted by _read_file_safe windowing.

    Gap markers look like '# ... (lines X-Y omitted) ...' or '# ... (N more lines)'.
    They are injected for display but must NOT appear in source sent to the repair LLM,
    because the LLM will copy them into original_code causing patch mismatches.
    """
    import re as _re
    lines = content.splitlines()
    cleaned = [l for l in lines if not _re.match(r'\s*#\s*\.\.\.\s*\(lines? ', l)
               and not _re.match(r'\s*#\s*\.\.\.\s*\(\d+ more lines', l)]
    return '\n'.join(cleaned)


def _find_file_in_repo(repo_path: Path, rel_path: str) -> Path | None:
    """Resolve a relative path to an actual file in the repo."""
    candidates = [
        repo_path / rel_path,
        repo_path / "src" / rel_path,
    ]
    if not any(c.exists() for c in candidates):
        matches = list(repo_path.rglob(f"*{Path(rel_path).name}"))
        candidates.extend(matches[:2])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if candidate.suffix.lower() not in _BINARY_EXTENSIONS:
                return candidate
    return None


def _check_graph_staleness(repo_name: str, repo_path: Path | None) -> int:
    """Return number of commits made since graph.json was last built.

    Returns 0 if graph doesn't exist, repo_path is None, or git is unavailable.
    A result > 10 means the blast radius data may miss recently added callers.
    """
    if not repo_path:
        return 0
    try:
        graph_path = DATA_DIR / repo_name / "graph.json"
        if not graph_path.exists():
            return 0
        from datetime import datetime
        graph_mtime = graph_path.stat().st_mtime
        since_dt = datetime.fromtimestamp(graph_mtime).strftime("%Y-%m-%dT%H:%M:%S")
        result = subprocess.run(
            ["git", "log", "--oneline", f"--after={since_dt}"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except Exception:
        pass
    return 0


def _load_graph_data(repo_name: str) -> tuple[dict, dict]:
    """Load graph.json and enriched_nodes.json for a repo."""
    graph_data: dict = {}
    enriched: dict = {}
    try:
        graph_path = DATA_DIR / repo_name / "graph.json"
        if graph_path.exists():
            graph_data = json.loads(graph_path.read_text())
    except Exception as e:
        logger.warning("Failed to load graph.json: %s", e)
    try:
        enriched_path = DATA_DIR / repo_name / "enriched_nodes.json"
        if enriched_path.exists():
            enriched = json.loads(enriched_path.read_text())
    except Exception as e:
        logger.warning("Failed to load enriched_nodes.json: %s", e)
    return graph_data, enriched


def _find_callers_from_graph(graph_data: dict, fault_files: list[str],
                             fault_functions: list[str]) -> list[str]:
    """Use the knowledge graph CALLS/IMPORTS edges to find caller files.

    Returns file paths that call/import the fault files or their functions.
    """
    edges = graph_data.get("edges", [])
    if not edges:
        return []

    # Build set of target node IDs (fault files and their symbols)
    target_ids: set[str] = set()
    for f in fault_files:
        target_ids.add(f)
        # Also match node IDs like "app/service/chat/crest_ai_services.py::CrestAIServices"
        stem = Path(f).stem
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if stem in tgt:
                target_ids.add(tgt)
            if stem in src:
                target_ids.add(src)

    for fn in fault_functions:
        target_ids.add(fn)

    # Find files that CALL or IMPORT target nodes
    caller_files: set[str] = set()
    fault_file_set = set(fault_files)
    _caller_noise = ("test_", "conftest", "/tests/", "/test/", "/__pycache__/")

    for edge in edges:
        etype = edge.get("type", "")
        if etype not in ("CALLS", "IMPORTS"):
            continue
        target = edge.get("target", "")
        source = edge.get("source", "")

        # Check if this edge points TO one of our targets
        if target in target_ids or any(t in target for t in target_ids):
            # Extract file path from source node ID (e.g., "file.py::ClassName.method")
            src_file = source.split("::")[0] if "::" in source else source
            if src_file and src_file not in fault_file_set:
                if not any(pat in src_file.lower() for pat in _caller_noise):
                    caller_files.add(src_file)

    return sorted(caller_files)[:8]


def _find_callers_via_grep(repo_path: Path, fault_files: list[str]) -> list[str]:
    """Fallback: grep for files that import the fault files."""
    caller_paths: list[str] = []
    seen: set[str] = set()
    _caller_noise = ("test_", "conftest", "/tests/", "/test/", "/__pycache__/")

    for rel_path in fault_files:
        stem = Path(rel_path).stem
        parts = Path(rel_path).with_suffix("").parts
        search_terms = [f"import {stem}", f"from {stem}"]
        if len(parts) > 1:
            for i in range(len(parts) - 1):
                search_terms.append(f"from {'.'.join(parts[i:])}")
        try:
            for term in search_terms:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", term, str(repo_path)],
                    capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    p = Path(line)
                    if p.exists() and str(p) not in seen:
                        if any(pat in str(p).lower() for pat in _caller_noise):
                            continue
                        rel = str(p.relative_to(repo_path))
                        if rel not in set(fault_files):
                            seen.add(str(p))
                            caller_paths.append(rel)
                            if len(caller_paths) >= 5:
                                return caller_paths
        except Exception:
            pass
    return caller_paths


def _load_business_rules(repo_name: str, fault_files: list[str]) -> str:
    """Load stored business rules + failure history relevant to the fault files.

    Reads auto-extracted and human-submitted rules from business_rules.json.
    Also queries Neo4j for FailureRecords linked to the fault files when connected.
    Returns a warning string if no context is found (never returns empty silently).
    """
    fault_functions = [Path(f).stem for f in fault_files]
    sections: list[str] = []

    # --- Business rules from flat file ---
    rules_path = DATA_DIR / repo_name / "business_rules.json"
    relevant_rules: list[str] = []
    if rules_path.exists():
        try:
            all_rules = json.loads(rules_path.read_text())
            for rule in all_rules:
                rule_file = rule.get("file", "")
                rule_func = rule.get("function_id", "")
                if any(f in rule_file or f in rule_func for f in fault_files):
                    severity = rule.get("severity", "medium").upper()
                    marker = "⚠️ DO NOT VIOLATE" if severity in ("CRITICAL", "HIGH") else ""
                    relevant_rules.append(
                        f"  [{severity}] {rule.get('description', '')[:300]} {marker}\n"
                        f"    Source: {rule.get('source', 'unknown')} | File: {rule_file}"
                    )
        except Exception:
            pass

    if relevant_rules:
        sections.append(
            "\n\nBUSINESS RULES (verified knowledge base — DO NOT VIOLATE):\n"
            + "\n".join(relevant_rules)
        )

    # --- FailureRecords from Neo4j ---
    try:
        if neo4j_client.is_connected():
            for file_path in fault_files:
                rows = neo4j_client.run(
                    "MATCH (fr:FailureRecord)-[:RESULTED_IN_CHANGE]->(n) "
                    "WHERE (n:Function OR n:File) "
                    "  AND (n.path ENDS WITH $file OR n.name IN $funcs) "
                    "  AND fr.repo = $repo "
                    "RETURN fr.message AS message, fr.date AS date, "
                    "       fr.issue_ref AS issue_ref, fr.severity_hint AS severity "
                    "ORDER BY fr.date DESC LIMIT 5",
                    {"file": file_path, "funcs": fault_functions, "repo": repo_name},
                )
                if rows:
                    failure_lines = []
                    for row in rows:
                        ref = f" ({row['issue_ref']})" if row.get("issue_ref") else ""
                        failure_lines.append(
                            f"  [{row.get('date', '?')}]{ref} {row.get('message', '')[:200]}"
                        )
                    sections.append(
                        f"\n\nPAST FAILURES touching {file_path}:\n"
                        + "\n".join(failure_lines)
                    )
    except Exception as exc:
        logger.debug("FailureRecord query failed (non-fatal): %s", exc)

    if not sections:
        # Inject warning so repair agent knows context is absent
        fault_desc = ", ".join(fault_files[:3]) or "target function"
        return (
            "\n\nWARNING: No business rules or failure history found for "
            f"{fault_desc}. Treat as high-risk — do not remove validation "
            "logic without explicit confirmation."
        )

    return "".join(sections)


def _build_enrichment_context(enriched: dict, fault_files: list[str],
                              fault_functions: list[str]) -> str:
    """Build extra context from enriched_nodes.json (docstrings, params, call info)."""
    sections = []
    for node_id, node in enriched.items():
        node_file = node.get("file", "")
        matches_file = any(f in node_file or node_file.endswith(f) for f in fault_files)
        matches_func = node.get("name", "") in fault_functions

        if not matches_file and not matches_func:
            continue

        ntype = node.get("type", "")
        name = node.get("name", "")
        if ntype == "function":
            raw_params = node.get("params", [])
            params = ", ".join(
                f"{p.get('name', '?')}: {p.get('type', 'Any')}" if isinstance(p, dict) else str(p)
                for p in raw_params
            )
            ret = node.get("return_type", "")
            doc = node.get("llm_summary") or node.get("summary") or node.get("docstring", "")
            calls = node.get("external_calls", [])
            section = f"  def {name}({params}) -> {ret}"
            if doc:
                section += f"\n    '''{doc[:200]}'''"
            if calls:
                section += f"\n    # Calls: {', '.join(calls[:10])}"
            sections.append(section)
        elif ntype == "class":
            methods = node.get("methods", [])
            bases = node.get("inherits", [])
            section = f"  class {name}({', '.join(bases)})"
            if methods:
                section += f"\n    # Methods: {', '.join(methods[:15])}"
            sections.append(section)

    if not sections:
        return ""
    return "\n\nENRICHED SYMBOL INFO (from knowledge graph):\n" + "\n".join(sections)




def _extract_function_source(source: str, function_name: str, context_lines: int = 2) -> str | None:
    """Extract a single named function from source using AST.

    Returns just the function body (plus a few context lines) so the repair
    LLM sees only what it needs to change — not the entire file.
    Falls back to None if AST parse fails or function not found.
    """
    import ast as _ast
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None

    src_lines = source.splitlines()

    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == function_name:
                start = max(0, node.lineno - 1 - context_lines)
                end = min(len(src_lines), getattr(node, "end_lineno", node.lineno) + context_lines)
                return "\n".join(src_lines[start:end])
    return None


def _build_source_section(source_code: dict) -> tuple[str, str]:
    """Build the source code prompt section from loaded source files."""
    source_section = ""
    enrichment_section = ""
    if not source_code:
        return "", ""

    fault_parts = []
    caller_parts = []
    business_rules_section = ""
    for fpath, code in source_code.items():
        if fpath == "__enrichment__":
            enrichment_section = code
            continue
        if fpath == "__business_rules__":
            business_rules_section = code
            continue
        lines = code.split('\n')
        # Fault files may already be pre-extracted to just target functions (short).
        # Callers/requested files truncate more aggressively to keep total tokens down.
        is_caller = "(caller)" in fpath or "(requested)" in fpath
        max_lines = 400 if is_caller else 200
        truncated = '\n'.join(lines[:max_lines])
        if len(lines) > max_lines:
            truncated += f"\n# ... truncated ({len(lines) - max_lines} more lines)"
        if is_caller:
            caller_parts.append(f"\n--- {fpath} ---\n{truncated}\n")
        else:
            fault_parts.append(f"\n--- {fpath} ---\n{truncated}\n")

    if fault_parts:
        source_section += "\n\nFAULT FILES (where the bug lives — add/modify functions here):\n"
        source_section += "".join(fault_parts)
    if caller_parts:
        source_section += "\n\nCALL SITES (where fault file functions are USED — wire your fix in here):\n"
        source_section += "".join(caller_parts)
    if enrichment_section:
        source_section += enrichment_section
    # Business rules go FIRST in the prompt — highest priority context
    if business_rules_section:
        source_section = business_rules_section + "\n" + source_section

    return source_section, enrichment_section


def _verify_and_fix_patches(
    patches: list[dict], source_code: dict, repo_path: Path | None,
    intent: dict, localization: dict, feedback_section: str,
) -> list[dict]:
    """Verify each patch matches the source. If not, re-read the target area and ask the LLM to fix it.

    This is the core agentic loop: try → observe failure → read more context → retry.
    """
    if not repo_path:
        return patches

    verified: list[dict] = []
    failed_patches: list[dict] = []

    for patch in patches:
        file_path = patch.get("file_path", "")
        original = patch.get("original_code", "")
        patched = patch.get("patched_code", "")

        if not file_path or not original or not patched:
            continue
        if original.strip() == patched.strip():
            continue

        # Try to find the file and match
        resolved = _find_file_in_repo(repo_path, file_path)
        if not resolved:
            logger.warning("Patch target not found: %s", file_path)
            failed_patches.append(patch)
            continue

        content = _read_file_safe(resolved, max_lines=10000)
        if not content:
            failed_patches.append(patch)
            continue

        # Test if the patch would apply
        result = _fuzzy_match_replace(content, original, patched)
        if result is not None:
            verified.append(patch)
            logger.info("Patch verified: %s", file_path)
        else:
            logger.warning("Patch does NOT match source in %s — will retry with actual code", file_path)
            failed_patches.append({**patch, "_actual_content": content})

    # For failed patches, re-ask the LLM with the actual file content
    if failed_patches and verified:
        # Some patches worked, some didn't — try to fix the failed ones
        for fp in failed_patches:
            actual = fp.pop("_actual_content", "")
            if not actual:
                continue

            # Try to extract just the target function to send in retry (Research #23)
            fault_fns = localization.get("fault_functions", [])
            fn_extract = ""
            for fn_name in fault_fns:
                fn_src = _extract_function_source(actual, fn_name)
                if fn_src:
                    fn_extract += f"\n\n# function: {fn_name}\n{fn_src}"
            target_section = fn_extract if fn_extract else chr(10).join(actual.split(chr(10))[:150])

            retry_prompt = f"""Your patch for `{fp['file_path']}` did not match. Here is the actual source.

BUG: {intent.get('actual_behavior', '')}
ROOT CAUSE: {localization.get('root_cause_hypothesis', '')}

ACTUAL SOURCE:
{target_section}

Produce a patch where original_code is copied EXACTLY from ACTUAL SOURCE above (start from `def`)."""

            try:
                retry_result = _structured_call("claude-sonnet-4-6", 4000, RepairResult, retry_prompt)
                for rp in retry_result.patches:
                    rp_dict = rp.model_dump()
                    if rp_dict.get("original_code", "").strip() != rp_dict.get("patched_code", "").strip():
                        # Verify the retry patch matches
                        if _fuzzy_match_replace(actual, rp_dict["original_code"], rp_dict["patched_code"]) is not None:
                            rp_dict["file_path"] = fp["file_path"]
                            verified.append(rp_dict)
                            logger.info("Retry patch verified: %s", fp["file_path"])
                            break
                        else:
                            logger.warning("Retry patch also failed to match: %s", fp["file_path"])
            except Exception as e:
                logger.warning("Retry patch generation failed for %s: %s", fp["file_path"], e)

    elif not verified and failed_patches:
        # ALL patches failed — do a full retry with actual file contents
        logger.warning("All %d patches failed to match — full agentic retry", len(failed_patches))
        extra_source = {}
        for fp in failed_patches:
            actual = fp.pop("_actual_content", "")
            if actual:
                extra_source[fp["file_path"]] = actual

        if extra_source:
            # Build a new source section with the actual content
            combined = dict(source_code)
            for fpath, content in extra_source.items():
                combined[fpath] = content  # Replace with full actual content

            # Build focused section from actual file content (Research #23)
            fault_fns = localization.get("fault_functions", [])
            focused_actual: dict = {}
            for fpath, content in extra_source.items():
                clean = _strip_gap_markers(content)
                if fault_fns:
                    parts = [src for fn in fault_fns if (src := _extract_function_source(clean, fn))]
                    focused_actual[fpath] = "\n\n".join(parts) if parts else clean
                else:
                    focused_actual[fpath] = clean
            for fpath, content in combined.items():
                if fpath not in focused_actual:
                    focused_actual[fpath] = content
            source_section, _ = _build_source_section(focused_actual)
            retry_prompt = f"""All patches failed to match. Below is the ACTUAL source re-read from disk.

BUG: {intent.get('actual_behavior', '')}
ROOT CAUSE: {localization.get('root_cause_hypothesis', '')}
TARGET FUNCTIONS: {fault_fns} in {localization.get('fault_files', [])}
{feedback_section}
{source_section}

Produce patches where original_code starts from the `def` line and is an EXACT substring of the source above."""

            try:
                retry_result = _structured_call("claude-sonnet-4-6", 8000, RepairResult, retry_prompt)
                for rp in retry_result.patches:
                    rp_dict = rp.model_dump()
                    orig = rp_dict.get("original_code", "").strip()
                    patched_c = rp_dict.get("patched_code", "").strip()
                    if orig == patched_c:
                        continue
                    # Verify against the actual file content we sent
                    fp_key = rp_dict.get("file_path", "")
                    actual = extra_source.get(fp_key, "")
                    if actual and _fuzzy_match_replace(actual, rp_dict["original_code"], rp_dict["patched_code"]) is not None:
                        verified.append(rp_dict)
                        logger.info("Full retry patch verified: %s", fp_key)
                    elif actual:
                        logger.warning("Full retry patch still does not match: %s", fp_key)
                    else:
                        verified.append(rp_dict)  # No content to verify against
                if verified:
                    logger.info("Full retry produced %d patches", len(verified))
            except Exception as e:
                logger.warning("Full retry failed: %s", e)

    return verified


def _generate_stub_tests(
    intent: dict, localization: dict, target_files: list[str],
) -> list[dict]:
    """Generate deterministic stub test patches from acceptance criteria and
    function signatures when the LLM fails to produce test_patches.

    Returns a list of patch dicts suitable for repair_dump["test_patches"].
    """
    acceptance = intent.get("acceptance_criteria", [])
    fault_fns = localization.get("fault_functions", [])
    if not fault_fns:
        return []

    # Derive test file path from first target file
    first_file = target_files[0] if target_files else "unknown.py"
    stem = Path(first_file).stem
    test_file = f"tests/test_{stem}.py"

    # Build stub test code from acceptance criteria
    test_lines = [
        f'"""Auto-generated stub tests for {stem} — acceptance criteria."""',
        "import pytest",
        "",
    ]
    for i, criterion in enumerate(acceptance):
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", criterion[:60]).strip("_").lower()
        test_lines.extend([
            f"def test_acceptance_{i + 1}_{safe_name}():",
            f'    """Verify: {criterion}"""',
            f"    # TODO: replace with real assertion",
            f'    pytest.skip("Stub test — implement assertion for: {criterion}")',
            "",
        ])

    # Fallback: if no acceptance criteria, generate from function names
    if not acceptance:
        for fn_name in fault_fns:
            test_lines.extend([
                f"def test_{fn_name}_fixed():",
                f'    """Verify {fn_name} works correctly after fix."""',
                f"    # TODO: replace with real assertion",
                f'    pytest.skip("Stub test — implement assertion for {fn_name}")',
                "",
            ])

    return [{
        "file_path": test_file,
        "original_code": "",
        "patched_code": "\n".join(test_lines),
        "explanation": "Auto-generated stub tests from acceptance criteria (LLM failed to produce tests)",
    }]


def _find_similar_fixes(repo_name: str, fault_files: list[str], fault_functions: list[str]) -> list[dict]:
    """Load fix_history.json and find the most similar past fixes."""
    try:
        fixes_path = DATA_DIR / repo_name / "fix_history.json"
        if not fixes_path.exists():
            return []
        history = json.loads(fixes_path.read_text())
        if not history:
            return []

        # Score each past fix by overlap with current fault location
        fault_file_stems = {Path(f).stem for f in fault_files}
        fault_func_set = set(fault_functions)

        scored = []
        for fix in history[-50:]:  # Check last 50 fixes
            fix_file_stems = {Path(f).stem for f in fix.get("fault_files", [])}
            fix_func_set = set(fix.get("fault_functions", []))

            file_overlap = len(fault_file_stems & fix_file_stems)
            func_overlap = len(fault_func_set & fix_func_set)
            score = file_overlap * 2 + func_overlap * 3

            if score > 0:
                scored.append((score, fix))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fix for _, fix in scored[:3]]
    except Exception as e:
        logger.debug("Failed to load fix history: %s", e)
        return []


def repair_node(state: AgentState) -> AgentState:
    """Stage 4: Agentic repair with per-file verification and sequential patching.

    Architecture (based on SWE-Agent/Agentless best practices):
    1. Generate all patches in one call (overview of the fix)
    2. Deduplicate overlapping patches for the same file
    3. Apply patches sequentially per file (prevents conflicts)
    4. Verify each patch against actual source on disk
    5. If patches fail to match, re-read the file and retry
    6. If LLM needs more files, read them and re-run
    7. Syntax-check is done in test_node before commit
    """
    _thread_local.current_stage = "repair"
    trace = _get_trace()
    if trace:
        trace.stage_start("repair")
    logger.info("=== REPAIR: Generating fix (iteration %d) ===", state.get("iteration_count", 0) + 1)
    state["status"] = PipelineStatus.REPAIRING
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    _report_progress(state)

    intent = state.get("intent", {})
    localization = state.get("localization", {})
    source_code = state.get("source_code", {})
    previous_review = state.get("review", {})
    work_order = state.get("work_order", {})
    repo_path = _resolve_repo_path(work_order)

    # Inject the full existing test file for each fault file so the agent
    # can ADD tests to it rather than replace it entirely.
    if repo_path:
        tests_dir = repo_path / "tests"
        if not tests_dir.exists():
            tests_dir = repo_path / "test"
        if tests_dir.exists():
            fault_files = localization.get("fault_files", [])
            source_code = dict(source_code)
            for fp in fault_files:
                stem = Path(fp).stem
                candidate = tests_dir / f"test_{stem}.py"
                key = f"tests/test_{stem}.py (EXISTING TEST FILE — preserve all tests, only ADD new ones)"
                if candidate.exists() and key not in source_code:
                    content = _read_file_safe(candidate, max_lines=5000)
                    if content:
                        source_code[key] = content

    # ── Phase 2B: Blast radius — load caller files for multi-file awareness ──
    repo_name_for_callers = work_order.get("repo_name", "")
    fault_files_for_callers = localization.get("fault_files", [])
    fault_fns_for_callers = localization.get("fault_functions", [])
    caller_files_loaded: list[str] = []
    if repo_path and fault_files_for_callers and repo_name_for_callers:
        # Staleness check: warn if graph is > 10 commits old
        stale = _check_graph_staleness(repo_name_for_callers, repo_path)
        if stale > 10:
            logger.warning(
                "Graph for %s is %d commits stale — blast radius may miss recent callers. "
                "Re-index: python -m agent.graph.build --repo=%s",
                repo_name_for_callers, stale, repo_path,
            )
        graph_data_c, _ = _load_graph_data(repo_name_for_callers)
        caller_paths = _find_callers_from_graph(
            graph_data_c, fault_files_for_callers, fault_fns_for_callers
        )
        if not caller_paths:
            caller_paths = _find_callers_via_grep(repo_path, fault_files_for_callers)
        source_code = dict(source_code)  # ensure mutable copy if not already
        fault_file_set = set(fault_files_for_callers)
        for cp in caller_paths[:5]:
            key = f"{cp} (caller)"
            if key in source_code or cp in fault_file_set:
                continue
            resolved = _find_file_in_repo(repo_path, cp)
            if not resolved:
                resolved_direct = repo_path / cp
                resolved = resolved_direct if resolved_direct.exists() else None
            if resolved and resolved.exists():
                content = _read_file_safe(resolved, max_lines=400)
                if content:
                    source_code[key] = content
                    caller_files_loaded.append(cp)
                    logger.info("Loaded caller file for blast radius: %s", cp)
    state["caller_files"] = caller_files_loaded

    # Build focused source section — send only target functions, not entire files (Research #23, #4)
    fault_functions = localization.get("fault_functions", [])
    focused_source: dict = {}
    for fpath, code in source_code.items():
        is_caller = "(caller)" in fpath or "(requested)" in fpath
        is_test = "EXISTING TEST FILE" in fpath
        if is_caller or is_test:
            focused_source[fpath] = code  # callers + test files stay full
            continue
        # For fault files: extract just the target functions
        if fault_functions:
            clean = _strip_gap_markers(code)
            extracted_parts: list[str] = []
            for fn_name in fault_functions:
                fn_src = _extract_function_source(clean, fn_name)
                if fn_src:
                    extracted_parts.append(fn_src)
                    logger.info("Extracted function %s from %s (%d chars)", fn_name, fpath, len(fn_src))
            if extracted_parts:
                focused_source[fpath] = "\n\n".join(extracted_parts)
                continue
        focused_source[fpath] = _strip_gap_markers(code)  # fallback: full file, no gap markers

    source_section, _ = _build_source_section(focused_source)
    if not source_section:
        ctx = state.get("context", "")
        source_section = f"\n\nCODEBASE CONTEXT (summaries only):\n{ctx[:6000]}"

    # Include review feedback + test failure output on retry
    feedback_section = ""
    if previous_review.get("feedback"):
        feedback_section = f"\nPREVIOUS REVIEW FEEDBACK:\n{previous_review['feedback'][:500]}\n"
        # Mandatory test enforcement: when reviewer flagged TESTS as FAIL,
        # explicitly require test_patches on retry (Finding: autoplan eng review)
        review_checks = previous_review.get("checks", [])
        tests_failed = any(
            (c.get("name") == "TESTS" and c.get("status") == "FAIL")
            for c in review_checks
        )
        if tests_failed:
            feedback_section += (
                "\nMANDATORY: You MUST produce test_patches in your response. "
                "The reviewer rejected your previous fix because it had NO tests. "
                "If you return empty test_patches again, the fix will be rejected. "
                "Generate at least one test that verifies the fixed behavior.\n"
            )
    test_result = state.get("test_result", "")
    if test_result and "fail" in test_result.lower():
        feedback_section += f"\nTEST FAILURE (fix your tests):\n{test_result[:1000]}\n"

    # Acceptance criteria from spec (Finding #21: verification from spec, not implementation)
    acceptance = intent.get("acceptance_criteria", [])
    criteria_section = ""
    if acceptance:
        criteria_section = "\nACCEPTANCE CRITERIA (your tests MUST verify these):\n"
        criteria_section += "\n".join(f"  - {c}" for c in acceptance)

    # Deduplicate fault_files by basename — exploration sometimes returns
    # both 'agent/feature_flags.py' and 'backend/agent/feature_flags.py'
    # for the same file. Keep the longest (most qualified) path per basename.
    raw_fault_files = localization.get('fault_files', [])
    by_basename: dict[str, str] = {}
    for fp in raw_fault_files:
        base = Path(fp).name
        if base not in by_basename or len(fp) > len(by_basename[base]):
            by_basename[base] = fp
    target_files = list(by_basename.values())

    # Build explicit completeness enforcement for multi-function fixes
    target_fns = localization.get('fault_functions', [])
    n_targets = len(target_fns)
    completeness_rule = ""
    if n_targets > 1:
        fn_list = ", ".join(f"`{fn}`" for fn in target_fns)
        completeness_rule = (
            f"\nCRITICAL — ALL {n_targets} TARGET FUNCTIONS MUST BE PATCHED: {fn_list}\n"
            f"The root cause affects every one of them. A fix that patches only some is INCOMPLETE\n"
            f"and will be rejected. You MUST produce one patch entry for EACH target function.\n"
        )

    # Multi-file caller awareness (Phase 2B-1)
    caller_instruction = ""
    if caller_files_loaded:
        caller_list = ", ".join(f"`{c}`" for c in caller_files_loaded[:5])
        caller_instruction = (
            f"\nMULTI-FILE AWARENESS: The CALL SITES shown above ({caller_list}) call the modified "
            f"functions. If your fix changes a function signature, adds/removes parameters, or "
            f"renames something, you MUST ALSO produce patches for those caller files. "
            f"If your fix is purely internal (same public interface), you can omit caller patches.\n"
        )

    # Load past fixes in the same area to guide repair (Change 3: fix history matching)
    repo_name = work_order.get("repo_name", "")
    fix_history_context = ""
    if repo_name:
        similar_fixes = _find_similar_fixes(repo_name, localization.get("fault_files", []), localization.get("fault_functions", []))
        if similar_fixes:
            fix_history_context = "\n\nPAST FIXES IN THE SAME AREA:\n"
            for fix in similar_fixes:
                fix_history_context += (
                    f"- Ticket {fix.get('ticket_id', '?')}: {fix.get('root_cause', '')}\n"
                    f"  Fix approach: {fix.get('fix_summary', '')}\n"
                )
            fix_history_context += "\nUse these as reference — similar bugs in the same files may share root causes.\n"

    # End-state prompt style (Research Finding #10 — Claude performs better with end-state descriptions)
    prompt = f"""Fix this bug by producing a RepairResult with correct patches.

BUG: {intent.get('actual_behavior', '')}
EXPECTED: {intent.get('expected_behavior', '')}
ROOT CAUSE: {localization.get('root_cause_hypothesis', '')}
TARGET FUNCTIONS: {target_fns} in {target_files}
{completeness_rule}
{caller_instruction}
{criteria_section}
{feedback_section}
{fix_history_context}
{source_section}

A correct RepairResult has:
- `patches`: one entry per target function — you MUST patch EVERY function in TARGET FUNCTIONS.
  - `original_code`: copy the ENTIRE function EXACTLY as shown above, starting from the `def` line
    through the LAST line of the function body. Include ALL lines — do NOT use a single-line snippet
    like `slug[:8]`. The original_code MUST be at least 3 lines. Character-for-character exact copy.
  - `patched_code`: the corrected ENTIRE function. Must differ from original_code.
  - `file_path`: exact file path as shown above.
- `test_patches`: adds tests WITHOUT removing existing ones.
  - If the test file is shown above ("EXISTING TEST FILE"):
    * `original_code`: the last ~5 lines of that file (verbatim, for unique matching)
    * `patched_code`: those same lines + new test functions appended at the end
  - If test file does not exist: `original_code` = "", `patched_code` = full new file.
  - Use `_save_flags` (NOT `_write_flags` — does not exist). Use `import agent.X as X` style.
- `explanation`: one sentence on what was wrong and how the patch fixes it.
- `needs_more_files`: list paths you need to see before you can produce patches.

The fix is complete when:
1. original_code is an EXACT substring of the source shown (start from `def`)
2. patched_code addresses the stated root cause
3. EVERY target function has a corresponding patch (no partial fixes)
4. new tests cover the fixed behaviour and the existing ones still pass

IMPORTANT: Generate 3 alternative patches (patch_alternatives) in order from most to least confident.
The first should be your primary fix. The second should be an alternative approach (e.g., fix at a different layer).
The third should be the most conservative fix (minimal change).

If one approach doesn't work, the system will try the next."""

    MAX_FILE_REQUESTS = 2

    try:
        current_source = dict(source_code)
        current_prompt = prompt
        repair_dump = {}
        raw_patches = []

        for file_round in range(MAX_FILE_REQUESTS + 1):
            result = _structured_call("claude-sonnet-4-6", 8000, RepairResult, current_prompt)
            repair_dump = result.model_dump()

            raw_patches = [
                p for p in repair_dump.get("patches", [])
                if p.get("original_code", "").strip() != p.get("patched_code", "").strip()
            ]

            # Handle file requests
            needs_files = repair_dump.get("needs_more_files", [])
            if needs_files and repo_path and file_round < MAX_FILE_REQUESTS:
                logger.info("LLM requests %d more files: %s", len(needs_files), needs_files)
                new_count = 0
                for req_path in needs_files[:5]:
                    if req_path in current_source:
                        continue
                    resolved = _find_file_in_repo(repo_path, req_path)
                    if resolved:
                        content = _read_file_safe(resolved, max_lines=500)
                        if content:
                            current_source[f"{req_path} (requested)"] = content
                            new_count += 1
                            logger.info("Read requested file: %s", req_path)

                if new_count > 0:
                    source_section, _ = _build_source_section(current_source)
                    current_prompt = f"""Fix the bug. Additional files have been loaded per your request.

BUG: {intent.get('actual_behavior', '')}
ROOT CAUSE: {localization.get('root_cause_hypothesis', '')}
{feedback_section}
{source_section}

Produce patches where original_code is an EXACT substring of source above (start from `def` line).
patched_code must fix the stated root cause."""
                    state["source_code"] = current_source
                    continue

            break

        # Retry if no patches — use a more targeted approach
        if not raw_patches and focused_source:
            logger.warning("No patches on first try — retrying with targeted prompt")
            explanation = repair_dump.get("explanation", "")
            fault_files_list = localization.get("fault_files") or [""]
            fault_fns_list = localization.get("fault_functions") or [""]
            fault_file = fault_files_list[0] if fault_files_list else ""
            fault_fn = fault_fns_list[0] if fault_fns_list else ""

            retry_prompt = f"""Patches array was empty. You must produce patches.

BUG: {intent.get('actual_behavior', '')}
TARGET: function `{fault_fn}` in file `{fault_file}`
Your analysis: {explanation[:200]}
{feedback_section}
{source_section}

The correct patch has:
- file_path: "{fault_file}"
- original_code: copy the ENTIRE `def {fault_fn}(...)` function from the source above,
  starting from `def` through the last line. Do NOT use a single-line snippet.
- patched_code: the corrected version of that entire function
- test_patches: add tests that verify the fix works

Produce the RepairResult with this patch now."""

            result2 = _structured_call("claude-sonnet-4-6", 8000, RepairResult, retry_prompt)
            repair_dump2 = result2.model_dump()
            raw_patches = [
                p for p in repair_dump2.get("patches", [])
                if p.get("original_code", "").strip() != p.get("patched_code", "").strip()
            ]
            if raw_patches:
                # Keep the original explanation but use new patches
                repair_dump2["explanation"] = repair_dump.get("explanation", repair_dump2.get("explanation", ""))
                repair_dump = repair_dump2
                logger.info("Targeted retry produced %d patches", len(raw_patches))

        if raw_patches:
            # Step 1: Deduplicate
            raw_patches = _deduplicate_patches(raw_patches)
            logger.info("After dedup: %d patches", len(raw_patches))

            # Step 2: Verify against actual source + retry mismatches
            if repo_path:
                verified = _verify_and_fix_patches(
                    raw_patches, current_source, repo_path, intent, localization, feedback_section,
                )
                logger.info("After verify: %d/%d patches", len(verified), len(raw_patches))
            else:
                verified = raw_patches

            # Dedup again after verify — retry merge can reintroduce duplicates
            if verified:
                verified = _deduplicate_patches(verified)
                logger.info("After verify+dedup: %d patches", len(verified))
            repair_dump["patches"] = verified

            # patch_alternatives fallback: if primary patches all failed to verify,
            # try each alternative set in order until one produces verified patches.
            if not verified:
                alternatives = repair_dump.get("patch_alternatives") or []
                for alt_idx, alt in enumerate(alternatives[:2]):
                    alt_patches = alt.get("patches", []) if isinstance(alt, dict) else []
                    alt_patches = [
                        p for p in alt_patches
                        if p.get("original_code", "").strip() != p.get("patched_code", "").strip()
                    ]
                    if not alt_patches:
                        continue
                    alt_patches = _deduplicate_patches(alt_patches)
                    if repo_path:
                        alt_verified = _verify_and_fix_patches(
                            alt_patches, current_source, repo_path, intent, localization, feedback_section,
                        )
                    else:
                        alt_verified = alt_patches
                    if alt_verified:
                        logger.info(
                            "patch_alternatives[%d] succeeded with %d patches (primary failed)",
                            alt_idx, len(alt_verified),
                        )
                        repair_dump["patches"] = alt_verified
                        verified = alt_verified
                        break
                    logger.info("patch_alternatives[%d] also failed to apply", alt_idx)

            # Trace: emit each patch candidate
            for p in verified:
                _emit_trace("patch_candidate", {
                    "file_path": p.get("file_path", ""),
                    "explanation": p.get("explanation", ""),
                    "has_original": bool(p.get("original_code")),
                    "has_patched": bool(p.get("patched_code")),
                })

        # Programmatic stub test fallback: if LLM produced patches but no
        # test_patches after 2+ iterations, generate stubs from acceptance
        # criteria and function signatures (autoplan eng review finding 5A)
        test_patches = repair_dump.get("test_patches", [])
        has_tests = any(
            tp.get("patched_code", "").strip()
            for tp in test_patches
        )
        if not has_tests and state.get("iteration_count", 0) >= 2 and repair_dump.get("patches"):
            logger.warning("No test_patches after %d iterations — generating stub tests", state["iteration_count"])
            stub_tests = _generate_stub_tests(intent, localization, target_files)
            if stub_tests:
                repair_dump["test_patches"] = stub_tests
                logger.info("Generated %d stub test patches", len(stub_tests))

        state["repair"] = repair_dump
    except Exception as e:
        logger.error("Repair failed: %s", e)
        _emit_trace("error", {"message": f"Repair failed: {e}"})
        state["repair"] = {
            "patches": [],
            "explanation": f"Repair generation failed: {e}",
            "tests_added": [],
        }

    if trace:
        trace.stage_end("repair")
    return state


def multi_file_coordinator_node(state: AgentState) -> AgentState:
    """Stage 4.5: Verify caller files after repair and patch any that need updating.

    After repair_node produces patches for fault files, checks whether the caller
    files identified by blast radius also need updates (e.g., changed signatures).
    Produces additional patches for affected callers and merges them into repair state.
    Non-blocking — failures are logged and the pipeline continues without coordinator patches.
    """
    _thread_local.current_stage = "coordinate"
    repair = state.get("repair", {})
    caller_files = state.get("caller_files", [])

    # Early exits — nothing to coordinate
    if not caller_files or not repair.get("patches"):
        return state

    patches = repair.get("patches", [])
    patched_files = {p.get("file_path", "") for p in patches}
    unpatched_callers = [c for c in caller_files if c not in patched_files]

    if not unpatched_callers:
        logger.info("Coordinator: all %d caller(s) already covered by patches", len(caller_files))
        return state

    logger.info("Coordinator: %d unpatched caller(s) to check: %s", len(unpatched_callers), unpatched_callers)

    work_order = state.get("work_order", {})
    localization = state.get("localization", {})
    repo_path = _resolve_repo_path(work_order)
    if not repo_path:
        return state

    # Load source of unpatched callers (cap at 3 to limit token spend)
    caller_source: dict[str, str] = {}
    for cp in unpatched_callers[:3]:
        resolved = _find_file_in_repo(repo_path, cp)
        if not resolved:
            direct = repo_path / cp
            resolved = direct if direct.exists() else None
        if resolved and resolved.exists():
            content = _read_file_safe(resolved, max_lines=400)
            if content:
                caller_source[cp] = content

    if not caller_source:
        return state

    patch_summary = "\n".join(
        f"- {p.get('file_path', '?')}: {repair.get('explanation', 'see fix')}"
        for p in patches[:5]
    )
    caller_sections = "".join(
        f"\n--- {path} ---\n{code}\n"
        for path, code in caller_source.items()
    )

    prompt = f"""A bug fix was applied to {localization.get('fault_files', [])}.

FIX APPLIED:
{patch_summary}
EXPLANATION: {repair.get('explanation', '')}

Check if these caller files need updating due to the fix (e.g., renamed function, changed signature):

CALLER FILES:
{caller_sections}

Rules:
- If callers DO NOT need changes (fix is internal, same public interface): return empty patches list.
- If a caller DOES need updating: return a patch with original_code as an EXACT substring of the
  caller source shown above, and patched_code with only the necessary change.
- Do NOT patch callers unless the fix directly breaks their call site."""

    try:
        result = _structured_call("claude-sonnet-4-6", 3000, RepairResult, prompt)
        caller_patches = []
        for p in result.patches:
            fp = p.get("file_path", "")
            orig = p.get("original_code", "").strip()
            patched = p.get("patched_code", "").strip()
            if not fp or not orig or orig == patched:
                continue
            if fp not in caller_source:
                logger.warning("Coordinator: patch for unknown file %s — skipping", fp)
                continue
            if orig in caller_source[fp]:
                caller_patches.append(p)
                logger.info("Coordinator: verified caller patch for %s", fp)
            else:
                logger.warning("Coordinator: patch for %s failed source verification — skipping", fp)

        if caller_patches:
            updated = dict(repair)
            updated["patches"] = patches + caller_patches
            state["repair"] = updated
            logger.info("Coordinator: added %d caller patch(es) to repair", len(caller_patches))
        else:
            logger.info("Coordinator: no caller patches needed")
    except Exception as e:
        logger.warning("multi_file_coordinator failed (non-blocking): %s", e)

    return state


def _build_reviewer_context(repo_name: str, modified_files: list[str]) -> str:
    """Build independent reviewer context from graph data.

    The reviewer must NOT see the developer's source_code to prevent inherited bias.
    Instead it independently queries business rules and blast radius from stored graph data.
    """
    graph_data, enriched = _load_graph_data(repo_name)
    sections = []

    # 1. Business rules relevant to modified files
    rules = []
    for nid, node in enriched.items():
        ntype = node.get("type", "")
        if ntype not in ("business_rule", "decision_point"):
            continue
        node_file = node.get("file", "") or node.get("function_id", "")
        if any(f in node_file for f in modified_files):
            if ntype == "business_rule":
                rules.append(f"  [{node.get('rule_type', 'policy')}] {node.get('content', node.get('name', ''))[:200]}")
            else:
                q = node.get("question_for_human", "")
                if q:
                    rules.append(f"  [decision] {node.get('name', '')}: {q[:200]}")

    if rules:
        sections.append("BUSINESS RULES & DECISION POINTS (for modified files):")
        sections.extend(rules[:15])

    # 2. Blast radius — downstream consumers
    callers = _find_callers_from_graph(graph_data, modified_files, [])
    if callers:
        sections.append("\nBLAST RADIUS (files that call/import the modified code):")
        for c in callers[:10]:
            sections.append(f"  - {c}")
        risk = "CRITICAL" if len(callers) > 5 else "HIGH" if len(callers) > 2 else "MEDIUM"
        sections.append(f"  Risk level: {risk}")
    else:
        sections.append("\nBLAST RADIUS: No downstream consumers detected. Risk: LOW")

    return "\n".join(sections) if sections else "No business rules or blast radius data available."


def review_node(state: AgentState) -> AgentState:
    """Stage 5: Independent review with Opus — fresh context, no developer bias."""
    _thread_local.current_stage = "review"
    trace = _get_trace()
    if trace:
        trace.stage_start("review")
    logger.info("=== REVIEW: Independent check with Opus ===")
    state["status"] = PipelineStatus.REVIEWING
    _report_progress(state)

    repair = state.get("repair", {})

    if not repair.get("patches"):
        logger.warning("No patches to review — repair failed, escalating")
        state["review"] = {
            "verdict": "ESCALATE",
            "confidence": 0.0,
            "checks": [{"name": "ROOT_CAUSE", "status": "FAIL", "comment": "No patches generated — repair stage failed."}],
            "feedback": f"Repair produced no patches: {repair.get('explanation', 'unknown error')}",
        }
        if trace:
            trace.stage_end("review")
        return state

    intent = state.get("intent", {})
    work_order = state.get("work_order", {})
    repo_name = work_order.get("repo_name", "")

    # Build INDEPENDENT context — reviewer does NOT see the developer's source_code
    modified_files = [p.get("file_path", "") for p in repair.get("patches", []) if p.get("file_path")]
    reviewer_context = _build_reviewer_context(repo_name, modified_files) if repo_name else ""

    # Clean patches for review (strip internal fields)
    clean_patches = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in repair.get("patches", [])
    ]

    # Include acceptance criteria from intake (Finding #21: verification from spec)
    acceptance = intent.get("acceptance_criteria", [])
    criteria_section = ""
    if acceptance:
        criteria_section = "\nACCEPTANCE CRITERIA (from the bug spec — the fix must satisfy these):\n"
        criteria_section += "\n".join(f"  - {c}" for c in acceptance)

    # Pass localization info to reviewer so it can check completeness
    localization = state.get("localization", {})
    fault_functions = localization.get("fault_functions", [])
    # Deduplicate fault_files by basename (same fix as repair_node)
    raw_ff = localization.get("fault_files", [])
    ff_by_base: dict[str, str] = {}
    for fp in raw_ff:
        base = Path(fp).name
        if base not in ff_by_base or len(fp) > len(ff_by_base[base]):
            ff_by_base[base] = fp
    fault_files = list(ff_by_base.values())
    patched_functions = [p.get("file_path", "") for p in clean_patches]
    n_fault = len(fault_functions)
    n_patched = len(clean_patches)

    completeness_data = ""
    if n_fault > 1:
        completeness_data = (
            f"\nCOMPLETENESS DATA: Localization identified {n_fault} fault functions: {fault_functions}\n"
            f"Patches provided: {n_patched} patch(es) touching: {patched_functions}\n"
            f"If any identified fault function is missing a patch, COMPLETENESS must FAIL.\n"
        )

    # Phase 2B-3: surface caller files that were checked by coordinator
    caller_files_state = state.get("caller_files", [])
    patched_file_set = {p.get("file_path", "") for p in clean_patches}
    unpatched_callers_for_review = [c for c in caller_files_state if c not in patched_file_set]
    caller_completeness_note = ""
    if unpatched_callers_for_review:
        caller_completeness_note = (
            f"\nCALLER FILES NOT PATCHED: {unpatched_callers_for_review}\n"
            f"These files call the modified code but have no patches. "
            f"BLAST_RADIUS should FAIL if the fix changes a public interface (signature/name). "
            f"BLAST_RADIUS can PASS if the fix is purely internal (no interface change).\n"
        )

    prompt = f"""Review this bug fix as an independent reviewer who has NOT seen the developer's code.

BUG: {intent.get('actual_behavior', '')}
EXPECTED: {intent.get('expected_behavior', '')}
IDENTIFIED FAULT LOCATIONS: functions={fault_functions} in files={fault_files}
{completeness_data}
{caller_completeness_note}
{criteria_section}

PROPOSED PATCHES:
{json.dumps(clean_patches, indent=2)}

TEST PATCHES:
{json.dumps([{k: v for k, v in tp.items() if not k.startswith("_")} for tp in repair.get("test_patches", [])], indent=2)}

FIX EXPLANATION: {repair.get('explanation', '')}

INDEPENDENT CONTEXT (from knowledge graph):
{reviewer_context}

A correct review produces:
- 6 checks (ROOT_CAUSE, BUSINESS_RULES, PATTERNS, COMPLETENESS, BLAST_RADIUS, TESTS), each PASS/FAIL/WARNING
- ROOT_CAUSE passes when the fix addresses why the bug happens, not just the symptom
- BUSINESS_RULES passes when no rules from the context above are violated
- PATTERNS passes when code follows existing conventions (naming, imports, style)
- COMPLETENESS: FAIL if localization identified {n_fault} fault function(s) but patches cover fewer.
  Count the patches: do they fix ALL of {fault_functions}? If ANY fault function is missing a patch,
  COMPLETENESS = FAIL and verdict MUST be CHANGES_REQUESTED with feedback naming the missing function(s).
- BLAST_RADIUS: FAIL if the fix changes a public interface (function signature, parameter names,
  return type) AND caller files listed in CALLER FILES NOT PATCHED are present but have no patches.
  PASS if the fix is purely internal (same public interface) OR all affected callers are patched.
- TESTS passes when test_patches contains real test code covering the fix
- verdict: APPROVE only if ALL checks pass. CHANGES_REQUESTED if any check is FAIL. ESCALATE if too complex."""

    try:
        # Use Opus for deeper reasoning — worth the cost for catching subtle issues
        result = _structured_call("claude-opus-4-6", 3000, ReviewResult, prompt)
        state["review"] = result.model_dump()
        logger.info("Review verdict: %s (%.0f%%) — %s",
                    result.verdict, result.confidence * 100,
                    result.feedback or ", ".join(f"{c.name}:{c.status}" for c in result.checks))
    except Exception as e:
        logger.error("Opus review failed, falling back to Sonnet: %s", e)
        try:
            result = _structured_call("claude-sonnet-4-6", 2000, ReviewResult, prompt)
            state["review"] = result.model_dump()
        except Exception as e2:
            logger.error("Review failed completely: %s", e2)
            state["review"] = {
                "verdict": "ESCALATE",
                "confidence": 0.0,
                "checks": [],
                "feedback": f"Review failed: {e2}",
            }

    if trace:
        trace.stage_end("review")
    return state


def _find_related_tests(modified_files: list[str], sandbox_path: Path) -> list[str]:
    """Find test files related to the modified source files."""
    related = []
    for src_file in modified_files:
        base = Path(src_file).stem
        # Common test file patterns
        patterns = [
            f"test_{base}.py",
            f"{base}_test.py",
            f"tests/test_{base}.py",
            f"test/test_{base}.py",
        ]
        for pattern in patterns:
            matches = list(sandbox_path.rglob(pattern))
            related.extend(str(m.relative_to(sandbox_path)) for m in matches)
    return list(set(related))


# ---------------------------------------------------------------------------
# Finding #9: Robust trivial-test detection
# ---------------------------------------------------------------------------

def _is_trivial_test(test_code: str) -> bool:
    """Detect whether generated test code is a trivial stub.

    A test is trivial if it:
      - Is empty or nearly empty (< 5 non-blank lines)
      - Contains only placeholder assertions (assert True, pass, skip, etc.)
      - Has zero real assertions (assert X == Y, assertEqual, etc.)
      - Uses only trivial assertion values (assert 1 == 1, assert True)
    """
    if not test_code or not test_code.strip():
        return True

    lines = [l.strip() for l in test_code.splitlines() if l.strip() and not l.strip().startswith("#")]

    # Very short test code is almost certainly a stub
    if len(lines) < 5:
        return True

    # Patterns that indicate a placeholder / stub test
    stub_patterns = [
        "assert True",
        "assert 1 == 1",
        "assert 1",
        "raise NotImplementedError",
        "pytest.skip",
        "unittest.skip",
        "self.skipTest",
        "pass",
        "...",
        "# TODO",
        "# FIXME",
        "# placeholder",
    ]

    # Count real vs stub lines inside test functions
    inside_test = False
    real_assertion_count = 0
    stub_count = 0
    test_func_count = 0

    for line in lines:
        if line.startswith("def test_") or line.startswith("async def test_"):
            inside_test = True
            test_func_count += 1
            continue

        if inside_test:
            # Check for stub patterns
            if any(line.startswith(p) or line == p for p in stub_patterns):
                stub_count += 1
                continue

            # Count real assertions (assert with comparison operators)
            if re.match(r"assert\s+.+\s*(==|!=|>=|<=|>|<|in|not\s+in|is\s+not|is)\s+", line):
                real_assertion_count += 1
            elif re.match(r"self\.assert(Equal|NotEqual|True|False|In|NotIn|Raises|Greater|Less)", line):
                real_assertion_count += 1
            elif re.match(r"assert\s+\w+\.\w+", line):  # assert obj.method() style
                real_assertion_count += 1
            elif "pytest.raises" in line or "assertRaises" in line:
                real_assertion_count += 1

    # Trivial if no test functions found
    if test_func_count == 0:
        return True

    # Trivial if no real assertions at all
    if real_assertion_count == 0:
        return True

    # Trivial if stubs outnumber real assertions
    if stub_count > 0 and stub_count >= real_assertion_count:
        return True

    return False


def test_node(state: AgentState) -> AgentState:
    """Stage 5.5: Create sandbox via git worktree, apply patches, run tests."""
    _thread_local.current_stage = "test"
    trace = _get_trace()
    if trace:
        trace.stage_start("test")
    logger.info("=== TEST: Creating sandbox and running tests ===")
    state["status"] = PipelineStatus.TESTING
    _report_progress(state)

    work_order = state.get("work_order", {})
    repair = state.get("repair", {})
    repo_path = _resolve_repo_path(work_order)
    ticket_id = work_order.get("ticket_id", "UNKNOWN")

    if not repo_path:
        logger.warning("No repo path — skipping sandbox and tests")
        state["test_result"] = "skipped: no repo path"
        state["sandbox_path"] = ""
        _emit_trace("test_output", {"result": "skipped: no repo path", "passed": False, "patches_applied": 0})
        if trace:
            trace.stage_end("test")
        return state

    # Sanitize ticket_id: keep only alphanumerics, hyphens, and underscores.
    # Prevents special characters (e.g. "/" or "..") from escaping into file
    # paths or git branch names.
    safe_ticket_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ticket_id).lower()

    # Generate unique branch name
    branch_suffix = uuid.uuid4().hex[:6]
    branch_name = f"fix/{safe_ticket_id}-{branch_suffix}"
    state["branch_name"] = branch_name

    try:
        # Get base branch
        base_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        state["base_branch"] = base_branch

        # Acquire per-repo file lock to eliminate the race between dirty check
        # and worktree creation when multiple agents run against the same repo.
        _repo_lock_file = open(repo_path / ".agent_lock", 'w')
        try:
            fcntl.flock(_repo_lock_file, fcntl.LOCK_EX)

            # Check for dirty repo — ignore untracked files (??) which don't affect worktree
            porcelain = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
            ).stdout
            dirty = "\n".join(
                l for l in porcelain.splitlines() if l and not l.startswith("??")
            ).strip()
            if dirty:
                logger.error("Repo has uncommitted changes — cannot create sandbox")
                state["test_result"] = "skipped: repo has uncommitted changes"
                state["sandbox_path"] = ""
                state["error"] = "Repository has uncommitted changes. Commit or stash them first."
                _emit_trace("test_output", {"result": "skipped: repo has uncommitted changes", "passed": False, "patches_applied": 0})
                if trace:
                    trace.stage_end("test")
                return state

            # Create worktree (safe_ticket_id has no special chars, path is safe)
            worktree_path = Path(f"/tmp/agent_sandbox_{safe_ticket_id}_{branch_suffix}")
            subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, str(worktree_path), base_branch],
                cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
            )
        finally:
            fcntl.flock(_repo_lock_file, fcntl.LOCK_UN)
            _repo_lock_file.close()

        state["sandbox_path"] = str(worktree_path)
        logger.info("Created worktree at %s on branch %s", worktree_path, branch_name)

        # Scope guard (Finding #22/#23): verify patches only touch expected files
        localization = state.get("localization", {})
        expected_files = set(localization.get("fault_files", []))
        for patch in repair.get("patches", []):
            pf = patch.get("file_path", "")
            pf_name = Path(pf).name if pf else ""
            if pf and not any(pf in ef or ef in pf or pf_name == Path(ef).name for ef in expected_files):
                logger.warning("SCOPE GUARD: patch touches unexpected file %s (expected: %s)", pf, expected_files)

        # Apply patches — use pre-merged content if available, otherwise fuzzy match
        patches_applied = 0
        for patch in repair.get("patches", []):
            file_path = patch.get("file_path", "")
            if not file_path:
                continue

            full_path = worktree_path / file_path
            if not full_path.exists():
                matches = list(worktree_path.rglob(f"*{Path(file_path).name}"))
                full_path = matches[0] if matches else full_path

            if not full_path.exists():
                logger.warning("File not found in worktree: %s", file_path)
                continue

            original = patch.get("original_code", "")
            patched = patch.get("patched_code", "")
            if not original or not patched:
                continue

            try:
                content = full_path.read_text()
            except UnicodeDecodeError:
                logger.warning("Cannot read binary file for patching: %s", file_path)
                continue

            new_content = _fuzzy_match_replace(content, original, patched)
            if new_content is not None:
                full_path.write_text(new_content)
                patches_applied += 1
                logger.info("Applied patch to %s", file_path)
                _emit_trace("info", {"message": f"Patch applied: {file_path}"})
            else:
                logger.warning("Patch could not be matched in %s (original=%d chars, file=%d chars)",
                               file_path, len(original), len(content))
                _emit_trace("error", {
                    "message": f"Patch FAILED to match in {file_path}",
                    "original_code_preview": original[:200],
                    "file_content_preview": content[:200],
                    "original_len": len(original),
                    "file_len": len(content),
                })

        state["patches_applied"] = patches_applied

        # Apply test patches — create or overwrite test files in the sandbox
        test_patches_applied = 0
        for tp in repair.get("test_patches", []):
            file_path = tp.get("file_path", "")
            patched = tp.get("patched_code", "")
            if not file_path or not patched:
                continue

            full_path = worktree_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            original = tp.get("original_code", "")
            if not original.strip():
                if full_path.exists():
                    # File exists — APPEND new tests, never overwrite
                    existing = full_path.read_text()
                    full_path.write_text(existing.rstrip() + "\n\n\n" + patched)
                    test_patches_applied += 1
                    logger.info("Appended tests to existing file: %s", file_path)
                else:
                    # Genuinely new file
                    full_path.write_text(patched)
                    test_patches_applied += 1
                    logger.info("Created test file: %s", file_path)
            else:
                # Update existing test file
                if full_path.exists():
                    content = full_path.read_text()
                    new_content = _fuzzy_match_replace(content, original, patched)
                    if new_content is not None:
                        full_path.write_text(new_content)
                        test_patches_applied += 1
                        logger.info("Updated test file: %s", file_path)
                    else:
                        # Replace entire file if fuzzy match fails
                        full_path.write_text(patched)
                        test_patches_applied += 1
                        logger.info("Replaced test file: %s", file_path)
                else:
                    full_path.write_text(patched)
                    test_patches_applied += 1
                    logger.info("Created test file: %s", file_path)

        if test_patches_applied:
            logger.info("Applied %d test patch(es) to sandbox", test_patches_applied)

        if patches_applied == 0:
            logger.warning("No patches applied — cleaning up worktree")
            _cleanup_worktree(repo_path, str(worktree_path))
            state["sandbox_path"] = ""
            state["test_result"] = "failed: no patches could be applied"
            state["error"] = "No patches could be applied to the source code."
            _emit_trace("test_output", {"result": "failed: no patches could be applied", "passed": False, "patches_applied": 0})
            if trace:
                trace.stage_end("test")
            return state

        # Syntax validation — check patched Python files compile
        syntax_errors = []
        for patch in repair.get("patches", []):
            fpath = patch.get("file_path", "")
            full_path = worktree_path / fpath
            if full_path.exists() and full_path.suffix == ".py":
                err = _check_syntax(full_path)
                if err:
                    syntax_errors.append(f"{fpath}: {err}")
                    logger.warning("Syntax error in patched file %s: %s", fpath, err)

        if syntax_errors:
            logger.error("Patched files have syntax errors — aborting commit")
            _cleanup_worktree(repo_path, str(worktree_path))
            state["sandbox_path"] = ""
            state["test_result"] = "failed: syntax errors in patched files\n" + "\n".join(syntax_errors)
            state["error"] = "Patches introduced syntax errors: " + "; ".join(syntax_errors)
            _emit_trace("test_output", {"result": "failed: syntax errors", "passed": False, "patches_applied": patches_applied})
            if trace:
                trace.stage_end("test")
            return state

        # Custom lint rules (Step 16) — run against patched files
        try:
            from agent.lint_rules import run_lint_on_patches
            repo_name = work_order.get("repo_name", "")
            lint_violations = run_lint_on_patches(repair.get("patches", []), worktree_path, repo_name)
            lint_errors = [v for v in lint_violations if v["severity"] == "error"]
            if lint_errors:
                lint_msg = "\n".join(f"  {v['file']}:{v['line']} [{v['rule_id']}] {v['message']}" for v in lint_errors)
                logger.warning("Lint errors in patched files:\n%s", lint_msg)
                # Don't abort — include in test_result so reviewer sees it
                state["test_result"] = f"lint_warnings:\n{lint_msg}\n"
            elif lint_violations:
                state["test_result"] = f"lint_ok ({len(lint_violations)} warnings)\n"
        except Exception as lint_err:
            logger.debug("Lint check failed (non-fatal): %s", lint_err)

        # Commit patches in worktree
        subprocess.run(
            ["git", "add", "-A"],
            cwd=worktree_path, capture_output=True, text=True, check=True, timeout=30,
        )
        commit_msg = f"fix({ticket_id}): {repair.get('explanation', 'Automated fix')[:200]}"
        subprocess.run(
            [
                "git", "-c", "user.email=agent@context-builder.ai",
                "-c", "user.name=Context Builder Agent",
                "commit", "-m", commit_msg,
            ],
            cwd=worktree_path, capture_output=True, text=True, check=True, timeout=30,
        )
        logger.info("Committed %d patches in sandbox", patches_applied)

        # Log repro steps availability for test generation guidance
        repro_steps = state.get("work_order", {}).get("repro_steps", [])
        if repro_steps:
            logger.info("Reproduction steps available (%d steps) — can be used for test generation", len(repro_steps))
            # Inject into any stub test generation prompts

        # Install repo dependencies in sandbox if requirements.txt present
        req_file = worktree_path / "requirements.txt"
        if req_file.exists():
            logger.info("Installing sandbox dependencies from requirements.txt")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=worktree_path, capture_output=True, text=True, timeout=120,
            )

        # Run targeted tests first (faster feedback), then the full suite
        fault_files = state.get("localization", {}).get("fault_files", [])
        related_tests = _find_related_tests(fault_files, worktree_path)
        run_full_suite = True
        if related_tests:
            targeted_cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"] + related_tests
            logger.info("Running targeted tests first (%d file(s)): %s", len(related_tests), related_tests)
            try:
                targeted_result = subprocess.run(
                    targeted_cmd,
                    cwd=worktree_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if targeted_result.returncode != 0:
                    logger.info("Targeted tests FAILED — skipping full suite")
                    _emit_trace("info", {"message": f"Targeted tests failed:\n{targeted_result.stdout[-2000:]}"})
                    test_result = f"failed (targeted): {targeted_result.stdout[-1000:]}"
                    state["test_result"] = test_result
                    run_full_suite = False
                else:
                    logger.info("Targeted tests pass (%d files). Running full suite...", len(related_tests))
            except subprocess.TimeoutExpired:
                logger.warning("Targeted test run timed out — proceeding to full suite")

        # Auto-detect and run tests (full suite, unless targeted run already failed)
        if run_full_suite:
            test_result = _run_tests(worktree_path)
        state["test_result"] = test_result
        _emit_trace("test_output", {
            "result": test_result,
            "passed": test_result.startswith("passed"),
            "patches_applied": patches_applied,
        })

    except subprocess.CalledProcessError as e:
        logger.error("Sandbox/test operation failed: %s — %s", e, e.stderr)
        state["test_result"] = f"error: {e.stderr}"
        state["error"] = f"Sandbox operation failed: {e.stderr}"
        _emit_trace("test_output", {"result": f"error: {e.stderr}", "passed": False, "patches_applied": 0})
        _cleanup_worktree(repo_path, state.get("sandbox_path", ""))
    except Exception as e:
        logger.error("Test node failed: %s", e)
        state["test_result"] = f"error: {e}"
        _emit_trace("test_output", {"result": f"error: {e}", "passed": False, "patches_applied": 0})
        _cleanup_worktree(repo_path, state.get("sandbox_path", ""))

    # Step 18: Enrich failed test results with business context
    _append_test_business_context(state, work_order)

    # If tests passed and review was CHANGES_REQUESTED only for TESTS, upgrade verdict
    test_result = state.get("test_result", "")
    review = state.get("review", {})
    if test_result.startswith("passed") and review.get("verdict") == "CHANGES_REQUESTED":
        checks = review.get("checks", [])
        non_test_fails = [c for c in checks if c.get("status") == "FAIL" and c.get("name", "").upper() != "TESTS"]
        if not non_test_fails:
            # Guard: don't upgrade if the test patches are trivial stubs
            test_code = "\n".join(
                tp.get("patched_code", "")
                for tp in (repair.get("test_patches") or [])
            )
            is_trivial = _is_trivial_test(test_code)

            if is_trivial:
                logger.info("Tests appear trivial — NOT auto-upgrading CHANGES_REQUESTED verdict to APPROVE")
            else:
                review = dict(review)
                review["verdict"] = "APPROVE"
                # Also update the TESTS check status so the UI shows green
                updated_checks = []
                for c in checks:
                    c = dict(c)
                    if c.get("name", "").upper() == "TESTS" and c.get("status") == "FAIL":
                        c["status"] = "PASS"
                        c["comment"] = f"Tests passed after fix: {test_result.splitlines()[0]}"
                    updated_checks.append(c)
                review["checks"] = updated_checks
                review["feedback"] = ""
                state["review"] = review
                logger.info("Upgraded review verdict to APPROVE — tests now pass")

    if trace:
        trace.stage_end("test")
    return state


def pr_creation_node(state: AgentState) -> AgentState:
    """Stage 6: Push branch and create GitHub PR from sandbox."""
    _thread_local.current_stage = "pr_creation"
    trace = _get_trace()
    if trace:
        trace.stage_start("pr_creation")
    logger.info("=== PR CREATION: Pushing branch and creating PR ===")
    state["status"] = PipelineStatus.PR_CREATING
    _report_progress(state)

    work_order = state.get("work_order", {})
    repair = state.get("repair", {})
    review = state.get("review", {})
    ticket_id = work_order.get("ticket_id", "UNKNOWN")
    sandbox_path = state.get("sandbox_path", "")
    branch_name = state.get("branch_name", "")
    base_branch = state.get("base_branch", "main")
    repo_path = _resolve_repo_path(work_order)

    if not sandbox_path or not Path(sandbox_path).exists():
        logger.warning("No sandbox available — cannot push or create PR")
        state["pr_url"] = f"branch://{branch_name}" if branch_name else ""
        state["error"] = state.get("error", "") or "No sandbox available for PR creation."
        state["status"] = PipelineStatus.DONE
        _report_progress(state)
        if trace:
            trace.stage_end("pr_creation")
        return state

    try:
        # Build PR body with blast radius analysis (always — useful in dry_run too)
        test_result = state.get("test_result", "not run")
        patches = repair.get("patches", [])
        files_changed = ", ".join(p.get("file_path", "?") for p in patches)
        repo_name = work_order.get("repo_name", "")

        # Compute blast radius for PR body
        blast_section = ""
        if repo_name:
            try:
                graph_data, _ = _load_graph_data(repo_name)
                modified_files = [p.get("file_path", "") for p in patches if p.get("file_path")]
                callers = _find_callers_from_graph(graph_data, modified_files, [])
                if callers:
                    risk = "CRITICAL" if len(callers) > 5 else "HIGH" if len(callers) > 2 else "MEDIUM" if callers else "LOW"
                    blast_section = (
                        f"## Blast Radius ({risk})\n"
                        "Files that call/import the modified code:\n"
                        + "\n".join(f"- `{c}`" for c in callers[:15])
                        + "\n\n"
                    )
                else:
                    blast_section = "## Blast Radius (LOW)\nNo downstream consumers detected.\n\n"
            except Exception:
                blast_section = ""

        pr_body = (
            f"## Root Cause\n{state.get('localization', {}).get('root_cause_hypothesis', 'N/A')}\n\n"
            f"## Fix\n{repair.get('explanation', 'N/A')}\n\n"
            f"## Files Changed\n{files_changed}\n\n"
            f"{blast_section}"
            f"## Review\n"
            f"- Verdict: {review.get('verdict', 'N/A')}\n"
            f"- Confidence: {review.get('confidence', 0):.0%}\n\n"
            f"## Tests\n```\n{test_result[:2000]}\n```\n\n"
            f"---\n*Generated by AI Deploy Agent ({ticket_id})*"
        )
        pr_title = f"fix({ticket_id}): {repair.get('explanation', 'Automated fix')[:60]}"

        # Dry run: return with PR body but skip push, feature flag, and PR creation.
        # The finally block still cleans up the worktree.
        # Do NOT call _enrich_from_fix — no real fix to record.
        if state.get("dry_run"):
            logger.info("DRY RUN — skipping push, feature flag, and PR creation. "
                        "Patch + PR body available in state.")
            state["pr_url"] = "(dry-run — no PR created)"
            state["status"] = PipelineStatus.DONE
            _report_progress(state)
            if trace:
                trace.stage_end("pr_creation")
            return state

        # Push branch to remote — inject GH_TOKEN into git credential helper
        gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
        push_env = {**os.environ}
        if gh_token:
            push_env["GIT_ASKPASS"] = "echo"
            push_env["GIT_USERNAME"] = "x-access-token"
            push_env["GIT_PASSWORD"] = gh_token
            # Rewrite remote URL to use token inline for HTTPS
            subprocess.run(
                ["git", "config", "url.https://x-access-token:" + gh_token + "@github.com/.insteadOf", "https://github.com/"],
                cwd=sandbox_path, capture_output=True, text=True,
            )
        push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=sandbox_path, capture_output=True, text=True, timeout=120, env=push_env,
        )
        if push_result.returncode != 0:
            logger.warning("Git push failed: %s", push_result.stderr)
            state["pr_url"] = f"branch://{branch_name} (push failed: {push_result.stderr[:200]})"
            state["status"] = PipelineStatus.DONE
            _report_progress(state)
            _cleanup_worktree(repo_path, sandbox_path)
            if trace:
                trace.stage_end("pr_creation")
            return state

        logger.info("Pushed branch %s to origin", branch_name)

        # Create feature flag for this change (Step 19)
        flag_name = ""
        if repo_name:
            try:
                modified_files = [p.get("file_path", "") for p in patches if p.get("file_path")]
                flag_name = _create_feature_flag(
                    repo_name=repo_name,
                    ticket_id=ticket_id,
                    description=repair.get("explanation", "automated fix")[:60],
                    files_changed=modified_files,
                )
                pr_body += (
                    f"\n\n## Feature Flag\n"
                    f"Flag name: `{flag_name}`\n"
                    f"Status: **disabled** (enable after verification)\n"
                )
                logger.info("Created feature flag %s for PR", flag_name)
            except Exception as exc:
                logger.warning("Feature flag creation failed (non-blocking): %s", exc)

        # Create PR via gh CLI (requires gh + GH_TOKEN)
        pr_result = subprocess.run(
            ["gh", "pr", "create",
             "--title", pr_title,
             "--body", pr_body,
             "--base", base_branch,
             "--head", branch_name],
            cwd=sandbox_path, capture_output=True, text=True, timeout=60,
            env={**os.environ, "GH_TOKEN": gh_token} if gh_token else None,
        )

        if pr_result.returncode == 0:
            pr_url = pr_result.stdout.strip()
            state["pr_url"] = pr_url
            logger.info("Created PR: %s", pr_url)
            # Update feature flag with PR URL
            if flag_name and repo_name:
                try:
                    _set_flag_pr_url(repo_name, flag_name, pr_url)
                except Exception:
                    pass
            # Self-enriching loop (Finding #26): store fix pattern
            _enrich_from_fix(state)
        else:
            logger.warning("gh pr create failed: %s", pr_result.stderr)
            state["pr_url"] = f"branch://{branch_name} (PR creation failed: {pr_result.stderr[:200]})"

    except subprocess.TimeoutExpired as e:
        logger.error("PR creation timed out: %s", e)
        state["pr_url"] = f"branch://{branch_name} (timed out)"
        state["error"] = "PR creation timed out"
    except Exception as e:
        logger.error("PR creation failed: %s", e)
        state["pr_url"] = ""
        state["error"] = f"PR creation failed: {e}"
    finally:
        # Clean up worktree
        _cleanup_worktree(repo_path, sandbox_path)

    state["status"] = PipelineStatus.DONE
    _report_progress(state)
    if trace:
        trace.stage_end("pr_creation")
    return state


def _enrich_from_fix(state: AgentState) -> None:
    """Self-enriching loop (Finding #26): store fix pattern after successful PR.

    Every successful fix permanently enriches the knowledge base so the agent
    never has to re-discover the same pattern. This is the compounding advantage.
    """
    work_order = state.get("work_order", {})
    repo_name = work_order.get("repo_name", "")
    if not repo_name:
        return

    repair = state.get("repair", {})
    localization = state.get("localization", {})
    pr_url = state.get("pr_url", "")

    fix_record = {
        "ticket_id": work_order.get("ticket_id", ""),
        "root_cause": localization.get("root_cause_hypothesis", "")[:200],
        "fix_summary": repair.get("explanation", "")[:200],
        "fault_files": localization.get("fault_files", []),
        "fault_functions": localization.get("fault_functions", []),
        "pr_url": pr_url,
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    try:
        fixes_path = DATA_DIR / repo_name / "fix_history.json"
        fixes_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if fixes_path.exists():
            existing = json.loads(fixes_path.read_text())
        existing.append(fix_record)
        fixes_path.write_text(json.dumps(existing, indent=2))
        logger.info("Stored fix pattern for %s → fix_history.json (%d total)",
                    work_order.get("ticket_id", ""), len(existing))
    except Exception as e:
        logger.debug("Failed to store fix pattern: %s", e)


def escalate_node(state: AgentState) -> AgentState:
    """Escalate to human when agent can't fix confidently."""
    _thread_local.current_stage = "escalate"
    trace = _get_trace()
    if trace:
        trace.stage_start("escalate")
    ticket_id = state.get("work_order", {}).get("ticket_id", "UNKNOWN")
    iterations = state.get("iteration_count", 0)
    reason = state.get("review", {}).get("feedback", "no feedback")
    declined_reason = f"Agent declined after {iterations} iterations: {reason}"
    # Explicit escalation signal — visible in logs and monitoring (BUG-6 instrumentation)
    logger.error(
        "AGENT_DECLINED ticket=%s iterations=%d reason=%r",
        ticket_id, iterations, reason,
    )
    _emit_trace("escalation", {
        "ticket_id": ticket_id,
        "iterations": iterations,
        "reason": reason,
        "declined_reason": declined_reason,
    })
    state["status"] = PipelineStatus.ESCALATED
    state["declined_reason"] = declined_reason
    state["error"] = declined_reason
    _report_progress(state)
    if trace:
        trace.stage_end("escalate")
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def should_iterate(state: AgentState) -> Literal["test", "retry_fix", "escalate"]:
    """Decide whether to test+PR, retry, or escalate."""
    review = state.get("review", {})
    iteration = state.get("iteration_count", 0)
    verdict = review.get("verdict", "ESCALATE")
    confidence = review.get("confidence", 0.0)

    if verdict == "APPROVE":
        return "test"
    elif verdict == "ESCALATE" or iteration >= MAX_ITERATIONS:
        return "escalate"
    elif verdict == "CHANGES_REQUESTED":
        # If only the TESTS check is failing and confidence is high, proceed anyway
        checks = review.get("checks", [])
        blocking_fails = [
            c for c in checks
            if c.get("status") == "FAIL" and c.get("name", "").upper() != "TESTS"
        ]
        if not blocking_fails and confidence >= 0.7:
            logger.info("Only TESTS check failing with %.0f%% confidence — proceeding to test", confidence * 100)
            return "test"
        return "retry_fix"
    else:
        return "retry_fix"


def should_retry_after_test(state: AgentState) -> str:
    """Route after test_node: block PR on test failures, retry on syntax errors."""
    test_result = state.get("test_result", "")
    iteration = state.get("iteration_count", 0)

    # Syntax errors → retry repair
    if "syntax error" in test_result.lower():
        if iteration >= MAX_ITERATIONS:
            logger.warning("Syntax errors after max iterations — escalating")
            return "escalate"
        logger.info("Syntax errors — routing back to repair (iteration %d)", iteration)
        return "retry_fix"

    # Test failures → retry repair (agent's fix or tests are wrong)
    if test_result.startswith("failed"):
        if iteration >= MAX_ITERATIONS:
            logger.warning("Tests still failing after max iterations — escalating")
            return "escalate"
        logger.info("Tests failed — routing back to repair (iteration %d)", iteration)
        return "retry_fix"

    # Tests passed, skipped, or errored (non-blocking) → proceed to PR
    return "create_pr"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph():
    """Build and compile the LangGraph state machine.

    Pipeline: intake → exploration → repair → review → test → PR
    Exploration is kick-started with graph + vector + failure signals,
    then the agent explores freely with grep/read/search tools.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("exploration", exploration_node)
    graph.add_node("repair", repair_node)
    graph.add_node("multi_file_coordinator", multi_file_coordinator_node)
    graph.add_node("review", review_node)
    graph.add_node("test", test_node)
    graph.add_node("create_pr", pr_creation_node)
    graph.add_node("escalate", escalate_node)
    graph.set_entry_point("intake")

    graph.add_edge("intake", "exploration")
    graph.add_edge("exploration", "repair")
    graph.add_edge("repair", "multi_file_coordinator")
    graph.add_edge("multi_file_coordinator", "review")
    graph.add_conditional_edges(
        "review",
        should_iterate,
        {"test": "test", "retry_fix": "repair", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "test",
        should_retry_after_test,
        {"create_pr": "create_pr", "retry_fix": "repair", "escalate": "escalate"},
    )
    graph.add_edge("create_pr", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# Module-level compiled graph
agent_app = build_agent_graph()


def run_ticket(
    work_order: dict,
    progress_cb: Callable[[AgentState], None] | None = None,
    trace: RunTrace | None = None,
    dry_run: bool = False,
) -> dict:
    """Run a bug ticket through the full agent pipeline.

    Args:
        work_order: Bug ticket work order dict.
        progress_cb: Optional callback for progress updates.
        trace: Optional RunTrace instance for observability.
        dry_run: If True, skip PR creation, feature flags, and enrichment.
                 Tests still run. Useful for validation and eval suites.
    """
    _thread_local.progress_callback = progress_cb
    _thread_local.trace = trace
    _thread_local.current_stage = "pending"

    initial_state: AgentState = {
        "work_order": work_order,
        "intent": {},
        "context": "",
        "context_nodes": 0,
        "source_code": {},
        "localization": {},
        "repair": {},
        "review": {},
        "iteration_count": 0,
        "status": PipelineStatus.PENDING,
        "error": "",
        "pr_url": "",
        "test_result": "",
        "sandbox_path": "",
        "branch_name": "",
        "base_branch": "",
        "patches_applied": 0,
        "exploration_log": [],
        "caller_files": [],
        "dry_run": dry_run,
    }

    try:
        result = agent_app.invoke(initial_state)
        result_dict = dict(result)
        # Record metrics (Step 20)
        try:
            from api.metrics import record_run
            record_run(result_dict)
        except Exception:
            pass
        return result_dict
    finally:
        _thread_local.progress_callback = None
        if trace:
            trace.complete()
        _thread_local.trace = None
