"""
react_pipeline.py — ReAct agent pipeline (3-node LangGraph).

Architecture:
    intake_node → react_agent_node → finalize_node → END

The react_agent_node runs a single while-loop where the LLM decides:
explore → localize → edit → test → review → submit.

This is the ReAct alternative to the fixed 8-node pipeline in pipeline.py.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from langgraph.graph import END, StateGraph

from agent.types import PipelineStatus, ReactAgentState

if TYPE_CHECKING:
    from agent.trace import RunTrace

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# Thread-local for progress + trace (reuse same pattern as pipeline.py)
import threading
_thread_local = threading.local()


def _get_trace():
    return getattr(_thread_local, "trace", None)


def _emit_trace(event_type: str, data: dict | None = None):
    trace = _get_trace()
    if trace:
        stage = getattr(_thread_local, "current_stage", "unknown")
        trace.emit(event_type, stage, data)


def _report_progress(state: ReactAgentState) -> None:
    cb = getattr(_thread_local, "progress_callback", None)
    if cb:
        try:
            cb(state)
        except Exception:
            pass


def _resolve_repo_path(work_order: dict) -> Path | None:
    """Resolve the actual filesystem path for a repo."""
    if work_order.get("repo_path"):
        p = Path(work_order["repo_path"])
        if p.exists():
            return p
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
    repos_base = os.environ.get("REPOS_BASE_DIR", "")
    if repos_base:
        p = Path(repos_base) / repo_name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Node 1: INTAKE (reused from pipeline.py)
# ---------------------------------------------------------------------------

def intake_node(state: ReactAgentState) -> ReactAgentState:
    """Translate bug ticket into technical spec — same logic as pipeline.py intake."""
    _thread_local.current_stage = "intake"
    trace = _get_trace()
    if trace:
        trace.stage_start("intake")
    logger.info("=== REACT INTAKE: Translating bug ticket ===")
    state["status"] = PipelineStatus.INTAKE
    _report_progress(state)

    work_order = state.get("work_order", {})

    from agent.llm import structured_call as _structured_call, INTAKE_MODEL
    from agent.intake_helpers import extract_stack_trace_hints, extract_repro_steps, classify_bug_category
    from agent.types import IntentAnalysis

    prompt = f"""Translate this bug ticket into a technical specification.

Ticket: {work_order.get('title', '')}
Description: {work_order.get('description', '')}
Priority: {work_order.get('priority', 'unknown')}
Component: {work_order.get('affected_component', 'unknown')}
Comments: {'; '.join(work_order.get('comments', []))}

Include acceptance_criteria: 2-4 testable assertions derived from the bug description
that prove the fix works. These must come from the SPEC (what the user reported),
not from guessing the implementation."""

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

    # Stack trace hints
    description = work_order.get("description", "")
    stack_hints = extract_stack_trace_hints(description)
    if stack_hints:
        work_order["stack_trace_hints"] = stack_hints
        state["work_order"] = work_order
        intent = state.get("intent", {})
        stack_note = "Stack trace points to: " + "; ".join(
            f"{h['file']} line {h['line']}" + (f" in {h['function']}" if h["function"] else "")
            for h in stack_hints
        )
        existing_notes = intent.get("notes", "")
        intent["notes"] = (existing_notes + "\n" + stack_note).strip() if existing_notes else stack_note
        state["intent"] = intent

    # Repro steps
    repro_steps = extract_repro_steps(description)
    if repro_steps:
        work_order["repro_steps"] = repro_steps
        state["work_order"] = work_order

    # Bug category
    bug_category = classify_bug_category(work_order.get("title", ""), description)
    work_order["bug_category"] = bug_category
    state["work_order"] = work_order

    if bug_category == "C":
        intent = state.get("intent", {})
        cat_c_note = (
            "WARNING: Bug category C — concurrency, performance, or multi-service. "
            "Auto-fix success rate is low; consider escalating early."
        )
        existing_notes = intent.get("notes", "")
        intent["notes"] = (existing_notes + "\n" + cat_c_note).strip() if existing_notes else cat_c_note
        state["intent"] = intent

    if trace:
        trace.stage_end("intake")
    return state


# ---------------------------------------------------------------------------
# Node 2: REACT AGENT (the core)
# ---------------------------------------------------------------------------

def react_agent_node(state: ReactAgentState) -> ReactAgentState:
    """Run the ReAct loop — agent explores, edits, tests, reviews, submits."""
    _thread_local.current_stage = "react_agent"
    trace = _get_trace()
    logger.info("=== REACT AGENT: Starting ReAct loop ===")
    state["status"] = PipelineStatus.EXPLORING
    _report_progress(state)

    work_order = state.get("work_order", {})
    intent = state.get("intent", {})
    repo_name = work_order.get("repo_name", "")
    repo_path = _resolve_repo_path(work_order)

    if not repo_path:
        logger.error("No repo_path — cannot run ReAct loop")
        state["escalated"] = True
        state["escalate_reason"] = "No repo path available"
        state["status"] = PipelineStatus.ESCALATED
        return state

    # Set tool context
    from agent.explore_tools import set_context, ALL_TOOLS as EXPLORE_TOOLS
    from agent.react_tools import set_react_context
    set_context(repo_name, repo_path, DATA_DIR)
    set_react_context(repo_name, repo_path, DATA_DIR)

    # Build orientation context
    from agent.graph_utils import build_kickstart_context, load_business_rules
    kickstart = build_kickstart_context(repo_name, str(repo_path), intent, DATA_DIR)

    # Add repo structure snapshot (saves 2-3 list_files calls)
    try:
        top_level = subprocess.run(
            ["find", str(repo_path), "-maxdepth", "2", "-name", "*.py", "-not", "-path", "*/.*"],
            capture_output=True, text=True, timeout=10,
        )
        if top_level.stdout:
            py_files = sorted(set(
                str(Path(f.strip()).relative_to(repo_path))
                for f in top_level.stdout.strip().split("\n")
                if f.strip() and "__pycache__" not in f
            ))[:30]
            if py_files:
                kickstart += "\n\nREPO STRUCTURE (top-level .py files):\n" + "\n".join(f"  {f}" for f in py_files)
    except Exception:
        pass

    # Load conventions and business rules
    from agent.react_prompt import (
        build_system_prompt,
        build_task_message,
        load_project_conventions,
    )
    conventions = load_project_conventions(repo_name)

    # Business rules (use hint files from intent for scoping)
    hint_files = intent.get("likely_affected_modules", [])[:5]
    business_rules = load_business_rules(repo_name, hint_files) if hint_files else ""

    system_prompt = build_system_prompt(
        work_order=work_order,
        intent=intent,
        kickstart_context=kickstart,
        conventions_section=conventions,
        business_rules_section=business_rules,
    )
    task_message = build_task_message(work_order, intent)

    # Run the ReAct loop
    from agent.react_loop import react_loop
    state = react_loop(
        state=state,
        system_prompt=system_prompt,
        task_message=task_message,
        explore_tools=list(EXPLORE_TOOLS),
        trace=trace,
    )

    return state


# ---------------------------------------------------------------------------
# Node 3: FINALIZE (PR creation or escalation)
# ---------------------------------------------------------------------------

def finalize_node(state: ReactAgentState) -> ReactAgentState:
    """Handle post-agent work: create PR if submitted, escalate otherwise."""
    _thread_local.current_stage = "finalize"
    trace = _get_trace()
    if trace:
        trace.stage_start("finalize")
    logger.info("=== FINALIZE: %s ===",
                "Creating PR" if state.get("submitted") else "Escalating")

    work_order = state.get("work_order", {})
    ticket_id = work_order.get("ticket_id", "UNKNOWN")
    sandbox_path = state.get("sandbox_path", "")
    branch_name = state.get("branch_name", "")
    base_branch = state.get("base_branch", "main")
    repo_path = _resolve_repo_path(work_order)
    dry_run = state.get("dry_run", False)

    if state.get("escalated"):
        logger.info("ESCALATED: %s — reason: %s",
                    ticket_id, state.get("escalate_reason", "unknown"))
        state["status"] = PipelineStatus.ESCALATED
        _report_progress(state)
        if trace:
            trace.stage_end("finalize")
        _cleanup_sandbox(sandbox_path, repo_path, branch_name)
        return state

    if not state.get("submitted"):
        state["status"] = PipelineStatus.ESCALATED
        state["escalated"] = True
        state["escalate_reason"] = "Agent did not submit or escalate"
        _report_progress(state)
        if trace:
            trace.stage_end("finalize")
        _cleanup_sandbox(sandbox_path, repo_path, branch_name)
        return state

    # Build PR body
    explanation = state.get("explanation", "Automated fix")
    review = state.get("review", {})
    test_result = state.get("test_result", "not run")

    # Get diff for PR body
    diff_stat = ""
    if sandbox_path and Path(sandbox_path).exists():
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                cwd=sandbox_path, capture_output=True, text=True, timeout=30,
            )
            diff_stat = result.stdout[:1000]
        except Exception:
            pass

    pr_body = (
        f"## Root Cause\n{explanation}\n\n"
        f"## Review\n"
        f"- Verdict: {review.get('verdict', 'N/A')}\n"
        f"- Confidence: {review.get('confidence', 0):.0%}\n\n"
        f"## Tests\n```\n{test_result[:2000]}\n```\n\n"
        f"## Changes\n```\n{diff_stat}\n```\n\n"
        f"---\n*Generated by AI Deploy Agent (ReAct) — {ticket_id}*"
    )
    pr_title = f"fix({ticket_id}): {explanation[:60]}"

    if dry_run:
        logger.info("DRY RUN — skipping PR creation")
        state["pr_url"] = "(dry-run — no PR created)"
        state["status"] = PipelineStatus.DONE

        # Populate repair + localization for eval compatibility
        if sandbox_path and Path(sandbox_path).exists():
            repair = _extract_repair_from_sandbox(sandbox_path)
            state["repair"] = repair
            # Derive localization from what the agent actually modified
            fault_files = [
                p["file_path"] for p in repair.get("patches", [])
                if not p["file_path"].startswith("test") and "/test" not in p["file_path"]
            ]
            if fault_files and not state.get("localization", {}).get("fault_files"):
                state["localization"] = {
                    "fault_files": fault_files,
                    "fault_functions": [],
                    "root_cause_hypothesis": state.get("explanation", ""),
                    "confidence": 0.8,
                }

        _report_progress(state)
        if trace:
            trace.stage_end("finalize")
        _cleanup_sandbox(sandbox_path, repo_path, branch_name)
        return state

    # Push and create PR
    if sandbox_path and Path(sandbox_path).exists() and branch_name:
        try:
            gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
            if gh_token:
                subprocess.run(
                    ["git", "config", "credential.helper", ""],
                    cwd=sandbox_path, capture_output=True, timeout=10,
                )
                # Set remote URL with token
                remote_url = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=sandbox_path, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                if remote_url and "github.com" in remote_url:
                    auth_url = remote_url.replace(
                        "https://", f"https://x-access-token:{gh_token}@"
                    )
                    subprocess.run(
                        ["git", "remote", "set-url", "origin", auth_url],
                        cwd=sandbox_path, capture_output=True, timeout=10,
                    )

            push_result = subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                cwd=sandbox_path, capture_output=True, text=True, timeout=60,
            )
            if push_result.returncode != 0:
                logger.error("Push failed: %s", push_result.stderr)
                state["error"] = f"Push failed: {push_result.stderr[:200]}"
                state["pr_url"] = f"branch://{branch_name}"
            else:
                # Create PR via gh CLI
                pr_result = subprocess.run(
                    ["gh", "pr", "create",
                     "--title", pr_title,
                     "--body", pr_body,
                     "--base", base_branch,
                     "--head", branch_name],
                    cwd=sandbox_path, capture_output=True, text=True, timeout=30,
                )
                if pr_result.returncode == 0:
                    pr_url = pr_result.stdout.strip()
                    state["pr_url"] = pr_url
                    logger.info("PR created: %s", pr_url)
                else:
                    logger.error("PR creation failed: %s", pr_result.stderr)
                    state["pr_url"] = f"branch://{branch_name}"
                    state["error"] = f"PR creation failed: {pr_result.stderr[:200]}"
        except Exception as e:
            state["error"] = f"PR creation error: {e}"
            state["pr_url"] = f"branch://{branch_name}"

    state["status"] = PipelineStatus.DONE

    # Populate repair + localization for eval compatibility
    if sandbox_path and Path(sandbox_path).exists():
        repair = _extract_repair_from_sandbox(sandbox_path)
        state["repair"] = repair
        fault_files = [
            p["file_path"] for p in repair.get("patches", [])
            if not p["file_path"].startswith("test") and "/test" not in p["file_path"]
        ]
        if fault_files and not state.get("localization", {}).get("fault_files"):
            state["localization"] = {
                "fault_files": fault_files,
                "fault_functions": [],
                "root_cause_hypothesis": state.get("explanation", ""),
                "confidence": 0.8,
            }

    _report_progress(state)
    if trace:
        trace.stage_end("finalize")
    _cleanup_sandbox(sandbox_path, repo_path, branch_name)
    return state


def _extract_repair_from_sandbox(sandbox_path: str) -> dict:
    """Extract repair info from sandbox diff for eval compatibility."""
    try:
        # Try committed diff first
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--name-only"],
            cwd=sandbox_path, capture_output=True, text=True, timeout=30,
        )
        files = [l.strip() for l in diff.stdout.splitlines() if l.strip()]
        if not files:
            # Fallback: uncommitted changes
            diff = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=sandbox_path, capture_output=True, text=True, timeout=30,
            )
            files = [l.strip() for l in diff.stdout.splitlines() if l.strip()]
            # Also check untracked
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=sandbox_path, capture_output=True, text=True, timeout=30,
            )
            for line in status.stdout.splitlines():
                if line.startswith("??"):
                    f = line[3:].strip()
                    if f not in files:
                        files.append(f)
        patches = [{"file_path": f} for f in files]
        return {"patches": patches, "explanation": "ReAct agent fix"}
    except Exception:
        return {"patches": []}


def _cleanup_sandbox(sandbox_path: str, repo_path: Path | None, branch_name: str) -> None:
    """Clean up the git worktree after finalization."""
    if not sandbox_path:
        return
    sp = Path(sandbox_path)
    try:
        if sp.exists():
            if repo_path:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(sp)],
                    cwd=repo_path, capture_output=True, timeout=30,
                )
            if sp.exists():
                shutil.rmtree(sp, ignore_errors=True)
    except Exception as e:
        logger.debug("Sandbox cleanup error: %s", e)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_react_graph():
    """Build the 3-node ReAct LangGraph state machine."""
    graph = StateGraph(ReactAgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "react_agent")
    graph.add_edge("react_agent", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Module-level compiled graph
react_app = build_react_graph()


def run_ticket_react(
    work_order: dict,
    progress_cb: Callable[[ReactAgentState], None] | None = None,
    trace: RunTrace | None = None,
    dry_run: bool = False,
) -> dict:
    """Run a bug ticket through the ReAct pipeline.

    Drop-in replacement for pipeline.run_ticket() but uses the ReAct architecture.
    """
    _thread_local.progress_callback = progress_cb
    _thread_local.trace = trace
    _thread_local.current_stage = "pending"

    initial_state: ReactAgentState = {
        "work_order": work_order,
        "intent": {},
        "sandbox_path": "",
        "branch_name": "",
        "base_branch": "",
        "submitted": False,
        "escalated": False,
        "escalate_reason": "",
        "explanation": "",
        "tool_call_count": 0,
        "cost_usd": 0.0,
        "messages": [],
        "localization": {},
        "repair": {},
        "review": {},
        "status": PipelineStatus.PENDING,
        "error": "",
        "pr_url": "",
        "test_result": "",
        "dry_run": dry_run,
    }

    try:
        result = react_app.invoke(initial_state)
        result_dict = dict(result)
        # Record metrics (same as fixed pipeline — keeps dashboard current)
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
