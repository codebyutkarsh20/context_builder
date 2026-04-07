"""
react_pipeline.py — ReAct agent pipeline (plain function chain).

Architecture:
    intake_node → react_agent_node → finalize_node

The react_agent_node runs a single while-loop where the LLM decides:
explore → localize → edit → test → review → submit.

No LangGraph dependency. Three functions called sequentially.
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


def _classify_community(repo_name: str, intent: dict, data_dir: Path) -> str | None:
    """Map a bug ticket to a Leiden community cluster in one cheap Haiku call.

    Reads communities.json built during `cli.py build` and asks Haiku which
    community best matches the bug description. Returns the community name or
    None if communities aren't available.
    """
    communities_path = data_dir / repo_name / "communities.json"
    if not communities_path.exists():
        return None

    import json as _json
    try:
        communities = _json.loads(communities_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read communities.json at %s: %s", communities_path, e)
        return None

    if not communities:
        return None

    from agent.llm import structured_call as _structured_call, INTAKE_MODEL
    from pydantic import BaseModel

    community_names = [c["name"] for c in communities]
    community_descriptions = []
    for c in communities:
        files_preview = ", ".join(c.get("dominant_files", [])[:3])
        community_descriptions.append(f"- {c['name']}: {files_preview} ({c.get('size', 0)} nodes)")

    class CommunityMatch(BaseModel):
        community_name: str
        confidence: float  # 0.0-1.0

    from agent.scout import _build_bug_text
    bug_text = _build_bug_text(work_order={}, intent=intent, max_chars=400)

    prompt = (
        f"Bug description: {bug_text}\n\n"
        f"Code communities available:\n" + "\n".join(community_descriptions) + "\n\n"
        "Which community does this bug most likely belong to? "
        "Pick the single best matching community_name from the list above."
    )

    try:
        result = _structured_call(INTAKE_MODEL, 100, CommunityMatch, prompt)
        if result.community_name in community_names and result.confidence > 0.4:
            return result.community_name
    except Exception as e:
        logger.debug("Community classification LLM call failed: %s", e)
    return None


def _prelocalize(repo_name: str, intent: dict, data_dir: Path) -> list[str]:
    """Lightweight pre-localization: ChromaDB + graph neighbors → top-5 confirmed files.

    Runs in intake_node before the ReAct loop starts. Narrows from all repo
    files down to the most likely 5, so the agent starts with a strong prior
    and skips most early exploration tool calls (~40% savings).

    Strategy (Agentless 2-phase pattern):
      1. ChromaDB semantic search on bug description → top-10 nodes
      2. Graph neighbor expansion from hint_files/functions → 5-8 files
      3. Union + score → top-5 by combined score
    """
    from agent.graph_utils import load_graph_data, find_callers_from_graph
    from collections import Counter

    hint_files = [f for f in intent.get("likely_affected_modules", [])[:5] if f]
    hint_functions = [f for f in intent.get("likely_affected_functions", [])[:5] if f]
    bug_query = " ".join(filter(None, [
        intent.get("actual_behavior", ""),
        intent.get("expected_behavior", ""),
        " ".join(hint_functions),
    ]))

    scores: Counter = Counter()

    # 1. Seed from LLM hints (highest confidence)
    for f in hint_files:
        scores[f] += 3

    # 2. ChromaDB vector search
    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo_name, data_dir)
        results = embedder.query(bug_query, n_results=10)
        for r in results:
            file_ = r.get("metadata", {}).get("file", "")
            if file_:
                scores[file_] += 2
    except Exception as e:
        logger.debug("ChromaDB pre-localization failed (non-fatal): %s", e)

    # 3. Graph neighbor expansion — files that call hint files/functions
    try:
        graph_data, _ = load_graph_data(repo_name)
        neighbors = find_callers_from_graph(graph_data, hint_files, hint_functions)
        for f in neighbors:
            scores[f] += 1
    except Exception as e:
        logger.debug("Graph neighbor expansion failed (non-fatal): %s", e)

    # Return top-5 unique file paths, excluding test files
    _test_noise = ("test_", "/tests/", "/test/", "conftest")
    ranked = [
        f for f, _ in scores.most_common(20)
        if f and not any(t in f.lower() for t in _test_noise)
    ]
    return ranked[:5]


def _find_repo_python(repo_path: Path) -> str:
    """Find the repo's virtualenv Python, falling back to sys.executable."""
    import sys
    for venv_dir in (".venv", "venv", "env"):
        candidate = repo_path / venv_dir / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


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
        except Exception as e:
            logger.debug("Progress callback error: %s", e)


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
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read graph.json for repo path: %s", e)
    repos_base = os.environ.get("REPOS_BASE_DIR", "")
    if repos_base:
        p = Path(repos_base) / repo_name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# intake_node helpers
# ---------------------------------------------------------------------------


def _translate_intent(work_order: dict) -> dict:
    """LLM call to translate a bug ticket into a structured intent; fallback on error."""
    from agent.llm import structured_call as _structured_call, INTAKE_MODEL
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
        return result.model_dump()
    except Exception as e:
        logger.error("Intent translation failed: %s", e)
        return {
            "expected_behavior": work_order.get("title", ""),
            "actual_behavior": work_order.get("description", ""),
            "likely_affected_modules": [],
            "likely_affected_functions": [],
            "fix_type": "bug_fix",
            "severity": work_order.get("priority", "medium"),
        }


def _enrich_with_stack_hints(state: ReactAgentState) -> ReactAgentState:
    """Extract stack trace hints from bug description and inject notes into intent."""
    from agent.intake_helpers import extract_stack_trace_hints

    work_order = state.get("work_order", {})
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
    return state


def _run_localization(state: ReactAgentState, repo_name: str) -> ReactAgentState:
    """Community classification + Scout FL + pre-localization fallback."""
    # Step 1: Community classifier — map bug ticket to a code cluster in 1 Haiku call.
    # Narrows the search space before any file-level localization.
    community_name: str | None = None
    try:
        if repo_name:
            community_name = _classify_community(repo_name, state.get("intent", {}), DATA_DIR)
            if community_name:
                intent = state.get("intent", {})
                intent["community"] = community_name
                state["intent"] = intent
                logger.info("Community classifier: bug maps to community '%s'", community_name)
    except Exception as e:
        logger.debug("Community classifier failed (non-fatal): %s", e)

    # Step 2: Scout FL pipeline (3-agent: Haiku extractor → Sonnet Graph-RAG → Opus re-ranker)
    # Produces top-5 suspicious locations with confidence scores.
    # Falls back gracefully to pre-localization if Scout fails.
    # Skipped when disable_scout=True (v2.0 baseline mode for A/B eval).
    _scout_disabled = getattr(_thread_local, "disable_scout", False)
    try:
        if repo_name and not _scout_disabled:
            from agent.scout import scout_localize
            scout_report = scout_localize(
                repo_name, state.get("work_order", {}), state.get("intent", {}), DATA_DIR,
                community_name=community_name,
            )
            top_locs = scout_report.get("top_locations", [])
            if top_locs:
                intent = state.get("intent", {})
                # Inject Scout's top files as confirmed_files (overrides simple pre-localization)
                scout_files = [loc["file"] for loc in top_locs if loc.get("file")][:5]
                if scout_files:
                    intent["confirmed_files"] = scout_files
                intent["scout_report"] = scout_report
                state["intent"] = intent
                logger.info(
                    "Scout FL: top locations %s (cost $%.4f)",
                    scout_files, scout_report.get("scout_cost_usd", 0),
                )
    except Exception as e:
        logger.debug("Scout FL pipeline failed (non-fatal): %s", e)
        # Fall back to simple pre-localization
        try:
            if repo_name:
                confirmed = _prelocalize(repo_name, state.get("intent", {}), DATA_DIR)
                if confirmed:
                    intent = state.get("intent", {})
                    intent["confirmed_files"] = confirmed
                    state["intent"] = intent
        except Exception as e:
            logger.debug("Pre-localization fallback also failed: %s", e)

    return state


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

    # 1. Translate ticket into structured intent
    state["intent"] = _translate_intent(work_order)

    # 2. Enrich with stack trace hints
    state = _enrich_with_stack_hints(state)

    # 3. Repro steps + bug category
    from agent.intake_helpers import extract_repro_steps, classify_bug_category
    description = work_order.get("description", "")

    repro_steps = extract_repro_steps(description)
    if repro_steps:
        work_order["repro_steps"] = repro_steps
        state["work_order"] = work_order

    bug_category = classify_bug_category(work_order.get("title", ""), description)
    work_order["bug_category"] = bug_category
    state["work_order"] = work_order

    if bug_category == "C":
        intent = state.get("intent", {})
        cat_c_note = (
            "NOTE: Bug shows complexity signals (category C). Attempt the fix — "
            "escalate only if you make no progress after 3+ explore/edit cycles, "
            "or the root cause spans 5+ files."
        )
        existing_notes = intent.get("notes", "")
        intent["notes"] = (existing_notes + "\n" + cat_c_note).strip() if existing_notes else cat_c_note
        state["intent"] = intent

    # 4. Localization: community detection + Scout FL + fallback
    repo_name = work_order.get("repo_name", "")
    state = _run_localization(state, repo_name)

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
    from agent.react_tools import set_react_context, _tls as _react_tls
    set_context(repo_name, repo_path, DATA_DIR)
    set_react_context(repo_name, repo_path, DATA_DIR)

    # Make BRTs accessible to the run_brt tool via thread-local
    _react_tls.brts = state.get("brts", [])

    # Build orientation context
    from agent.graph_utils import build_kickstart_context, load_business_rules
    kickstart = build_kickstart_context(repo_name, str(repo_path), intent, DATA_DIR)

    # Load conventions and business rules
    from agent.react_prompt import (
        build_system_prompt,
        build_task_message,
        load_project_conventions,
    )
    conventions = load_project_conventions(repo_name)

    # Business rules (use hint files from intent for scoping)
    hint_files = intent.get("likely_affected_modules", [])[:5]
    hint_functions = intent.get("likely_affected_functions", [])[:5]
    business_rules = load_business_rules(repo_name, hint_files) if hint_files else ""

    system_prompt = build_system_prompt(
        work_order=work_order,
        intent=intent,
        kickstart_context=kickstart,
        conventions_section=conventions,
        business_rules_section=business_rules,
        brts=state.get("brts", []),
    )
    task_message = build_task_message(work_order, intent)

    # Emit prompt_build for observability
    if trace:
        trace.emit("prompt_build", "react_agent", {
            "system_prompt_chars": len(system_prompt),
            "system_prompt_tokens_approx": len(system_prompt) // 4,
            "task_message_chars": len(task_message),
            "kickstart_chars": len(kickstart),
            "conventions_chars": len(conventions),
            "business_rules_chars": len(business_rules),
            "hint_files": hint_files,
            "hint_functions": hint_functions,
            "repo_name": repo_name,
        })

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
# finalize_node helpers
# ---------------------------------------------------------------------------


def _build_pr_body(state: ReactAgentState, ticket_id: str, sandbox_path: str) -> tuple[str, str]:
    """Construct the PR body markdown and title from agent state. Returns (pr_title, pr_body)."""
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
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("Failed to get diff stat: %s", e)

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
    return pr_title, pr_body


def _push_and_create_pr(
    sandbox_path: str, branch_name: str, base_branch: str, pr_title: str, pr_body: str,
) -> dict:
    """Git push + gh pr create. Returns dict with 'pr_url' and optional 'error'."""
    result_info: dict[str, str] = {}
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
            result_info["error"] = f"Push failed: {push_result.stderr[:200]}"
            result_info["pr_url"] = f"branch://{branch_name}"
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
                result_info["pr_url"] = pr_result.stdout.strip()
                logger.info("PR created: %s", result_info["pr_url"])
            else:
                logger.error("PR creation failed: %s", pr_result.stderr)
                result_info["pr_url"] = f"branch://{branch_name}"
                result_info["error"] = f"PR creation failed: {pr_result.stderr[:200]}"
    except Exception as e:
        result_info["error"] = f"PR creation error: {e}"
        result_info["pr_url"] = f"branch://{branch_name}"
    return result_info


def _populate_repair_and_localization(
    state: ReactAgentState, sandbox_path: str, repo_path: Path | None,
) -> ReactAgentState:
    """Extract repair info from sandbox diff and backfill localization if missing."""
    if not sandbox_path or not Path(sandbox_path).exists():
        return state
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

    # Build PR title + body
    pr_title, pr_body = _build_pr_body(state, ticket_id, sandbox_path)

    if dry_run:
        logger.info("DRY RUN — skipping PR creation")
        state["pr_url"] = "(dry-run — no PR created)"
        state["status"] = PipelineStatus.DONE

        # Populate repair + localization for eval compatibility
        state = _populate_repair_and_localization(state, sandbox_path, repo_path)

        # Run ground-truth tests before cleanup — only if the agent didn't already
        # pass them (avoids double-running). This is the authoritative test signal
        # for eval scoring (full_pass metric).
        repair = state.get("repair", {})
        test_already_passed = (state.get("test_result") or "").strip().lower().startswith("passed")
        if not test_already_passed and repair.get("patches"):
            try:
                from agent.sandbox import run_tests as _run_tests
                gt_result = _run_tests(Path(sandbox_path), repo_path=repo_path)
                state["test_result"] = gt_result[:2000]
                logger.info("Ground-truth tests (dry-run eval): %s", gt_result[:120])
            except Exception as e:
                logger.debug("Ground-truth test execution skipped: %s", e)

        _report_progress(state)
        if trace:
            trace.stage_end("finalize")
        _cleanup_sandbox(sandbox_path, repo_path, branch_name)
        return state

    # Push and create PR
    if sandbox_path and Path(sandbox_path).exists() and branch_name:
        pr_info = _push_and_create_pr(sandbox_path, branch_name, base_branch, pr_title, pr_body)
        state["pr_url"] = pr_info.get("pr_url", f"branch://{branch_name}")
        if "error" in pr_info:
            state["error"] = pr_info["error"]

    state["status"] = PipelineStatus.DONE

    # Populate repair + localization for eval compatibility
    state = _populate_repair_and_localization(state, sandbox_path, repo_path)

    _report_progress(state)
    if trace:
        trace.stage_end("finalize")
    _cleanup_sandbox(sandbox_path, repo_path, branch_name)
    return state


# ---------------------------------------------------------------------------
# brt_node helpers
# ---------------------------------------------------------------------------


def _read_brt_source_snippets(repo_path: Path, hint_files: list[str]) -> list[str]:
    """Read source code from hint files, capped to 3000 chars each for Haiku context."""
    source_snippets: list[str] = []
    for fpath in hint_files[:2]:
        full = repo_path / fpath
        if full.exists() and full.suffix == ".py":
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                # Cap to 3000 chars per file to stay within Haiku context
                source_snippets.append(f"# {fpath}\n{content[:3000]}")
            except OSError as e:
                logger.debug("Failed to read BRT source file %s: %s", fpath, e)
    return source_snippets


def _run_brt_candidates(candidates: list, repo_path: Path, repo_python: str) -> list[dict]:
    """Execute BRT candidates against the original repo; return confirmed BRTs (max 3)."""
    import tempfile

    confirmed_brts: list[dict] = []
    for cand in candidates:
        if len(confirmed_brts) >= 3:
            break
        code = cand.test_code.strip()
        if not code.startswith("def test_") and "def test_" not in code:
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", prefix="brt_", dir=str(repo_path), mode="w",
                encoding="utf-8", delete=False,
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            # File is closed before subprocess reads it (avoids TOCTOU race)
            result = subprocess.run(
                [repo_python, "-m", "pytest", tmp_path, "--tb=short", "-x", "-q", "--no-header"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Exit code 1 = test ran and FAILED = confirmed BRT (catches the bug!)
            if result.returncode == 1:
                confirmed_brts.append({
                    "code": code,
                    "description": cand.description[:200],
                    "target_function": cand.target_function,
                    "fail_output": (result.stdout + result.stderr)[:500],
                })
                logger.info("BRT confirmed: '%s' (fails on current code)", cand.description[:80])
            else:
                logger.debug(
                    "BRT candidate not confirmed (exit %d): %s",
                    result.returncode, cand.description[:60],
                )
        except subprocess.TimeoutExpired:
            logger.debug("BRT candidate timed out: %s", cand.description[:60])
        except Exception as e:
            logger.debug("BRT candidate error: %s", e)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return confirmed_brts


# ---------------------------------------------------------------------------
# Node 4: BRT (Bug Reproduction Test)
# ---------------------------------------------------------------------------

def brt_node(state: ReactAgentState) -> ReactAgentState:
    """Bug Reproduction Test (BRT) generator — runs before the fix loop.

    Google Passerine / TDD-Bench Verified pattern:
    Before writing a single line of fix code, generate tests that FAIL on the
    broken codebase. These confirmed BRTs become the objective function for the
    Engineer: the fix is correct when all BRTs pass.

    Flow:
      1. Read suspected functions from confirmed_files / hint_files
      2. Haiku generates 5-7 test candidates from the bug description + code
      3. Each candidate is run against the ORIGINAL repo (not a sandbox)
      4. Confirmed BRTs: exit code 1 (assertion failure) = test catches the bug
      5. Up to 3 confirmed BRTs stored in state["brts"], injected into system prompt

    Only runs for Python repos (pytest). Skips gracefully for JS/other.
    Non-fatal: if BRT generation fails, agent proceeds without BRTs.
    """
    _thread_local.current_stage = "brt"
    trace = _get_trace()
    if trace:
        trace.stage_start("brt")

    work_order = state.get("work_order", {})
    intent = state.get("intent", {})
    repo_path = _resolve_repo_path(work_order)

    if not repo_path:
        logger.debug("BRT node: no repo_path — skipping")
        if trace:
            trace.stage_end("brt")
        return state

    # Only generate BRTs for Python repos (check for pytest marker files)
    _pytest_markers = ("pytest.ini", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini")
    is_python_repo = any((repo_path / m).exists() for m in _pytest_markers)
    if not is_python_repo:
        logger.debug("BRT node: non-Python repo — skipping BRT generation")
        if trace:
            trace.stage_end("brt")
        return state

    hint_files = (
        intent.get("confirmed_files", []) or
        intent.get("likely_affected_modules", [])
    )[:3]

    if not hint_files:
        logger.debug("BRT node: no hint files — skipping BRT generation")
        if trace:
            trace.stage_end("brt")
        return state

    # 1. Read source code of suspected functions
    source_snippets = _read_brt_source_snippets(repo_path, hint_files)
    if not source_snippets:
        if trace:
            trace.stage_end("brt")
        return state

    source_context = "\n\n".join(source_snippets)

    # 2. Generate BRT candidates with Haiku (cheap, fast)
    from agent.llm import structured_call as _structured_call, INTAKE_MODEL
    from pydantic import BaseModel

    class BRTCandidate(BaseModel):
        test_code: str         # Complete pytest test function (must start with "def test_")
        description: str       # One sentence: what bug this reproduces
        target_function: str   # Function name being tested

    class BRTBatch(BaseModel):
        candidates: list[BRTCandidate]

    hint_functions = intent.get("likely_affected_functions", [])[:3]
    prompt = (
        f"BUG: {work_order.get('title', '')}\n"
        f"DESCRIPTION: {work_order.get('description', '')[:500]}\n"
        f"EXPECTED: {intent.get('expected_behavior', '')[:200]}\n"
        f"ACTUAL: {intent.get('actual_behavior', '')[:200]}\n"
        f"FUNCTIONS TO TEST: {hint_functions}\n\n"
        f"SOURCE CODE:\n{source_context}\n\n"
        "Generate 4-5 pytest test functions that:\n"
        "1. FAIL on the CURRENT (broken) code (they should catch the bug)\n"
        "2. Would PASS after a correct fix\n"
        "3. Are self-contained — they import from the source files using the repo root as working dir\n"
        "4. Use assert statements, NOT pytest.raises (unless the bug IS an unexpected exception)\n"
        "5. Each test MUST start with 'def test_' and be a plain function (no fixtures)\n\n"
        "Write tests that directly exercise the buggy behaviour. "
        "Import using the relative module path shown in the source (e.g. 'from backend.app import foo')."
    )

    try:
        batch = _structured_call(INTAKE_MODEL, 3500, BRTBatch, prompt)
        candidates = batch.candidates[:6]
        logger.info("BRT node: generated %d candidates", len(candidates))
    except Exception as e:
        logger.debug("BRT candidate generation failed: %s", e)
        if trace:
            trace.stage_end("brt")
        return state

    # 3. Run candidates against original repo — keep only confirmed BRTs
    # Fix #12: Use the repo's virtualenv python if available, not sys.executable
    repo_python = _find_repo_python(repo_path)
    confirmed_brts = _run_brt_candidates(candidates, repo_path, repo_python)

    if confirmed_brts:
        state["brts"] = confirmed_brts
        logger.info(
            "BRT node: %d/%d candidates confirmed as BRTs",
            len(confirmed_brts), len(candidates),
        )
        if trace:
            trace.emit("brt_confirmed", "brt", {
                "confirmed": len(confirmed_brts),
                "total_candidates": len(candidates),
                "descriptions": [b["description"] for b in confirmed_brts],
            })
    else:
        logger.info("BRT node: no candidates confirmed (all passed or errored) — proceeding without BRTs")

    if trace:
        trace.stage_end("brt")
    return state


def verifier_node(state: ReactAgentState) -> ReactAgentState:
    """Independent verifier subagent — fresh-context review before PR creation.

    Implements two patterns from research:
    - OpenHands independent verifier: fresh LLM with only diff + tests votes pass/fail
    - cavekit speculative review: starts early, adds zero latency on the happy path

    Input: bug description + git diff + test result
    Output: verifier_verdict ('APPROVE'/'REJECT'), verifier_explanation, verifier_confidence
    Budget: 1 LLM call, no tools (fresh context only)
    """
    if not state.get("submitted"):
        return state  # Only verify if agent produced a patch

    sandbox_path = state.get("sandbox_path", "")
    if not sandbox_path or not Path(sandbox_path).exists():
        return state

    work_order = state.get("work_order", {})
    test_result = state.get("test_result", "not run")
    explanation = state.get("explanation", "")

    # Get the diff
    diff_text = ""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1"],
            cwd=sandbox_path, capture_output=True, text=True, timeout=30,
        )
        diff_text = result.stdout[:8000]
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Failed to get diff for verifier: %s", e)

    if not diff_text:
        # Nothing to verify — no committed changes
        return state

    from agent.llm import structured_call as _structured_call
    from pydantic import BaseModel

    class VerifierResult(BaseModel):
        verdict: str          # "APPROVE" or "REJECT"
        confidence: float     # 0.0-1.0
        explanation: str      # Why APPROVE/REJECT in 2-3 sentences
        regression_risk: str  # "LOW", "MEDIUM", "HIGH"

    prompt = (
        f"You are an independent code reviewer. Review this patch with fresh eyes.\n\n"
        f"BUG: {work_order.get('title', '')}\n"
        f"DESCRIPTION: {work_order.get('description', '')[:500]}\n\n"
        f"PATCH (what the agent changed):\n```diff\n{diff_text}\n```\n\n"
        f"TESTS: {test_result[:500]}\n\n"
        f"AGENT EXPLANATION: {explanation[:300]}\n\n"
        "Evaluate:\n"
        "1. Does this patch actually fix the described bug?\n"
        "2. Are there any obvious regressions or side effects?\n"
        "3. Is the change minimal and correct?\n\n"
        "APPROVE if: patch clearly fixes the bug, no obvious regressions.\n"
        "REJECT if: patch doesn't address the bug, introduces new bugs, or is dangerously wrong."
    )

    # Run confirmed BRTs against the patched sandbox (speculative EPR check)
    brts = state.get("brts", [])
    brt_pass_count = 0
    brt_total = len(brts)
    if brts and sandbox_path and Path(sandbox_path).exists():
        import tempfile
        sandbox_python = _find_repo_python(Path(sandbox_path))
        for brt in brts:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".py", prefix="brt_verify_", dir=sandbox_path,
                    mode="w", encoding="utf-8", delete=False,
                ) as tmp:
                    tmp.write(brt["code"])
                    tmp_path = tmp.name
                # File closed before subprocess reads it
                result_brt = subprocess.run(
                    [sandbox_python, "-m", "pytest", tmp_path, "--tb=short", "-x", "-q", "--no-header"],
                    cwd=sandbox_path,
                    capture_output=True, text=True, timeout=30,
                )
                if result_brt.returncode == 0:
                    brt_pass_count += 1
                    logger.info("BRT passed in sandbox: %s", brt["description"][:60])
                else:
                    logger.warning("BRT still failing after fix: %s", brt["description"][:60])
            except Exception as e:
                logger.debug("BRT sandbox run error: %s", e)
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

        state["brt_pass_count"] = brt_pass_count
        state["brt_total"] = brt_total
        epr = brt_pass_count / brt_total if brt_total > 0 else 0.0
        state["epr_score"] = round(epr, 3)
        logger.info("EPR score: %.0f%% (%d/%d BRTs pass)", epr * 100, brt_pass_count, brt_total)

    try:
        result = _structured_call("claude-sonnet-4-6", 1200, VerifierResult, prompt)

        # Validate the Pydantic result has sane values
        if result.verdict not in ("APPROVE", "REJECT"):
            logger.warning("Verifier returned invalid verdict '%s' — treating as REJECT", result.verdict)
            result.verdict = "REJECT"
        result.confidence = max(0.0, min(1.0, result.confidence))

        state["verifier_verdict"] = result.verdict
        state["verifier_confidence"] = result.confidence
        state["verifier_explanation"] = result.explanation
        state["verifier_regression_risk"] = result.regression_risk

        logger.info(
            "Verifier verdict: %s (confidence: %.0f%%, risk: %s)",
            result.verdict, result.confidence * 100, result.regression_risk,
        )

        # If verifier rejects with high confidence, flag it but don't block
        # (the main agent's review already approved — verifier is advisory)
        if result.verdict == "REJECT" and result.confidence > 0.8:
            existing_notes = state.get("explanation", "")
            state["explanation"] = (
                existing_notes + f"\n\n⚠ VERIFIER FLAGGED: {result.explanation}"
            ).strip()
            logger.warning("Verifier rejected with high confidence — flagged in PR body")

    except Exception as e:
        logger.warning("Verifier node failed (non-fatal): %s", e)
        state["verifier_verdict"] = "SKIP"
        state["verifier_explanation"] = f"Verifier error: {e}"

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
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Failed to extract repair from sandbox: %s", e)
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
# Pipeline execution — plain function chain, no LangGraph needed.
# The old StateGraph added complexity for 3 linear nodes with no
# conditional routing. A simple function sequence does the same thing.
# ---------------------------------------------------------------------------

def run_ticket_react(
    work_order: dict,
    progress_cb: Callable[[ReactAgentState], None] | None = None,
    trace: RunTrace | None = None,
    dry_run: bool = False,
    best_of_n: int = 1,
    disable_brt: bool = False,
    disable_scout: bool = False,
) -> dict:
    """Run a bug ticket through the ReAct pipeline.

    Stages:
      1. intake_node    — translate ticket to intent + pre-localization
      2. brt_node       — generate failing tests before fix (skipped if disable_brt)
      3. react_agent_node — ReAct loop with tools
      4. verifier_node  — independent fresh-context review (cavekit speculative pattern)
      5. finalize_node  — create PR or escalate

    disable_brt / disable_scout: v2.0 baseline flags used by eval runner for A/B.
    best_of_n > 1: runs N parallel instances, picks winner by test pass then
    review confidence (SWE-agent best-of-N pattern, +10-15pp submit rate).
    """
    if best_of_n > 1:
        return _run_best_of_n(work_order, progress_cb, trace, dry_run, best_of_n)

    _thread_local.progress_callback = progress_cb
    _thread_local.trace = trace
    _thread_local.current_stage = "pending"
    # Store feature flags so nodes can read them
    _thread_local.disable_brt = disable_brt
    _thread_local.disable_scout = disable_scout

    state: ReactAgentState = {
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
        state = intake_node(state)
        if not getattr(_thread_local, "disable_brt", False):
            state = brt_node(state)    # BRT-first: generate failing tests before fix
        state = react_agent_node(state)
        state = verifier_node(state)   # Independent reviewer (speculative pattern)
        state = finalize_node(state)

        result_dict = dict(state)
        try:
            from api.metrics import record_run
            record_run(result_dict)
        except Exception as e:
            logger.debug("Metrics recording failed: %s", e)
        return result_dict
    finally:
        _thread_local.progress_callback = None
        if trace:
            trace.complete()
        _thread_local.trace = None


def _run_best_of_n(
    work_order: dict,
    progress_cb: Callable[[ReactAgentState], None] | None,
    trace: RunTrace | None,
    dry_run: bool,
    n: int,
) -> dict:
    """Run N parallel agent instances and pick the best patch.

    Uses ProcessPoolExecutor to avoid thread-local state pollution across
    concurrent instances. Each process gets its own thread-local namespace.

    Selection order (SWE-agent pattern):
      1. tests_passed (test_result starts with "passed")
      2. verifier APPROVE
      3. review confidence score
      4. cost (cheapest of ties)
    """
    import concurrent.futures

    n = min(n, 5)  # Hard cap: 5 instances max
    logger.info("Best-of-%d: launching %d parallel agent instances", n, n)

    def _run_one(seed: int) -> dict:
        # Give each instance a unique ticket suffix so sandbox branches don't collide
        wo = {**work_order, "ticket_id": f"{work_order.get('ticket_id', 'UNKNOWN')}_bon{seed}"}
        return run_ticket_react(wo, progress_cb=None, trace=None, dry_run=dry_run, best_of_n=1)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(_run_one, i) for i in range(n)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.warning("Best-of-N instance failed: %s", e)

    if not results:
        # All instances failed — return a stub escalated result
        return {
            "work_order": work_order, "submitted": False, "escalated": True,
            "escalate_reason": "All best-of-N instances failed",
            "status": PipelineStatus.ESCALATED,
        }

    def _score(r: dict) -> tuple:
        test_ok = (r.get("test_result") or "").strip().lower().startswith("passed")
        verifier_ok = r.get("verifier_verdict", "") == "APPROVE"
        confidence = (r.get("review") or {}).get("confidence", 0.0)
        cost = -(r.get("cost_usd") or 0.0)  # Negative so lower cost = higher rank
        submitted = r.get("submitted", False)
        return (submitted, test_ok, verifier_ok, confidence, cost)

    best = max(results, key=_score)
    submitted_count = sum(1 for r in results if r.get("submitted"))
    test_pass_count = sum(1 for r in results if (r.get("test_result") or "").startswith("passed"))
    logger.info(
        "Best-of-%d complete: %d/%d submitted, %d/%d tests passed — selected by %s",
        n, submitted_count, n, test_pass_count, n,
        "tests" if (best.get("test_result") or "").startswith("passed") else "review",
    )

    # Clean up losing sandboxes (Fix #14: N-1 sandboxes were leaked)
    repo_path = _resolve_repo_path(work_order)
    best_sandbox = best.get("sandbox_path", "")
    for r in results:
        sb = r.get("sandbox_path", "")
        if sb and sb != best_sandbox:
            _cleanup_sandbox(sb, repo_path, r.get("branch_name", ""))

    best["best_of_n_stats"] = {
        "n": n, "submitted": submitted_count, "test_pass": test_pass_count,
    }

    try:
        from api.metrics import record_run
        record_run(best)
    except Exception as e:
        logger.debug("Metrics recording failed: %s", e)
    return best
