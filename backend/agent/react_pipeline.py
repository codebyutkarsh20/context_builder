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
    """Lightweight pre-localization: LLM hints + graph neighbors → top-5 confirmed files.

    Runs in intake_node before the ReAct loop starts. Narrows from all repo
    files down to the most likely 5, so the agent starts with a strong prior
    and skips most early exploration tool calls (~40% savings).

    Strategy (Agentless 2-phase pattern):
      1. Seed from LLM hints (highest confidence)
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

    # 2. Graph neighbor expansion — files that call hint files/functions
    try:
        graph_data, _ = load_graph_data(repo_name)
        neighbors = find_callers_from_graph(graph_data, hint_files, hint_functions)
        for f in neighbors:
            scores[f] += 1
    except Exception as e:
        logger.debug("Graph neighbor expansion failed (non-fatal): %s", e)

    # 3. Flow boost — files in high-criticality flows touching hint area
    try:
        import json as _json_f
        flows_path = data_dir / repo_name / "flows.json"
        if flows_path.exists():
            flows_data = _json_f.loads(flows_path.read_text())
            hint_set = set(hint_files)
            for flow in flows_data.get("flows", []):
                flow_files = set(flow.get("files", []))
                if flow_files & hint_set:
                    crit = flow.get("criticality", 0)
                    for ff in flow_files:
                        scores[ff] += crit  # higher criticality → bigger boost
    except Exception as e:
        logger.debug("Flow boost failed (non-fatal): %s", e)

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

    # Natural-language mode: description uses business/symptom language with no code terms.
    # Guide the LLM to focus on the observable symptom and broad module areas, not specific
    # function names it can't know from the description alone.
    nl_mode = work_order.get("_natural_lang", False)
    nl_guidance = ""
    if nl_mode:
        nl_guidance = (
            "\n\nNOTE: This ticket is written in business/user language with no code terms. "
            "For likely_affected_modules: list broad module paths (e.g. 'auth/', 'api/routes.py') based on the "
            "feature area described — do NOT guess specific function names. "
            "For likely_affected_functions: leave EMPTY unless you can infer a highly likely function from the symptom. "
            "The agent will use get_file_structure to find the actual function names at runtime."
        )

    prompt = f"""Translate this bug ticket into a technical specification.

Ticket: {work_order.get('title', '')}
Description: {work_order.get('description', '')}
Priority: {work_order.get('priority', 'unknown')}
Component: {work_order.get('affected_component', 'unknown')}
Comments: {'; '.join(work_order.get('comments', []))}

Include acceptance_criteria: 2-4 testable assertions derived from the bug description
that prove the fix works. These must come from the SPEC (what the user reported),
not from guessing the implementation.{nl_guidance}"""

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
                # Inject Scout's top files — but VALIDATE they exist first.
                # Scout can hallucinate paths based on bug description.
                repo_path_for_check = _resolve_repo_path(state.get("work_order", {}))
                scout_files_raw = [loc["file"] for loc in top_locs if loc.get("file")][:5]
                scout_files = []
                for sf in scout_files_raw:
                    if repo_path_for_check and (repo_path_for_check / sf).exists():
                        scout_files.append(sf)
                    else:
                        logger.warning("Scout hallucinated path (doesn't exist): %s", sf)
                if scout_files:
                    intent["confirmed_files"] = scout_files
                    intent["likely_affected_modules"] = scout_files  # Override with validated paths
                else:
                    logger.warning("All scout paths hallucinated — falling back to pre-localization")
                intent["scout_report"] = scout_report
                state["intent"] = intent
                logger.info(
                    "Scout FL: validated %d/%d locations %s (cost $%.4f)",
                    len(scout_files), len(scout_files_raw),
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

    # Clean up stale sandboxes from previous runs (best-effort, non-blocking)
    repo_path = _resolve_repo_path(state.get("work_order", {}))
    if repo_path:
        from agent.sandbox import cleanup_stale_worktrees
        try:
            cleanup_stale_worktrees(repo_path)
        except Exception as e:
            logger.debug("Stale worktree cleanup failed (non-fatal): %s", e)
    # Auto-detect project type and write .agent_config.json if missing.
    # This ensures sandbox.run_tests, check_syntax, and linters all know
    # the repo's language, test runner, and lint tool — even for repos the
    # agent has never seen before. Existing .agent_config.json takes priority
    # (detection only fills gaps).
    if repo_path and repo_path.is_dir():
        try:
            from agent.repo_detection import write_agent_config_from_detection
            config_path = repo_path / ".agent_config.json"
            if not config_path.exists():
                write_agent_config_from_detection(repo_path)
                logger.info("Auto-detected project config for %s", repo_path.name)
        except Exception as e:
            logger.debug("repo_detection failed (non-fatal): %s", e)

    state["status"] = PipelineStatus.INTAKE
    _report_progress(state)

    work_order = state.get("work_order", {})

    # 1. Translate ticket into structured intent
    state["intent"] = _translate_intent(work_order)

    # 1b. Concept-to-code mapping: query BusinessRule nodes for ticket keywords
    #     and inject any matched functions/files as localization hints.
    #     This bridges business-language tickets (e.g. "requisition approval flow")
    #     to code entities when grep-based search would find nothing.
    from agent.graph_utils import query_concept_to_code
    repo_name_for_c2c = work_order.get("repo_name", "")
    title = work_order.get("title", "")
    description = work_order.get("description", "")
    c2c = query_concept_to_code(title, description, repo_name_for_c2c)
    if c2c.get("matched_rules"):
        intent = state.get("intent", {})
        # Merge graph-derived hints into the intent, deduplicating against LLM hints
        existing_funcs = intent.get("likely_affected_functions", []) or []
        existing_mods = intent.get("likely_affected_modules", []) or []
        merged_funcs = list(dict.fromkeys(existing_funcs + c2c["hint_functions"]))
        merged_mods = list(dict.fromkeys(existing_mods + c2c["hint_files"]))
        intent["likely_affected_functions"] = merged_funcs[:8]
        intent["likely_affected_modules"] = merged_mods[:6]
        # Stash concept section so react_agent_node can inject it into kickstart
        intent["_concept_section"] = c2c["concept_section"]
        state["intent"] = intent
        logger.info(
            "Concept-to-code: merged %d rules → intent.likely_affected_functions=%s",
            len(c2c["matched_rules"]), merged_funcs[:4],
        )

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
    fix_type = intent.get("fix_type", "bug_fix") if intent else "bug_fix"
    set_react_context(repo_name, repo_path, DATA_DIR, fix_type=fix_type)

    # Make BRTs accessible to the run_brt tool via thread-local
    _react_tls.brts = state.get("brts", [])

    # Build orientation context
    from agent.graph_utils import build_kickstart_context, load_business_rules
    kickstart = build_kickstart_context(repo_name, str(repo_path), intent, DATA_DIR)

    # Learn-from-fix: prepend lessons from past runs on this repo. The
    # learn_from_fix module reads {DATA_DIR}/{repo_name}/agent_lessons.md
    # (populated by record_lesson in finalize_node on previous runs).
    # Returns "" if no lessons exist or feature is disabled.
    try:
        from agent.learn_from_fix import load_lessons
        lessons_section = load_lessons(repo_name)
        if lessons_section:
            # Prepend so it's among the first things the agent sees
            kickstart = lessons_section + "\n\n" + kickstart
            logger.info(
                "Injected %d chars of past-run lessons for %s",
                len(lessons_section), repo_name,
            )
    except Exception as e:
        logger.debug("load_lessons failed (non-fatal): %s", e)

    # BUILD STRUCTURED CODE MAP for localized files — gives the agent a compact
    # overview (function signatures + line numbers) instead of dumping whole files.
    # Research: Composio FQDN maps use ~500 tokens vs 25K for whole files.
    # The agent reads specific sections via read_file when it needs the actual code.
    hint_files_for_map = intent.get("likely_affected_modules", [])[:5]
    code_map_sections = []
    for hf in hint_files_for_map:
        try:
            fpath = repo_path / hf
            if not fpath.exists():
                continue
            content = fpath.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            # Extract function/class signatures with line numbers
            import re
            sigs = []
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                lineno = i + 1
                # Python patterns
                if re.match(r"^\s*(class\s+\w+|(?:async\s+)?def\s+\w+)", stripped):
                    sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                # Decorators (show what the function does)
                elif re.match(r"^\s*@(router|app)\.", stripped):
                    sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
            if sigs:
                code_map_sections.append(
                    f"{hf} ({len(lines)} lines):\n" + "\n".join(sigs)
                )
                logger.info("Code map for %s: %d signatures from %d lines", hf, len(sigs), len(lines))
        except Exception as e:
            logger.debug("Could not build code map for %s: %s", hf, e)

    if code_map_sections:
        kickstart += (
            "\n\n## CODE MAP (function signatures + line numbers for localized files)\n"
            "Use read_file(file, start_line, end_line) to read the specific section you need.\n\n"
            + "\n\n".join(code_map_sections)
        )
        logger.info("Code map: %d files, %d total chars", len(code_map_sections), sum(len(s) for s in code_map_sections))
    else:
        # No code map — scout paths didn't match real files. Give the agent a directory listing instead.
        logger.warning("No code map built — hint files don't exist. Adding directory listing as fallback.")
        try:
            routers_dir = repo_path / "backend" / "app" / "routers"
            if routers_dir.exists():
                py_files = sorted(routers_dir.glob("*.py"))
                listing = "\n".join(f"  {f.relative_to(repo_path)} ({f.stat().st_size // 1000}KB)" for f in py_files)
                kickstart += f"\n\n## REPO FILES (no code map available — start with get_file_structure)\n{listing}"
            else:
                # Generic fallback — list top-level Python files
                py_files = sorted(repo_path.rglob("*.py"))[:20]
                listing = "\n".join(f"  {f.relative_to(repo_path)}" for f in py_files)
                kickstart += f"\n\n## REPO FILES (no code map available — start with get_file_structure)\n{listing}"
        except Exception:
            pass

    # Inject concept-to-code section into kickstart if present (set during intake)
    concept_section = intent.get("_concept_section", "")
    if concept_section:
        kickstart += f"\n\n{concept_section}"
        logger.info("Injected concept-to-code section (%d chars) into kickstart", len(concept_section))

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

    static_block, dynamic_block = build_system_prompt(
        work_order=work_order,
        intent=intent,
        kickstart_context=kickstart,
        conventions_section=conventions,
        business_rules_section=business_rules,
        brts=state.get("brts", []),
    )
    task_message = build_task_message(work_order, intent)

    # Emit prompt_build for observability — include full prompt text for replay/iteration
    if trace:
        import subprocess
        git_sha = ""
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
            ).decode().strip()[:12]
        except Exception:
            pass

        full_prompt = static_block + "\n\n" + dynamic_block
        trace.emit("prompt_build", "react_agent", {
            "system_prompt_chars": len(full_prompt),
            "system_prompt_tokens_approx": len(full_prompt) // 4,
            "system_prompt_text": full_prompt,
            "static_block_chars": len(static_block),
            "dynamic_block_chars": len(dynamic_block),
            "task_message_text": task_message,
            "task_message_chars": len(task_message),
            "kickstart_chars": len(kickstart),
            "conventions_chars": len(conventions),
            "business_rules_chars": len(business_rules),
            "hint_files": hint_files,
            "hint_functions": hint_functions,
            "repo_name": repo_name,
            "agent_git_sha": git_sha,
        })

    # Run the ReAct loop
    from agent.react_loop import react_loop
    state = react_loop(
        state=state,
        static_block=static_block,
        dynamic_block=dynamic_block,
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
            # Check if origin exists and points to GitHub
            remote_url = ""
            try:
                remote_url = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=sandbox_path, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception:
                pass

            if remote_url and "github.com" in remote_url:
                # Origin exists and is GitHub — add auth token
                auth_url = remote_url.replace(
                    "https://", f"https://x-access-token:{gh_token}@"
                )
                subprocess.run(
                    ["git", "remote", "set-url", "origin", auth_url],
                    cwd=sandbox_path, capture_output=True, timeout=10,
                )
            elif not remote_url or "github.com" not in remote_url:
                # No origin or origin is not GitHub — try to detect GitHub repo via gh CLI
                try:
                    gh_repo = subprocess.run(
                        ["gh", "repo", "view", "--json", "url", "-q", ".url"],
                        cwd=sandbox_path, capture_output=True, text=True, timeout=15,
                    ).stdout.strip()
                    if gh_repo and "github.com" in gh_repo:
                        auth_url = gh_repo.replace(
                            "https://github.com/",
                            f"https://x-access-token:{gh_token}@github.com/"
                        ) + ".git"
                        if remote_url:
                            subprocess.run(
                                ["git", "remote", "set-url", "origin", auth_url],
                                cwd=sandbox_path, capture_output=True, timeout=10,
                            )
                        else:
                            subprocess.run(
                                ["git", "remote", "add", "origin", auth_url],
                                cwd=sandbox_path, capture_output=True, timeout=10,
                            )
                        logger.info("Set origin to GitHub: %s", gh_repo)
                except Exception as gh_err:
                    logger.debug("Could not detect GitHub repo: %s", gh_err)

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

    if state.get("escalated") or not state.get("submitted"):
        reason = state.get("escalate_reason", "Agent did not submit or escalate")
        logger.info("ESCALATED: %s — reason: %s", ticket_id, reason)
        state["status"] = PipelineStatus.ESCALATED
        state["escalated"] = True
        state["escalate_reason"] = reason

        # Even on escalation, create a PR if the agent made edits and got review
        # approval. This preserves the work for human review instead of discarding it.
        review = state.get("review", {})
        has_edits = sandbox_path and Path(sandbox_path).exists() and branch_name
        review_approved = review.get("verdict") == "APPROVE"
        if has_edits and review_approved and not dry_run:
            logger.info("Escalated but review approved — creating PR for human review")
            pr_title, pr_body = _build_pr_body(state, ticket_id, sandbox_path)
            pr_body += "\n\n> **Note:** Agent escalated after making this fix. " \
                       "Review approved but agent could not complete the full pipeline. " \
                       f"Reason: {reason[:200]}"
            pr_info = _push_and_create_pr(
                sandbox_path, branch_name, base_branch, pr_title, pr_body,
            )
            state["pr_url"] = pr_info.get("pr_url", f"branch://{branch_name}")
            if "error" in pr_info:
                logger.warning("PR creation on escalation failed: %s", pr_info["error"])

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
                from agent.agent_config import AgentConfig

                # Use test config from work_order if provided (eval bugs carry
                # test_command/setup_commands from bugs.json).  Live repos
                # without these fields fall back to auto-detection as before.
                eval_test_cmd = work_order.get("test_command", "")
                if eval_test_cmd:
                    eval_cfg = AgentConfig({
                        "test_command": eval_test_cmd,
                        "setup_commands": work_order.get("setup_commands", []),
                        "test_timeout": work_order.get("test_timeout", 300),
                    })
                    gt_result = _run_tests(
                        Path(sandbox_path), repo_path=repo_path, agent_config=eval_cfg,
                    )
                else:
                    gt_result = _run_tests(Path(sandbox_path), repo_path=repo_path)
                state["test_result"] = gt_result[:2000]
                logger.info("Ground-truth tests (dry-run eval): %s", gt_result[:120])
            except Exception as e:
                logger.debug("Ground-truth test execution skipped: %s", e)

        _report_progress(state)
        # Record a lesson for future runs (escalated path)
        _safe_record_lesson(state)
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
    # Record a lesson for future runs (successful path)
    _safe_record_lesson(state)
    if trace:
        trace.stage_end("finalize")
    _cleanup_sandbox(sandbox_path, repo_path, branch_name)
    return state


def _safe_record_lesson(state: dict) -> None:
    """Call learn_from_fix.record_lesson with all exceptions swallowed.

    record_lesson already swallows its own errors, but we add a second
    layer of safety here since this runs during finalize (post-submit)
    and any crash would break the pipeline.
    """
    try:
        from agent.learn_from_fix import record_lesson
        record_lesson(state)
    except Exception as e:
        logger.debug("record_lesson wrapper failed (non-fatal): %s", e)


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

    # If the agent produced a plan, include it in the verifier prompt so the
    # verifier can cross-check whether the diff matches the declared plan.
    plan_section = ""
    try:
        from agent.react_tools import get_current_plan
        plan = get_current_plan()
        if plan:
            plan_section = (
                "\n=== AGENT'S DECLARED PLAN ===\n"
                f"Root cause: {plan.get('root_cause', '')}\n"
                f"Target files: {', '.join(plan.get('target_files', []))}\n"
                f"Approach: {plan.get('approach', '')}\n"
                f"Success criteria: {plan.get('success_criteria', '')}\n"
                f"Risk: {plan.get('risk', 'LOW')}\n\n"
                "**Cross-check**: does the diff match this plan? If the diff "
                "touches files NOT in target_files, or implements a different "
                "approach than declared, that's a REJECT signal — the agent "
                "deviated from its plan without justification.\n"
            )
    except Exception as e:
        logger.debug("Plan fetch for verifier failed (non-fatal): %s", e)

    # Hardened verifier prompt (ported from Claude Code's verificationAgent).
    # Anti-rationalization framing + required adversarial reasoning before APPROVE.
    prompt = (
        "You are a verification specialist. Your job is NOT to confirm the patch works — "
        "it's to try to break it on paper.\n\n"
        "You have two documented failure patterns:\n"
        "1. **Verification avoidance**: when faced with a check, you find reasons not "
        "to run it — you read the diff, narrate what you would test, write 'APPROVE,' "
        "and move on.\n"
        "2. **Seduced by the first 80%**: you see a passing test result and feel "
        "inclined to approve, not noticing that the patch only handles the happy path, "
        "or that the test only covers one of three reported symptoms.\n\n"
        "Your entire value is in finding the last 20%.\n\n"
        "=== INPUT ===\n"
        f"BUG: {work_order.get('title', '')}\n"
        f"DESCRIPTION: {work_order.get('description', '')[:500]}\n\n"
        f"PATCH (what the agent changed):\n```diff\n{diff_text}\n```\n\n"
        f"TEST RESULT: {test_result[:500]}\n\n"
        f"AGENT EXPLANATION: {explanation[:300]}\n"
        f"{plan_section}\n"
        "=== EVALUATION ===\n"
        "Walk through these checks IN ORDER. For each, write your reasoning before "
        "moving on — do not skip.\n\n"
        "1. **Does the diff address the reported symptom?** Quote the line in the "
        "description that describes the bug, then point to the diff lines that change "
        "the behavior. If you can't tie diff lines to symptom lines, that's a REJECT.\n"
        "2. **Adversarial probe** (REQUIRED): Pick at least ONE break attempt and "
        "reason about whether the patch handles it:\n"
        "   - Boundary inputs: empty, None, 0, -1, very large, unicode, malformed\n"
        "   - Concurrency: what if this is called twice in parallel?\n"
        "   - Idempotency: what if the same call is repeated?\n"
        "   - Related code: does another caller of the changed function rely on the "
        "old behavior?\n"
        "   State explicitly which probe you ran and what you concluded.\n"
        "3. **Test coverage**: does the test result actually exercise the fix? A "
        "passing test on an unrelated path is not evidence. If the test command was "
        "`pytest -x` on a file with pre-existing failures, the test may have stopped "
        "before reaching the fix — note this.\n"
        "4. **Side effects in the diff**: does the patch touch anything the bug didn't "
        "ask for? Removed validation? Disabled checks? Renamed exports? These are "
        "REJECT signals unless explicitly justified.\n\n"
        "=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===\n"
        "If you catch yourself thinking any of these, REJECT or escalate:\n"
        "- 'The diff looks correct' → reading is not verification, did you check "
        "behavior?\n"
        "- 'The test passed so it's fine' → verify the test exercised the fix\n"
        "- 'The agent's explanation is plausible' → plausible is not verified\n"
        "- 'This is probably an edge case nobody hits' → not your call\n\n"
        "=== VERDICTS ===\n"
        "APPROVE: only if (a) diff lines tie to symptom, (b) at least one adversarial "
        "probe was reasoned about and the patch survives it, (c) test result exercises "
        "the fix, (d) no unrequested side effects.\n"
        "REJECT: any of the above fail, or you spot regression risk.\n\n"
        "Confidence: how sure are you? Use 0.5 if you couldn't run the adversarial "
        "probe (e.g., diff is truncated and you can't see context). Use 0.9+ only if "
        "you ran a probe and it survived.\n\n"
        "Set regression_risk=HIGH if the patch removes validation, changes a public "
        "API signature, or touches >5 files. MEDIUM if it changes shared utilities. "
        "LOW if it's a localized fix in one function.\n\n"
        "Your `explanation` field MUST mention which adversarial probe you ran. If "
        "the explanation has no probe, the caller will treat it as REJECT regardless "
        "of your verdict."
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
        # Try the cache-preserving forked path first — verifier inherits the
        # main agent's prompt prefix, paying only for the new task message +
        # response. Falls back to fresh structured_call if no parent params
        # are cached (e.g., react loop crashed before saving them).
        result: Any = None
        forked_used = False
        try:
            from agent.forked_subagent import (
                run_forked_subagent,
                get_last_cache_safe_params,
            )
            if get_last_cache_safe_params() is not None:
                fork_result = run_forked_subagent(
                    task=prompt,
                    schema=VerifierResult,
                    max_tokens=1200,
                )
                if fork_result.get("parsed") is not None and fork_result.get("error") is None:
                    result = fork_result["parsed"]
                    forked_used = True
                    cache_read = fork_result.get("cache_read_tokens", 0)
                    logger.info(
                        "Verifier ran via forked subagent — cache_read=%d tokens",
                        cache_read,
                    )
        except Exception as e:
            logger.debug("Forked verifier path failed (%s) — falling back to fresh call", e)

        if result is None:
            # Fallback: fresh-context structured_call (pre-fork behavior)
            result = _structured_call("claude-sonnet-4-6", 1200, VerifierResult, prompt)
            logger.info("Verifier ran via fresh structured_call (no fork)")

        state["verifier_used_fork"] = forked_used

        # Validate the Pydantic result has sane values
        if result.verdict not in ("APPROVE", "REJECT"):
            logger.warning("Verifier returned invalid verdict '%s' — treating as REJECT", result.verdict)
            result.verdict = "REJECT"
        result.confidence = max(0.0, min(1.0, result.confidence))

        # Anti-rationalization gate: if the verifier APPROVES but the explanation
        # contains no evidence of an adversarial probe, downgrade to REJECT with
        # low confidence. This enforces the prompt requirement that the explanation
        # must mention which probe was run.
        if result.verdict == "APPROVE":
            probe_signals = (
                "boundary", "concurrency", "idempoten", "parallel", "edge",
                "empty", "none", "unicode", "malformed", "probe", "considered",
                "checked", "if called", "what if", "negative",
            )
            explanation_lower = (result.explanation or "").lower()
            if not any(sig in explanation_lower for sig in probe_signals):
                logger.warning(
                    "Verifier APPROVE without adversarial-probe evidence — "
                    "downgrading to REJECT (explanation: %s)",
                    result.explanation[:120],
                )
                result.verdict = "REJECT"
                result.confidence = min(result.confidence, 0.4)
                result.explanation = (
                    "[downgraded] Original APPROVE lacked adversarial-probe evidence. "
                    + (result.explanation or "")
                )

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
        epr = r.get("epr_score", 0.0)
        verifier_ok = r.get("verifier_verdict", "") == "APPROVE"
        confidence = (r.get("review") or {}).get("confidence", 0.0)
        cost = -(r.get("cost_usd") or 0.0)  # Negative so lower cost = higher rank
        submitted = r.get("submitted", False)
        return (submitted, test_ok, epr, verifier_ok, confidence, cost)

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
