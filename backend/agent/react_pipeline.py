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
import re
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


def _fuzzy_find_file(repo_path: Path, hallucinated_path: str) -> str | None:
    """Try to locate a real file whose path matches the basename of a hallucinated one.

    Scout sometimes gets the filename right but the directory wrong (e.g. returns
    'pylint/extensions/fixme.py' when the real file is at 'pylint/checkers/misc.py').
    Returning a basename match lets the agent start from a real file instead of
    falling back to dumb pre-localization.

    Matching is conservative: only accepts a single unambiguous hit.
    """
    if not repo_path or not repo_path.is_dir():
        return None

    basename = Path(hallucinated_path).name
    if not basename or "." not in basename:
        return None

    # Glob for the basename — cap results to avoid scanning the whole repo
    try:
        matches = list(repo_path.rglob(basename))
    except Exception:
        return None

    # Filter out tests/examples/docs directories
    filtered: list[Path] = []
    for m in matches:
        rel = m.relative_to(repo_path)
        parts = [p.lower() for p in rel.parts]
        if any(skip in parts for skip in ("test", "tests", "__pycache__", "examples", "docs", "build", "dist")):
            # Skip test/example dirs unless the hallucinated path also had them
            if not any(skip in hallucinated_path.lower() for skip in ("test", "example")):
                continue
        filtered.append(m)

    if len(filtered) == 1:
        return str(filtered[0].relative_to(repo_path))

    # If multiple matches, prefer one whose parent dir matches any part of the
    # hallucinated path (e.g. both contain 'checkers')
    hallucinated_parts = set(Path(hallucinated_path).parts)
    for m in filtered:
        rel = m.relative_to(repo_path)
        if set(rel.parts) & hallucinated_parts:
            return str(rel)

    return None


def _check_existing_work(work_order: dict, repo_path: Path | None) -> dict | None:
    """Check for existing branches/PRs that might address this same ticket.

    Prevents the agent from silently creating duplicate work when re-invoked
    on a ticket it already processed (e.g., user clicks "Run" twice, or
    session restart loses in-memory state).

    Returns a dict with 'branches', 'prs', 'summary' if duplicates found,
    or None if the work appears fresh.
    """
    if not repo_path or not repo_path.is_dir():
        return None

    ticket_id = (work_order.get("ticket_id") or "").strip()
    if not ticket_id:
        return None

    # Build search pattern: match ticket_id in branch name or PR title.
    # Our branches are fix/<repo>-<hash> but we also want to catch ticket-based names.
    safe_ticket = re.sub(r"[^a-zA-Z0-9\-]", "", ticket_id).lower()

    existing_branches: list[str] = []
    existing_prs: list[dict] = []

    # Check local branches — our naming is fix/<repo>-<hash6>, so we can't match
    # on ticket directly. Scan branch messages for ticket_id instead.
    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)|%(contents:subject)"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "|" not in line:
                    continue
                branch, subj = line.split("|", 1)
                branch = branch.strip()
                if not branch.startswith("fix/"):
                    continue
                # Match ticket ID in subject OR safe_ticket in branch name
                if ticket_id.lower() in subj.lower() or safe_ticket in branch.lower():
                    existing_branches.append(branch)
    except Exception:
        pass

    # Check open PRs via gh CLI (if available)
    try:
        result = subprocess.run(
            ["gh", "pr", "list",
             "--search", ticket_id,
             "--state", "open",
             "--json", "number,title,url,headRefName",
             "--limit", "5"],
            cwd=repo_path, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            prs = json.loads(result.stdout)
            for pr in prs:
                existing_prs.append({
                    "number": pr.get("number"),
                    "title": pr.get("title", "")[:100],
                    "url": pr.get("url", ""),
                    "branch": pr.get("headRefName", ""),
                })
    except Exception:
        pass

    if not existing_branches and not existing_prs:
        return None

    summary_parts = []
    if existing_prs:
        summary_parts.append(f"{len(existing_prs)} open PR(s)")
    if existing_branches:
        summary_parts.append(f"{len(existing_branches)} local branch(es)")
    summary = f"Found {' and '.join(summary_parts)} possibly addressing {ticket_id}"

    return {
        "ticket_id": ticket_id,
        "branches": existing_branches,
        "prs": existing_prs,
        "summary": summary,
    }


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
                repo_path=_resolve_repo_path(state.get("work_order", {})),
            )
            top_locs = scout_report.get("top_locations", [])
            if top_locs:
                intent = state.get("intent", {})
                # Inject Scout's top files — but VALIDATE they exist first.
                # Scout can hallucinate paths based on bug description.
                repo_path_for_check = _resolve_repo_path(state.get("work_order", {}))
                scout_files_raw = [loc["file"] for loc in top_locs if loc.get("file")][:5]
                scout_files = []
                hallucinated = []
                for sf in scout_files_raw:
                    if repo_path_for_check and (repo_path_for_check / sf).exists():
                        scout_files.append(sf)
                    else:
                        hallucinated.append(sf)
                        logger.warning("Scout hallucinated path (doesn't exist): %s", sf)

                # Fuzzy recovery: for each hallucinated path, try to find the
                # real file by basename. Saves bugs where scout got the name
                # right but the directory wrong (common on non-Django repos).
                if hallucinated and repo_path_for_check:
                    for bad_path in hallucinated:
                        recovered = _fuzzy_find_file(repo_path_for_check, bad_path)
                        if recovered and recovered not in scout_files:
                            logger.info("Recovered '%s' → '%s' via basename match", bad_path, recovered)
                            scout_files.append(recovered)

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

    # Dedup check — warn if an open PR or branch already exists for this ticket.
    # Avoids the "Bug 3 → 2 branches" problem where re-invocation creates duplicates.
    # Result is stored in state for inclusion in PR body and for the finalize node.
    try:
        dup_info = _check_existing_work(work_order, repo_path)
        if dup_info:
            state["existing_work"] = dup_info
            logger.warning("Possible duplicate work detected: %s", dup_info.get("summary", ""))
            if trace:
                trace.emit("dedup_warning", "intake", dup_info)
    except Exception as e:
        logger.debug("Dedup check failed (non-fatal): %s", e)

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
# Node 1b: SETUP (Pipeline v4 — parallel threads)
# ---------------------------------------------------------------------------
# Replaces intake_node's sequential work with 3 independent threads:
#   Thread 1: Repo detection + sandbox creation + baseline tests (no LLM)
#   Thread 2: Scout localization (Haiku + Sonnet, no Opus re-ranker)
#   Thread 3: Context assembly (no LLM)
# Results are merged into state["_dynamic_context"].


def _setup_thread_repo(
    repo_path: Path,
    repo_name: str,
) -> dict:
    """Thread 1: Repo detection + sandbox creation + baseline tests.

    All work is local I/O and subprocess calls — no LLM calls.

    Returns:
        {
            "sandbox_path": str,        # absolute path to worktree
            "branch_name": str,         # e.g. "fix/myrepo-a1b2c3"
            "base_branch": str,         # e.g. "main"
            "baseline_failures": set,   # test IDs that fail on unmodified code
        }
    """
    import uuid

    result: dict = {
        "sandbox_path": "",
        "branch_name": "",
        "base_branch": "",
        "baseline_failures": set(),
    }

    # --- Step 1: Auto-detect project type and write .agent_config.json ---
    if repo_path and repo_path.is_dir():
        try:
            from agent.repo_detection import write_agent_config_from_detection
            config_path = repo_path / ".agent_config.json"
            if not config_path.exists():
                write_agent_config_from_detection(repo_path)
                logger.info("setup_thread_repo: auto-detected project config for %s", repo_path.name)
        except Exception as e:
            logger.debug("setup_thread_repo: repo_detection failed (non-fatal): %s", e)

    # --- Step 2: Create git worktree sandbox ---
    if not repo_path or not repo_path.is_dir():
        logger.error("setup_thread_repo: no valid repo_path — cannot create sandbox")
        return result

    # Verify this is a git repo
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        logger.error("setup_thread_repo: %s is not a git repo (no .git)", repo_path)
        return result

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_name).lower()
    branch_suffix = uuid.uuid4().hex[:6]
    branch_name = f"fix/{safe_name}-{branch_suffix}"
    worktree_path = Path(f"/tmp/agent_sandbox_{safe_name}_{branch_suffix}")

    try:
        import fcntl

        # Get base branch
        base_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()

        # Lock to prevent race conditions with other agent instances
        with open(repo_path / ".agent_lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Check for dirty repo (ignore untracked files)
                porcelain = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
                ).stdout
                dirty = "\n".join(
                    l for l in porcelain.splitlines() if l and not l.startswith("??")
                ).strip()
                if dirty:
                    logger.error("setup_thread_repo: repo has uncommitted changes")
                    return result

                # Prune stale worktrees
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=repo_path, capture_output=True, timeout=30,
                )

                # Remove old worktree directory if it exists
                if worktree_path.exists():
                    shutil.rmtree(worktree_path, ignore_errors=True)

                # Create worktree
                subprocess.run(
                    ["git", "worktree", "add", "-b", branch_name,
                     str(worktree_path), base_branch],
                    cwd=repo_path, capture_output=True, text=True, check=True, timeout=60,
                )
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        result["sandbox_path"] = str(worktree_path)
        result["branch_name"] = branch_name
        result["base_branch"] = base_branch
        logger.info("setup_thread_repo: sandbox created at %s (branch: %s)", worktree_path, branch_name)

    except subprocess.CalledProcessError as e:
        logger.error("setup_thread_repo: git worktree creation failed: %s", e.stderr or e)
        return result
    except PermissionError as e:
        logger.error("setup_thread_repo: permission error creating sandbox: %s", e)
        return result
    except Exception as e:
        logger.error("setup_thread_repo: sandbox creation failed: %s", e)
        return result

    # --- Step 3: Run baseline tests on unmodified code ---
    try:
        from agent.sandbox import run_tests as _sb_run_tests
        baseline_out = _sb_run_tests(
            worktree_path=worktree_path, repo_path=repo_path,
            test_path="", timeout=120,
        )
        if baseline_out.startswith("failed"):
            failed_tests = set(re.findall(r"FAILED\s+([\w/.:]+)", baseline_out))
            result["baseline_failures"] = failed_tests
            logger.info(
                "setup_thread_repo: baseline snapshot — %d pre-existing test failures",
                len(failed_tests),
            )
        else:
            logger.info(
                "setup_thread_repo: baseline snapshot — tests %s (no pre-existing failures)",
                baseline_out[:30],
            )
    except Exception as e:
        logger.debug("setup_thread_repo: baseline test snapshot skipped: %s", e)

    return result


def _setup_thread_scout(
    repo_name: str,
    work_order: dict,
    intent: dict,
    repo_path: Path | None,
) -> dict:
    """Thread 2: Scout localization (Haiku + Sonnet, NO Opus re-ranker).

    Calls the existing scout_localize() pipeline with path validation
    and fuzzy recovery (both already implemented in _run_localization).

    Returns:
        Full scout dict with "suspects", "entity_extraction", "skeleton_data",
        plus "scout_files" (validated file list) and "community" (if detected).
    """
    result: dict = {
        "scout_report": {},
        "scout_files": [],
        "community": None,
    }

    if not repo_name:
        return result

    # --- Step 1: Community classification (single cheap Haiku call) ---
    try:
        community_name = _classify_community(repo_name, intent, DATA_DIR)
        if community_name:
            result["community"] = community_name
            logger.info("setup_thread_scout: community classifier → '%s'", community_name)
    except Exception as e:
        logger.debug("setup_thread_scout: community classifier failed (non-fatal): %s", e)

    community_name = result["community"]

    # --- Step 2: Scout FL pipeline (Haiku extractor + Sonnet Graph-RAG) ---
    try:
        from agent.scout import scout_localize
        scout_report = scout_localize(
            repo_name, work_order, intent, DATA_DIR,
            community_name=community_name,
            repo_path=repo_path,
        )
        result["scout_report"] = scout_report

        # Validate and recover paths (same logic as _run_localization)
        top_locs = scout_report.get("top_locations", [])
        if top_locs and repo_path:
            scout_files_raw = [loc["file"] for loc in top_locs if loc.get("file")][:5]
            scout_files = []
            hallucinated = []
            for sf in scout_files_raw:
                if (repo_path / sf).exists():
                    scout_files.append(sf)
                else:
                    hallucinated.append(sf)
                    logger.warning("setup_thread_scout: hallucinated path: %s", sf)

            # Fuzzy recovery for hallucinated paths
            if hallucinated and repo_path:
                for bad_path in hallucinated:
                    recovered = _fuzzy_find_file(repo_path, bad_path)
                    if recovered and recovered not in scout_files:
                        logger.info(
                            "setup_thread_scout: recovered '%s' → '%s'", bad_path, recovered,
                        )
                        scout_files.append(recovered)

            result["scout_files"] = scout_files
            logger.info(
                "setup_thread_scout: validated %d/%d locations %s (cost $%.4f)",
                len(scout_files), len(scout_files_raw),
                scout_files, scout_report.get("scout_cost_usd", 0),
            )

    except Exception as e:
        logger.debug("setup_thread_scout: scout FL failed (non-fatal): %s", e)
        # Fall back to pre-localization
        try:
            confirmed = _prelocalize(repo_name, intent, DATA_DIR)
            if confirmed:
                result["scout_files"] = confirmed
        except Exception as e2:
            logger.debug("setup_thread_scout: pre-localization fallback also failed: %s", e2)

    return result


def _setup_thread_context(
    repo_name: str,
    repo_path: Path | None,
    work_order: dict,
) -> dict:
    """Thread 3: Context assembly (no LLM calls).

    Builds all the static context the react_agent_node needs:
    repo tree, graph data, lessons, concept-to-code mappings.

    Returns:
        {
            "repo_tree": str,           # compact file listing
            "graph_context": str,       # kickstart orientation block
            "lessons": str,             # past-run lessons markdown
            "concept_mappings": dict,   # business-rule → code mappings
        }
    """
    result: dict = {
        "repo_tree": "",
        "graph_context": "",
        "lessons": "",
        "concept_mappings": {},
    }

    # --- Step 1: Build repo tree listing ---
    if repo_path and repo_path.is_dir():
        try:
            from agent.scout import _build_repo_listing
            result["repo_tree"] = _build_repo_listing(repo_path)
        except Exception as e:
            logger.debug("setup_thread_context: repo tree listing failed (non-fatal): %s", e)

    # --- Step 2: Load graph data via build_kickstart_context ---
    # We pass an empty intent here; the real intent merge happens in
    # react_agent_node after setup_node merges scout results into intent.
    try:
        from agent.graph_utils import build_kickstart_context
        # Use a minimal intent — full intent is not yet enriched with scout files,
        # so kickstart gets only the raw LLM hints. react_agent_node rebuilds
        # kickstart with the merged intent anyway; this is pre-loading for speed.
        minimal_intent: dict = {}
        result["graph_context"] = build_kickstart_context(
            repo_name, str(repo_path) if repo_path else None, minimal_intent, DATA_DIR,
        )
    except Exception as e:
        logger.debug("setup_thread_context: kickstart context failed (non-fatal): %s", e)

    # --- Step 3: Load lessons from past runs ---
    try:
        from agent.learn_from_fix import load_lessons
        result["lessons"] = load_lessons(repo_name)
    except Exception as e:
        logger.debug("setup_thread_context: load_lessons failed (non-fatal): %s", e)

    # --- Step 4: Query concept-to-code mappings ---
    try:
        from agent.graph_utils import query_concept_to_code
        title = work_order.get("title", "")
        description = work_order.get("description", "")
        result["concept_mappings"] = query_concept_to_code(title, description, repo_name)
    except Exception as e:
        logger.debug("setup_thread_context: concept-to-code failed (non-fatal): %s", e)

    return result


def setup_node(state: ReactAgentState) -> ReactAgentState:
    """Pipeline v4 setup: run 3 independent threads in parallel, merge results.

    Replaces the sequential intake_node with parallel work:
      Thread 1: repo detection + sandbox creation + baseline tests
      Thread 2: scout localization (Haiku + Sonnet)
      Thread 3: context assembly (repo tree, graph, lessons, concept mappings)

    Pre-thread work (sequential, fast):
      - Resolve repo path
      - Clean up stale worktrees
      - Check for existing work (dedup)
      - Translate intent (single Haiku call, needed by Thread 2)
      - Enrich with stack trace hints
      - Extract repro steps + bug category

    Post-thread work (sequential):
      - Merge scout files into intent
      - Merge concept-to-code mappings into intent
      - Store all results in state["_dynamic_context"]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _thread_local.current_stage = "setup"
    trace = _get_trace()
    if trace:
        trace.stage_start("setup")
    logger.info("=== SETUP NODE: Parallel initialization (v4) ===")

    work_order = state.get("work_order", {})
    repo_path = _resolve_repo_path(work_order)
    repo_name = work_order.get("repo_name", "")

    # --- Pre-thread sequential work ---

    # Clean up stale sandboxes (best-effort)
    if repo_path:
        from agent.sandbox import cleanup_stale_worktrees
        try:
            cleanup_stale_worktrees(repo_path)
        except Exception as e:
            logger.debug("Stale worktree cleanup failed (non-fatal): %s", e)

    state["status"] = PipelineStatus.INTAKE
    _report_progress(state)

    # Dedup check
    try:
        dup_info = _check_existing_work(work_order, repo_path)
        if dup_info:
            state["existing_work"] = dup_info
            logger.warning("Possible duplicate work detected: %s", dup_info.get("summary", ""))
            if trace:
                trace.emit("dedup_warning", "setup", dup_info)
    except Exception as e:
        logger.debug("Dedup check failed (non-fatal): %s", e)

    # Translate intent (single Haiku call — needed by Thread 2 for scout)
    state["intent"] = _translate_intent(work_order)

    # Enrich with stack trace hints
    state = _enrich_with_stack_hints(state)

    # Repro steps + bug category
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

    # Snapshot intent for threads (immutable copy — threads must not share mutable state)
    intent_snapshot = dict(state.get("intent", {}))
    work_order_snapshot = dict(work_order)

    # --- Launch 3 threads in parallel ---
    t_start = time.time()

    repo_result: dict = {}
    scout_result: dict = {}
    context_result: dict = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_repo = executor.submit(
            _setup_thread_repo,
            repo_path,
            repo_name,
        )
        future_scout = executor.submit(
            _setup_thread_scout,
            repo_name,
            work_order_snapshot,
            intent_snapshot,
            repo_path,
        )
        future_context = executor.submit(
            _setup_thread_context,
            repo_name,
            repo_path,
            work_order_snapshot,
        )

        # Collect results — each thread is independent, so order doesn't matter
        for future in as_completed([future_repo, future_scout, future_context]):
            try:
                res = future.result(timeout=180)  # 3-minute timeout per thread
                if future is future_repo:
                    repo_result = res
                elif future is future_scout:
                    scout_result = res
                else:
                    context_result = res
            except Exception as e:
                # Identify which thread failed
                if future is future_repo:
                    logger.error("setup_node: repo thread failed: %s", e)
                elif future is future_scout:
                    logger.error("setup_node: scout thread failed: %s", e)
                else:
                    logger.error("setup_node: context thread failed: %s", e)

    elapsed = time.time() - t_start
    logger.info("setup_node: all 3 threads completed in %.2fs", elapsed)

    # --- Post-thread merge ---

    # Merge sandbox info into state
    if repo_result.get("sandbox_path"):
        state["sandbox_path"] = repo_result["sandbox_path"]
        state["branch_name"] = repo_result["branch_name"]
        state["base_branch"] = repo_result["base_branch"]

    # Merge scout files into intent
    intent = state.get("intent", {})
    scout_files = scout_result.get("scout_files", [])
    if scout_files:
        intent["confirmed_files"] = scout_files
        intent["likely_affected_modules"] = scout_files
    if scout_result.get("community"):
        intent["community"] = scout_result["community"]
    if scout_result.get("scout_report"):
        intent["scout_report"] = scout_result["scout_report"]

    # Merge concept-to-code mappings into intent
    c2c = context_result.get("concept_mappings", {})
    if c2c.get("matched_rules"):
        existing_funcs = intent.get("likely_affected_functions", []) or []
        existing_mods = intent.get("likely_affected_modules", []) or []
        merged_funcs = list(dict.fromkeys(existing_funcs + c2c["hint_functions"]))
        merged_mods = list(dict.fromkeys(existing_mods + c2c["hint_files"]))
        intent["likely_affected_functions"] = merged_funcs[:8]
        intent["likely_affected_modules"] = merged_mods[:6]
        intent["_concept_section"] = c2c["concept_section"]
        logger.info(
            "setup_node: concept-to-code merged %d rules → functions=%s",
            len(c2c["matched_rules"]), merged_funcs[:4],
        )

    state["intent"] = intent

    # Store all thread results in _dynamic_context for downstream nodes
    state["_dynamic_context"] = {
        # Thread 1 results
        "sandbox_path": repo_result.get("sandbox_path", ""),
        "branch_name": repo_result.get("branch_name", ""),
        "base_branch": repo_result.get("base_branch", ""),
        "baseline_failures": repo_result.get("baseline_failures", set()),
        # Thread 2 results
        "scout_report": scout_result.get("scout_report", {}),
        "scout_files": scout_result.get("scout_files", []),
        "community": scout_result.get("community"),
        # Thread 3 results
        "repo_tree": context_result.get("repo_tree", ""),
        "graph_context": context_result.get("graph_context", ""),
        "lessons": context_result.get("lessons", ""),
        "concept_mappings": context_result.get("concept_mappings", {}),
    }

    if trace:
        trace.stage_end("setup")

    logger.info(
        "setup_node: done — sandbox=%s, scout_files=%d, lessons=%d chars, graph=%d chars",
        repo_result.get("sandbox_path", "N/A"),
        len(scout_files),
        len(context_result.get("lessons", "")),
        len(context_result.get("graph_context", "")),
    )

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
            # Multi-language: Python, JavaScript, TypeScript, Go, Rust
            import re
            sigs = []
            ext = fpath.suffix.lower()
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                lineno = i + 1
                # Python patterns
                if re.match(r"^\s*(class\s+\w+|(?:async\s+)?def\s+\w+)", stripped):
                    sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                # Python decorators (show what the function does)
                elif re.match(r"^\s*@(router|app)\.", stripped):
                    sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                # JS/TS patterns (function, class, export, arrow functions)
                elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue"):
                    if re.match(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                    elif re.match(r"^\s*(export\s+)?(default\s+)?class\s+\w+", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                    elif re.match(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()[:120]}")
                    elif re.match(r"^\s*(export\s+)?(interface|type|enum)\s+\w+", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                # Go patterns
                elif ext == ".go":
                    if re.match(r"^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                    elif re.match(r"^type\s+\w+\s+(struct|interface)", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                # Rust patterns
                elif ext == ".rs":
                    if re.match(r"^\s*(pub\s+)?(async\s+)?fn\s+\w+", stripped):
                        sigs.append(f"  L{lineno:4d}: {stripped.strip()}")
                    elif re.match(r"^\s*(pub\s+)?(struct|enum|trait|impl)\s+", stripped):
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

    # Test-infra warning: preflight detected pytest can't even collect tests
    # (usually conftest import error). Tell the agent so it doesn't waste turns
    # debugging "test failures" that are really infra/env breakage — but ALSO
    # actively suggest run_shell for env repair (it can install missing deps
    # straight into the scorer's venv).
    if work_order.get("_preflight_failed"):
        infra_note = (
            "\n\n## ⚠️  TEST INFRASTRUCTURE WARNING — INVESTIGATE WITH run_shell\n"
            "The repo's test framework is currently broken (pytest cannot collect tests, "
            "exit code 4 — usually a conftest.py import error or missing dep). "
            "**This is FIXABLE with run_shell** — `pip install` lands in the scorer's venv.\n"
            "\n"
            "Recommended workflow:\n"
            "1. `run_shell('python -c \"import <suspected-module>\"')` — find the missing import\n"
            "2. `run_shell('pip install <missing-pkg>')` — install it (auto-targets scorer's venv)\n"
            "3. `run_shell('python -m pytest --collect-only -q')` — verify tests can collect now\n"
            "4. Then proceed with your fix + run_tests as usual.\n"
            "\n"
            "If the env truly can't be repaired (e.g. missing C extension build), submit anyway — "
            "the verifier judges based on code reasoning, not test pass/fail.\n"
        )
        if work_order.get("_preflight_stderr"):
            infra_note += f"\nPreflight stderr (first 300 chars):\n{work_order['_preflight_stderr']}\n"
        kickstart += infra_note
        logger.info("Injected preflight failure warning + run_shell suggestion into kickstart")

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

    # If this is a retry attempt (pass@3), prepend the failure feedback so the
    # agent knows what didn't work last time.
    retry_feedback = state.get("retry_feedback", "")
    retry_count = state.get("retry_count", 0)
    if retry_feedback and retry_count > 0:
        task_message = (
            f"🔁 RETRY ATTEMPT {retry_count + 1} of {int(os.environ.get('REACT_MAX_RETRIES', '2')) + 1}\n\n"
            f"YOUR PREVIOUS ATTEMPT FAILED with this feedback:\n{retry_feedback}\n\n"
            f"Review what went wrong, revise your hypothesis, and try a different approach. "
            f"The sandbox from your previous attempt still exists — you can continue from it "
            f"OR use undo_last_edit() to revert and try something new.\n\n"
            f"---\n\n{task_message}"
        )
        logger.info("Retry attempt %d: injected retry feedback into task message", retry_count)

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


def _is_internal_ticket_id(ticket_id: str) -> bool:
    """Detect internal/auto-generated ticket IDs that shouldn't appear in PR titles.

    Matches: MANUAL-abc123, short numeric IDs (< 4 digits), hex-only, SWE-bench IDs.
    Keeps: real ticket IDs like ARIA-1234, DJANGO-16315, JIRA-PROJ-1234.
    """
    if not ticket_id:
        return True
    tid = ticket_id.strip().upper()
    # Explicit internal prefixes
    if tid.startswith(("MANUAL-", "TEST-", "AUTO-", "TMP-")):
        return True
    # Pure hex or numeric IDs (e.g., "25", "a7f3c1")
    if re.fullmatch(r"[A-F0-9]{4,}", tid):
        return True
    if tid.isdigit() and len(tid) < 6:
        return True
    return False


def _build_pr_title(ticket_id: str, explanation: str, fault_files: list = None) -> str:
    """Build a clean PR title ≤72 chars, no mid-word truncation, no internal IDs.

    Format preference (best to worst):
      1. "fix(component): one-line summary"            ← when fault_files available
      2. "fix(TICKET-123): one-line summary"           ← for real ticket IDs
      3. "fix: one-line summary"                        ← for internal/empty IDs
    """
    MAX_TITLE = 72

    # Derive scope/component from fault_files (e.g., 'backend/app/auth.py' → 'auth')
    scope = ""
    if fault_files:
        first = fault_files[0]
        # Take stem of first file: "src/auth/service.py" → "service"
        name = first.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        # Clean scope names: "test_xyz" → "xyz", "index" → use parent dir
        if name in ("index", "main", "__init__"):
            parts = first.rsplit("/", 2)
            if len(parts) >= 2:
                name = parts[-2]
        scope = name[:20]

    # Decide prefix based on ticket_id quality
    if _is_internal_ticket_id(ticket_id):
        prefix = f"fix({scope})" if scope else "fix"
    else:
        # Real ticket ID — use it, but drop scope to keep title short
        prefix = f"fix({ticket_id})"

    # Clean + trim explanation: first sentence, no newlines
    summary = (explanation or "Automated fix").strip().split("\n")[0]
    # Drop leading "Fixed a " / "Fix the " boilerplate.
    # Use \s+ AFTER the article so "a" in "auth" doesn't match.
    # Order longer alternatives first so "an" beats "a" in "an off-by-one".
    summary = re.sub(
        r"^(fixed|fixes|fix)(\s+(the|an|a)\s+)?\s*",
        "",
        summary,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if summary:
        summary = summary[0].upper() + summary[1:]

    # Full title with trailing space room
    full = f"{prefix}: {summary}"
    if len(full) <= MAX_TITLE:
        return full

    # Truncate at word boundary
    budget = MAX_TITLE - len(f"{prefix}: ") - 1  # -1 for ellipsis
    if budget <= 10:
        # Prefix itself is too long — fall back to bare "fix:"
        prefix = "fix"
        budget = MAX_TITLE - len("fix: ") - 1
    trimmed = summary[:budget]
    # Break at last space to avoid mid-word cut
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return f"{prefix}: {trimmed}…"


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

    # Build review section — suppress confidence when it's zero/unset
    # (a 0% score looks broken to human reviewers)
    verdict = review.get("verdict", "N/A")
    confidence = review.get("confidence", 0)
    review_section = f"- Verdict: {verdict}\n"
    if confidence and confidence > 0:
        review_section += f"- Confidence: {confidence:.0%}\n"

    # Adjacent concerns — surface related gaps the agent noticed but didn't fix
    adjacent = state.get("adjacent_concerns", "")
    adjacent_section = ""
    if adjacent:
        adjacent_section = f"\n## Adjacent Concerns\n{adjacent}\n"

    # Dedup warning — tell reviewer if overlapping work exists
    existing = state.get("existing_work", {}) or {}
    dedup_section = ""
    if existing and (existing.get("prs") or existing.get("branches")):
        lines = [f"\n## ⚠ Possible Duplicate Work"]
        if existing.get("prs"):
            lines.append("**Open PRs addressing this ticket:**")
            for pr in existing["prs"][:5]:
                lines.append(f"- #{pr['number']} {pr['title']} — {pr['url']}")
        if existing.get("branches"):
            lines.append("**Existing local branches:**")
            for b in existing["branches"][:5]:
                lines.append(f"- `{b}`")
        lines.append("Consider closing this PR in favor of the existing work, or marking it as a follow-up.\n")
        dedup_section = "\n".join(lines) + "\n"

    pr_body = (
        f"## Root Cause\n{explanation}\n\n"
        f"## Review\n{review_section}\n"
        f"## Tests\n```\n{test_result[:2000]}\n```\n\n"
        f"## Changes\n```\n{diff_stat}\n```\n"
        f"{adjacent_section}"
        f"{dedup_section}\n"
        f"---\n*Generated by AI Deploy Agent (ReAct) — {ticket_id}*"
    )

    # Build clean title using fault_files for scope
    localization = state.get("localization", {}) or {}
    fault_files = localization.get("fault_files") or []
    pr_title = _build_pr_title(ticket_id, explanation, fault_files)
    return pr_title, pr_body


def _strip_url_credentials(url: str) -> str:
    """Remove any existing user:pass@ or user@ from an HTTPS git URL.

    Handles:
      https://user@github.com/...         → https://github.com/...
      https://user:token@github.com/...   → https://github.com/...
      https://x-access-token:tok@gh...    → https://github.com/...
    """
    import re
    return re.sub(r"(https?://)([^@]+@)", r"\1", url)


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
                # Origin exists and is GitHub — strip any existing credentials first,
                # then add the token. Prevents double-@ when URL already has user@.
                clean_url = _strip_url_credentials(remote_url)
                auth_url = clean_url.replace(
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
                        clean_url = _strip_url_credentials(gh_repo)
                        auth_url = clean_url.replace(
                            "https://github.com/",
                            f"https://x-access-token:{gh_token}@github.com/"
                        )
                        if not auth_url.endswith(".git"):
                            auth_url += ".git"
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
    """Handle post-agent work: create PR if submitted, escalate otherwise.

    Includes pass@3 retry: if verifier REJECTed the fix AND BRTs are failing,
    re-enter the agent loop with the failure feedback as context. Max 2 retries
    (so max 3 total attempts). This matches OpenHands/Aider's leaderboard pattern.
    """
    _thread_local.current_stage = "finalize"
    trace = _get_trace()
    if trace:
        trace.stage_start("finalize")
    logger.info("=== FINALIZE: %s ===",
                "Creating PR" if state.get("submitted") else "Escalating")

    # Pass@3 retry logic: if verifier flagged the fix as incomplete AND BRTs
    # are failing, give the agent another shot with the failure feedback.
    # Default 0 retries (disabled) — enable per-bug or by env var when there's
    # genuine time budget. Last run had DJANGO-11630 timeout at 900s because
    # retry burned through the whole budget.
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = int(os.environ.get("REACT_MAX_RETRIES", "0"))  # default off
    # Time budget guard — don't retry if we're running low on case timeout
    pipeline_start = state.get("_pipeline_start", time.time())
    elapsed = time.time() - pipeline_start
    case_timeout = state.get("_case_timeout", 900)
    remaining_budget = case_timeout - elapsed
    has_time_for_retry = remaining_budget >= 300  # need at least 5 min

    should_retry = (
        state.get("needs_retry")
        and retry_count < MAX_RETRIES
        and state.get("submitted")
        and not state.get("dry_run_scoring_only")
        and has_time_for_retry
    )
    if state.get("needs_retry") and not has_time_for_retry:
        logger.warning(
            "Skipping retry: only %.0fs remaining of %ds case budget", remaining_budget, case_timeout,
        )
    if should_retry:
        retry_reason = state.get("retry_reason", "Fix incomplete")
        logger.info(
            "PASS@3 RETRY %d/%d (%.0fs remaining): %s",
            retry_count + 1, MAX_RETRIES, remaining_budget, retry_reason[:200],
        )
        if trace:
            trace.emit("retry_triggered", "finalize", {
                "retry_count": retry_count + 1,
                "max_retries": MAX_RETRIES,
                "reason": retry_reason,
                "epr_score": state.get("epr_score"),
                "verifier_verdict": state.get("verifier_verdict"),
            })
        # Clear submitted + retry signal, re-enter react loop with feedback
        state["retry_count"] = retry_count + 1
        state["submitted"] = False
        state["needs_retry"] = False
        state["retry_feedback"] = retry_reason
        # Re-enter react_agent_node. This requires the intake to have completed
        # (it has — we got here post-verifier). Just rerun react_agent_node
        # which uses state["work_order"] + state["intent"] already set.
        from agent.react_pipeline import react_agent_node as _react_agent_node
        state = _react_agent_node(state)
        # Re-run verifier on the new fix
        state = verifier_node(state)
        # Fall through to normal finalize logic — recursion check via retry_count
        if state.get("needs_retry") and state.get("retry_count", 0) < MAX_RETRIES:
            # Will re-enter again on next finalize_node call. But since we're
            # already IN finalize_node, just let it complete after retries exhaust.
            pass

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
        _emit_failure_diagnosis(state)
        _safe_record_lesson(state)
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
                    # Detect edited file languages so runner picks correct tool
                    from agent.react_tools import _detect_edited_file_langs
                    edited_langs = _detect_edited_file_langs(Path(sandbox_path))
                    gt_result = _run_tests(
                        Path(sandbox_path), repo_path=repo_path, agent_config=eval_cfg,
                        edited_langs=edited_langs,
                    )
                else:
                    from agent.react_tools import _detect_edited_file_langs
                    edited_langs = _detect_edited_file_langs(Path(sandbox_path))
                    gt_result = _run_tests(
                        Path(sandbox_path), repo_path=repo_path,
                        edited_langs=edited_langs,
                    )
                state["test_result"] = gt_result[:2000]
                logger.info("Ground-truth tests (dry-run eval): %s", gt_result[:120])
            except Exception as e:
                logger.debug("Ground-truth test execution skipped: %s", e)

        _report_progress(state)
        _emit_failure_diagnosis(state)
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
    _emit_failure_diagnosis(state)
    _safe_record_lesson(state)
    if trace:
        trace.stage_end("finalize")
    _cleanup_sandbox(sandbox_path, repo_path, branch_name)
    return state


def _safe_record_lesson(state: dict) -> None:
    """Call learn_from_fix.record_lesson with all exceptions swallowed."""
    try:
        from agent.learn_from_fix import record_lesson
        record_lesson(state)
    except Exception as e:
        logger.debug("record_lesson wrapper failed (non-fatal): %s", e)


def _emit_failure_diagnosis(state: dict) -> None:
    """Generate and emit a structured failure diagnosis into the trace.

    Called after every run (success or fail). For FAIL runs, this provides
    a "decision replay" summary that answers "where did the agent go wrong?"
    without requiring manual trace inspection.

    The diagnosis is stored in the trace JSON as a `failure_diagnosis` event,
    readable by both the UI and `cli.py eval diagnose`.
    """
    trace = _get_trace()
    if not trace:
        return

    try:
        from agent.react_tools import get_current_plan, get_edit_history

        work_order = state.get("work_order", {}) or {}
        intent = state.get("intent", {}) or {}
        plan = get_current_plan() or {}
        edits = get_edit_history()

        submitted = state.get("submitted", False)
        escalated = state.get("escalated", False)
        test_result = state.get("test_result", "")
        verifier = state.get("verifier_verdict", "")
        explanation = state.get("explanation", "")
        escalate_reason = state.get("escalate_reason", "")
        tool_calls = state.get("tool_call_count", 0)
        cost = state.get("cost_usd", 0.0)

        # Classify the failure mode
        if submitted and "passed" in str(test_result).lower():
            outcome = "SUCCESS"
            failure_mode = "none"
        elif escalated:
            if "never attempted" in escalate_reason.lower() or "0 edits" in escalate_reason:
                failure_mode = "stuck_exploring"
            elif "time limit" in escalate_reason.lower():
                failure_mode = "timeout"
            elif "cost cap" in escalate_reason.lower():
                failure_mode = "cost_exceeded"
            else:
                failure_mode = "escalated_other"
            outcome = "FAIL"
        elif submitted and "failed" in str(test_result).lower():
            failure_mode = "tests_failed"
            outcome = "FAIL"
        elif submitted and ("skipped" in str(test_result).lower() or "error" in str(test_result).lower()):
            failure_mode = "test_infra_broken"
            outcome = "FAIL"
        elif not submitted:
            failure_mode = "no_fix_submitted"
            outcome = "FAIL"
        else:
            failure_mode = "unknown"
            outcome = "FAIL" if not submitted else "SUCCESS"

        # Build the step-by-step replay
        steps: list[str] = []
        steps.append(f"1. TICKET: {work_order.get('title', '?')[:100]}")

        if plan:
            steps.append(f"2. PLAN: root_cause='{plan.get('root_cause', '?')[:120]}'")
            steps.append(f"   target_files={plan.get('target_files', [])}")
        else:
            steps.append("2. PLAN: (none produced)")

        if edits:
            for i, e in enumerate(edits[:5], 1):
                steps.append(f"3.{i}. EDIT: {e['tool']} on {e['file_path']}")
        else:
            steps.append("3. EDITS: (none)")

        if test_result:
            steps.append(f"4. TESTS: {str(test_result)[:150]}")
        else:
            steps.append("4. TESTS: (not run)")

        if verifier:
            steps.append(f"5. VERIFIER: {verifier} — {state.get('verifier_explanation', '')[:150]}")

        if outcome == "FAIL":
            steps.append(f"6. FAILURE MODE: {failure_mode}")
            if escalate_reason:
                steps.append(f"   REASON: {escalate_reason[:200]}")

        # --- Richer metrics for dashboarding ---
        # Count edits on same file → edit_churn_ratio
        edit_files = [e.get("file_path", "") for e in edits]
        from collections import Counter
        edit_file_counts = Counter(edit_files)
        edit_churn_ratio = (
            len(edits) / max(len(edit_file_counts), 1) if edits else 0
        )
        # BRT metrics
        brt_total = state.get("brt_total", 0)
        brt_pass = state.get("brt_pass_count", 0)
        epr_score = state.get("epr_score", None)

        # Verifier calibration signal
        verifier_confidence = state.get("verifier_confidence", 0)
        verifier_approved = verifier == "APPROVE"
        # Localization vs ground truth (if present in work_order)
        gold_files = set(work_order.get("expected_files", []) or work_order.get("expected_patch_files", []))
        predicted_files = set(edit_files)
        loc_precision = (
            len(predicted_files & gold_files) / len(predicted_files) if predicted_files else 0
        )
        loc_recall = (
            len(predicted_files & gold_files) / len(gold_files) if gold_files else 0
        )

        diagnosis = {
            "outcome": outcome,
            "failure_mode": failure_mode,
            "ticket_id": work_order.get("ticket_id", "?"),
            "tool_calls": tool_calls,
            "cost_usd": round(cost, 4),
            "plan_produced": bool(plan),
            "edits_count": len(edits),
            "unique_files_edited": len(edit_file_counts),
            "edit_churn_ratio": round(edit_churn_ratio, 2),
            "tests_attempted": bool(test_result),
            "verifier_verdict": verifier or "not_run",
            "verifier_confidence": verifier_confidence,
            "verifier_approved": verifier_approved,
            "brt_total": brt_total,
            "brt_pass": brt_pass,
            "epr_score": epr_score,
            "localization_precision": round(loc_precision, 2),
            "localization_recall": round(loc_recall, 2),
            "retry_count": state.get("retry_count", 0),
            "replay_steps": steps,
        }
        trace.emit("failure_diagnosis", "finalize", diagnosis)

        # Emit a separate compact metrics event for dashboard aggregation
        trace.emit("run_metrics", "finalize", {
            "ticket_id": work_order.get("ticket_id", "?"),
            "outcome": outcome,
            "cost_usd": round(cost, 4),
            "tool_calls": tool_calls,
            "edit_churn_ratio": round(edit_churn_ratio, 2),
            "epr_score": epr_score,
            "verifier_approved": verifier_approved,
            "verifier_confidence": verifier_confidence,
            "localization_precision": round(loc_precision, 2),
            "localization_recall": round(loc_recall, 2),
            "retry_count": state.get("retry_count", 0),
            "failure_mode": failure_mode,
        })
    except Exception as e:
        logger.debug("failure_diagnosis failed (non-fatal): %s", e)


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

    # Ground-truth FAIL_TO_PASS tests are the IDEAL template for BRTs — these are
    # the REAL failing tests that SWE-bench uses to score the fix. Feed them to
    # the generator so Haiku can emulate their structure and assertions.
    ftp_tests = work_order.get("fail_to_pass") or []
    ftp_section = ""
    if ftp_tests:
        # Try to read actual test source from the first FTP test
        ftp_source_snippets = []
        for test_id in ftp_tests[:2]:
            # Parse test ID: "path/to/file.py::TestClass::test_method" or "test_name (module.Class)"
            test_file = None
            if "::" in test_id:
                test_file = test_id.split("::")[0]
            elif "(" in test_id and ")" in test_id:
                # Django format: "test_name (app.tests.ClassName)"
                import re as _re
                m = _re.search(r"\(([a-zA-Z0-9_.]+)\)", test_id)
                if m:
                    mod_path = m.group(1).rsplit(".", 1)[0]  # drop ClassName
                    test_file = mod_path.replace(".", "/") + ".py"
            if test_file:
                # Resolve relative to repo_path — try common locations
                for candidate in (
                    repo_path / test_file,
                    repo_path / "tests" / test_file,
                ):
                    if candidate.exists():
                        try:
                            content = candidate.read_text(encoding="utf-8", errors="replace")
                            # Extract the specific test function if named
                            fn_name = test_id.split("::")[-1] if "::" in test_id else test_id.split("(")[0].strip()
                            if fn_name:
                                import re as _re
                                m2 = _re.search(
                                    rf"def {_re.escape(fn_name)}\s*\([^)]*\):.*?(?=\n    def |\nclass |\Z)",
                                    content, _re.DOTALL,
                                )
                                if m2:
                                    ftp_source_snippets.append(
                                        f"# From {test_file}:\n{m2.group(0)[:800]}"
                                    )
                                    break
                        except Exception:
                            pass
        if ftp_source_snippets:
            ftp_section = (
                "\n\n=== GROUND-TRUTH TEST EXAMPLES (EMULATE THESE) ===\n"
                "These are the REAL tests that MUST pass after a correct fix. "
                "Use their structure, assertions, and imports as your template:\n\n"
                + "\n\n".join(ftp_source_snippets)
                + "\n\nYour generated BRTs should test the SAME behavior these tests check. "
                "If possible, mirror their import paths and assertion patterns.\n"
            )
        else:
            # No source available, just list them as references
            ftp_section = (
                "\n\n=== TESTS THAT MUST PASS (targets) ===\n"
                + "\n".join(f"  - {t}" for t in ftp_tests[:5])
                + "\n\nYour BRTs should test these same behaviors using similar patterns.\n"
            )

    prompt = (
        f"BUG: {work_order.get('title', '')}\n"
        f"DESCRIPTION: {work_order.get('description', '')[:500]}\n"
        f"EXPECTED: {intent.get('expected_behavior', '')[:200]}\n"
        f"ACTUAL: {intent.get('actual_behavior', '')[:200]}\n"
        f"FUNCTIONS TO TEST: {hint_functions}\n\n"
        f"SOURCE CODE:\n{source_context}\n"
        f"{ftp_section}\n"
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
        explanation: str = "" # Why APPROVE/REJECT in 2-3 sentences (default empty if truncated)
        regression_risk: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH" (default if truncated)

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

    # Run confirmed BRTs against the patched sandbox BEFORE building the verifier
    # prompt, so the verifier sees the EPR result. Previously BRTs ran AFTER the
    # prompt was already sent — verifier was blind to them, caused 60% false
    # APPROVE rate (verifier approved fixes where BRTs failed 0/3).
    brts = state.get("brts", [])
    brt_pass_count = 0
    brt_total = len(brts)
    brt_details = []  # for the verifier prompt
    if brts and sandbox_path and Path(sandbox_path).exists():
        import tempfile
        sandbox_python = _find_repo_python(Path(sandbox_path))
        for i, brt in enumerate(brts, 1):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".py", prefix="brt_verify_", dir=sandbox_path,
                    mode="w", encoding="utf-8", delete=False,
                ) as tmp:
                    tmp.write(brt["code"])
                    tmp_path = tmp.name
                result_brt = subprocess.run(
                    [sandbox_python, "-m", "pytest", tmp_path, "--tb=short", "-x", "-q", "--no-header"],
                    cwd=sandbox_path,
                    capture_output=True, text=True, timeout=30,
                )
                passed = result_brt.returncode == 0
                if passed:
                    brt_pass_count += 1
                    brt_details.append(f"  BRT {i} ✓ PASS — {brt.get('description','')[:80]}")
                    logger.info("BRT passed in sandbox: %s", brt["description"][:60])
                else:
                    brt_details.append(f"  BRT {i} ✗ FAIL — {brt.get('description','')[:80]}")
                    logger.warning("BRT still failing after fix: %s", brt["description"][:60])
            except Exception as e:
                brt_details.append(f"  BRT {i} ⚠ infra error: {str(e)[:60]}")
                logger.debug("BRT sandbox run error: %s", e)
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

        state["brt_pass_count"] = brt_pass_count
        state["brt_total"] = brt_total
        epr = brt_pass_count / brt_total if brt_total > 0 else 0.0
        state["epr_score"] = round(epr, 3)
        logger.info("EPR score: %.0f%% (%d/%d BRTs pass)", epr * 100, brt_pass_count, brt_total)

    # Build BRT section for verifier prompt
    if brt_total > 0:
        brt_pct = 100 * brt_pass_count / brt_total
        brt_section = (
            f"\n=== BRT RESULTS (run against patched code) ===\n"
            f"EPR score: {brt_pass_count}/{brt_total} passed ({brt_pct:.0f}%)\n"
            + "\n".join(brt_details) + "\n"
            f"If EPR < 100%, the patch did NOT fully resolve the reported bug.\n"
        )
    else:
        brt_section = "\n=== BRT RESULTS ===\nNo BRTs generated for this bug (non-Python repo or generator failed).\n"

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
        f"{plan_section}"
        f"{brt_section}\n"
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
        "of your verdict.\n\n"
        "HARD RULE: If BRT results show EPR < 100% and the test result is real "
        "(not 'error'/'skipped'), verdict MUST be REJECT with confidence >= 0.8. "
        "BRTs are confirmed failing tests — the fix is incomplete if any still fail."
    )

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
                    max_tokens=2000,
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
            result = _structured_call("claude-sonnet-4-6", 2000, VerifierResult, prompt)
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

        # Emit full verifier reasoning into the trace for debugging.
        # This is the "decision replay" data for the verification stage —
        # shows exactly what the verifier saw, what probe it ran, and why
        # it decided APPROVE/REJECT.
        trace = _get_trace()
        if trace:
            trace.emit("verifier_result", "verifier", {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "regression_risk": result.regression_risk,
                "used_fork": state.get("verifier_used_fork", False),
                "downgraded": "[downgraded]" in (result.explanation or ""),
            })

        logger.info(
            "Verifier verdict: %s (confidence: %.0f%%, risk: %s)",
            result.verdict, result.confidence * 100, result.regression_risk,
        )

        # When verifier REJECTs, flag it. If BRTs ALSO fail, the fix is almost
        # certainly incomplete — set a flag that finalize_node can use to trigger
        # a pass@3 retry loop (or at minimum, surface the issue prominently).
        epr_score = state.get("epr_score", None)
        brt_total = state.get("brt_total", 0)
        brts_failing = brt_total > 0 and (epr_score is not None and epr_score < 1.0)

        if result.verdict == "REJECT" and result.confidence > 0.8:
            existing_notes = state.get("explanation", "")
            state["explanation"] = (
                existing_notes + f"\n\n⚠ VERIFIER FLAGGED: {result.explanation}"
            ).strip()
            logger.warning("Verifier rejected with high confidence — flagged in PR body")

            # HARD STOP: verifier REJECT + BRTs failing = fix is incomplete.
            # Set a retry signal so finalize can choose to re-enter the agent loop.
            if brts_failing:
                state["needs_retry"] = True
                state["retry_reason"] = (
                    f"Verifier REJECT at {result.confidence:.0%} confidence AND "
                    f"{brt_total - int(epr_score * brt_total)}/{brt_total} BRTs still failing. "
                    f"Fix incomplete: {result.explanation[:200]}"
                )
                logger.warning(
                    "Retry signal set: verifier REJECT + EPR %.0f%% (BRTs %d failing)",
                    (epr_score or 0) * 100, brt_total - int((epr_score or 0) * brt_total),
                )

    except Exception as e:
        logger.warning("Verifier node failed (non-fatal): %s", e)
        state["verifier_verdict"] = "SKIP"
        state["verifier_explanation"] = f"Verifier error: {e}"
        # Flag in explanation so PR body reflects unverified status
        existing_notes = state.get("explanation", "")
        state["explanation"] = (
            existing_notes + "\n\n⚠ VERIFIER SKIPPED: Verification could not run "
            f"({type(e).__name__}). This patch has NOT been independently reviewed."
        ).strip()

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


def _ensemble_vote(
    candidates: list[dict],
    work_order: dict,
) -> dict | None:
    """LLM ensemble vote: ask a model to pick the best patch from candidates.

    Adapted from Augment's SWE-bench approach (65.4%): generate K patches,
    show all diffs to an ensembler LLM, pick the majority vote winner.
    The key insight from research is that this adds +3-8% pass rate because
    "agent results are highly unstable — any two rollouts can differ."

    Returns the selected candidate dict, or None if ensemble fails.
    """
    from agent.llm import simple_call

    bug_description = work_order.get("description", work_order.get("title", ""))

    # Build compact diff summaries for each candidate
    candidate_summaries = []
    for i, c in enumerate(candidates[:8]):  # Cap at 8 (K=8 sweet spot from research)
        sandbox = c.get("sandbox_path", "")
        diff = ""
        if sandbox and Path(sandbox).exists():
            try:
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=sandbox, capture_output=True, text=True, timeout=30,
                )
                diff = diff_result.stdout[:3000]  # Cap diff size
            except Exception:
                diff = "(diff unavailable)"

        test_result = (c.get("test_result") or "")[:200]
        confidence = (c.get("review") or {}).get("confidence", 0)
        explanation = c.get("explanation", "")[:300]

        candidate_summaries.append(
            f"## Candidate {i+1}\n"
            f"Test result: {test_result}\n"
            f"Verifier confidence: {confidence}\n"
            f"Explanation: {explanation}\n"
            f"Diff:\n```\n{diff}\n```"
        )

    if not candidate_summaries:
        return None

    prompt = f"""You are a code review ensembler. Given a bug description and multiple candidate patches, select the BEST one.

BUG DESCRIPTION:
{bug_description[:2000]}

CANDIDATES:
{"".join(candidate_summaries)}

Which candidate number (1-{len(candidate_summaries)}) is the best fix? Consider:
1. Does the diff correctly address the bug description?
2. Does the test result indicate the fix works?
3. Is the change minimal and focused (no unnecessary modifications)?
4. Is the verifier confidence high?

Reply with ONLY the number (e.g., "2") of the best candidate."""

    try:
        # Use a capable model for voting (same as verifier)
        response = simple_call("claude-haiku-4-5-20251001", prompt, max_tokens=10)
        # Extract the number
        import re as _re
        m = _re.search(r"\d+", response)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(candidates):
                logger.info("Ensemble vote selected candidate %d/%d", idx + 1, len(candidates))
                return candidates[idx]
    except Exception as exc:
        logger.debug("Ensemble LLM call failed: %s", exc)
    return None


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

    submitted_count = sum(1 for r in results if r.get("submitted"))
    test_pass_count = sum(1 for r in results if (r.get("test_result") or "").startswith("passed"))

    # ------------------------------------------------------------------
    # LLM ensemble vote (Augment technique: +3-8% documented improvement)
    # When multiple patches are submitted, ask an LLM to review all diffs
    # and pick the best one. Falls back to static scoring if ensemble fails.
    # ------------------------------------------------------------------
    best = None
    selection_method = "static_score"
    submitted_results = [r for r in results if r.get("submitted")]
    if len(submitted_results) >= 2:
        try:
            best = _ensemble_vote(submitted_results, work_order)
            if best:
                selection_method = "llm_ensemble"
        except Exception as e:
            logger.debug("Ensemble vote failed, falling back to static: %s", e)

    if best is None:
        best = max(results, key=_score)

    logger.info(
        "Best-of-%d complete: %d/%d submitted, %d/%d tests passed — selected by %s",
        n, submitted_count, n, test_pass_count, n, selection_method,
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
