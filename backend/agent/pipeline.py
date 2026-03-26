"""
pipeline.py — LangGraph state machine for the AI Deploy Agent.

Flow: Intake → Context Assembly → Localization → [Confidence Gate]
      → Read Source → Repair → Review → [Dev Loop] → Test → PR
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from agent.feature_flags import create_flag as _create_feature_flag, set_pr_url as _set_flag_pr_url
from agent.types import (
    AgentState,
    IntentAnalysis,
    LocalizationResult,
    Patch,
    PipelineStatus,
    RepairResult,
    ReviewResult,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
MIN_CONFIDENCE_TO_REPAIR = 0.3
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# Thread-local storage for per-run progress callback
_thread_local = threading.local()

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
    r'(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|'
    r'secret[_-]?key|password|passwd|private[_-]?key|credentials)'
    r'\s*[=:]\s*["\']?[A-Za-z0-9+/=_\-]{16,}["\']?'
)


def _redact_secrets(text: str) -> str:
    """Redact potential secrets/tokens from source code before sending to LLM."""
    return _SECRETS_RE.sub("[REDACTED]", text)


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

    llm = ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120.0, max_retries=2)
    structured = llm.with_structured_output(schema)

    try:
        return structured.invoke(prompt)
    except Exception as first_err:
        if retries <= 0:
            raise
        logger.warning("Structured output failed (%s), retrying", first_err)
        # Truncate error message so it doesn't blow up the context window, but
        # keep enough to diagnose which fields failed (ValidationError dumps all fields).
        error_msg = str(first_err)[:1000]
        retry_prompt = (
            f"Your previous response failed: {error_msg}\n"
            "Please try again. Respond with the exact structured data requested.\n\n"
            + prompt
        )
        return structured.invoke(retry_prompt)


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


def _fuzzy_match_replace(content: str, original: str, patched: str) -> str | None:
    """Try multiple matching strategies, from strict to fuzzy.

    Strategies (in order):
    1. Exact substring match
    2. Whitespace-normalized line-by-line match
    3. Stripped-whitespace match (ignores leading indentation differences)
    4. Best sliding-window match (tolerates minor differences like variable names)
    """
    if not original or not original.strip():
        return None

    # Strategy 1: Exact match
    if original in content:
        return content.replace(original, patched, 1)

    def normalize_line(s: str) -> str:
        return s.rstrip().expandtabs(4)

    orig_lines = [normalize_line(l) for l in original.splitlines()]
    content_lines = content.splitlines()
    norm_content_lines = [normalize_line(l) for l in content_lines]

    if not orig_lines:
        return None

    # Strategy 2: Whitespace-normalized exact match
    for i in range(len(norm_content_lines) - len(orig_lines) + 1):
        if norm_content_lines[i:i + len(orig_lines)] == orig_lines:
            new_lines = content_lines[:i] + patched.splitlines() + content_lines[i + len(orig_lines):]
            return '\n'.join(new_lines)

    # Strategy 3: Stripped match (ignores leading whitespace differences entirely)
    stripped_orig = [l.strip() for l in orig_lines if l.strip()]
    stripped_content = [l.strip() for l in content_lines]

    if len(stripped_orig) >= 2:
        for i in range(len(stripped_content) - len(stripped_orig) + 1):
            window = [l for l in stripped_content[i:i + len(stripped_orig) + 5] if l][:len(stripped_orig)]
            if window == stripped_orig:
                # Find the actual line range in content_lines
                matched = 0
                j = i
                start_j = None
                while j < len(content_lines) and matched < len(stripped_orig):
                    if content_lines[j].strip() == stripped_orig[matched]:
                        if start_j is None:
                            start_j = j
                        matched += 1
                    elif content_lines[j].strip():
                        break  # Non-blank mismatch
                    j += 1
                if matched == len(stripped_orig) and start_j is not None:
                    new_lines = content_lines[:start_j] + patched.splitlines() + content_lines[j:]
                    return '\n'.join(new_lines)

    # Strategy 4: Best sliding-window match with similarity scoring
    #   Tolerates minor differences (e.g., variable name changes between
    #   focus-windowed source and actual file)
    if len(orig_lines) >= 3:
        import difflib
        best_score = 0.0
        best_pos = -1
        window_size = len(orig_lines)

        for i in range(len(norm_content_lines) - window_size + 1):
            window = norm_content_lines[i:i + window_size]
            ratio = difflib.SequenceMatcher(None,
                '\n'.join(orig_lines), '\n'.join(window)).ratio()
            if ratio > best_score:
                best_score = ratio
                best_pos = i

        # Only accept if similarity is very high (>85%) — avoids wrong-location patches
        if best_score >= 0.85 and best_pos >= 0:
            new_lines = content_lines[:best_pos] + patched.splitlines() + content_lines[best_pos + window_size:]
            return '\n'.join(new_lines)

    # Strategy 5: Anchor-based matching with adaptive region sizing
    #   Find the most unique/distinctive line in original_code, locate it in content,
    #   then find the ACTUAL extent of the statement in the real file (which may span
    #   more or fewer lines than the LLM's paraphrased version).
    if len(orig_lines) >= 1:
        import re as _re

        def _anchor_score(line: str) -> float:
            s = line.strip()
            if not s or s in ('{', '}', 'pass', 'return', 'else:', 'try:', 'except:'):
                return 0
            score = len(s)
            score += len(_re.findall(r'\w+\.\w+', s)) * 20
            score += len(_re.findall(r'await |return_exceptions|raise |async ', s)) * 15
            return score

        scored = [(i, _anchor_score(l)) for i, l in enumerate(orig_lines)]
        scored.sort(key=lambda x: -x[1])

        for anchor_idx, anchor_score in scored[:3]:
            if anchor_score < 10:
                continue
            anchor = orig_lines[anchor_idx].strip()
            if len(anchor) < 8:
                continue

            # Find this anchor in content
            candidates = []
            dotted = _re.findall(r'\w+\.\w+', anchor)
            for ci, cl in enumerate(content_lines):
                if anchor in cl.strip() or cl.strip() in anchor:
                    candidates.append(ci)
                elif dotted and any(d in cl for d in dotted):
                    if ci not in candidates:
                        candidates.append(ci)

            for ci in candidates:
                # Find the statement boundary in the ACTUAL file
                # Start: go back from the anchor line to find the statement start
                # (look for the line that starts the assignment/call — same or lower indent)
                anchor_indent = len(content_lines[ci]) - len(content_lines[ci].lstrip())
                region_start = ci - anchor_idx  # naive estimate
                region_start = max(0, region_start)

                # Walk back to find the actual statement start (matching indent level)
                stmt_start = ci
                for k in range(ci - 1, max(ci - 10, -1), -1):
                    line_k = content_lines[k]
                    if not line_k.strip():
                        continue
                    indent_k = len(line_k) - len(line_k.lstrip())
                    if indent_k <= anchor_indent:
                        stmt_start = k
                        break
                    elif indent_k > anchor_indent:
                        stmt_start = k  # part of the same multi-line expression
                    else:
                        break

                # Walk forward to find the statement end
                # Look for a closing paren/bracket that balances, or next statement at same indent
                stmt_end = ci + 1
                # Check if there are unclosed parens/brackets
                region_text = '\n'.join(content_lines[stmt_start:stmt_end])
                open_count = region_text.count('(') - region_text.count(')')
                open_count += region_text.count('[') - region_text.count(']')

                while open_count > 0 and stmt_end < min(len(content_lines), ci + 20):
                    line_text = content_lines[stmt_end]
                    open_count += line_text.count('(') - line_text.count(')')
                    open_count += line_text.count('[') - line_text.count(']')
                    stmt_end += 1

                if stmt_start < 0 or stmt_end > len(content_lines):
                    continue

                # Verify: the region should contain the anchor's key elements
                region = content_lines[stmt_start:stmt_end]
                region_text = '\n'.join(region)
                if dotted and not any(d in region_text for d in dotted):
                    continue
                if 'asyncio.gather' not in anchor and 'await' not in anchor:
                    # Generic anchor — need higher similarity check
                    import difflib
                    ratio = difflib.SequenceMatcher(None,
                        '\n'.join(l.strip() for l in orig_lines),
                        '\n'.join(l.strip() for l in region)).ratio()
                    if ratio < 0.35:
                        continue

                # Re-indent the patched code to match the actual file's indentation
                actual_indent = len(content_lines[stmt_start]) - len(content_lines[stmt_start].lstrip())
                patch_lines = patched.splitlines()
                if patch_lines:
                    patch_indent = len(patch_lines[0]) - len(patch_lines[0].lstrip())
                    indent_diff = actual_indent - patch_indent
                    if indent_diff > 0:
                        patch_lines = [(' ' * indent_diff + l) if l.strip() else l for l in patch_lines]
                    elif indent_diff < 0:
                        patch_lines = [l[-indent_diff:] if l[:(-indent_diff)].strip() == '' else l for l in patch_lines]

                new_lines = content_lines[:stmt_start] + patch_lines + content_lines[stmt_end:]
                return '\n'.join(new_lines)

    return None


def _run_tests(worktree_path: Path) -> str:
    """Auto-detect test runner and execute tests.

    Searches both the worktree root and one level of subdirectories so that
    projects with a backend/ or src/ layout (where pytest.ini lives in a subdir)
    are found correctly.
    """
    pytest_markers = ("pytest.ini", "pyproject.toml", "setup.py", "setup.cfg")

    # Check root first, then immediate subdirs (e.g. backend/, src/)
    test_cwd = worktree_path
    cmd = None
    for search_dir in [worktree_path] + sorted(worktree_path.iterdir()):
        if not search_dir.is_dir():
            continue
        if any((search_dir / m).exists() for m in pytest_markers):
            cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
            test_cwd = search_dir
            break
        if (search_dir / "package.json").exists() and cmd is None:
            cmd = ["npm", "test"]
            test_cwd = search_dir
        if (search_dir / "Makefile").exists() and cmd is None:
            cmd = ["make", "test"]
            test_cwd = search_dir

    if cmd is None:
        logger.info("No test runner detected — skipping tests")
        return "skipped: no test runner found"

    logger.info("Running tests in %s: %s", test_cwd, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=test_cwd, capture_output=True, text=True, timeout=300,
        )
        raw_output = (result.stdout + "\n" + result.stderr).strip()

        if result.returncode == 0:
            logger.info("Tests passed")
            # On success, keep summary short (Spotify: short success message)
            summary_lines = [l for l in raw_output.splitlines() if "passed" in l.lower() or "ok" in l.lower()]
            return "passed\n" + ("\n".join(summary_lines[-5:]) if summary_lines else raw_output[:500])
        else:
            logger.warning("Tests failed (exit code %d)", result.returncode)
            # On failure, extract only error-relevant lines (Spotify: regex to extract relevant errors)
            error_lines = []
            for line in raw_output.splitlines():
                line_lower = line.lower()
                if any(kw in line_lower for kw in ("error", "fail", "assert", "exception", "traceback", "syntaxerror", "nameerror", "import")):
                    error_lines.append(line)
                elif line.startswith("E ") or line.startswith("> "):  # pytest error/assertion lines
                    error_lines.append(line)
                elif line.startswith("FAILED "):
                    error_lines.append(line)
            parsed = "\n".join(error_lines[:40]) if error_lines else raw_output[:3000]
            return f"failed (exit code {result.returncode})\n{parsed}"
    except subprocess.TimeoutExpired:
        logger.warning("Tests timed out after 5 minutes")
        return "failed: timed out after 5 minutes"
    except Exception as e:
        logger.warning("Test execution error: %s", e)
        return f"error: {e}"


def _cleanup_worktree(repo_path: Path | None, sandbox_path: str) -> None:
    """Clean up a git worktree."""
    if not sandbox_path or not repo_path:
        return
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", sandbox_path],
            cwd=repo_path, capture_output=True, text=True, timeout=30,
        )
        logger.info("Cleaned up worktree %s", sandbox_path)
    except Exception as e:
        logger.warning("Failed to clean up worktree %s: %s", sandbox_path, e)


def _append_test_business_context(state: AgentState, work_order: dict) -> None:
    """Step 18: If tests failed, look up business-intent enrichments and
    append context to state['test_result'] so the reviewer understands WHY."""
    test_result = state.get("test_result", "")
    if not test_result or "fail" not in test_result.lower():
        return

    repo = work_order.get("repo", work_order.get("repo_name", ""))
    if not repo:
        return

    try:
        from enricher.test_enricher import lookup_failed_tests, format_failure_context

        # Extract failed test names from pytest output (e.g. "FAILED tests/test_x.py::test_foo")
        failed_names: list[str] = []
        for line in test_result.splitlines():
            if "FAILED" in line:
                # pytest format: "FAILED tests/test_x.py::test_foo - ..."
                match = re.search(r"FAILED\s+\S+::(\w+)", line)
                if match:
                    failed_names.append(match.group(1))

        if not failed_names:
            return

        enrichments = lookup_failed_tests(repo, failed_names)
        if enrichments:
            context_str = format_failure_context(enrichments)
            state["test_result"] = test_result + context_str
            logger.info("Appended business context for %d failed tests", len(enrichments))
    except Exception as e:
        logger.debug("Could not append test business context: %s", e)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> AgentState:
    """Stage 1: Translate bug ticket into technical spec via structured output."""
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
        result = _structured_call("claude-sonnet-4-6", 1000, IntentAnalysis, prompt)
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

    state["iteration_count"] = 0
    return state


def context_assembly_node(state: AgentState) -> AgentState:
    """Stage 2: Query Graph RAG to assemble targeted context."""
    logger.info("=== CONTEXT ASSEMBLY: Building targeted context via Graph RAG ===")
    state["status"] = PipelineStatus.CONTEXT
    _report_progress(state)

    work_order = state.get("work_order", {})
    intent = state.get("intent", {})
    repo_name = work_order.get("repo_name", "")

    search_terms = []
    search_terms.append(intent.get("actual_behavior", ""))
    search_terms.extend(intent.get("likely_affected_modules", []))
    search_terms.extend(intent.get("likely_affected_functions", []))
    query = " ".join(search_terms)

    try:
        from rag.retriever import GraphRAGRetriever
        from rag.context_assembler import ContextAssembler

        retriever = GraphRAGRetriever(repo_name, DATA_DIR)
        result = retriever.retrieve(query, max_nodes=40)

        assembler = ContextAssembler(repo_name, DATA_DIR)
        context = assembler.assemble(
            primary_ids=result.primary_nodes,
            expanded_ids=result.expanded_nodes,
            edges=result.edges,
            scores=result.scores,
            token_budget=20000,
        )

        state["context"] = context
        state["context_nodes"] = len(result.all_node_ids)
        logger.info("Context assembled: %d nodes, ~%d tokens", len(result.all_node_ids), len(context) // 4)

    except Exception as e:
        logger.warning("Graph RAG failed, falling back to summary: %s", e)
        summary_path = DATA_DIR / repo_name / "summary.md"
        if summary_path.exists():
            state["context"] = summary_path.read_text()
        else:
            state["context"] = "No context available."
        state["context_nodes"] = 0

    return state


def exploration_node(state: AgentState) -> AgentState:
    """Stage 2b: Agentic exploration — agent uses tools to find the bug itself.

    Like Claude Code: the agent gets grep, read_file, read_function, list_files,
    search_code, get_function_info. It explores until confident, then summarises
    what it found (fault files, functions, source snippets) into state.

    Replaces the old context_assembly + localization + read_source combo.
    """
    logger.info("=== EXPLORATION: Agent actively exploring the codebase ===")
    state["status"] = PipelineStatus.EXPLORING
    _report_progress(state)

    work_order = state.get("work_order", {})
    intent = state.get("intent", {})
    repo_name = work_order.get("repo_name", "")
    repo_path = _resolve_repo_path(work_order)

    if not repo_path:
        logger.warning("No repo_path — falling back to context_assembly")
        return context_assembly_node(state)

    # Set per-run context for tools
    from agent.explore_tools import set_context, ALL_TOOLS
    set_context(repo_name, repo_path, DATA_DIR)

    # Load non-inferable business rules for this repo (Finding #4: only non-inferable context helps)
    business_context = ""
    try:
        br_path = DATA_DIR / repo_name / "business_rules.json"
        if br_path.exists():
            import json as _json
            br_data = _json.loads(br_path.read_text())
            if br_data:
                top_rules = [f"  - [{r.get('rule_type','rule')}] {r.get('content','')[:120]}"
                             for r in br_data[:10]]
                business_context = "\nBUSINESS RULES (non-inferable — cannot be discovered from code):\n" + "\n".join(top_rules)
    except Exception:
        pass

    # End-state prompt (Findings #5, #7, #10: no file tree, no step-by-step, describe outcome)
    system_prompt = f"""You are debugging a production bug in repo `{repo_name}` at `{repo_path}`.

You have tools: grep_repo, read_file, read_function, list_files, search_code, get_function_info, get_file_structure.

A successful exploration ends with you writing a summary that contains:
- The exact fault file path(s) and function name(s)
- A root cause hypothesis explaining WHY the bug occurs
- The relevant source code of the buggy function(s) and their callers

Stop exploring as soon as you have enough evidence. Do not read files you don't need.
{business_context}
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
    while tool_call_count < MAX_TOOL_CALLS:
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

            # Find and invoke the tool
            result_str = f"Tool '{tool_name}' not found"
            for t in ALL_TOOLS:
                if t.name == tool_name:
                    try:
                        result_str = t.invoke(tool_args)
                    except Exception as te:
                        result_str = f"Tool error: {te}"
                    break

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

Extract the most likely fault location. If both the exploration and semantic search agree on a file, weight it higher."""
            loc = _structured_call("claude-sonnet-4-6", 800, LocalizationResult, parse_prompt)

            # Merge: ensure embedding-suggested files appear in fault_files if relevant
            merged_fault_files = list(loc.fault_files)
            for ef in embedding_files:
                if ef not in merged_fault_files and len(merged_fault_files) < 5:
                    # Only add if not already covered
                    merged_fault_files.append(ef)

            loc_dict = loc.model_dump()
            loc_dict["fault_files"] = merged_fault_files
            state["localization"] = loc_dict
            logger.info("Exploration localization: confidence=%.2f files=%s",
                        loc.confidence, merged_fault_files)
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
    return state


def localization_node(state: AgentState) -> AgentState:
    """Stage 3: Localize the fault using structured output."""
    logger.info("=== LOCALIZATION: Finding the fault site ===")
    state["status"] = PipelineStatus.LOCALIZING
    _report_progress(state)

    intent = state.get("intent", {})
    context = state.get("context", "")

    prompt = f"""You are a senior developer localizing a bug.

BUG: {intent.get('actual_behavior', '')}
EXPECTED: {intent.get('expected_behavior', '')}
HINTS: modules={intent.get('likely_affected_modules', [])}, functions={intent.get('likely_affected_functions', [])}

CODEBASE CONTEXT:
{context}

Based on the context above, identify the most likely fault locations.
Be specific about file paths and function names visible in the context."""

    try:
        result = _structured_call("claude-sonnet-4-6", 1500, LocalizationResult, prompt)
        state["localization"] = result.model_dump()
    except Exception as e:
        logger.error("Localization failed: %s", e)
        state["localization"] = {
            "fault_files": intent.get("likely_affected_modules", []),
            "fault_functions": intent.get("likely_affected_functions", []),
            "fault_classes": [],
            "root_cause_hypothesis": "Could not determine root cause.",
            "confidence": 0.1,
            "evidence": [],
        }

    return state


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
                caller_files.add(src_file)

    return sorted(caller_files)[:8]


def _find_callers_via_grep(repo_path: Path, fault_files: list[str]) -> list[str]:
    """Fallback: grep for files that import the fault files."""
    caller_paths: list[str] = []
    seen: set[str] = set()

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
                        if '__pycache__' in str(p) or '/test' in str(p).lower():
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
    """Load stored business rules relevant to the fault files.

    These are rules created from human answers (Step 8/21 of the guide).
    They contain knowledge the agent literally cannot discover from code alone.
    """
    rules_path = DATA_DIR / repo_name / "business_rules.json"
    if not rules_path.exists():
        return ""

    try:
        all_rules = json.loads(rules_path.read_text())
    except Exception:
        return ""

    relevant = []
    for rule in all_rules:
        rule_file = rule.get("file", "")
        rule_func = rule.get("function_id", "")
        if any(f in rule_file or f in rule_func for f in fault_files):
            severity = rule.get("severity", "medium").upper()
            marker = "⚠️ DO NOT VIOLATE" if severity in ("CRITICAL", "HIGH") else ""
            relevant.append(
                f"  [{severity}] {rule.get('description', '')[:300]} {marker}\n"
                f"    Source: {rule.get('source', 'unknown')} | File: {rule_file}"
            )

    if not relevant:
        return ""
    return "\n\nBUSINESS RULES (from human-verified knowledge base — DO NOT VIOLATE):\n" + "\n".join(relevant)


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


def read_source_node(state: AgentState) -> AgentState:
    """Stage 3.5: Read source code using the knowledge graph for smart discovery.

    Strategy:
    1. Read fault files from disk
    2. Use graph CALLS/IMPORTS edges to discover call sites (callers)
    3. Read caller files (where the fix needs to be wired in)
    4. Add enriched symbol info (docstrings, params, call chains)
    """
    logger.info("=== READ SOURCE: Loading code via knowledge graph ===")
    state["status"] = PipelineStatus.READING_SOURCE
    _report_progress(state)

    work_order = state.get("work_order", {})
    localization = state.get("localization", {})
    fault_files = localization.get("fault_files", [])
    fault_functions = localization.get("fault_functions", [])
    repo_name = work_order.get("repo_name", "")

    repo_path = _resolve_repo_path(work_order)
    if repo_path and not work_order.get("repo_path"):
        work_order = dict(work_order)
        work_order["repo_path"] = str(repo_path)
        state["work_order"] = work_order

    if not repo_path:
        logger.warning("Could not resolve repo path for %s — using context only", repo_name)
        state["source_code"] = {}
        return state

    # Load the knowledge graph for smart caller discovery
    graph_data, enriched = _load_graph_data(repo_name)
    has_graph = bool(graph_data.get("edges"))

    source_code: dict[str, str] = {}

    # 1. Read the fault files themselves (with focus on fault functions)
    #    Use graph node line_start/line_end for precise function extraction
    for rel_path in fault_files[:5]:
        resolved = _find_file_in_repo(repo_path, rel_path)
        if resolved:
            # Collect focus lines from multiple sources:
            focus_lines: list[int] = []

            # a) Graph nodes with line numbers for fault functions in this file
            for node in graph_data.get("nodes", []):
                nid = node.get("id", "")
                if node.get("type") != "function":
                    continue
                nfile = nid.split("::")[0] if "::" in nid else ""
                if nfile != rel_path:
                    continue
                # Check if this function is in the fault_functions list
                short_name = nid.split("::")[-1]
                if any(short_name == ff or short_name in ff or ff in nid for ff in fault_functions):
                    ls = node.get("line_start", 0)
                    le = node.get("line_end", 0)
                    if ls:
                        focus_lines.append(ls)
                    if le:
                        focus_lines.append(le)

            # b) Decision points in this file
            for dp in graph_data.get("decision_points", []):
                if dp.get("file", "") == rel_path or rel_path in dp.get("function_id", ""):
                    line = dp.get("line", 0)
                    if line:
                        focus_lines.append(line)

            # c) Line hints from the root cause text (e.g. "line 688")
            root_cause = localization.get("root_cause_hypothesis", "")
            import re as _re
            for m in _re.finditer(r'lines?\s+(\d+)', root_cause):
                focus_lines.append(int(m.group(1)))

            content = _read_file_safe(resolved, max_lines=3000, focus_lines=focus_lines or None)
            if content:
                # Strip gap markers so LLM sees only real source lines it can copy from
                content = _strip_gap_markers(content)
                source_code[rel_path] = content
                logger.info("Read fault file: %s (%d lines, %d focus points)",
                           rel_path, len(content.split('\n')), len(focus_lines))

    # 2. Discover callers — graph first, grep fallback
    if has_graph:
        caller_files = _find_callers_from_graph(graph_data, fault_files, fault_functions)
        logger.info("Graph found %d caller files: %s", len(caller_files), caller_files)
    else:
        caller_files = _find_callers_via_grep(repo_path, fault_files)
        logger.info("Grep found %d caller files: %s", len(caller_files), caller_files)

    # 3. Read caller files
    for rel in caller_files:
        if rel in source_code:
            continue
        resolved = _find_file_in_repo(repo_path, rel)
        if resolved:
            content = _read_file_safe(resolved, max_lines=3000)
            if content:
                content = _strip_gap_markers(content)
                source_code[f"{rel} (caller)"] = content
                logger.info("Read caller: %s (%d lines)", rel, len(content.split('\n')))

    # 4. Add enriched context (function signatures, docstrings, call chains)
    enrichment = _build_enrichment_context(enriched, fault_files, fault_functions)
    if enrichment:
        source_code["__enrichment__"] = enrichment
        logger.info("Added enriched symbol info for %d fault functions", len(fault_functions))

    # 5. Load business rules from human answers (Step 8/21)
    business_rules = _load_business_rules(repo_name, fault_files)
    if business_rules:
        source_code["__business_rules__"] = business_rules
        logger.info("Loaded business rules for fault files")

    state["source_code"] = source_code
    logger.info("Loaded %d files total (%d fault + %d callers) using %s",
                len(source_code) - (1 if enrichment else 0),
                min(len(fault_files), 5), len(caller_files),
                "knowledge graph" if has_graph else "grep fallback")
    return state


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


def _check_syntax(file_path: Path) -> str | None:
    """Check Python file for syntax errors. Returns error message or None if OK.

    Uses ``ast.parse`` in-process instead of spawning a subprocess — avoids
    any shell/Python injection risk from unusual file paths and is ~100x faster.
    """
    if file_path.suffix != ".py":
        return None
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return str(e)
    return None


def _deduplicate_patches(patches: list[dict]) -> list[dict]:
    """Deduplicate patches: one patch per unique (file_path, original_code) pair.

    When multiple patches target the same region of the same file, keep the last one
    (which is typically the most complete, as the LLM refines its approach).
    """
    seen: dict[str, dict] = {}  # key: (file_path, normalized_original) → patch
    for p in patches:
        key = f"{p.get('file_path', '')}::{p.get('original_code', '').strip()[:200]}"
        seen[key] = p  # Last one wins
    return list(seen.values())


def _pick_best_patch_per_file(patches: list[dict], repo_path: Path | None) -> list[dict]:
    """When multiple patches target the same file, pick the most complete one.

    Based on SWE-Agent research: one edit per file per turn prevents cascading conflicts.
    Selection criteria: longest patched_code that actually applies to the source.
    """
    if not repo_path:
        return patches

    by_file: dict[str, list[dict]] = {}
    for p in patches:
        by_file.setdefault(p.get("file_path", ""), []).append(p)

    result: list[dict] = []
    for file_path, file_patches in by_file.items():
        if len(file_patches) == 1:
            result.append(file_patches[0])
            continue

        # Multiple patches for same file — pick the one with the most new code that verifies
        resolved = _find_file_in_repo(repo_path, file_path)
        content = _read_file_safe(resolved, max_lines=5000) if resolved else None

        best = None
        best_score = -1
        for p in file_patches:
            original = p.get("original_code", "")
            patched = p.get("patched_code", "")
            # Score: how much new code does this patch add?
            score = len(patched) - len(original)
            # Verify it applies
            if content and _fuzzy_match_replace(content, original, patched) is not None:
                score += 1000  # Huge bonus for actually applying
            if score > best_score:
                best_score = score
                best = p

        if best:
            result.append(best)
            logger.info("Picked best patch for %s (%d candidates, score=%d)", file_path, len(file_patches), best_score)

    return result


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
    test_result = state.get("test_result", "")
    if test_result and "fail" in test_result.lower():
        feedback_section += f"\nTEST FAILURE (fix your tests):\n{test_result[:1000]}\n"

    # Acceptance criteria from spec (Finding #21: verification from spec, not implementation)
    acceptance = intent.get("acceptance_criteria", [])
    criteria_section = ""
    if acceptance:
        criteria_section = "\nACCEPTANCE CRITERIA (your tests MUST verify these):\n"
        criteria_section += "\n".join(f"  - {c}" for c in acceptance)

    # End-state prompt style (Research Finding #10 — Claude performs better with end-state descriptions)
    prompt = f"""Fix this bug by producing a RepairResult with correct patches.

BUG: {intent.get('actual_behavior', '')}
EXPECTED: {intent.get('expected_behavior', '')}
ROOT CAUSE: {localization.get('root_cause_hypothesis', '')}
TARGET FUNCTIONS: {localization.get('fault_functions', [])} in {localization.get('fault_files', [])}
{criteria_section}
{feedback_section}
{source_section}

A correct RepairResult has:
- `patches`: one entry per function you change.
  - `original_code`: copy the function EXACTLY as shown above, starting from the `def` line.
    Character-for-character. Same indentation, same newlines. Do NOT truncate or paraphrase.
  - `patched_code`: the corrected function. Must differ from original_code.
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
3. new tests cover the fixed behaviour and the existing ones still pass"""

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
            fault_file = localization.get("fault_files", [""])[0]
            fault_fn = localization.get("fault_functions", [""])[0]

            retry_prompt = f"""Patches array was empty. You must produce patches.

BUG: {intent.get('actual_behavior', '')}
TARGET: function `{fault_fn}` in file `{fault_file}`
Your analysis: {explanation[:200]}

{source_section}

The correct patch has:
- file_path: "{fault_file}"
- original_code: copy the `def {fault_fn}` function EXACTLY from the source above
- patched_code: the corrected version of that function

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

            repair_dump["patches"] = verified

        state["repair"] = repair_dump
    except Exception as e:
        logger.error("Repair failed: %s", e)
        state["repair"] = {
            "patches": [],
            "explanation": f"Repair generation failed: {e}",
            "tests_added": [],
        }

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

    prompt = f"""Review this bug fix as an independent reviewer who has NOT seen the developer's code.

BUG: {intent.get('actual_behavior', '')}
EXPECTED: {intent.get('expected_behavior', '')}
{criteria_section}

PROPOSED PATCHES:
{json.dumps(clean_patches, indent=2)}

FIX EXPLANATION: {repair.get('explanation', '')}

INDEPENDENT CONTEXT (from knowledge graph):
{reviewer_context}

A correct review produces:
- 6 checks (ROOT_CAUSE, BUSINESS_RULES, PATTERNS, COMPLETENESS, BLAST_RADIUS, TESTS), each PASS/FAIL/WARNING
- ROOT_CAUSE passes when the fix addresses why the bug happens, not just the symptom
- BUSINESS_RULES passes when no rules from the context above are violated
- PATTERNS passes when code follows existing conventions (naming, imports, style)
- COMPLETENESS passes when every changed function is wired into its call sites (no dead code)
- BLAST_RADIUS passes when downstream consumers (listed above) are not broken
- TESTS passes when test_patches contains real test code covering the fix
- verdict: APPROVE (all pass), CHANGES_REQUESTED (concrete failures), ESCALATE (too complex)"""

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

    return state


def test_node(state: AgentState) -> AgentState:
    """Stage 5.5: Create sandbox via git worktree, apply patches, run tests."""
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
            return state

        # Create worktree (safe_ticket_id has no special chars, path is safe)
        worktree_path = Path(f"/tmp/agent_sandbox_{safe_ticket_id}_{branch_suffix}")
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(worktree_path), base_branch],
            cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
        )
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
            else:
                logger.warning("Patch could not be matched in %s", file_path)

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
            ["git", "commit", "-m", commit_msg],
            cwd=worktree_path, capture_output=True, text=True, check=True, timeout=30,
        )
        logger.info("Committed %d patches in sandbox", patches_applied)

        # Auto-detect and run tests
        test_result = _run_tests(worktree_path)
        state["test_result"] = test_result

    except subprocess.CalledProcessError as e:
        logger.error("Sandbox/test operation failed: %s — %s", e, e.stderr)
        state["test_result"] = f"error: {e.stderr}"
        state["error"] = f"Sandbox operation failed: {e.stderr}"
        _cleanup_worktree(repo_path, state.get("sandbox_path", ""))
    except Exception as e:
        logger.error("Test node failed: %s", e)
        state["test_result"] = f"error: {e}"
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
            review = dict(review)
            review["verdict"] = "APPROVE"
            state["review"] = review
            logger.info("Upgraded review verdict to APPROVE — tests now pass")

    return state


def pr_creation_node(state: AgentState) -> AgentState:
    """Stage 6: Push branch and create GitHub PR from sandbox."""
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
        return state

    try:
        # Push branch to remote
        push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=sandbox_path, capture_output=True, text=True, timeout=120,
        )
        if push_result.returncode != 0:
            logger.warning("Git push failed: %s", push_result.stderr)
            state["pr_url"] = f"branch://{branch_name} (push failed: {push_result.stderr[:200]})"
            state["status"] = PipelineStatus.DONE
            _report_progress(state)
            _cleanup_worktree(repo_path, sandbox_path)
            return state

        logger.info("Pushed branch %s to origin", branch_name)

        # Build PR body with blast radius analysis
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
    logger.info("=== ESCALATE: Agent cannot fix with confidence ===")
    state["status"] = PipelineStatus.ESCALATED
    state["error"] = (
        f"Escalated after {state.get('iteration_count', 0)} iterations. "
        f"Last review: {state.get('review', {}).get('feedback', 'no feedback')}"
    )
    _report_progress(state)
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_read_source_or_escalate(state: AgentState) -> Literal["read_source", "escalate"]:
    """Confidence gate: proceed only if localization is confident enough."""
    confidence = state.get("localization", {}).get("confidence", 0)
    fault_files = state.get("localization", {}).get("fault_files", [])

    if confidence < MIN_CONFIDENCE_TO_REPAIR or not fault_files:
        logger.info("Low confidence (%.0f%%) or no fault files — escalating", confidence * 100)
        return "escalate"
    return "read_source"


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
    """Route after test_node: retry repair on syntax errors, else proceed to PR."""
    test_result = state.get("test_result", "")
    iteration = state.get("iteration_count", 0)

    if "syntax error" in test_result.lower():
        if iteration >= MAX_ITERATIONS:
            logger.warning("Syntax errors after max iterations — escalating")
            return "escalate"
        logger.info("Syntax errors in patched files — routing back to repair (iteration %d)", iteration)
        return "retry_fix"

    return "create_pr"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph():
    """Build and compile the LangGraph state machine.

    AGENT_MODE env var controls exploration strategy:
      'explore'  — agentic tool loop (grep/read/search), agent finds context itself (new)
      'rag'      — Graph RAG push context upfront (legacy default)
    """
    mode = os.environ.get("AGENT_MODE", "explore")
    logger.info("Building agent graph in mode: %s", mode)

    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("repair", repair_node)
    graph.add_node("review", review_node)
    graph.add_node("test", test_node)
    graph.add_node("create_pr", pr_creation_node)
    graph.add_node("escalate", escalate_node)
    graph.set_entry_point("intake")

    if mode == "explore":
        # New: agentic exploration replaces context_assembly + localization + read_source
        graph.add_node("exploration", exploration_node)
        graph.add_edge("intake", "exploration")
        graph.add_edge("exploration", "repair")
    else:
        # Legacy: push context upfront via Graph RAG
        graph.add_node("context_assembly", context_assembly_node)
        graph.add_node("localization", localization_node)
        graph.add_node("read_source", read_source_node)
        graph.add_edge("intake", "context_assembly")
        graph.add_edge("context_assembly", "localization")
        graph.add_conditional_edges(
            "localization",
            should_read_source_or_escalate,
            {"read_source": "read_source", "escalate": "escalate"},
        )
        graph.add_edge("read_source", "repair")

    graph.add_edge("repair", "review")
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


def run_ticket(work_order: dict, progress_cb: Callable[[AgentState], None] | None = None) -> dict:
    """Run a bug ticket through the full agent pipeline."""
    _thread_local.progress_callback = progress_cb

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
