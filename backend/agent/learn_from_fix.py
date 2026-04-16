"""
learn_from_fix.py — Per-repo persistent learnings from completed agent runs.

This is the "memory" layer that makes our autonomous agent ADAPT over time.
After each run (success OR fail), we extract a short structured lesson
and append it to `{DATA_DIR}/{repo_name}/agent_lessons.md`. Future runs
on the same repo read that file and include relevant lessons in the
system prompt, so the agent doesn't re-discover the same gotchas.

Design principles:
  - **Cheap**: single Haiku call per run (~$0.01), capped at 400 output
    tokens. Fallback to a rule-based extraction if the Haiku call fails.
  - **Bounded**: agent_lessons.md capped at 25 lessons. Oldest evicted
    when cap is hit. Keeps the file size predictable.
  - **Relevant**: when loaded into the next run, only the 5 most recent
    lessons are injected. Avoids flooding the prompt with stale wisdom.
  - **No cross-repo leakage**: lessons are strictly per-repo_name. A
    lesson about Django test settings stays out of Flask runs.

Ports the spirit of Claude Code's extractMemories + SessionMemory
(services/extractMemories, services/SessionMemory) adapted for our
one-shot-per-bug model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
LESSONS_FILENAME = "agent_lessons.md"
GLOBAL_LESSONS_DIR = "_global"  # cross-repo lessons stored here
MAX_LESSONS_STORED = 25
MAX_LESSONS_INJECTED = 5
MAX_GLOBAL_LESSONS_INJECTED = 3
LESSON_MAX_CHARS = 600

# Feature flag — enabled by default, disable via env var for tests / comparison
LEARN_FROM_FIX_ENABLED = os.environ.get("DISABLE_LEARN_FROM_FIX", "") not in (
    "1", "true", "True",
)


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------

def _derive_tests_passed(state: dict) -> bool:
    """Derive whether tests passed from the state dict.

    The ReAct loop emits `run_outcome.tests_passed` as a trace event, but
    it's not stored on the state dict. So we check multiple fields:
    test_result starts with "passed"/"PASSED"/"PASS"/"passing", OR
    verifier_verdict == APPROVE, OR the eval scorer's full_pass flag.
    """
    test_result = str(state.get("test_result", "") or "").strip().lower()
    # Handle multiple test runner output formats:
    #   pytest: "passed" / "PASSED"
    #   jest:   "PASS" / "Tests: X passed"
    #   mocha:  "passing"
    if test_result.startswith(("passed", "pass")):
        return True
    if "passing" in test_result[:50]:
        return True
    # Fallback: verifier approved with high confidence counts as "tests passed"
    # since the verifier assesses the full diff + test evidence
    if state.get("verifier_verdict") == "APPROVE" and \
            state.get("verifier_confidence", 0) >= 0.7:
        return True
    return False


def _build_lesson_extraction_prompt(state: dict, successful: bool) -> str:
    """Build the prompt for the Haiku extractor."""
    work_order = state.get("work_order", {}) or {}
    intent = state.get("intent", {}) or {}
    review = state.get("review", {}) or {}

    ticket_id = work_order.get("ticket_id", "UNKNOWN")
    title = work_order.get("title", "")
    fix_type = intent.get("fix_type", "bug_fix")

    # Get the plan the agent actually followed
    from agent.react_tools import get_current_plan
    plan = get_current_plan() or {}

    # Build short summary of the run
    tests_passed = _derive_tests_passed(state)
    status = "SUCCESS" if successful else (
        "FAIL-NO-FIX" if not state.get("submitted")
        else "FAIL-TESTS" if not tests_passed
        else "FAIL-OTHER"
    )
    tool_call_count = state.get("tool_call_count", 0)

    return f"""You are extracting a LESSON from a completed bug-fix attempt so that
future runs on the same codebase can avoid repeating mistakes (or can
repeat what worked).

Write ONE concise lesson (max {LESSON_MAX_CHARS} chars), structured:

  **Pattern** (one short phrase — what was this bug about?)
  **Lesson** (1-3 sentences — what to remember for similar bugs)
  **Tactic** (optional — a specific tool/approach that worked or failed)

DO NOT re-summarize the ticket. DO NOT mention "the agent" — write in
second person ("when you see X, do Y"). Focus on what's TRANSFERABLE.

=== RUN DETAILS ===
Ticket: {ticket_id}
Title: {title[:150]}
Fix type: {fix_type}
Outcome: {status}
Review verdict: {review.get("verdict", "n/a")}
Verifier verdict: {state.get("verifier_verdict", "n/a")} (confidence: {state.get("verifier_confidence", 0)})
Test pass: {tests_passed}
Tool calls used: {tool_call_count}

=== PLAN THE AGENT FOLLOWED ===
Root cause: {plan.get("root_cause", "(no plan produced)")[:300]}
Target files: {plan.get("target_files", [])}
Approach: {plan.get("approach", "")[:300]}

=== WHAT TO WRITE ===
If SUCCESS: what specifically worked? (tool choice, file discovery pattern,
  test-selection strategy)
If FAIL: what went wrong? Was the hypothesis right but the execution bad,
  or was the hypothesis itself wrong? What would you do differently?

Your entire response MUST be valid markdown in the exact format:

**Pattern**: <phrase>
**Lesson**: <sentences>
**Tactic**: <optional specific technique>
"""


def _extract_lesson_via_haiku(state: dict, successful: bool) -> str:
    """Run a Haiku call to extract a structured lesson from the run state.

    Returns the lesson markdown or empty string on any failure.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        prompt = _build_lesson_extraction_prompt(state, successful)
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            timeout=30.0,
            max_retries=1,
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = (
            str(resp.content)
            if not isinstance(resp.content, list)
            else " ".join(str(b.get("text", "")) for b in resp.content if isinstance(b, dict))
        )
        return (text or "").strip()[:LESSON_MAX_CHARS]
    except Exception as exc:
        logger.debug("Haiku lesson extraction failed (non-fatal): %s", exc)
        return ""


def _fallback_lesson(state: dict, successful: bool) -> str:
    """Rule-based lesson when Haiku is unavailable. Keeps the feature alive
    even without the LLM call. Structure matches the Haiku prompt's format
    so downstream consumers see the same shape.
    """
    from agent.react_tools import get_current_plan
    work_order = state.get("work_order", {}) or {}
    plan = get_current_plan() or {}

    ticket = work_order.get("ticket_id", "UNKNOWN")
    tool_calls = state.get("tool_call_count", 0)

    if successful:
        pattern = f"{plan.get('target_files', ['unknown'])[0] if plan.get('target_files') else 'unknown'} fix"
        lesson = (
            f"[{ticket}] Successful fix in {tool_calls} calls. "
            f"Root cause: {plan.get('root_cause', 'unknown')[:180]}"
        )
        tactic = plan.get("approach", "")[:180] or "see plan"
    else:
        reason = state.get("escalate_reason", "") or "tests failed"
        pattern = f"{plan.get('target_files', ['unknown'])[0] if plan.get('target_files') else 'unknown'} — unresolved"
        lesson = (
            f"[{ticket}] Failed after {tool_calls} calls. "
            f"Hypothesis: {plan.get('root_cause', 'unknown')[:180]}. "
            f"Reason: {reason[:140]}"
        )
        tactic = "reconsider the root cause — hypothesis may be wrong"

    return (
        f"**Pattern**: {pattern[:100]}\n"
        f"**Lesson**: {lesson[:400]}\n"
        f"**Tactic**: {tactic[:180]}"
    )


# ---------------------------------------------------------------------------
# Storage — append-trim-persist
# ---------------------------------------------------------------------------

def _lessons_path(repo_name: str) -> Path:
    return DATA_DIR / repo_name / LESSONS_FILENAME


def _parse_lessons(text: str) -> list[dict]:
    """Parse existing lessons markdown into a list of entries.

    Each entry looks like:
      ## [ticket_id] YYYY-MM-DD status
      **Pattern**: ...
      **Lesson**: ...
      **Tactic**: ...
    Robust to minor format drift.
    """
    if not text.strip():
        return []
    entries: list[dict] = []
    # Split on "## " headers (each lesson starts with one)
    chunks = re.split(r"\n(?=## )", text.strip())
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        header_match = re.match(r"## \[([^\]]+)\]\s*([\d\-]+)?\s*(\w+)?", chunk)
        header = chunk.split("\n", 1)[0]
        body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
        if header_match:
            entries.append({
                "ticket_id": header_match.group(1),
                "date": header_match.group(2) or "",
                "status": header_match.group(3) or "",
                "header": header,
                "body": body.strip(),
            })
        else:
            # Unrecognized header — keep as-is so we don't drop it
            entries.append({
                "ticket_id": "",
                "date": "",
                "status": "",
                "header": header,
                "body": body.strip(),
            })
    return entries


def _format_lessons(entries: list[dict]) -> str:
    """Serialize a list of entries back to markdown."""
    parts: list[str] = []
    for e in entries:
        parts.append(e["header"])
        if e.get("body"):
            parts.append(e["body"])
        parts.append("")  # blank line separator
    return "\n".join(parts).rstrip() + "\n"


def record_lesson(state: dict) -> str | None:
    """Record a lesson from a completed run. Called from finalize_node.

    Returns the lesson markdown that was appended (for logging/testing),
    or None if the feature is disabled or no lesson could be written.
    """
    if not LEARN_FROM_FIX_ENABLED:
        return None

    work_order = state.get("work_order", {}) or {}
    repo_name = work_order.get("repo_name")
    if not repo_name:
        logger.debug("record_lesson: no repo_name in work_order — skipping")
        return None

    successful = bool(
        state.get("submitted")
        and _derive_tests_passed(state)
        and state.get("review", {}).get("verdict") == "APPROVE"
    )
    status_label = "SUCCESS" if successful else "FAIL"

    # Try Haiku first, fall back to rule-based
    lesson_body = _extract_lesson_via_haiku(state, successful)
    if not lesson_body:
        lesson_body = _fallback_lesson(state, successful)

    ticket_id = work_order.get("ticket_id", "UNKNOWN")
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = f"## [{ticket_id}] {date_str} {status_label}"
    entry_text = f"{header}\n{lesson_body.strip()}\n"

    # Load existing lessons, append, trim, write back.
    # Use file locking to prevent concurrent writes from losing data
    # (parallel eval runs on the same repo).
    import fcntl
    lessons_path = _lessons_path(repo_name)
    try:
        lessons_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode to create if needed, then lock
        with open(lessons_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                existing_text = f.read()
                existing = _parse_lessons(existing_text)
                # Add the new entry at the tail
                new_entry = {
                    "ticket_id": ticket_id,
                    "date": date_str,
                    "status": status_label,
                    "header": header,
                    "body": lesson_body.strip(),
                }
                existing.append(new_entry)
                # Trim to MAX_LESSONS_STORED (keep the newest)
                if len(existing) > MAX_LESSONS_STORED:
                    existing = existing[-MAX_LESSONS_STORED:]
                f.seek(0)
                f.truncate()
                f.write(_format_lessons(existing))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        logger.info(
            "Recorded lesson for %s/%s (%s), %d lessons total in %s",
            repo_name, ticket_id, status_label, len(existing), lessons_path,
        )
        return entry_text
    except Exception as exc:
        logger.warning("record_lesson failed (non-fatal): %s", exc)
        return None

    # Cross-repo: check if this lesson is generalizable (applies beyond this repo).
    # If yes, also store in _global/agent_lessons.md so OTHER repos benefit.
    try:
        _maybe_record_global_lesson(repo_name, ticket_id, date_str, status_label, lesson_body)
    except Exception as exc:
        logger.debug("Global lesson recording failed (non-fatal): %s", exc)

    return entry_text


def _maybe_record_global_lesson(
    repo_name: str, ticket_id: str, date_str: str,
    status_label: str, lesson_body: str,
) -> None:
    """Check if a lesson is generalizable and store it in the global tier.

    A lesson is generalizable if it's about a PATTERN (regex, API design,
    test strategy) rather than a repo-specific detail (Django's ORM,
    Flask's blueprint system). Haiku makes the call — cheap ($0.005).

    Global lessons are injected into ALL future runs (from any repo),
    giving the agent cross-repo wisdom like "always check regex DOTALL
    for multiline strings" even when working on a new repo.
    """
    if not LEARN_FROM_FIX_ENABLED or status_label != "SUCCESS":
        return  # Only promote successful lessons to global

    # Ask Haiku: is this lesson generalizable?
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        prompt = f"""Is this lesson GENERALIZABLE to other codebases, or is it specific to this one repo ({repo_name})?

LESSON:
{lesson_body}

Answer ONLY "GENERAL" or "SPECIFIC".
- GENERAL: the lesson is about a universal pattern (regex, testing, API design, error handling, concurrency) that applies to any codebase
- SPECIFIC: the lesson is about this particular codebase's internal structure, naming, or conventions

One word answer:"""

        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            timeout=15.0,
            max_retries=1,
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        answer = str(resp.content).strip().upper()

        if "GENERAL" not in answer:
            logger.debug("Lesson for %s/%s classified as SPECIFIC — not promoting", repo_name, ticket_id)
            return
    except Exception:
        return  # Can't classify → don't promote

    # Append to global lessons file
    import fcntl
    global_path = DATA_DIR / GLOBAL_LESSONS_DIR / LESSONS_FILENAME
    global_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"## [{ticket_id}] {date_str} {status_label} (from {repo_name})"
    entry_text = f"{header}\n{lesson_body.strip()}\n"

    try:
        with open(global_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                existing_text = f.read()
                existing = _parse_lessons(existing_text)

                # Dedup: skip if a lesson from the same ticket already exists
                existing_tickets = {e.get("ticket_id", "") for e in existing}
                if ticket_id in existing_tickets:
                    return

                new_entry = {
                    "ticket_id": ticket_id,
                    "date": date_str,
                    "status": status_label,
                    "header": header,
                    "body": lesson_body.strip(),
                }
                existing.append(new_entry)
                if len(existing) > MAX_LESSONS_STORED:
                    existing = existing[-MAX_LESSONS_STORED:]
                f.seek(0)
                f.truncate()
                f.write(_format_lessons(existing))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        logger.info("Promoted lesson %s to global tier (from %s)", ticket_id, repo_name)
    except Exception as exc:
        logger.debug("Global lesson write failed: %s", exc)


# ---------------------------------------------------------------------------
# Loading — read + filter for next run
# ---------------------------------------------------------------------------

def _load_global_lessons(
    repo_name: str,
    max_entries: int = MAX_GLOBAL_LESSONS_INJECTED,
    exclude_tickets: set | None = None,
) -> list[dict]:
    """Load cross-repo lessons from the global tier.

    Excludes lessons that originated from the current repo (those are
    already in per-repo lessons) and any specific ticket IDs in
    exclude_tickets (to avoid duplication).
    """
    global_path = DATA_DIR / GLOBAL_LESSONS_DIR / LESSONS_FILENAME
    if not global_path.exists():
        return []
    try:
        text = global_path.read_text(encoding="utf-8")
        entries = _parse_lessons(text)
        # Filter: skip lessons from the same repo (already in per-repo)
        # The header contains "(from {repo_name})" — check for it.
        filtered = []
        for e in entries:
            if f"from {repo_name})" in e.get("header", ""):
                continue  # Same repo — skip, already in per-repo
            if exclude_tickets and e.get("ticket_id") in exclude_tickets:
                continue  # Already included from per-repo
            filtered.append(e)
        return filtered[-max_entries:]  # Most recent N
    except Exception as exc:
        logger.debug("load_global_lessons: read failed (%s)", exc)
        return []


def load_lessons(repo_name: str, max_entries: int = MAX_LESSONS_INJECTED) -> str:
    """Load relevant past lessons for injection into the next run's prompt.

    Two tiers:
      1. Per-repo lessons (5 most recent from this repo's agent_lessons.md)
      2. Global lessons (3 most recent from _global/agent_lessons.md, from OTHER repos)

    Global lessons are generalizable patterns (regex, testing, API design)
    that Haiku classified as cross-repo transferable. They give the agent
    wisdom from Django when working on Sympy, for example.

    Returns a markdown section or empty string if no lessons exist.
    """
    if not LEARN_FROM_FIX_ENABLED or not repo_name:
        return ""

    parts = []
    per_repo_tickets = set()

    # --- Tier 1: Per-repo lessons ---
    lessons_path = _lessons_path(repo_name)
    if lessons_path.exists():
        try:
            text = lessons_path.read_text(encoding="utf-8")
            entries = _parse_lessons(text)
            if entries:
                selected = entries[-max_entries:]
                per_repo_tickets = {e.get("ticket_id", "") for e in selected}
                parts.append("## LESSONS FROM PAST RUNS IN THIS REPO")
                parts.append("")
                parts.append(
                    f"{len(selected)} most recent lesson(s) from this repo. "
                    "Each is a transferable pattern from a prior fix attempt."
                )
                parts.append("")
                for e in selected:
                    parts.append(e["header"])
                    if e.get("body"):
                        parts.append(e["body"])
                    parts.append("")
        except Exception as exc:
            logger.debug("load_lessons: per-repo read failed (%s)", exc)

    # --- Tier 2: Global cross-repo lessons ---
    global_entries = _load_global_lessons(
        repo_name, MAX_GLOBAL_LESSONS_INJECTED, per_repo_tickets,
    )
    if global_entries:
        parts.append("## CROSS-REPO PATTERNS (from other codebases)")
        parts.append("")
        parts.append(
            f"{len(global_entries)} universal pattern(s) learned from other repos. "
            "These apply broadly — not specific to this codebase."
        )
        parts.append("")
        for e in global_entries:
            parts.append(e["header"])
            if e.get("body"):
                parts.append(e["body"])
            parts.append("")

    if not parts:
        return ""

    full = "\n".join(parts).rstrip() + "\n"
    # Cap total: 3K per-repo + 2K global = 5K max
    if len(full) > 5000:
        full = full[:5000] + "\n[... older lessons truncated]\n"
    return full
