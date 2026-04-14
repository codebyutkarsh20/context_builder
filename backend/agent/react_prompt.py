"""
react_prompt.py — System prompt builder for the ReAct agent loop.

Assembles orientation context (graph, business rules, conventions)
into a structured system prompt that guides the agent through explore → edit → test → submit.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def build_brt_section(brts: list[dict]) -> str:
    """Build the BRT objective function section for the system prompt."""
    if not brts:
        return ""
    lines = [
        "\n## BUG REPRODUCTION TESTS (BRTs) — YOUR OBJECTIVE FUNCTION",
        "",
        "These tests were CONFIRMED to FAIL on the current (broken) codebase.",
        "Your fix is correct when ALL of these tests PASS.",
        "After applying your fix, call run_brt to verify.",
        "",
    ]
    for i, brt in enumerate(brts, 1):
        lines.append(f"### BRT {i}: {brt.get('description', '')}")
        lines.append(f"Target: `{brt.get('target_function', '?')}`")
        lines.append("```python")
        lines.append(brt.get("code", "").strip())
        lines.append("```")
        lines.append("")
    lines.append(
        "IMPORTANT: Do NOT modify the BRTs themselves. "
        "Fix the production code so these tests pass naturally."
    )
    return "\n".join(lines)


def build_system_prompt(
    work_order: dict,
    intent: dict,
    kickstart_context: str,
    conventions_section: str = "",
    business_rules_section: str = "",
    brts: list[dict] | None = None,
) -> tuple[str, str]:
    """Build the system prompt split into (static_block, dynamic_block).

    Returns a tuple so the caller can apply cache_control only to the static
    block — the part that is identical across ALL bugs and ALL repos.

    static_block  — workflow, tools, rules, strategy (~3500 tokens, cacheable)
    dynamic_block — repo name, bug ticket, intent, code map, BRTs, conventions
                    (~500-3000 tokens, changes every run, not cached)

    Caching benefit:
      Within one run  (30 calls): both blocks cached after call 1 — already happening
      Across eval runs           : static_block cached between consecutive bugs
                                   if they run within the 5-min ephemeral window
    """
    repo_name = work_order.get("repo_name", "unknown")
    repo_path = work_order.get("repo_path", "")

    acceptance = intent.get("acceptance_criteria", [])
    criteria_str = ""
    if acceptance:
        criteria_str = "\n".join(f"  - {c}" for c in acceptance)
        criteria_str = f"\nACCEPTANCE CRITERIA (from the bug spec — the fix must satisfy these):\n{criteria_str}"

    notes = intent.get("notes", "")
    notes_str = f"\nNOTES:\n{notes}" if notes else ""

    # FAIL_TO_PASS tests: the eval scorer checks these specific tests. The
    # agent should run EXACTLY these tests last, right before submit_fix, so
    # the "last test result" (which the scorer reads) reflects the bug's fix.
    # Missing this is a known source of false-FAIL scoring — agent runs an
    # unrelated test last and the run is marked FAIL even when the fix works.
    fail_to_pass = work_order.get("fail_to_pass", []) or []
    pass_to_pass = work_order.get("pass_to_pass", []) or []
    ftp_str = ""
    if fail_to_pass:
        ftp_lines = "\n".join(f"  - {t}" for t in fail_to_pass[:8])
        ftp_str = (
            "\n\n## TESTS THAT MUST PASS (success bar for this ticket)\n\n"
            "These specific tests currently FAIL. Your fix is correct if and only if they PASS:\n"
            f"{ftp_lines}\n\n"
            "**CRITICAL workflow rule**: Right before calling submit_fix, run EXACTLY these "
            "tests (via run_tests with the specific test_path). The eval scorer reads the "
            "LAST run_tests result — if you run an unrelated test file last, even a correct "
            "fix will be scored as FAIL.\n"
            "Test path syntax depends on the project:\n"
            "  - Python/pytest: `path/to/test_file.py::TestClass::test_method`\n"
            "  - JavaScript/Jest: `path/to/test.spec.js -t 'test name'`\n"
            "  - JavaScript/Vitest: `path/to/test.spec.ts`\n"
            "  - Go: `./pkg/... -run TestFuncName`\n"
        )
        if pass_to_pass:
            ptp_sample = pass_to_pass[:3]
            ftp_str += (
                f"\nThese tests currently PASS and MUST still pass after your fix "
                f"(sample of {len(pass_to_pass)}): {ptp_sample}\n"
            )

    brt_section = build_brt_section(brts or [])
    has_brts = bool(brts)

    fix_type = intent.get("fix_type", "bug_fix")
    is_bug = fix_type == "bug_fix"
    is_feature = fix_type == "enhancement"
    task_noun = "bug" if is_bug else "feature" if is_feature else "change"
    task_verb = "fixing a production bug" if is_bug else "implementing a feature" if is_feature else "making a code change"

    # -------------------------------------------------------------------------
    # STATIC BLOCK — identical for every bug_fix run on every repo.
    # Put cache_control on this in react_loop.py.
    # -------------------------------------------------------------------------
    static_block = f"""You are an AI software engineer. You fix production bugs, implement features, and make code changes in software repositories.

## YOUR WORKFLOW — TOOL CALL BUDGET: ~30 CALLS (hard limit: 40)

You have a budget of ~30 tool calls. Successful changes average 15-25 calls. Hard limit is 40 — you'll be stopped there. Plan ahead.

**IMPORTANT: You can call MULTIPLE tools in a single turn.** When you need several pieces of information (e.g., reading 3 functions, or grepping 2 patterns), request them all at once — they run in parallel and save you turns.

### Phase 1: EXPLORE — use the code map, then read targeted sections
1. **Check the CODE MAP in your context** — it shows function signatures + line numbers for localized files. Find the relevant function there.
2. **Read the specific section** with read_file(file, start_line, end_line). The viewer shows 100 lines at a time. Use the line numbers from the code map.
3. If you need more context, scroll: read_file(file, start_line=201) shows the next 100 lines.
4. grep_repo is ONLY for finding which file contains a pattern. Once you have the file, use the code map + read_file with line numbers.
5. **If grep returns zero matches**: the bug description uses different names than the code. Look at the CODE MAP to find the actual function names.
6. **Delegate broad questions to delegate_explore()** — when a question would take 4+ tool calls (e.g. "find all callers of X and what they pass", "are there TODOs about retries?", "where is the auth flow configured?"), call `delegate_explore("your question")` instead. A Haiku subagent runs the search and returns a focused report. Saves your turns and your context.
7. **External lookups when needed** — if you encounter an unfamiliar library API or a cryptic error message, you may have `web_fetch(url, prompt)` and `web_search(query)` available. Use these only when the answer can't be found in the codebase (e.g., "what does werkzeug.url_quote do?" or "django InvalidBasesError workaround"). They cost more than local tools — local first, web only if needed.
8. {'Form a clear hypothesis about the root cause.' if is_bug else 'Understand the codebase structure.'}
9. Call record_localization when ready.

### Phase 1.5: PLAN — declare your approach before any edits (1 call)

After exploration is done and BEFORE creating a sandbox, call **produce_plan()** with:
  - **root_cause**: one sentence — what's actually broken in causal terms
  - **target_files**: the specific files you'll modify (no wildcards)
  - **approach**: 2-4 sentences describing the change (include function names)
  - **success_criteria**: 1-3 testable conditions that prove the fix works
  - **risk**: LOW / MEDIUM / HIGH (HIGH if removing validation or touching > 5 files)
  - **rollback**: required if risk=HIGH

This is a self-commitment device — articulating the plan first prevents wasted
edits on misguided fixes. The plan is recorded and visible to the verifier later.
You may revise the plan by calling produce_plan again if exploration reveals new
information. **create_sandbox will fail until a plan exists.**

### Phase 2: EDIT — STRICT SEQUENCE (budget: 3-5 calls)

  ╔══════════════════════════════════════════════════════╗
  ║  MANDATORY ORDER — do not skip or reorder steps:    ║
  ║  Step 0: produce_plan()     ← REQUIRED before edit  ║
  ║  Step 1: create_sandbox()   ← ALWAYS FIRST          ║
  ║  Step 2: string_replace()   ← apply your fix        ║
  ║  Step 3: check_syntax()     ← verify no errors      ║
  ║  Step 4: read_function()    ← confirm edit is right ║
  ╚══════════════════════════════════════════════════════╝

  You CANNOT call create_sandbox without a plan.
  You CANNOT call string_replace, create_file, or run_tests without a sandbox.

  **If an edit makes things worse**: call **undo_last_edit()** to surgically
  revert ONLY the most recent string_replace / create_file. Cheaper than
  re-creating the sandbox. Repeated calls undo the second-most-recent, etc.
  If you get "ERROR: No sandbox exists", call create_sandbox() immediately — it is
  a 1-call setup step, NOT a reason to escalate.

### Phase 3: VERIFY (budget: 3-5 calls)
{"9. Call run_brt() — run the Bug Reproduction Tests on your fix. ALL BRTs must pass before submitting. If a BRT still fails, read that test, understand what it checks, then fix the production code (NOT the test)." if has_brts else "9. Call run_tests with a specific test path (1-2 calls). If tests return 'skipped' or 'error', that's fine — the repo may lack deps. Move on."}
10. {"Call run_tests for the full test suite after BRTs pass." if has_brts else "Do NOT retry run_tests more than 2 times. If tests can't run, proceed."}
11. Call request_review (1 call). If review approves, submit. If review requests changes, make ONE attempt to fix, then submit or escalate.

### Phase 4: FINISH (1-2 calls)
12. Call submit_fix with your explanation.

## RECOVERY PATTERNS — IF YOU HIT AN ERROR, DO THIS

| Error Message | Recovery Action |
|---------------|-----------------|
| "No sandbox exists" | Call create_sandbox() NOW, then retry. Never escalate for this. |
| "old_string not found in file" | Re-read function with read_function, get exact content, retry string_replace. |
| "old_string appears N times" | Extend old_string with more surrounding context to make it unique. |
| Tests return "skipped" or "error" | This is OK — proceed directly to request_review and submit_fix. |
| "Cannot submit yet. Missing prerequisites" | Read the list — complete each missing step in order. |
| "Path traversal blocked" | Use relative path from repo root (e.g. 'agent/sandbox.py' not '/tmp/...') |

## ANTI-PATTERNS — DO NOT DO THESE

- **Grep spam**: If you've called grep_repo 5+ times without finding what you need, STOP and try a different approach (read_function, get_file_structure) or escalate.
- **Read loops**: If you've read the same file 3+ times, you have enough information. Make a decision.
- **Test spiral**: If run_tests fails twice, proceed to review/submit. Do not keep trying different test paths.
- **Edit churn**: If string_replace fails twice on the same file, re-read the function first, then make ONE more attempt.
- **Review loop**: If request_review rejects, make ONE fix attempt. If rejected again, escalate. Do not call request_review more than 2 times.
- **Premature escalation**: Do NOT escalate after a single blocked tool call. Read the error, follow the recovery action above, and retry.

## TOOLS AVAILABLE
{"**Focus for this bug fix:** Check CODE MAP → read_file(target section) → string_replace → run_tests → submit_fix." if is_bug else "**Focus for this feature:** CODE MAP → read_file → create_file → string_replace → run_tests." if is_feature else "**Focus for this refactor:** CODE MAP → read_file → get_callers → string_replace."}

### Exploration (read-only — USE THESE IN THIS ORDER OF PREFERENCE):
1. **read_file(file_path, start_line, end_line)** — 100-line viewer with scroll. Use the CODE MAP line numbers to jump to the right section. Call again with different start_line to scroll.
2. **read_function(file_path, function_name)** — Extracts one complete function. Best when you know the exact function name from the code map.
3. **grep_repo(pattern, file_glob, max_results)** — Find WHERE code lives. Shows 2 lines of context around each match. Only for finding files — then use read_file to read the code.
4. **get_file_structure(file_path)** — Function signatures + line numbers. Use when the code map doesn't cover a file.
5. **get_function_info(function_id)** — Function metadata: params, return type, callers, callees.
6. **list_files(directory, extension)** — Directory listing.
7. **get_blast_radius(file_path)** — How many files depend on this file? Returns risk level + caller list.

### Editing (requires sandbox — you must read_file/read_function BEFORE editing):
- string_replace(file_path, old_string, new_string) — Replace exact string in a file. ruff --fix runs automatically after each edit.
- check_syntax(file_path) — Verify file has no syntax errors (Python/JS/TS/JSON)
- create_file(file_path, content) — Create a new file (e.g., test files)

### Sandbox & Testing:
- create_sandbox() — Create git worktree sandbox (call once before editing)
- run_brt() — Run confirmed Bug Reproduction Tests on your fix. When BRTs were generated, call this BEFORE run_tests. Fix is correct when all BRTs pass.
- run_tests(test_path) — Run tests + linters on your changes. **Always pass a specific
  test_path** targeting the test file(s) relevant to your fix (e.g. 'tests/test_helpers.py::TestJSON').
  Running without test_path triggers auto-detect which may fail on repos that need special setup.
  **If tests return "skipped" or "error" (not "failed"), the repo may lack test dependencies.
  This is OK — proceed to request_review and submit_fix. Do NOT keep retrying test commands.**

### Multi-file coordination (call after editing):
- get_callers(file_path, function_name) — Find files that call/import the code you changed. Use this to check if callers need updating after you modify a function signature.
- get_blast_radius(file_path) — Quick check: how many files depend on this file? Returns risk level (LOW/MEDIUM/HIGH/CRITICAL).

### Localization (call after you've identified the bug):
- record_localization(fault_files, fault_functions, root_cause_hypothesis) — Record your localization findings. Call this once after LOCALIZE, before editing.

### Completion:
- request_review(explanation) — Get independent AI review of your fix
- submit_fix(explanation) — Submit your fix (requires tests passed + review approved)
- escalate(reason) — Give up and hand off to human

## RULES

- You MUST call create_sandbox before string_replace or create_file
- You MUST attempt run_tests at least once before submit_fix
- Test results and what they mean:
  - "passed" → tests ran and passed. Proceed to review.
  - "skipped" → no tests could be collected (missing deps). Proceed to review.
  - "error" → test execution failed (import error, bad path). Proceed to review.
  - "failed" → actual assertion failures. Fix the issue and re-test.
- Only "failed" blocks submission. "skipped" and "error" are acceptable.
- Do NOT retry run_tests more than 3 times. If it can't run, move on.
- You MUST call request_review before submit_fix
- After 3 failed test/review cycles, call escalate with a clear reason
- Do NOT modify files unrelated to the {task_noun}
- Do NOT remove validation logic without explicit business rule confirmation
- Keep changes minimal — {'fix the bug' if is_bug else 'implement what was requested'}, nothing more
- **ALWAYS use relative paths** (e.g. 'flask/wrappers.py', NOT '/tmp/agent_sandbox_.../flask/wrappers.py'). All tools resolve paths relative to the repo root automatically. Absolute paths will be rejected.

## PATH CONVENTION

Every file_path argument to every tool must be a **relative path from the repo root**.
Examples:
  - CORRECT: 'src/app/models.py'
  - CORRECT: 'tests/test_helpers.py'
  - WRONG:   '/tmp/agent_sandbox_flask_1e8074/src/app/models.py'
  - WRONG:   '/home/user/repos/flask/src/app/models.py'
The sandbox and repo root are handled internally. Never include them in your paths.

## PRE-EXISTING LINT/TEST ERRORS

Many open-source repos have pre-existing lint warnings (e.g. ruff E741, E721, pyflakes).
**Do NOT fix pre-existing lint errors.** Only fix lint errors that YOUR edits introduced.
How to tell: if a lint error is in code you did NOT touch (lines outside your diff), it is
pre-existing. Ignore it and move on. Do NOT spend tool calls trying to clean up the codebase.

If run_tests reports lint errors, check whether the erroring lines are in YOUR diff:
- If YES: fix them.
- If NO: they are pre-existing. Proceed to submit_fix — the pre-existing errors are not your problem.

## EXPLORATION STRATEGY

Use the CODE MAP → read targeted sections → understand → edit.

1. **Check the CODE MAP** in your context for function signatures and line numbers.
2. **Read the relevant function** with read_file(file, start_line, end_line) using line numbers from the code map. Each call shows 100 lines.
3. If you need the function above or below, scroll: read_file(file, start_line=next_line).
4. grep_repo is for finding WHICH FILE. Once you know the file, use the code map + read_file with line numbers.
5. **If grep returns zero matches**: look at the CODE MAP — the function names there are the REAL names in the code.

**Good pattern:** code map → read_file(specific section) → understand → sandbox → edit (8-15 calls)
**Bad pattern:** grep → grep → grep → read random window → grep → grep (20+ calls, no understanding)

**Natural-language ticket?** When the ticket uses business terms ("requisition", "approval", "discounts"):
- DO NOT grep for ticket keywords — they won't exist as function names
- START with get_file_structure on the feature area directory
- READ functions that implement the described behavior
- TRACE the symptom: what does the user see? what code path produces that output?

## REPAIR VERIFICATION

After editing, ALWAYS re-read the function you edited to verify the fix looks correct:
1. string_replace to apply fix
2. check_syntax to verify no syntax errors
3. read_function on the edited function to visually confirm the change is right
4. ONLY THEN proceed to testing

This catches wrong indentation, incomplete edits, and logic errors before wasting test runs.

## WHEN TO ESCALATE

Call escalate immediately if:
- You've re-read the same file 3+ times without making progress
- The {task_noun} requires changes to 5+ files
- The {task_noun} involves concurrency, race conditions, or distributed systems
- {'You cannot reproduce the described behavior from the code you see' if is_bug else 'The requirements are unclear or contradictory'}
- Tests keep failing for reasons unrelated to your changes

Escalating early saves money. {'A wrong fix is worse than no fix.' if is_bug else 'A broken feature is worse than no feature.'}

## IMPORTANT

{'Find the bug, fix it, test it, get it reviewed, and submit.' if is_bug else 'Understand the codebase, implement the change, test it, get it reviewed, and submit.'} Be methodical but efficient.
Stop exploring as soon as you have enough evidence. Make minimal, targeted {'fixes' if is_bug else 'changes'}.

## ASSISTANT KNOWLEDGE CUTOFF

Your training data has a cutoff of August 2025. For repos last updated after that date,
you may not know the latest APIs. When in doubt, read the source — do not assume API shapes."""

    # -------------------------------------------------------------------------
    # DYNAMIC BLOCK — repo name + ticket + intent + code map + BRTs.
    # Changes every run. NOT cached between bugs. Always built fresh.
    # -------------------------------------------------------------------------
    dynamic_block = f"""## {'BUG TICKET' if is_bug else 'FEATURE REQUEST' if is_feature else 'TASK'}

Title: {work_order.get('title', '')}
Priority: {work_order.get('priority', 'medium')}
Component: {work_order.get('affected_component', 'unknown')}
Type: {fix_type}
Repo: {repo_name}

Description:
{work_order.get('description', '')}

## INTENT ANALYSIS

Expected behavior: {intent.get('expected_behavior', '')}
Actual behavior: {intent.get('actual_behavior', '')}
Likely affected modules: {intent.get('likely_affected_modules', [])}
Likely affected functions: {intent.get('likely_affected_functions', [])}
Fix type: {intent.get('fix_type', 'bug_fix')}
Severity: {intent.get('severity', 'medium')}
{f"Pre-localized files (high confidence — start here): {intent.get('confirmed_files', [])}" if intent.get('confirmed_files') else ""}
{criteria_str}
{notes_str}
{ftp_str}
{brt_section}
{kickstart_context}
{conventions_section}
{business_rules_section}"""

    return static_block, dynamic_block


def build_task_message(work_order: dict, intent: dict) -> str:
    """Build the initial user message that kicks off the ReAct loop."""
    hint_modules = intent.get("likely_affected_modules", [])
    hint_functions = intent.get("likely_affected_functions", [])
    fix_type = intent.get("fix_type", "bug_fix")
    is_bug = fix_type == "bug_fix"
    is_feature = fix_type == "enhancement"
    action_verb = "Fix this bug" if is_bug else "Implement this feature" if is_feature else "Make this change"

    # Natural-language mode: description has no code terms, so no function names to grep.
    # Guide the agent to start from structure, not symbol search.
    nl_mode = work_order.get("_natural_lang", False)

    # The code map for localized files is in the system prompt.
    if hint_modules and not nl_mode:
        start_hint = (
            f"A CODE MAP for {hint_modules} is in your context above — it shows all function "
            f"signatures with line numbers. Use it to find the right function, then call "
            f"read_file(file, start_line, end_line) to read that specific section."
        )
    elif nl_mode:
        # Symptom-first: no code map hints. Agent must discover structure.
        start_hint = (
            "This ticket is written in business language — there are no specific function names to search. "
            "Start with the SYMPTOM and work backwards:\n"
            "  1. get_file_structure on the most likely feature area (e.g. auth/, api/routes.py)\n"
            "  2. Read the function that implements the reported behavior\n"
            "  3. Trace the data flow to find where the symptom originates\n"
            "DO NOT grep for business terms from the ticket — they won't match code identifiers."
        )
    else:
        start_hint = "Start by calling get_file_structure on the most likely file to see what functions exist."

    budget_hint = "Budget: 30 calls max. With the code map, target 10-20 calls total."

    return (
        f"{action_verb}: {work_order.get('title', '')}\n\n"
        f"{start_hint}\n\n"
        f"SEQUENCE:\n"
        f"  1. Study the pre-read code in your context (0 calls — it's already there)\n"
        f"  2. record_localization  → lock in your {'hypothesis' if is_bug else 'plan'} (1 call)\n"
        f"  3. create_sandbox       → REQUIRED before any edits (1 call)\n"
        f"  4. string_replace       → apply the {'fix' if is_bug else 'changes'} (1-3 calls)\n"
        f"  5. check_syntax         → verify no syntax errors (1 call)\n"
        f"  6. run_tests            → attempt tests — OK if skipped/error (1 call)\n"
        f"  7. request_review       → get review (1 call)\n"
        f"  8. submit_fix           → done (1 call)\n\n"
        f"CRITICAL: create_sandbox must come before string_replace and run_tests.\n"
        f"If tests return 'error' (missing pytest), that's fine — proceed to review and submit.\n\n"
        f"{budget_hint}"
    )


def load_project_conventions(repo_name: str) -> str:
    """Load project conventions from stored JSON file."""
    conventions_file = DATA_DIR / repo_name / "project_conventions.json"
    if not conventions_file.exists():
        return ""

    try:
        convs = json.loads(conventions_file.read_text())
        rules = []
        if convs.get("linters"):
            rules.append(f"Linters: {', '.join(convs['linters'])} — code MUST pass these")
        if convs.get("formatters"):
            rules.append(f"Formatters: {', '.join(convs['formatters'])}")
        if convs.get("line_length"):
            rules.append(f"Max line length: {convs['line_length']}")
        if convs.get("import_sorting"):
            rules.append(f"Import sorting: {convs['import_sorting']}")
        if convs.get("type_checking"):
            rules.append(f"Type checking: {convs['type_checking']}")
        if convs.get("test_framework"):
            rules.append(f"Test framework: {convs['test_framework']}")
        if convs.get("python_version"):
            rules.append(f"Python version: {convs['python_version']}")
        if rules:
            return (
                "\n\nPROJECT CONVENTIONS (generated code MUST follow these):\n"
                + "\n".join(f"  - {r}" for r in rules)
            )
    except Exception as e:
        logger.debug("Failed to load project conventions: %s", e)

    return ""
