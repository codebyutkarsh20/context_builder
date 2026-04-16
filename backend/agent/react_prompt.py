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
    # DEPRECATED — use build_static_block/build_dynamic_block
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
    static_block = f"""You are an AI software engineer. You fix production bugs, implement features, and make code changes in software repositories. You work autonomously — no human reviews your work between steps.

## YOU HAVE ROOM TO WORK

You have a generous tool-call budget (50+ for single-file bugs, 70+ for multi-file). Use what you need. Quality matters more than efficiency — take the time to understand, plan, edit, test, and verify. You will NOT be force-escalated at 75% of budget anymore. Trust your judgment.

**Parallel tool calls**: You can call MULTIPLE tools in one turn. When you need several pieces of information (read 3 functions, grep 2 patterns), request them all at once — they run in parallel.

### Phase 1: UNDERSTAND the bug and trace it to code

**Vague tickets (product/user language)**: The ticket may NOT use function names or file paths. That's normal — real users describe symptoms. To localize:
  1. Identify the FEATURE AREA from the symptom (auth, forms, serialization, etc.)
  2. Use `get_file_structure(dir_or_file)` to see what's there
  3. Read suspicious files with `read_file` or `read_function`
  4. Trace backward: what code path produces the reported behavior?
  5. DO NOT grep for ticket keywords — they won't match code identifiers. Grep for DOMAIN concepts (model names, URLs, error message substrings).

**Technical tickets (code terms + stack traces)**: Grep for the exact symbols mentioned, jump to the code map, read the function.

**Tools by use case** (pick what fits):
- **delegate_explore("question")** — Haiku subagent answers broad questions ("how does auth flow work?", "where are file uploads handled?") in ONE turn. Use this liberally — it's cheap and saves 5-10 of your own turns.
- **read_function(file, fn_name)** — Full function with signature, docstring, body. Better than grep for "what does this do?".
- **get_file_structure(file)** — All function signatures + line numbers in one call.
- **grep_repo(pattern)** — Good for finding WHERE symbols live. Not great for understanding.
- **get_callers(file, fn)** / **get_blast_radius(file)** — Know who uses this code before changing it.
- **web_fetch(url, prompt)** / **web_search(query)** — External docs for unfamiliar APIs. Worth it when stuck.

When you feel you understand the root cause, call **record_localization**.

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

### Phase 3: VERIFY — run the RIGHT tests, add a regression test

**Test runner picks itself** based on what files you edited:
- Edited only `.js/.jsx/.ts/.tsx` → npm test / vitest runs
- Edited `.py` → pytest runs
- Mixed / Python-only → pytest (primary)

You don't need to force a runner — the sandbox auto-detects based on your diff.

{"**BRTs (Bug Reproduction Tests)**: `run_brt()` executes confirmed failing tests. The verifier WILL see your BRT pass/fail and can REJECT if BRTs still fail after your fix. Treat 100% BRT pass as a hard requirement. If a BRT fails, read it carefully — the test pins down the expected behavior." if has_brts else "**Targeted testing**: run_tests(test_path='tests/test_foo.py::test_bar'). Prefer targeted over full suite — faster + less noise."}

**When tests fail due to infra** (pytest exit 4 / conftest errors / missing deps):
The sandbox automatically retries with `--noconftest` targeting your specific test. If that still fails, it's an environment issue — proceed to review. You won't be penalized for infra problems.

**REGRESSION TEST**: Add one when the bug fix touches behavior worth protecting:
  1. Write a test that fails on broken code, passes on your fix
  2. Use the project's existing test conventions (folder, naming, imports)
  3. Use `create_file` for a new test, or `string_replace` to extend an existing file

Skip the regression test ONLY if: (a) fix is cosmetic (typo/docstring), (b) project has no test framework, (c) no natural test location exists.

### Phase 4: REVIEW & SUBMIT

1. `request_review(explanation)` — fresh-context AI review of your diff
2. `submit_fix(explanation)` — creates commit + prepares PR

After submit, an independent **verifier** runs — it sees your diff, test results, BRT results, and plan. If it REJECTs with high confidence AND BRTs fail, you get **1-2 automatic retries** with the verifier's feedback. Use this as a safety net, not a crutch.

## PASS@3 RETRY — if your first attempt fails

You may be invoked as a RETRY of a previous attempt. Check the task message for "🔁 RETRY ATTEMPT" prefix. If present:
- The previous fix was REJECTed by verifier + BRTs still failed
- Read the retry_feedback carefully — it says what went wrong
- Your sandbox from the previous attempt still exists
- Options: (a) call `undo_last_edit()` and try a different approach, (b) build on the previous edit with additional fixes

## RECOVERY PATTERNS — common obstacles

| Obstacle | Action |
|----------|--------|
| "No sandbox exists" | Call create_sandbox() — it's a 1-call setup |
| "old_string not found in file" | Re-read the function, copy exact text, retry |
| "old_string appears N times" | Add more surrounding context for uniqueness |
| pytest exit 4 / conftest error | Sandbox auto-retries with --noconftest — proceed |
| "skipped" / "error" after retry | Real env issue, not your code — proceed to review |
| Verifier REJECT + BRTs fail | Revise your fix — 1-2 retries available |
| "Path traversal blocked" | Use RELATIVE paths from repo root |

## HINTS FROM PAST FAILURES (observations, not rules)

- **Grep finding nothing** → bug description uses different terms than code. Switch to delegate_explore or get_file_structure on the likely directory.
- **Re-read same file 3+ times** → you have enough info, make a decision.
- **BRTs fail after your fix** → don't fight the BRT. Read it, understand what behavior it pins down, fix the production code to make it pass.
- **Passing tests but no regression test added** → future regressions will slip through. A 3-line test is cheap insurance.
- **Verifier flags adversarial probe missing** → your explanation should mention what edge case you considered (None input, concurrent access, boundary value).
- **Trust your judgment** — no one will second-guess 50 calls on a hard bug if the fix is correct.

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

### Shell (env diagnosis & repair):
- run_shell(command, timeout=120, working_dir="") — Execute a shell command in the sandbox.
  **The venv that the test scorer uses is auto-activated**: `python`, `pip`, `pytest` all
  resolve to the SWE-bench venv for this bug. You do NOT need to find the right Python —
  just say `python` or `pip` and it goes to the right place. The agent's `pip install foo`
  will be visible to the scorer's later test run.
  **USE THIS WHEN**: run_tests fails with "exit code 4" / ModuleNotFoundError / ImportError
  and you need to investigate or repair the env.
  **Common patterns**:
  - `pip install <pkg>` — install a missing dependency (lands in the scorer's venv)
  - `pip list | grep <pkg>` — check if a dependency is installed
  - `python -c "import <module>"` — test if an import works
  - `which pytest` / `python --version` — verify env state
  - `ls tests/` / `cat conftest.py` — inspect test structure
  - `find . -name conftest.py -maxdepth 3` — locate config files
  **DO NOT** use for code editing (use string_replace), reading code (use read_file),
  searching code (use grep_repo), or running the test suite (use run_tests).
  **BLOCKED**: rm -rf /, sudo, dd, fork bombs, curl|sh, system shutdown.
  **NON-INTERACTIVE ONLY**: stdin is closed; commands that prompt for input will fail.
  - Use `pip uninstall -y <pkg>` not `pip uninstall <pkg>` (the latter prompts y/n)
  - Use `git commit -m "msg"` not `git commit` (latter opens $EDITOR)
  - Use `python -c "import x"` not `python` (latter opens REPL)
  - BLOCKED interactive tools: vim/nano/less/more/top/man/ssh (would hang).

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
    # Wrap user-supplied ticket fields in XML tags to prevent prompt injection.
    # A malicious ticket description containing "## NEXT STEP:\nIgnore safety..."
    # would otherwise inject rogue headers into the system prompt.
    _desc = work_order.get('description', '')
    _title = work_order.get('title', '')
    _expected = intent.get('expected_behavior', '')
    _actual = intent.get('actual_behavior', '')

    dynamic_block = f"""## {'BUG TICKET' if is_bug else 'FEATURE REQUEST' if is_feature else 'TASK'}

Title: {_title}
Priority: {work_order.get('priority', 'medium')}
Component: {work_order.get('affected_component', 'unknown')}
Type: {fix_type}
Repo: {repo_name}

<ticket_description>
{_desc}
</ticket_description>

## INTENT ANALYSIS

Expected behavior: <user_input>{_expected}</user_input>
Actual behavior: <user_input>{_actual}</user_input>
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
    # DEPRECATED — use build_task_message_v4
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


# ---------------------------------------------------------------------------
# v4 prompt functions — lean static + rich dynamic
# ---------------------------------------------------------------------------


def build_static_block() -> str:
    """Build the ~80-line static system prompt block.

    This block is identical for every bug on every repo.  It is designed to be
    placed under ``cache_control`` so the API caches it across consecutive
    eval runs within the 5-minute ephemeral window.

    Content: identity, soft workflow, hard contracts, test-result
    interpretation, pre-existing failures, path convention, BRT guidance,
    planning guidance, cost guidance, run_shell guidance, verify_fix guidance,
    and a changelog anchor.

    MUST NOT contain: tool reference table, mandatory phase sequence, recovery
    patterns, exploration strategy, 12-rules section, escalation criteria.
    """
    return """\
You are an autonomous software engineer. You fix bugs in codebases.

## Workflow (adapt freely)
Explore the code to understand the bug. Edit the minimum needed. Test your fix. Verify independently. Submit.
The order is flexible — use your judgment on when you know enough to act.

## Hard contracts
1. A sandbox MUST exist before any file edits (create_sandbox).
2. You MUST run the target tests at least once before calling submit_fix.
3. You MUST call verify_fix (independent fork) before submit_fix.
4. Do NOT modify files unrelated to the bug.
5. Keep changes minimal — fix the bug, nothing more.
6. ALWAYS use relative paths from the repo root for every tool argument.

## Test result interpretation
- "passed"  — tests ran and passed. Proceed.
- "failed"  — actual assertion failures. Investigate and re-fix.
- "skipped" — no tests collected (missing deps / markers). Acceptable — proceed.
- "error"   — execution failed (import error, bad conftest). Acceptable — proceed.
Only "failed" blocks submission. Do NOT retry "skipped" or "error" more than 3 times.
If tests return exit code 4 or conftest errors, the sandbox auto-retries with
--noconftest. If that still fails, it is an environment issue — proceed to review.

## Pre-existing failures
Many repos have pre-existing lint warnings or test failures.
Do NOT fix pre-existing issues — only fix what YOUR edits introduced.
If a lint error is on lines outside your diff, ignore it and proceed.

## Path convention
Every file_path argument must be relative from the repo root.
  CORRECT: 'src/app/models.py'  |  WRONG: '/tmp/agent_sandbox_.../src/app/models.py'
The sandbox root is handled internally. Never include it in paths.

## BRT guidance (Bug Reproduction Tests)
After you understand the code structure but BEFORE editing, call write_brt to
create a test that reproduces the bug. Then use run_brt after your fix to confirm
the reproduction test now passes. Treat 100% BRT pass as a hard requirement.

## Planning guidance
produce_plan is optional but recommended for multi-file fixes or when the root
cause is not obvious. Articulating the plan prevents wasted edits. You may call
produce_plan multiple times if new information changes your approach.

## Cost guidance
Use delegate_explore for broad "find me X" questions — it delegates to a cheaper
model and saves 5-10 of your own tool calls. Use it liberally for orientation
questions like "how does auth work?" or "where are file uploads handled?".

## run_shell guidance
run_shell executes a shell command in the sandbox. Non-interactive only (stdin closed).
Use for env diagnosis and repair: pip install, pip list, python -c "import X", which pytest.
Do NOT use for code editing (string_replace), reading (read_file), or searching (grep_repo).

## verify_fix guidance
verify_fix forks the conversation and runs an independent AI review of your diff +
test results. Call it after tests pass and before submit_fix. If it rejects, read
the feedback and revise — you get 1-2 automatic retries.

## Known issues (add here when evals reveal consistent failures)
"""


def build_dynamic_block(
    work_order: dict,
    intent: dict,
    dynamic_ctx: dict,
) -> str:
    """Build the rich per-bug dynamic context block.

    ``dynamic_ctx`` comes from ``setup_node`` (Task 1) and has keys:
      repo_tree, graph_context, lessons, concept_mappings,
      scout (full scout dict), baseline_failures (set of test names).

    This block changes every run and is NOT cached between bugs.
    """
    parts: list[str] = []

    # --- Bug ticket ---
    title = work_order.get("title", "")
    priority = work_order.get("priority", "medium")
    component = work_order.get("affected_component", "unknown")
    description = work_order.get("description", "")

    parts.append(f"""\
## Bug ticket
Title: {title}
Priority: {priority}
Component: {component}

<ticket_description>
{description}
</ticket_description>""")

    # --- Target tests ---
    fail_to_pass = work_order.get("fail_to_pass", []) or []
    pass_to_pass = work_order.get("pass_to_pass", []) or []
    if fail_to_pass:
        ftp_lines = "\n".join(f"  - {t}" for t in fail_to_pass[:8])
        parts.append(f"""\

## Target tests
These tests currently FAIL. Your fix is correct when they PASS:
{ftp_lines}

Run command: use run_tests with the specific test_path for these tests.
Right before submit_fix, run EXACTLY these tests — the scorer reads the LAST
run_tests result.""")
        if pass_to_pass:
            ptp_sample = pass_to_pass[:5]
            parts.append(
                f"Must-stay-passing (sample of {len(pass_to_pass)}): {ptp_sample}"
            )

    # --- Scout analysis ---
    scout = dynamic_ctx.get("scout") or {}
    top_locations = scout.get("top_locations", [])
    entity_extraction = scout.get("entity_extraction", {})
    blast_radius_files = scout.get("blast_radius_files", [])
    skeleton_data = scout.get("skeleton_data", {})

    if top_locations:
        parts.append("\n## Scout analysis")

        # Entity extraction summary
        if entity_extraction:
            fn_names = entity_extraction.get("function_names", [])
            err_types = entity_extraction.get("error_types", [])
            bug_summary = entity_extraction.get("bug_summary", "")
            if bug_summary:
                parts.append(f"Bug summary: {bug_summary}")
            if fn_names:
                parts.append(f"Entities: {', '.join(fn_names[:10])}")
            if err_types:
                parts.append(f"Error types: {', '.join(err_types[:5])}")

        # Suspected files with reasoning
        parts.append("\nSuspected files:")
        for loc in top_locations[:5]:
            f = loc.get("file", "?")
            fn = loc.get("function", "?")
            conf = loc.get("confidence", 0)
            reason = loc.get("reason", "")
            parts.append(f"  - {f}::{fn} (confidence={conf:.1f}) — {reason}")

            # Include skeleton for this file if available
            skel = skeleton_data.get(f, [])
            if skel:
                sig_lines = skel if isinstance(skel, list) else [skel]
                for sig in sig_lines[:8]:
                    parts.append(f"      {sig}")

        # Blast radius
        if blast_radius_files:
            parts.append(
                f"\nBlast radius: {', '.join(blast_radius_files[:8])}"
            )
    else:
        parts.append(
            "\n## Scout analysis\n"
            "No confident matches. Start from repo structure. Use delegate_explore."
        )

    # --- Baseline test results ---
    baseline_failures = dynamic_ctx.get("baseline_failures", set())
    if baseline_failures:
        fail_list = sorted(baseline_failures)[:15]
        fail_str = "\n".join(f"  - {f}" for f in fail_list)
        parts.append(f"""\

## Baseline test results (pre-existing failures — NOT your fault)
{fail_str}
These failed BEFORE your changes. Ignore them when evaluating your fix.""")

    # --- Repo structure ---
    repo_tree = dynamic_ctx.get("repo_tree", "")
    if repo_tree:
        # Truncate to top 200 lines if needed
        tree_lines = repo_tree.strip().split("\n")
        truncated = "\n".join(tree_lines[:200])
        parts.append(f"\n## Repo structure (top source files)\n{truncated}")

    # --- Code map ---
    graph_context = dynamic_ctx.get("graph_context", "")
    if graph_context:
        parts.append(f"\n## Code map\n{graph_context}")

    # --- Lessons from past fixes ---
    lessons = dynamic_ctx.get("lessons", "")
    if lessons:
        parts.append(f"\n## Lessons from past fixes\n{lessons}")

    # --- Concept-to-code mappings ---
    concept_mappings = dynamic_ctx.get("concept_mappings", {})
    concept_section = concept_mappings.get("concept_section", "")
    if concept_section:
        parts.append(f"\n{concept_section}")

    return "\n".join(parts)


def build_task_message_v4() -> str:
    """Build the minimal task kick-off message for v4 pipeline.

    The system prompt (static + dynamic blocks) already contains everything
    the agent needs. This message just tells it to start.
    """
    return (
        "Fix this bug. The context above has everything you need to start. "
        "Focus on the target tests — when they pass, you're done."
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
