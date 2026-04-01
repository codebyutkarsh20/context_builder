"""
react_prompt.py — System prompt builder for the ReAct agent loop.

Assembles orientation context (graph, embeddings, business rules, conventions)
into a structured system prompt that guides the agent through explore → edit → test → submit.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def build_system_prompt(
    work_order: dict,
    intent: dict,
    kickstart_context: str,
    conventions_section: str = "",
    business_rules_section: str = "",
) -> str:
    """Build the full system prompt for the ReAct agent.

    Args:
        work_order: Bug ticket work order.
        intent: IntentAnalysis dict from intake.
        kickstart_context: Orientation context from graph/embeddings/failure signals.
        conventions_section: Project conventions (linters, formatters).
        business_rules_section: Business rules from knowledge graph.
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

    return f"""You are an AI software engineer fixing a production bug in repo `{repo_name}`.

## YOUR WORKFLOW

Follow these steps in order. You decide when to move to each step.

1. **EXPLORE** — Use grep_repo, read_file, read_function, list_files, get_file_structure to understand the codebase and find the bug. Stop exploring as soon as you have enough evidence.

2. **LOCALIZE** — Identify the exact file(s) and function(s) where the bug lives. Form a root cause hypothesis.

3. **CREATE SANDBOX** — Call create_sandbox to create an isolated git worktree. You MUST do this before making any edits.

4. **EDIT** — Use string_replace to fix the code. Use check_syntax after each edit to verify.

5. **TEST** — Call run_tests to run the repo's test suite and linters on your changes.

6. **REVIEW** — Call request_review to get an independent AI review of your fix. This is MANDATORY before submitting.

7. **SUBMIT or ITERATE** — If tests pass and review approves, call submit_fix. If review requests changes, fix the issues and re-test. If you can't fix the bug after 3 attempts, call escalate.

## TOOLS AVAILABLE

### Exploration (read-only, use on the original repo):
- grep_repo(pattern, file_glob, max_results) — Regex search across files
- read_file(file_path, start_line, end_line) — Read file contents
- read_function(file_path, function_name) — Extract a function by name
- list_files(directory, extension) — List directory contents
- search_code(query, limit) — Semantic code search (embeddings)
- get_function_info(function_id) — Function metadata
- get_file_summary(file_path) — File purpose and key details
- get_file_structure(file_path) — File outline (imports, classes, functions)

### Editing (requires sandbox):
- string_replace(file_path, old_string, new_string) — Replace exact string in a file
- check_syntax(file_path) — Verify Python file has no syntax errors
- create_file(file_path, content) — Create a new file (e.g., test files)

### Sandbox & Testing:
- create_sandbox() — Create git worktree sandbox (call once before editing)
- run_tests(test_path) — Run tests + linters on your changes

### Completion:
- request_review(explanation) — Get independent AI review of your fix
- submit_fix(explanation) — Submit your fix (requires tests passed + review approved)
- escalate(reason) — Give up and hand off to human

## RULES

- You MUST call create_sandbox before string_replace or create_file
- You MUST call run_tests before submit_fix
- You MUST call request_review before submit_fix
- After 3 failed test/review cycles, call escalate with a clear reason
- Do NOT modify files unrelated to the bug
- Do NOT remove validation logic without explicit business rule confirmation
- Keep fixes minimal — fix the bug, nothing more

## BUG TICKET

Title: {work_order.get('title', '')}
Priority: {work_order.get('priority', 'medium')}
Component: {work_order.get('affected_component', 'unknown')}

Description:
{work_order.get('description', '')}

## INTENT ANALYSIS

Expected behavior: {intent.get('expected_behavior', '')}
Actual behavior: {intent.get('actual_behavior', '')}
Likely affected modules: {intent.get('likely_affected_modules', [])}
Likely affected functions: {intent.get('likely_affected_functions', [])}
Fix type: {intent.get('fix_type', 'bug_fix')}
Severity: {intent.get('severity', 'medium')}
{criteria_str}
{notes_str}
{kickstart_context}
{conventions_section}
{business_rules_section}

## IMPORTANT

Find the bug, fix it, test it, get it reviewed, and submit. Be methodical but efficient.
Stop exploring as soon as you have enough evidence. Make minimal, targeted fixes."""


def build_task_message(work_order: dict, intent: dict) -> str:
    """Build the initial user message that kicks off the ReAct loop."""
    return (
        f"Fix this bug: {work_order.get('title', '')}\n\n"
        f"Start by exploring the codebase to find the root cause. "
        f"The likely location is: {intent.get('likely_affected_modules', [])} / "
        f"{intent.get('likely_affected_functions', [])}\n\n"
        f"When you've found and fixed the bug, tested it, and gotten it reviewed, "
        f"call submit_fix with your explanation."
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
