"""
test_enricher.py — Step 18: Business-Intent Test Enhancement.

Scans a target repo for test files, links each test function to the
business rules it verifies, and stores enrichment data so that when a
test fails in the agent sandbox the failure message includes the
business rule being verified.

Enrichments are stored in DATA_DIR/{repo}/test_enrichments.json (never
modifies the target repo's test files directly).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# Pytest file patterns
_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _extract_test_functions(file_path: Path) -> list[dict[str, Any]]:
    """Parse a Python test file and return metadata for each test function."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        logger.debug("Could not parse %s", file_path)
        return []

    tests: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue

        docstring = ast.get_docstring(node) or ""
        # Collect function names called inside the test body
        called_names: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    called_names.append(child.func.attr)
                elif isinstance(child.func, ast.Name):
                    called_names.append(child.func.id)

        tests.append({
            "function_name": node.name,
            "docstring": docstring,
            "called_names": called_names,
            "lineno": node.lineno,
        })

    return tests


def _find_test_files(repo_path: Path) -> list[Path]:
    """Recursively find all pytest-convention test files."""
    results: list[Path] = []
    for pattern in _TEST_FILE_PATTERNS:
        results.extend(repo_path.rglob(pattern))
    # Deduplicate and skip hidden/venv dirs
    seen: set[Path] = set()
    filtered: list[Path] = []
    for p in results:
        if p in seen:
            continue
        seen.add(p)
        parts = p.relative_to(repo_path).parts
        if any(part.startswith(".") or part in ("venv", ".venv", "node_modules", "__pycache__") for part in parts):
            continue
        filtered.append(p)
    return sorted(filtered)


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------

def _load_business_rules(repo: str) -> list[dict]:
    """Load business rules from the knowledge base."""
    rules_path = _DATA_DIR / repo / "business_rules.json"
    if rules_path.exists():
        try:
            return json.loads(rules_path.read_text())
        except Exception:
            pass
    return []


def _load_graph_functions(repo: str) -> dict[str, dict]:
    """Load function nodes from graph.json, keyed by function name (short)."""
    graph_path = _DATA_DIR / repo / "graph.json"
    if not graph_path.exists():
        return {}
    try:
        data = json.loads(graph_path.read_text())
    except Exception:
        return {}

    funcs: dict[str, dict] = {}
    for node in data.get("nodes", []):
        if node.get("type") == "function":
            func_id = node.get("id", "")
            # Extract short name: "app/services/refund.py::process_refund" → "process_refund"
            short_name = func_id.split("::")[-1] if "::" in func_id else func_id
            funcs[short_name] = {
                "id": func_id,
                "file": func_id.split("::")[0] if "::" in func_id else "",
                "name": short_name,
            }
    return funcs


def _match_test_to_rules(
    test_info: dict,
    rules: list[dict],
    graph_functions: dict[str, dict],
) -> list[dict]:
    """Match a test function to business rules via the functions it calls.

    Returns a list of enrichment entries (one per matched rule).
    """
    enrichments: list[dict] = []
    called = set(test_info["called_names"])

    for called_name in called:
        func_meta = graph_functions.get(called_name)
        if not func_meta:
            continue

        func_id = func_meta["id"]
        func_file = func_meta["file"]

        # Find rules linked to this function (by function_id or file)
        for rule in rules:
            rule_func = rule.get("function_id", "")
            rule_file = rule.get("file", "")
            if rule_func and rule_func == func_id:
                enrichments.append(_build_enrichment(test_info, func_id, rule))
            elif rule_file and rule_file == func_file and not rule_func:
                enrichments.append(_build_enrichment(test_info, func_id, rule))

    return enrichments


def _build_enrichment(test_info: dict, func_id: str, rule: dict) -> dict:
    severity = rule.get("severity", "medium")
    description = rule.get("description", "")
    return {
        "test_function": test_info["function_name"],
        "tests_function_id": func_id,
        "business_rule": description,
        "severity": severity,
        "failure_meaning": (
            f"If this test fails, the constraint may be violated: {description}. "
            f"Severity: {severity}."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_tests(repo: str, repo_path: str | Path | None = None) -> list[dict]:
    """Scan a repo's tests and generate business-intent enrichments.

    Args:
        repo: Repository identifier (used for knowledge base lookups).
        repo_path: Path to the repository source. If None, attempts to
                   resolve from DATA_DIR/{repo}/repo_path.

    Returns:
        List of enrichment dicts, also persisted to
        DATA_DIR/{repo}/test_enrichments.json.
    """
    rules = _load_business_rules(repo)
    graph_functions = _load_graph_functions(repo)

    if not rules:
        logger.info("No business rules found for repo '%s' — nothing to enrich", repo)
        return []

    # Resolve repo path
    if repo_path is None:
        meta_path = _DATA_DIR / repo / "repo_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                repo_path = meta.get("repo_path", "")
            except Exception:
                pass
        if not repo_path:
            logger.warning("No repo_path provided and none found in metadata for '%s'", repo)
            return []

    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        logger.warning("Repo path does not exist: %s", repo_path)
        return []

    test_files = _find_test_files(repo_path)
    logger.info("Found %d test files in %s", len(test_files), repo_path)

    all_enrichments: list[dict] = []

    for test_file in test_files:
        rel_path = str(test_file.relative_to(repo_path))
        test_functions = _extract_test_functions(test_file)

        for test_info in test_functions:
            matches = _match_test_to_rules(test_info, rules, graph_functions)
            for enrichment in matches:
                enrichment["test_file"] = rel_path
                all_enrichments.append(enrichment)

    # Persist
    out_path = _DATA_DIR / repo / "test_enrichments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_enrichments, indent=2))
    logger.info("Wrote %d test enrichments to %s", len(all_enrichments), out_path)

    return all_enrichments


def load_enrichments(repo: str) -> list[dict]:
    """Load previously generated test enrichments for a repo."""
    p = _DATA_DIR / repo / "test_enrichments.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def lookup_failed_tests(repo: str, failed_test_names: list[str]) -> list[dict]:
    """Look up enrichment data for specific failed test functions.

    Args:
        repo: Repository identifier.
        failed_test_names: List of test function names that failed
            (e.g. ["test_refund_within_30_days", "test_discount_cap"]).

    Returns:
        Enrichment entries for the matching tests.
    """
    enrichments = load_enrichments(repo)
    if not enrichments:
        return []

    failed_set = set(failed_test_names)
    return [e for e in enrichments if e.get("test_function") in failed_set]


def format_failure_context(enrichments: list[dict]) -> str:
    """Format enrichment data into a human-readable string for the agent.

    Appended to state['test_result'] when tests fail.
    """
    if not enrichments:
        return ""

    lines = ["\n--- Business Context for Failed Tests ---"]
    for e in enrichments:
        lines.append(
            f"\n[{e.get('severity', 'unknown').upper()}] {e.get('test_function', '?')}"
            f"\n  Tests: {e.get('tests_function_id', '?')}"
            f"\n  Rule: {e.get('business_rule', 'N/A')}"
            f"\n  Impact: {e.get('failure_meaning', 'N/A')}"
        )
    lines.append("\n--- End Business Context ---")
    return "\n".join(lines)
