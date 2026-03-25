"""
lint_rules.py — Custom per-repo lint rules for the AI Deploy Agent.

Step 16 from the implementation guide:
Run codebase-specific lint rules against agent-generated patches.
Each rule explains WHY it exists so the agent can self-correct.

Rules are stored per-repo in DATA_DIR/{repo}/lint_rules.json.
Default rules are generated from codebase analysis (detected patterns).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


class LintViolation(NamedTuple):
    rule_id: str
    file: str
    line: int
    message: str
    severity: str  # error, warning
    fix_hint: str


# ---------------------------------------------------------------------------
# Built-in rule checks
# ---------------------------------------------------------------------------

_BUILTIN_RULES = {
    "no-sync-http-in-async": {
        "pattern": r"\brequests\.(get|post|put|delete|patch)\b",
        "file_pattern": r"\.py$",
        "context_pattern": r"async\s+def",
        "message": "Synchronous HTTP call (requests.*) inside an async function. Use httpx or aiohttp instead.",
        "severity": "error",
        "fix_hint": "Replace `requests.get(url)` with `async with httpx.AsyncClient() as client: await client.get(url)`",
    },
    "no-float-for-money": {
        "pattern": r"\bfloat\s*\(",
        "file_pattern": r"\.py$",
        "context_keywords": ["price", "amount", "balance", "total", "cost", "fee", "tax", "revenue"],
        "message": "Using float() for financial calculation. Use Decimal for money to avoid rounding errors.",
        "severity": "error",
        "fix_hint": "Replace `float(value)` with `Decimal(str(value))` and import from decimal module.",
    },
    "no-hardcoded-secrets": {
        "pattern": r"""(?:api[_-]?key|secret|password|token)\s*=\s*['"]((?!os\.environ|settings\.|config\.)[^'"]{8,})['"]\s*""",
        "file_pattern": r"\.py$",
        "message": "Possible hardcoded secret. Use environment variables or a secrets manager.",
        "severity": "error",
        "fix_hint": "Replace hardcoded value with `os.environ.get('KEY_NAME')` or `settings.KEY_NAME`.",
    },
    "no-bare-except": {
        "pattern": r"\bexcept\s*:",
        "file_pattern": r"\.py$",
        "message": "Bare except clause catches all exceptions including SystemExit and KeyboardInterrupt.",
        "severity": "warning",
        "fix_hint": "Use `except Exception:` to catch only standard exceptions.",
    },
    "no-print-in-production": {
        "pattern": r"^\s*print\s*\(",
        "file_pattern": r"\.py$",
        "exclude_files": ["test_", "conftest", "debug", "script"],
        "message": "print() found in production code. Use logging instead.",
        "severity": "warning",
        "fix_hint": "Replace `print(msg)` with `logger.info(msg)` using the logging module.",
    },
}


def _load_repo_rules(repo_name: str) -> list[dict]:
    """Load custom lint rules for a specific repo."""
    rules_path = DATA_DIR / repo_name / "lint_rules.json"
    if rules_path.exists():
        try:
            return json.loads(rules_path.read_text())
        except Exception:
            pass
    return []


def _save_repo_rules(repo_name: str, rules: list[dict]):
    """Save custom lint rules for a repo."""
    rules_path = DATA_DIR / repo_name / "lint_rules.json"
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(json.dumps(rules, indent=2))


def check_file(file_path: Path, content: str, repo_name: str = "") -> list[LintViolation]:
    """Run all lint rules against a file's content."""
    violations: list[LintViolation] = []
    fname = file_path.name
    fstr = str(file_path)

    # Combine builtin + repo-specific rules
    all_rules = dict(_BUILTIN_RULES)
    for custom in _load_repo_rules(repo_name):
        all_rules[custom.get("id", f"custom-{len(all_rules)}")] = custom

    lines = content.split("\n")

    for rule_id, rule in all_rules.items():
        # Check file pattern
        file_pat = rule.get("file_pattern", "")
        if file_pat and not re.search(file_pat, fstr):
            continue

        # Check exclude files
        excludes = rule.get("exclude_files", [])
        if any(ex in fname for ex in excludes):
            continue

        pattern = rule.get("pattern", "")
        if not pattern:
            continue

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            continue

        # Context check — only flag if surrounding code matches context
        context_pattern = rule.get("context_pattern", "")
        context_keywords = rule.get("context_keywords", [])

        for i, line in enumerate(lines, 1):
            if not regex.search(line):
                continue

            # Context pattern check (e.g., must be inside async def)
            if context_pattern:
                # Look at surrounding 20 lines for context
                context_window = "\n".join(lines[max(0, i - 20):i])
                if not re.search(context_pattern, context_window):
                    continue

            # Context keyword check (e.g., line must relate to money)
            if context_keywords:
                nearby = "\n".join(lines[max(0, i - 5):min(len(lines), i + 5)]).lower()
                if not any(kw in nearby for kw in context_keywords):
                    continue

            violations.append(LintViolation(
                rule_id=rule_id,
                file=str(file_path),
                line=i,
                message=rule.get("message", "Lint violation"),
                severity=rule.get("severity", "warning"),
                fix_hint=rule.get("fix_hint", ""),
            ))

    return violations


def run_lint_on_patches(patches: list[dict], repo_path: Path, repo_name: str = "") -> list[dict]:
    """Run lint rules on all patched files. Returns violations as dicts."""
    all_violations: list[dict] = []

    for patch in patches:
        file_path = patch.get("file_path", "")
        if not file_path:
            continue

        full_path = repo_path / file_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text()
        except Exception:
            continue

        violations = check_file(Path(file_path), content, repo_name)
        for v in violations:
            all_violations.append({
                "rule_id": v.rule_id,
                "file": v.file,
                "line": v.line,
                "message": v.message,
                "severity": v.severity,
                "fix_hint": v.fix_hint,
            })

    return all_violations


def generate_default_rules(repo_name: str) -> list[dict]:
    """Generate default lint rules by analyzing the repo's patterns.

    Detects: async patterns, ORM usage, logging style, etc.
    """
    graph_path = DATA_DIR / repo_name / "graph.json"
    if not graph_path.exists():
        return []

    try:
        data = json.loads(graph_path.read_text())
    except Exception:
        return []

    tech = [t.lower() for t in data.get("stats", {}).get("tech_stack", [])]
    rules = []

    if "fastapi" in tech or "asyncio" in tech:
        rules.append({
            "id": "async-consistency",
            "pattern": r"\brequests\.(get|post|put|delete)\b",
            "file_pattern": r"\.py$",
            "context_pattern": r"async\s+def",
            "message": "This codebase uses async/await (FastAPI). Don't use synchronous requests inside async functions.",
            "severity": "error",
            "fix_hint": "Use `httpx.AsyncClient` or `aiohttp.ClientSession` instead of `requests`.",
        })

    if "sqlalchemy" in tech:
        rules.append({
            "id": "no-raw-sql",
            "pattern": r'(?:execute|text)\s*\(\s*["\'](?:SELECT|INSERT|UPDATE|DELETE)',
            "file_pattern": r"\.py$",
            "message": "Raw SQL detected. This codebase uses SQLAlchemy ORM. Use model queries instead.",
            "severity": "warning",
            "fix_hint": "Replace raw SQL with SQLAlchemy model.query() or session.execute(select(Model)).",
        })

    if "redis" in tech:
        rules.append({
            "id": "redis-key-prefix",
            "pattern": r'\.set\(\s*["\'][a-z]',
            "file_pattern": r"\.py$",
            "message": "Redis key without namespace prefix. Use 'app:module:key' format to avoid collisions.",
            "severity": "warning",
            "fix_hint": "Prefix Redis keys with the module name: f'{MODULE_PREFIX}:{key}'.",
        })

    if rules:
        _save_repo_rules(repo_name, rules)
        logger.info("Generated %d default lint rules for %s", len(rules), repo_name)

    return rules
