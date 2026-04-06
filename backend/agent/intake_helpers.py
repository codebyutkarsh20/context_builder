"""
intake_helpers.py — Bug ticket parsing utilities extracted from pipeline.py.

Stack trace extraction, reproduction step parsing, and bug categorization.
"""

from __future__ import annotations

import re


def extract_stack_trace_hints(text: str) -> list[dict]:
    """Extract file:line:function hints from stack traces in ticket text."""
    hints = []

    python_pattern = re.compile(
        r'File ["\']([^"\']+\.py)["\'],\s+line\s+(\d+),\s+in\s+(\w+)'
    )
    for m in python_pattern.finditer(text):
        hints.append({"file": m.group(1), "line": int(m.group(2)), "function": m.group(3)})

    java_pattern = re.compile(r'at\s+([\w.]+)\((\w+\.java):(\d+)\)')
    for m in java_pattern.finditer(text):
        hints.append({"file": m.group(2), "line": int(m.group(3)), "function": m.group(1).split('.')[-1]})

    file_line_pattern = re.compile(r'([\w/.-]+\.(?:py|js|ts|go|rb|java)):(\d+)')
    for m in file_line_pattern.finditer(text):
        hints.append({"file": m.group(1), "line": int(m.group(2)), "function": None})

    seen: set[tuple[str, int]] = set()
    deduped = []
    for h in hints:
        key = (h["file"], h["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    return deduped[:10]


def extract_repro_steps(description: str) -> list[str]:
    """Extract reproduction steps from ticket description."""
    steps = []
    section_patterns = [
        r'(?:steps?\s+to\s+reproduce|how\s+to\s+reproduce|reproduction\s+steps?|repro\s+steps?)\s*:?\s*\n([\s\S]+?)(?:\n\n|\n#{1,3}|\Z)',
        r'(?:to\s+reproduce|reproduce)\s*:?\s*\n([\s\S]+?)(?:\n\n|\n#{1,3}|\Z)',
    ]
    for pattern in section_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            section = match.group(1)
            items = re.findall(r'(?:^|\n)\s*(?:\d+\.|[-*•])\s*(.+)', section)
            if items:
                steps = [item.strip() for item in items[:10]]
                break
    if not steps:
        numbered = re.findall(r'(?:^|\n)\s*(\d+)\.\s+(.+)', description)
        if len(numbered) >= 2:
            steps = [item for _, item in numbered[:10]]
    return steps


def classify_bug_category(title: str, description: str) -> str:
    """Classify bug into Category A (auto-fix), B (might work), C (skip).

    Category A: Clear logic/value bugs — high auto-fix success rate.
    Category B: Moderate complexity — worth attempting.
    Category C: Concurrency, perf, multi-service, UI — low success rate.
    """
    text = f"{title} {description}".lower()

    # Multi-word phrases — safe to substring-match
    c_signals_phrase = [
        "race condition", "concurrency", "deadlock",
        "memory leak", "database migration", "schema change",
        "multi-service", "works locally", "works in dev", "only in prod",
    ]
    if any(s in text for s in c_signals_phrase):
        return "C"

    # Single-word signals — require word boundaries to avoid false positives
    # e.g. "slow" in "tests/slow", "timeout" in "test_timeout", "env" in "environment"
    c_signals_word = [
        r"\bslow\b", r"\btimeout\b", r"\bperformance\b", r"\bui\b", r"\bcss\b",
        r"\blayout\b", r"\bvisual\b", r"\banimation\b", r"\barchitecture\b",
        r"\bredesign\b", r"\brefactor\b", r"\bkafka\b", r"\brabbitmq\b",
        r"\bn\+1\b",
    ]
    if any(re.search(pat, text) for pat in c_signals_word):
        return "C"

    a_signals = [
        "traceback", "exception", "error:", "typeerror", "attributeerror",
        "valueerror", "keyerror", "indexerror", "none", "null", "undefined",
        "missing", "import", "not found", "wrong value", "incorrect value",
        "returns wrong", "should return", "off by one", "index out",
        "str.split", "shlex", "incorrect", "broken", "fails silently",
    ]
    a_count = sum(1 for s in a_signals if s in text)
    if a_count >= 2:
        return "A"

    return "B"
