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
    """Classify bug into Category A (auto-fix), B (might work), C (skip)."""
    text = f"{title} {description}".lower()

    c_signals = [
        "race condition", "concurrency", "deadlock", "performance", "slow", "timeout",
        "n+1", "memory leak", "migration", "database migration", "schema change",
        "multi-service", "event", "kafka", "rabbitmq", "queue", "environment",
        "works locally", "works in dev", "only in prod", "ui", "visual", "layout",
        "animation", "css", "architecture", "redesign", "refactor",
    ]
    if any(s in text for s in c_signals):
        return "C"

    a_signals = [
        "traceback", "exception", "error:", "typeerror", "attributeerror",
        "none", "null", "undefined", "missing", "import", "not found",
        "wrong value", "incorrect value", "returns wrong", "should return",
        "off by one", "index out", "keyerror", "valueerror",
    ]
    a_count = sum(1 for s in a_signals if s in text)
    if a_count >= 2:
        return "A"

    return "B"
