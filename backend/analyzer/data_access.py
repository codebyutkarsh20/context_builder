"""
data_access.py — Pattern-based detector for data access operations in Python code.

Scans function bodies to identify reads/writes to databases, files, APIs, and caches.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_READ_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("database", re.compile(r"\.query\s*\(|\.filter\s*\(|\.filter_by\s*\(|\.get\s*\(|\.first\s*\(|\.all\s*\(|\.one\s*\(|\.scalar\s*\(|\.count\s*\(")),
    ("database", re.compile(r"objects\.(get|filter|exclude|all|values|annotate|aggregate)\s*\(")),
    ("database", re.compile(r"cursor\.execute\s*\(.*SELECT", re.IGNORECASE)),
    ("database", re.compile(r"\.fetchone\s*\(|\.fetchall\s*\(|\.fetchmany\s*\(")),
    ("file", re.compile(r"\.read_text\s*\(|\.read_bytes\s*\(|\.read\s*\(")),
    ("file", re.compile(r"open\s*\([^)]*['\"]r['\"]|open\s*\([^)]*mode\s*=\s*['\"]r")),
    ("file", re.compile(r"json\.load\s*\(|yaml\.safe_load\s*\(|csv\.reader\s*\(|toml\.load\s*\(")),
    ("api", re.compile(r"requests\.get\s*\(|httpx\.get\s*\(|aiohttp.*\.get\s*\(")),
    ("api", re.compile(r"\.fetch\s*\(|urllib\.request")),
    ("cache", re.compile(r"cache\.get\s*\(|redis.*\.get\s*\(|redis.*\.hget\s*\(|redis.*\.mget\s*\(")),
    ("cache", re.compile(r"memcache.*\.get\s*\(")),
    ("environment", re.compile(r"os\.environ\s*\[|os\.environ\.get\s*\(|os\.getenv\s*\(")),
]

_WRITE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("database", re.compile(r"\.add\s*\(|\.add_all\s*\(|\.merge\s*\(|\.commit\s*\(|\.flush\s*\(")),
    ("database", re.compile(r"\.save\s*\(|\.create\s*\(|\.bulk_create\s*\(|\.update\s*\(|\.delete\s*\(")),
    ("database", re.compile(r"\.insert\s*\(|\.update_or_create\s*\(|\.get_or_create\s*\(")),
    ("database", re.compile(r"cursor\.execute\s*\(.*(?:INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)", re.IGNORECASE)),
    ("file", re.compile(r"\.write_text\s*\(|\.write_bytes\s*\(|\.write\s*\(")),
    ("file", re.compile(r"open\s*\([^)]*['\"]w['\"]|open\s*\([^)]*mode\s*=\s*['\"]w|open\s*\([^)]*['\"]a['\"]")),
    ("file", re.compile(r"json\.dump\s*\(|yaml\.dump\s*\(|csv\.writer\s*\(|shutil\.")),
    ("api", re.compile(r"requests\.post\s*\(|requests\.put\s*\(|requests\.patch\s*\(|requests\.delete\s*\(")),
    ("api", re.compile(r"httpx\.post\s*\(|httpx\.put\s*\(|httpx\.patch\s*\(|httpx\.delete\s*\(")),
    ("api", re.compile(r"aiohttp.*\.post\s*\(|aiohttp.*\.put\s*\(")),
    ("cache", re.compile(r"cache\.set\s*\(|cache\.delete\s*\(|redis.*\.set\s*\(|redis.*\.hset\s*\(|redis.*\.delete\s*\(")),
    ("cache", re.compile(r"redis.*\.expire\s*\(|redis.*\.publish\s*\(")),
    ("logging", re.compile(r"logger\.(info|warning|error|critical|exception)\s*\(|logging\.(info|warning|error|critical)\s*\(")),
    ("event", re.compile(r"\.send_task\s*\(|\.apply_async\s*\(|\.delay\s*\(|\.publish\s*\(|\.emit\s*\(")),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_data_access(parsed_files: list[dict]) -> dict[str, dict[str, list[str]]]:
    """
    Scan all parsed files for data access patterns.

    Returns: {function_id: {"reads_from": ["database", "cache"], "writes_to": ["database", "file"]}}
    """
    results: dict[str, dict[str, list[str]]] = {}

    for pf in parsed_files:
        rel = pf["path"]
        abs_path = pf.get("abs_path", "")
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (OSError, FileNotFoundError):
            continue

        # Top-level functions
        for fn in pf.get("functions", []):
            if "line_start" not in fn or "line_end" not in fn:
                continue
            func_id = f"{rel}::{fn['name']}"
            body = _slice(lines, fn["line_start"], fn["line_end"])
            access = _scan_body(body)
            if access["reads_from"] or access["writes_to"]:
                results[func_id] = access

        # Class methods
        for cls in pf.get("classes", []):
            for method in cls.get("methods", []):
                if "line_start" not in method or "line_end" not in method:
                    continue
                func_id = f"{rel}::{cls['name']}.{method['name']}"
                body = _slice(lines, method["line_start"], method["line_end"])
                access = _scan_body(body)
                if access["reads_from"] or access["writes_to"]:
                    results[func_id] = access

    return results


def _slice(lines: list[str], start: int, end: int) -> str:
    """Extract lines from 1-indexed start to end inclusive."""
    return "".join(lines[max(0, start - 1):end])


def _scan_body(body: str) -> dict[str, list[str]]:
    """Scan a function body text for read/write patterns."""
    reads: set[str] = set()
    writes: set[str] = set()

    for category, pattern in _READ_PATTERNS:
        if pattern.search(body):
            reads.add(category)

    for category, pattern in _WRITE_PATTERNS:
        if pattern.search(body):
            writes.add(category)

    return {
        "reads_from": sorted(reads),
        "writes_to": sorted(writes),
    }
