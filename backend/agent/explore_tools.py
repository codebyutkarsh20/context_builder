"""
explore_tools.py — Tools the agent uses to actively explore a codebase.

These are callable by Claude during the exploration phase, like Claude Code uses
grep, read, glob etc. The agent decides what to look at — nothing is pushed upfront.
"""

from __future__ import annotations

import json
import logging
import os
import re
import re as _re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared read-only JSON cache (mtime-based invalidation).
# Module-level (not thread-local) because the data is read-only and safe to
# share across concurrent pipeline runs — avoids re-parsing 10-50 MB files on
# every tool call.
# ---------------------------------------------------------------------------
import shutil as _shutil

_json_file_cache: dict[tuple[str], tuple[float, object]] = {}
_json_cache_lock = threading.Lock()


def _load_json_cached(path: "Path") -> object:
    """Load JSON from *path* with mtime-based cache invalidation.

    Returns the parsed object, or None if the file doesn't exist or is
    unreadable.  The cache is keyed on the full resolved path string so
    different repos never alias each other.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    key = (str(path),)
    with _json_cache_lock:
        if key in _json_file_cache:
            cached_mtime, cached_data = _json_file_cache[key]
            if cached_mtime == mtime:
                return cached_data

    # Load from disk outside the lock so we don't block other threads while
    # doing I/O.
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    with _json_cache_lock:
        _json_file_cache[key] = (mtime, data)

    return data

# ---------------------------------------------------------------------------
# Issue #1: Thread-local storage replaces module-level globals so concurrent
# pipeline runs cannot overwrite each other's context.
# ---------------------------------------------------------------------------
_tls = threading.local()

_BINARY_EXTENSIONS = frozenset({
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.o', '.a', '.dylib',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.woff', '.woff2',
    '.ttf', '.zip', '.tar', '.gz', '.db', '.sqlite', '.sqlite3', '.DS_Store',
    '.pdf', '.mp3', '.mp4',
})

_MAX_OUTPUT = 8000  # chars — cap any single tool response

# ---------------------------------------------------------------------------
# Issue #13: Local secret-redaction — avoids circular import from pipeline.py
# ---------------------------------------------------------------------------

_EXPLORE_SECRETS_RE = _re.compile(
    r'(?i)(?:password|passwd|secret|api[_-]?key|auth[_-]?token|access[_-]?key)'
    r'\s*[=:]\s*["\']?([A-Za-z0-9+/=_\-]{8,})["\']?'
)
_ADDITIONAL_SECRET_PATTERNS = [
    _re.compile(r'AKIA[A-Z0-9]{16}'),                              # AWS access keys
    _re.compile(r'(?:Bearer|token)\s+[A-Za-z0-9\-._~+/]+=*', _re.I),
    _re.compile(r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+'),      # JWT
    _re.compile(r'sk-[a-zA-Z0-9]{20,}'),                          # OpenAI/Stripe keys
    _re.compile(r'ghp_[A-Za-z0-9]{36}'),                          # GitHub PATs
]


def _redact_content(text: str) -> str:
    """Redact credential-like values from text before returning it to the LLM."""
    text = _EXPLORE_SECRETS_RE.sub(
        lambda m: m.group(0).replace(m.group(1), '***REDACTED***'), text
    )
    for pat in _ADDITIONAL_SECRET_PATTERNS:
        text = pat.sub('***REDACTED***', text)
    return text


def _cap(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + f"\n... [truncated — {len(text) - _MAX_OUTPUT} more chars]"


def _safe_relpath(p: Path) -> str:
    repo_path = getattr(_tls, 'repo_path', None)
    if repo_path:
        try:
            return str(p.relative_to(repo_path))
        except ValueError:
            pass
    return str(p)


# ---------------------------------------------------------------------------
# Issue #4: Path traversal protection helper
# ---------------------------------------------------------------------------

def _safe_resolve(file_path: str) -> "Path | None":
    """Resolve file_path relative to repo root, rejecting path traversal."""
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return None
    try:
        resolved = (repo_path / file_path).resolve()
        if not str(resolved).startswith(str(repo_path.resolve())):
            logger.warning("Path traversal attempt blocked: %s", file_path)
            return None
        return resolved
    except Exception:
        return None


def _safe_resolve_rglob(match: Path) -> "Path | None":
    """Validate that an rglob result is inside the repo root."""
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return None
    try:
        resolved = match.resolve()
        if not str(resolved).startswith(str(repo_path.resolve())):
            logger.warning("Path traversal (rglob) blocked: %s", match)
            return None
        return resolved
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ripgrep acceleration — use rg when available (5-10x faster, respects
# .gitignore automatically).  Falls back to GNU grep otherwise.
# ---------------------------------------------------------------------------

_HAS_RIPGREP = _shutil.which("rg") is not None


def _build_search_cmd(
    pattern: str,
    repo_path: "Path",
    file_glob: str,
    max_results: int,
) -> list[str]:
    """Return a grep or ripgrep command list for the given search parameters."""
    if _HAS_RIPGREP:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if file_glob:
            cmd.extend(["-g", file_glob])
        else:
            # Default: common source extensions
            for ext in ["py", "js", "ts", "go", "java", "rb", "rs"]:
                cmd.extend(["-g", f"*.{ext}"])
        # rg respects .gitignore automatically — no --exclude-dir needed.
        # Cap results at the caller's limit.
        cmd.extend(["-m", str(max_results)])
        cmd.extend(["--", pattern, str(repo_path)])
    else:
        if file_glob:
            cmd = ["grep", "-rn", "--color=never", f"--include={file_glob}",
                   "-m", str(max_results), "--", pattern, str(repo_path)]
        else:
            cmd = [
                "grep", "-rn", "--color=never",
                "--include=*.py", "--include=*.js", "--include=*.ts",
                "--include=*.go", "--include=*.java", "--include=*.rb", "--include=*.rs",
                "-m", str(max_results),
            ]
            # Exclude common noise directories
            for excl in ["node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"]:
                cmd.extend(["--exclude-dir", excl])
            cmd.extend(["--", pattern, str(repo_path)])
    return cmd


# ---------------------------------------------------------------------------
# Tool 1 — grep_repo
# ---------------------------------------------------------------------------

@tool
def grep_repo(pattern: str, file_glob: str = "", max_results: int = 25) -> str:
    """
    Search for a regex pattern across all source files in the repo.
    Returns matching lines with file:line_number format.

    Args:
        pattern: Regex or literal string to search for
        file_glob: Optional glob filter e.g. '*.py', '*.ts' (default: all source files)
        max_results: Max number of matching lines to return (default 25)
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path or not repo_path.exists():
        return "ERROR: repo path not set"

    try:
        cmd = _build_search_cmd(pattern, repo_path, file_glob, max_results)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern: {pattern}"

        # Make paths relative
        lines = []
        for line in output.split("\n")[:max_results]:
            if repo_path:
                line = line.replace(str(repo_path) + "/", "")
            lines.append(line)

        return _cap(f"Found {len(lines)} matches:\n" + "\n".join(lines))
    except subprocess.TimeoutExpired:
        return "ERROR: grep timed out"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 2 — read_file
# ---------------------------------------------------------------------------

@tool
def read_file(file_path: str, start_line: int = 1, end_line: int = 100) -> str:
    """
    Read a 100-line window of a file from the repo.
    IMPORTANT: Jump, don't scroll — if you know the function name, use read_function instead.
    Use grep_repo first to find the exact line numbers, then read only that window.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
        start_line: First line to read (1-indexed, default 1)
        end_line: Last line to read (default 100 — one focused window at a time)
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    resolved = _safe_resolve(file_path)
    if resolved is None or not resolved.exists():
        # Try rglob fallback — only accept results inside the repo
        matches = list(repo_path.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = _safe_resolve_rglob(m)
            if candidate is not None:
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found: {file_path}"

    if resolved.suffix.lower() in _BINARY_EXTENSIONS:
        return f"ERROR: Binary file skipped: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total = len(lines)

        s = max(0, start_line - 1)
        e = min(total, end_line)
        selected = lines[s:e]

        header = f"=== {file_path} (lines {s+1}-{e} of {total}) ===\n"
        numbered = "\n".join(f"{s+1+i:4d} | {ln}" for i, ln in enumerate(selected))
        footer = ""
        if e < total:
            footer = f"\n... [{total - e} more lines — call read_file with start_line={e+1}]"

        # Issue #13: redact secrets before returning content to the LLM
        return _redact_content(_cap(header + numbered + footer))
    except Exception as e:
        return f"ERROR reading {file_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 3 — read_function
# ---------------------------------------------------------------------------

@tool
def read_function(file_path: str, function_name: str) -> str:
    """
    Extract the complete source code of a specific function or method from a file.
    Much more precise than read_file — extracts just the function body.

    Args:
        file_path: Relative path from repo root
        function_name: Name of the function or method to extract
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    resolved = _safe_resolve(file_path)
    if resolved is None or not resolved.exists():
        # Try rglob fallback — only accept results inside the repo
        matches = list(repo_path.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = _safe_resolve_rglob(m)
            if candidate is not None:
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")

        # -------------------------------------------------------------------
        # Issue #11: AST-based extraction for Python files
        # -------------------------------------------------------------------
        if resolved.suffix == '.py':
            import ast as _ast
            try:
                tree = _ast.parse(content)
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        if node.name == function_name:
                            file_lines = content.splitlines(keepends=True)
                            start = node.lineno - 1        # 0-indexed
                            end = node.end_lineno          # end_lineno is 1-indexed inclusive
                            extracted = "".join(file_lines[start:end])
                            numbered = "\n".join(
                                f"{start+1+i:4d} | {ln.rstrip()}"
                                for i, ln in enumerate(file_lines[start:end])
                            )
                            result = (
                                f"=== {function_name} in {file_path} "
                                f"(lines {start+1}-{end}) ===\n{numbered}"
                            )
                            # Issue #13: redact secrets
                            return _redact_content(_cap(result))
            except SyntaxError:
                pass  # Fall through to regex/indent heuristic

        # -------------------------------------------------------------------
        # Regex + indent heuristic fallback (non-Python or parse failure)
        # -------------------------------------------------------------------
        lines = content.split("\n")

        # Find function definition line
        pattern = re.compile(
            rf"^\s*(async\s+)?(def|function|func|fn)\s+{re.escape(function_name)}\s*[\(:]",
            re.MULTILINE
        )
        match = pattern.search(content)
        if not match:
            # Try class method pattern
            pattern2 = re.compile(
                rf"^\s+def\s+{re.escape(function_name)}\s*\(",
                re.MULTILINE
            )
            match = pattern2.search(content)

        if not match:
            return f"Function '{function_name}' not found in {file_path}. Try grep_repo to locate it."

        start_line = content[:match.start()].count("\n")
        indent = len(lines[start_line]) - len(lines[start_line].lstrip())

        # Walk forward to find end of function (dedent back to same level)
        end_line = start_line + 1
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip() == "":
                end_line += 1
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent and end_line > start_line + 1:
                break
            end_line += 1

        selected = lines[start_line:end_line]
        numbered = "\n".join(f"{start_line+1+i:4d} | {ln}" for i, ln in enumerate(selected))
        result = f"=== {function_name} in {file_path} (lines {start_line+1}-{end_line}) ===\n{numbered}"
        # Issue #13: redact secrets
        return _redact_content(_cap(result))

    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 4 — list_files
# ---------------------------------------------------------------------------

@tool
def list_files(directory: str = "", extension: str = "") -> str:
    """
    List files in the repo, optionally filtered by directory or extension.
    Use this to explore the repo structure and find relevant files.

    Args:
        directory: Subdirectory to list e.g. 'app/services' (default: repo root)
        extension: Filter by extension e.g. '.py', '.ts' (default: all source files)
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    if directory:
        base_resolved = _safe_resolve(directory)
        if base_resolved is None:
            return f"ERROR: Directory not found or path traversal blocked: {directory}"
        base = base_resolved
    else:
        base = repo_path

    if not base.exists():
        return f"ERROR: Directory not found: {directory}"

    try:
        source_exts = {'.py', '.js', '.ts', '.go', '.java', '.rb', '.rs', '.c', '.cpp',
                       '.jsx', '.tsx', '.cs', '.php', '.swift', '.kt'}

        files = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                # Validate each result is within the repo
                candidate = _safe_resolve_rglob(p)
                if candidate is None:
                    continue
                if p.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                if "__pycache__" in str(p) or "node_modules" in str(p) or ".git" in str(p):
                    continue
                if extension and p.suffix.lower() != extension.lower():
                    continue
                if not extension and p.suffix.lower() not in source_exts:
                    continue
                files.append(_safe_relpath(p))

        if not files:
            return f"No source files found in {directory or 'repo root'}"

        output = f"Files in {directory or 'repo root'} ({len(files)} total):\n"
        output += "\n".join(files[:100])
        if len(files) > 100:
            output += f"\n... and {len(files) - 100} more files"
        return _cap(output)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 5 — search_code
# ---------------------------------------------------------------------------

@tool
def search_code(query: str, limit: int = 10) -> str:
    """
    Semantic search across the codebase using natural language.
    Finds functions, classes and files by meaning — not just keyword matching.
    Use this when you don't know the exact name but know what you're looking for.

    Args:
        query: Natural language description e.g. 'payment validation logic'
        limit: Number of results (default 10, max 20)
    """
    repo_name = getattr(_tls, 'repo_name', None)
    data_dir = getattr(_tls, 'data_dir', None)
    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo_name, data_dir)
        info = embedder.collection_info()
        if info.get("count", 0) == 0:
            return "Semantic search not available — embeddings not built for this repo. Use grep_repo instead."

        results = embedder.query(text=query, n_results=min(limit, 20))
        if not results:
            return f"No results for: {query}"

        lines = [f"Semantic search results for: '{query}'\n"]
        for r in results:
            nid = r.get("id", "")
            meta = r.get("metadata", {})
            score = r.get("score", 0)
            lines.append(
                f"  [{score:.3f}] {meta.get('type','?')} — {nid}\n"
                f"          File: {meta.get('file', '')}"
            )
        return _cap("\n".join(lines))
    except Exception as e:
        return f"Semantic search unavailable: {e}. Use grep_repo instead."


# ---------------------------------------------------------------------------
# Tool 6 — get_function_info
# ---------------------------------------------------------------------------

@tool
def get_function_info(function_id: str) -> str:
    """
    Get structural info about a function from the knowledge graph:
    who calls it, what it calls, what business rules apply, decision points inside it.

    Args:
        function_id: Full function ID e.g. 'app/services/payment.py::validate_amount'
                     or just the function name to search broadly.
    """
    repo_name = getattr(_tls, 'repo_name', None)
    data_dir = getattr(_tls, 'data_dir', None)
    try:
        graph_path = data_dir / repo_name / "graph.json"
        enriched_path = data_dir / repo_name / "enriched_nodes.json"
        rules_path = data_dir / repo_name / "business_rules.json"

        if not graph_path.exists():
            return f"Graph not found for repo '{repo_name}'. Repo may not be analyzed yet."

        graph = _load_json_cached(graph_path) or {}
        enriched = _load_json_cached(enriched_path) if enriched_path.exists() else {}
        enriched = enriched or {}
        rules = _load_json_cached(rules_path) if rules_path.exists() else []
        rules = rules or []

        # Find matching node (exact or partial)
        node = enriched.get(function_id)
        if not node:
            # Search by suffix (just function name)
            matches = [k for k in enriched if k.endswith(f"::{function_id}") or function_id in k]
            if matches:
                function_id = matches[0]
                node = enriched[function_id]
            else:
                return f"Function '{function_id}' not found in knowledge graph. Try grep_repo to find its file."

        edges = graph.get("edges", [])
        callers = [e["source"] for e in edges if e.get("target") == function_id and e.get("type") == "CALLS"]
        callees = [e["target"] for e in edges if e.get("source") == function_id and e.get("type") == "CALLS"]

        file_path = node.get("file", function_id.split("::")[0] if "::" in function_id else "")
        matched_rules = [
            f"  [{r.get('severity','?')}] {r.get('description','')}"
            for r in rules
            if r.get("function_id") == function_id or r.get("file") == file_path
        ]

        dps = [
            f"  L{dp.get('line',0)}: `{dp.get('condition','')}` ({dp.get('condition_type','')})"
            for dp in graph.get("decision_points", [])
            if dp.get("function_id") == function_id
        ]

        lines = [
            f"=== {function_id} ===",
            f"File: {file_path}",
            f"Summary: {node.get('llm_summary') or node.get('docstring') or 'N/A'}",
            f"PageRank: {node.get('pagerank', 0):.4f}",
            f"\nCallers ({len(callers)}):",
            *[f"  {c}" for c in callers[:10]],
            f"\nCallees ({len(callees)}):",
            *[f"  {c}" for c in callees[:10]],
            f"\nBusiness Rules ({len(matched_rules)}):",
            *(matched_rules or ["  None"]),
            f"\nDecision Points ({len(dps)}):",
            *(dps or ["  None"]),
        ]
        return _cap("\n".join(lines))
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 7 — get_file_summary
# ---------------------------------------------------------------------------

@tool
def get_file_summary(file_path: str) -> str:
    """
    Get the enriched summary of a file from the knowledge graph —
    what it does, its key classes/functions, and which files import it.
    Faster than reading the file when you just need an overview.

    Args:
        file_path: Relative path e.g. 'app/services/payment.py'
    """
    repo_name = getattr(_tls, 'repo_name', None)
    data_dir = getattr(_tls, 'data_dir', None)
    try:
        enriched_path = data_dir / repo_name / "enriched_nodes.json"
        graph_path = data_dir / repo_name / "graph.json"

        if not enriched_path.exists():
            return f"Knowledge graph not found for repo '{repo_name}'. Use read_file instead."

        enriched = _load_json_cached(enriched_path) or {}
        graph = _load_json_cached(graph_path) if graph_path.exists() else {}
        graph = graph or {}

        # Find file node
        file_node = enriched.get(file_path)
        if not file_node:
            matches = [k for k, v in enriched.items() if v.get("type") == "file" and file_path in k]
            if matches:
                file_path = matches[0]
                file_node = enriched[file_path]

        # Get all functions/classes in this file
        members = [
            f"  {v.get('type','?')}: {v.get('name','?')} — {(v.get('docstring') or '')[:80]}"
            for k, v in enriched.items()
            if v.get("file") == file_path and v.get("type") in ("function", "class")
        ]

        # Who imports this file
        edges = graph.get("edges", [])
        importers = list({
            e["source"].split("::")[0]
            for e in edges
            if e.get("type") == "IMPORTS" and e.get("target", "").startswith(file_path)
        })[:10]

        summary = (file_node or {}).get("llm_summary") or (file_node or {}).get("docstring") or "No summary available"

        lines = [
            f"=== {file_path} ===",
            f"Summary: {summary}",
            f"\nMembers ({len(members)}):",
            *(members[:20] or ["  (none found in graph)"]),
            f"\nImported by ({len(importers)}):",
            *(importers or ["  (none)"]),
        ]
        return _cap("\n".join(lines))
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 8 — get_file_structure
# ---------------------------------------------------------------------------

@tool
def get_file_structure(file_path: str) -> str:
    """
    Get the compressed structure of a file — class and function signatures only, no bodies.
    Much cheaper than read_file. Use this first to understand what's in a file,
    then use read_function to read specific functions you care about.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    resolved = _safe_resolve(file_path)
    if resolved is None or not resolved.exists():
        # Try rglob fallback — only accept results inside the repo
        matches = list(repo_path.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = _safe_resolve_rglob(m)
            if candidate is not None:
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found: {file_path}"

    if resolved.suffix.lower() in _BINARY_EXTENSIONS:
        return f"ERROR: Binary file skipped: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total = len(lines)

        structure_lines = []
        imports: list[str] = []
        in_import = False

        # Patterns for Python
        class_re = re.compile(r"^(\s*)(class\s+\w+[^:]*:)")
        func_re = re.compile(r"^(\s*)((?:async\s+)?def\s+\w+[^:]*:)")
        import_re = re.compile(r"^(?:import|from)\s+")

        for i, line in enumerate(lines):
            stripped = line.rstrip()
            lineno = i + 1

            if import_re.match(stripped):
                imports.append(f"  {stripped}")
                continue

            m = class_re.match(stripped)
            if m:
                structure_lines.append(f"L{lineno:4d}: {m.group(1)}{m.group(2)}")
                continue

            m = func_re.match(stripped)
            if m:
                # Include the docstring (first line after def) if present
                docstring = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        docstring = f"  # {next_line[:80]}"
                structure_lines.append(f"L{lineno:4d}: {m.group(1)}{m.group(2)}{docstring}")
                continue

        header = f"=== {file_path} structure ({total} lines) ===\n"
        if imports:
            header += f"Imports ({len(imports)}):\n" + "\n".join(imports[:15])
            if len(imports) > 15:
                header += f"\n  ... {len(imports) - 15} more"
            header += "\n\n"

        body = "\n".join(structure_lines) if structure_lines else "  (no classes/functions found)"
        return _cap(header + body)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 9 — string_replace
# ---------------------------------------------------------------------------

@tool
def string_replace(file_path: str, old_string: str, new_string: str) -> str:
    """
    Replace an exact string in a file — position-independent, no line numbers needed.
    The old_string must be a unique, exact substring of the file (including whitespace).
    Use this after read_function to make precise edits.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
        old_string: The exact text to replace (must be unique in the file)
        new_string: The replacement text
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    if not old_string or not old_string.strip():
        return "ERROR: old_string must not be empty"

    if old_string == new_string:
        return "ERROR: old_string and new_string are identical — no change made"

    resolved = _safe_resolve(file_path)
    if resolved is None or not resolved.exists():
        matches = list(repo_path.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = _safe_resolve_rglob(m)
            if candidate is not None:
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found: {file_path}"

    if resolved.suffix.lower() in _BINARY_EXTENSIONS:
        return f"ERROR: Binary file skipped: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)

        if count == 0:
            # Try whitespace-normalized match
            import difflib
            lines = content.splitlines()
            old_lines = old_string.splitlines()
            norm_content = [l.rstrip() for l in lines]
            norm_old = [l.rstrip() for l in old_lines]
            for i in range(len(norm_content) - len(norm_old) + 1):
                if norm_content[i:i + len(norm_old)] == norm_old:
                    # Found a whitespace-normalized match
                    actual_old = "\n".join(lines[i:i + len(norm_old)])
                    content = content.replace(actual_old, new_string, 1)
                    resolved.write_text(content, encoding="utf-8")
                    return f"OK: replaced (whitespace-normalized match) in {file_path}"
            return (
                f"ERROR: old_string not found in {file_path}.\n"
                f"The string may have changed or you may have copied it incorrectly.\n"
                f"Use read_file or read_function to get the current exact content."
            )
        elif count > 1:
            return (
                f"ERROR: old_string appears {count} times in {file_path}. "
                f"Make it longer/more unique so only one instance matches."
            )

        new_content = content.replace(old_string, new_string, 1)
        resolved.write_text(new_content, encoding="utf-8")

        # Count lines changed
        old_line_count = old_string.count("\n") + 1
        new_line_count = new_string.count("\n") + 1
        delta = new_line_count - old_line_count
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        return f"OK: replaced 1 occurrence in {file_path} ({delta_str} lines)"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Exported tool list + context setter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool 10 — check_syntax
# ---------------------------------------------------------------------------

@tool
def check_syntax(file_path: str) -> str:
    """
    Check a Python file for syntax errors after editing it.
    Always run this after string_replace to verify your edit is valid Python.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    resolved = _safe_resolve(file_path)
    if resolved is None or not resolved.exists():
        matches = list(repo_path.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = _safe_resolve_rglob(m)
            if candidate is not None:
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found: {file_path}"

    if resolved.suffix.lower() != ".py":
        return f"OK: syntax check only available for .py files (skipped {file_path})"

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import ast; ast.parse(open({repr(str(resolved))}).read())"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"OK: {file_path} — no syntax errors"
        else:
            return f"SYNTAX ERROR in {file_path}:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "ERROR: syntax check timed out"
    except Exception as e:
        return f"ERROR: {e}"


# Exploration tools — READ-ONLY. The agent uses these to understand the codebase.
# string_replace and check_syntax are intentionally excluded: direct edits during
# exploration leave the main repo dirty, which blocks git worktree creation in the
# test node. Patching happens exclusively in the repair stage via the sandbox worktree.
ALL_TOOLS = [
    grep_repo,
    read_file,
    read_function,
    list_files,
    search_code,
    get_function_info,
    get_file_summary,
    get_file_structure,
]


def set_context(repo_name: str, repo_path: str | Path, data_dir: Path | None = None) -> None:
    """Set the per-run context so tools know which repo to operate on.

    Stores values in thread-local storage so concurrent pipeline runs are isolated.
    """
    _tls.repo_name = repo_name
    _tls.repo_path = Path(repo_path) if repo_path else None
    if data_dir:
        _tls.data_dir = data_dir
    else:
        # Preserve the default only if not already set on this thread
        if not hasattr(_tls, 'data_dir'):
            _tls.data_dir = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
