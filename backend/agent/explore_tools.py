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

from agent.path_safety import safe_resolve, safe_resolve_rglob

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
    """Internal safety cap. The real per-tool cap is applied by cap_tool_output() in react_loop.py
    using the tool_metadata registry. This is just a backstop for very large outputs."""
    MAX = 40_000  # 40K chars backstop — individual tool caps are lower
    if len(text) <= MAX:
        return text
    return text[:MAX] + f"\n... [truncated — {len(text) - MAX} more chars]"


def _safe_relpath(p: Path) -> str:
    repo_path = getattr(_tls, 'repo_path', None)
    if repo_path:
        try:
            return str(p.relative_to(repo_path))
        except ValueError:
            pass
    return str(p)


# ---------------------------------------------------------------------------
# Issue #4: Path traversal protection helper  (delegated to path_safety.py)
# ---------------------------------------------------------------------------

def _safe_resolve(file_path: str) -> "Path | None":
    """Resolve file_path relative to repo root, rejecting path traversal."""
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return None
    return safe_resolve(file_path, repo_path)


def _safe_resolve_rglob(match: Path) -> "Path | None":
    """Validate that an rglob result is inside the repo root."""
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return None
    return safe_resolve_rglob(match, repo_path)


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
# Issue #11: Multi-language function extraction helpers
# ---------------------------------------------------------------------------

def _extract_function_treesitter_js(
    content: str, function_name: str, suffix: str
) -> tuple[int, int] | None:
    """Extract function boundaries from JS/TS using tree-sitter if available.

    Returns (start_line_0indexed, end_line_0indexed_exclusive) or None.
    """
    try:
        import tree_sitter
    except ImportError:
        return None

    # Map suffix to tree-sitter language
    lang_map = {
        '.js': 'tree_sitter_javascript',
        '.jsx': 'tree_sitter_javascript',
        '.ts': 'tree_sitter_typescript',
        '.tsx': 'tree_sitter_typescript',
    }
    lang_module_name = lang_map.get(suffix)
    if not lang_module_name:
        return None

    try:
        lang_mod = __import__(lang_module_name)
        lang = tree_sitter.Language(lang_mod.language())
        parser = tree_sitter.Parser(lang)
        tree = parser.parse(content.encode())

        # Walk tree looking for function declarations / arrow functions / methods
        def _find_function(node):
            if node.type in ('function_declaration', 'method_definition', 'function',
                             'arrow_function', 'generator_function_declaration'):
                # Check name
                for child in node.children:
                    if child.type in ('identifier', 'property_identifier') and child.text.decode() == function_name:
                        return (node.start_point[0], node.end_point[0] + 1)
            # Variable declarator with arrow function: const foo = () => {...}
            if node.type == 'variable_declarator':
                name_node = node.child_by_field_name('name')
                value_node = node.child_by_field_name('value')
                if (name_node and name_node.text.decode() == function_name
                        and value_node and value_node.type in ('arrow_function', 'function')):
                    parent = node.parent  # Get the full variable_declaration
                    if parent and parent.type == 'lexical_declaration':
                        return (parent.start_point[0], parent.end_point[0] + 1)
                    return (node.start_point[0], value_node.end_point[0] + 1)
            # Recurse
            for child in node.children:
                result = _find_function(child)
                if result:
                    return result
            return None

        return _find_function(tree.root_node)
    except Exception:
        return None


def _extract_function_brace_counting(
    content: str, function_name: str
) -> tuple[int, int] | None:
    """Extract function boundaries using brace counting for C-family languages.

    Works for JS/TS/Go/Java/Rust/C/C++/C#/Kotlin/Swift.
    Returns (start_line_0indexed, end_line_0indexed_exclusive) or None.
    """
    lines = content.split("\n")
    escaped_name = re.escape(function_name)

    # Patterns for function definitions in various languages
    patterns = [
        # JS/TS: function foo(...) { / async function foo(...)
        rf"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+{escaped_name}\s*[\(<]",
        # JS/TS: const foo = (...) => / const foo = function
        rf"^\s*(export\s+)?(const|let|var)\s+{escaped_name}\s*=\s*(async\s+)?(function|\()",
        # Go: func foo(...) / func (r *Receiver) foo(...)
        rf"^\s*func\s+(\([^)]*\)\s+)?{escaped_name}\s*\(",
        # Java/C#/Kotlin: public void foo(...) / fun foo(...)
        rf"^\s*(public|private|protected|internal|static|override|suspend|fun|inline)\s+.*\b{escaped_name}\s*[\(<]",
        # Rust: fn foo(...) / pub fn foo(...)  / async fn foo(...)
        rf"^\s*(pub(\s*\([^)]*\))?\s+)?(async\s+)?fn\s+{escaped_name}\s*[\(<]",
        # C/C++: type foo(...) { — broad catch
        rf"^\s*[\w:*&<>\[\]]+\s+{escaped_name}\s*\(",
        # Class method: foo(...) { — inside class body
        rf"^\s+(async\s+)?{escaped_name}\s*\(",
    ]

    start_line = None
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.match(pat, line):
                start_line = i
                break
        if start_line is not None:
            break

    if start_line is None:
        return None

    # Count braces from the definition line to find the closing brace
    brace_depth = 0
    found_open = False
    end_line = start_line

    for i in range(start_line, len(lines)):
        line = lines[i]
        # Strip string literals and comments to avoid counting braces inside them
        stripped = re.sub(r'"(?:[^"\\]|\\.)*"', '', line)
        stripped = re.sub(r"'(?:[^'\\]|\\.)*'", '', stripped)
        stripped = re.sub(r'`(?:[^`\\]|\\.)*`', '', stripped)
        stripped = re.sub(r'//.*$', '', stripped)

        for ch in stripped:
            if ch == '{':
                brace_depth += 1
                found_open = True
            elif ch == '}':
                brace_depth -= 1
                if found_open and brace_depth <= 0:
                    return (start_line, i + 1)

        end_line = i

    # If we never found balanced braces, return up to 100 lines as best effort
    if found_open:
        return (start_line, min(start_line + 100, len(lines)))
    return None


# ---------------------------------------------------------------------------
# Tool 1 — grep_repo
# ---------------------------------------------------------------------------

@tool
def grep_repo(pattern: str, file_glob: str = "", max_results: int = 20, context_lines: int = 2) -> str:
    """
    Search for a regex pattern across source files. Returns matches with
    surrounding context lines so you can understand the code around each match.

    Args:
        pattern: Regex or literal string to search for
        file_glob: Optional glob filter e.g. '*.py' to narrow search
        max_results: Max matches to return (default 20)
        context_lines: Lines of context before/after each match (default 2)
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path or not repo_path.exists():
        return "ERROR: repo path not set"

    try:
        cmd = _build_search_cmd(pattern, repo_path, file_glob, max_results)
        # Add context lines for better understanding
        if context_lines > 0 and _HAS_RIPGREP:
            # Insert -C flag before the pattern
            idx = cmd.index("--")
            cmd.insert(idx, f"-C{context_lines}")
        elif context_lines > 0:
            # GNU grep
            cmd.insert(1, f"-C{context_lines}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern: {pattern}"

        # Make paths relative and cap line length
        lines = []
        repo_str = str(repo_path) + "/"
        for line in output.split("\n"):
            line = line.replace(repo_str, "")
            # Cap individual lines at 500 chars (prevents base64/minified noise)
            if len(line) > 500:
                line = line[:500] + "..."
            lines.append(line)

        # Apply head limit
        if len(lines) > max_results * (1 + 2 * context_lines):
            lines = lines[:max_results * (1 + 2 * context_lines)]
            lines.append(f"... (truncated to {max_results} matches)")

        return _cap(f"Found matches for '{pattern}':\n" + "\n".join(lines))
    except subprocess.TimeoutExpired:
        return "ERROR: grep timed out"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool 2 — read_file
# ---------------------------------------------------------------------------

@tool
def read_file(file_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """
    Read a file with a 100-line viewer window. Shows line numbers for navigation.
    Use the code map in your context to find the right line numbers, then read
    the section you need. Call again with different start_line to scroll.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
        start_line: First line to read (1-indexed, default 1)
        end_line: Last line to read (0 = start_line + 100). Set explicitly for larger sections.
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

        # Default: 100-line window from start_line (SWE-agent proven design)
        s = max(0, start_line - 1)
        e = min(total, end_line) if end_line > 0 else min(total, s + 100)
        selected = lines[s:e]

        header = f"=== {file_path} (lines {s+1}-{e} of {total}) ===\n"
        numbered = "\n".join(f"{s+1+i:4d} | {ln}" for i, ln in enumerate(selected))
        footer = ""
        if e < total:
            footer = f"\n... [{total - e} more lines — read_file('{file_path}', start_line={e+1}) to scroll down]"
        if s > 0:
            prev_start = max(1, s - 99)
            footer = f"\n[scroll up: read_file('{file_path}', start_line={prev_start})]" + footer

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
        # Issue #11: Tree-sitter extraction for JS/TS (if parsers available)
        # -------------------------------------------------------------------
        if resolved.suffix in ('.js', '.jsx', '.ts', '.tsx'):
            try:
                extracted = _extract_function_treesitter_js(content, function_name, resolved.suffix)
                if extracted:
                    start, end = extracted
                    file_lines = content.splitlines(keepends=True)
                    numbered = "\n".join(
                        f"{start+1+i:4d} | {ln.rstrip()}"
                        for i, ln in enumerate(file_lines[start:end])
                    )
                    result = (
                        f"=== {function_name} in {file_path} "
                        f"(lines {start+1}-{end}) ===\n{numbered}"
                    )
                    return _redact_content(_cap(result))
            except Exception:
                pass  # Fall through to regex/brace-counting

        # -------------------------------------------------------------------
        # Brace-counting extraction for C-family languages (JS/TS/Go/Java/Rust/C/C++)
        # -------------------------------------------------------------------
        if resolved.suffix in ('.js', '.jsx', '.ts', '.tsx', '.go', '.java', '.rs', '.c', '.cpp', '.cs', '.kt', '.swift'):
            extracted = _extract_function_brace_counting(content, function_name)
            if extracted:
                start, end = extracted
                file_lines = content.split("\n")
                selected = file_lines[start:end]
                numbered = "\n".join(f"{start+1+i:4d} | {ln}" for i, ln in enumerate(selected))
                result = f"=== {function_name} in {file_path} (lines {start+1}-{end}) ===\n{numbered}"
                return _redact_content(_cap(result))

        # -------------------------------------------------------------------
        # Regex + indent heuristic fallback (Ruby, unknown languages)
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
# Tool 5 — get_function_info
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

        ext = resolved.suffix.lower()

        # Python patterns
        py_class_re = re.compile(r"^(\s*)(class\s+\w+[^:]*:)")
        py_func_re = re.compile(r"^(\s*)((?:async\s+)?def\s+\w+[^:]*:)")
        py_import_re = re.compile(r"^(?:import|from)\s+")
        # JS/TS patterns
        js_import_re = re.compile(r"^(?:import\s|const\s+\{.*\}\s*=\s*require|require\()")
        js_func_re = re.compile(r"^(\s*)((?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+\w+)")
        js_class_re = re.compile(r"^(\s*)((?:export\s+)?(?:default\s+)?class\s+\w+)")
        js_arrow_re = re.compile(r"^(\s*)((?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()")
        js_type_re = re.compile(r"^(\s*)((?:export\s+)?(?:interface|type|enum)\s+\w+)")
        # Go patterns
        go_func_re = re.compile(r"^(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+)")
        go_type_re = re.compile(r"^(type\s+\w+\s+(?:struct|interface))")
        # Rust patterns
        rs_func_re = re.compile(r"^(\s*)((?:pub\s+)?(?:async\s+)?fn\s+\w+)")
        rs_type_re = re.compile(r"^(\s*)((?:pub\s+)?(?:struct|enum|trait|impl)\s+)")

        for i, line in enumerate(lines):
            stripped = line.rstrip()
            lineno = i + 1

            # Imports (all languages)
            if ext == ".py" and py_import_re.match(stripped):
                imports.append(f"  {stripped}")
                continue
            if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs") and js_import_re.match(stripped):
                imports.append(f"  {stripped[:120]}")
                continue
            if ext == ".go" and stripped.startswith("import"):
                imports.append(f"  {stripped}")
                continue

            # Python structures
            if ext == ".py":
                m = py_class_re.match(stripped)
                if m:
                    structure_lines.append(f"L{lineno:4d}: {m.group(1)}{m.group(2)}")
                    continue
                m = py_func_re.match(stripped)
                if m:
                    docstring = ""
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            docstring = f"  # {next_line[:80]}"
                    structure_lines.append(f"L{lineno:4d}: {m.group(1)}{m.group(2)}{docstring}")
                    continue

            # JS/TS structures
            if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue"):
                for pat in (js_class_re, js_func_re, js_arrow_re, js_type_re):
                    m = pat.match(stripped)
                    if m:
                        structure_lines.append(f"L{lineno:4d}: {stripped.strip()[:120]}")
                        break
                else:
                    continue
                continue

            # Go structures
            if ext == ".go":
                for pat in (go_func_re, go_type_re):
                    m = pat.match(stripped)
                    if m:
                        structure_lines.append(f"L{lineno:4d}: {stripped.strip()}")
                        break
                continue

            # Rust structures
            if ext == ".rs":
                for pat in (rs_func_re, rs_type_re):
                    m = pat.match(stripped)
                    if m:
                        structure_lines.append(f"L{lineno:4d}: {stripped.strip()}")
                        break

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
    Check a source file for syntax errors after editing.
    Supports Python (.py), JavaScript (.js/.jsx/.mjs/.cjs), TypeScript (.ts/.tsx),
    and JSON (.json). Always run this after string_replace to verify your edit.

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

    ext = resolved.suffix.lower()

    # Python: AST parse
    if ext == ".py":
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

    # JavaScript: node --check (parse only, no execution)
    if ext in (".js", ".jsx", ".mjs", ".cjs"):
        try:
            result = subprocess.run(
                ["node", "--check", str(resolved)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return f"OK: {file_path} — no syntax errors (node --check)"
            else:
                return f"SYNTAX ERROR in {file_path}:\n{(result.stderr or result.stdout).strip()}"
        except FileNotFoundError:
            return f"OK: node not available — skipped syntax check for {file_path}"
        except subprocess.TimeoutExpired:
            return "ERROR: syntax check timed out"

    # TypeScript: tsc --noEmit or fallback to node parse
    if ext in (".ts", ".tsx"):
        # Try tsc first (type-aware)
        try:
            result = subprocess.run(
                ["npx", "-y", "tsc", "--noEmit", "--allowJs", "--esModuleInterop",
                 "--jsx", "react", str(resolved)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"OK: {file_path} — no TypeScript errors (tsc --noEmit)"
            # Filter to only errors from THIS file (tsc reports all deps)
            lines = [l for l in (result.stdout or "").splitlines()
                     if resolved.name in l]
            if lines:
                return f"TS ERROR in {file_path}:\n" + "\n".join(lines[:10])
            # No errors from our file — other deps may have issues, that's OK
            return f"OK: {file_path} — no syntax errors in this file"
        except FileNotFoundError:
            # No tsc — try basic node parse as fallback
            try:
                result = subprocess.run(
                    ["node", "-e", f"require('fs').readFileSync('{resolved}','utf8')"],
                    capture_output=True, text=True, timeout=10,
                )
                return f"OK: {file_path} — basic parse OK (tsc not available)"
            except Exception:
                return f"OK: skipped — no node/tsc available for {file_path}"
        except subprocess.TimeoutExpired:
            return "ERROR: TypeScript check timed out"

    # JSON: stdlib parse
    if ext == ".json":
        try:
            import json as _json
            _json.loads(resolved.read_text(encoding="utf-8"))
            return f"OK: {file_path} — valid JSON"
        except _json.JSONDecodeError as e:
            return f"JSON ERROR in {file_path}:\n{e}"

    return f"OK: syntax check not available for {ext} files (skipped {file_path})"


# ---------------------------------------------------------------------------
# Graph-native tools — query the knowledge graph directly (no grep needed)
# ---------------------------------------------------------------------------

@tool
def get_call_chain(function_name: str, depth: int = 2) -> str:
    """
    Get the call chain around a function: who calls it (callers) and what
    it calls (callees), up to the specified hop depth. Uses the pre-built
    knowledge graph — much faster than grep.

    Args:
        function_name: Function name to look up (e.g. 'verify_password')
        depth: Number of hops to traverse (default 2, max 3)
    """
    from agent.graph_utils import load_graph_data
    data_dir = getattr(_tls, 'data_dir', None)
    repo_name = getattr(_tls, 'repo_name', '')
    if not repo_name:
        return "ERROR: repo context not set"

    depth = min(int(depth), 3)
    graph_data, _ = load_graph_data(repo_name)
    nodes = {n.get("id", ""): n for n in graph_data.get("nodes", [])}
    edges = graph_data.get("edges", [])

    fn_lower = function_name.lower()
    seed_ids = {
        nid for nid, n in nodes.items()
        if (n.get("label") or nid.split("::")[-1]).lower() == fn_lower
    }
    if not seed_ids:
        # Partial match fallback
        seed_ids = set(list({
            nid for nid, n in nodes.items()
            if fn_lower in (n.get("label") or nid.split("::")[-1]).lower()
        })[:5])

    if not seed_ids:
        return f"No function named '{function_name}' found in graph."

    callers: list[str] = []
    callees: list[str] = []

    for hop in range(depth):
        current = set(seed_ids) if hop == 0 else set()
        for edge in edges:
            if edge.get("type") not in ("CALLS", "IMPORTS"):
                continue
            src, tgt = edge.get("source", ""), edge.get("target", "")
            if tgt in seed_ids:
                name = (nodes.get(src, {}).get("label") or src.split("::")[-1])
                file_ = nodes.get(src, {}).get("file", src.split("::")[0])
                ls = nodes.get(src, {}).get("line_start", "")
                entry = f"  {'  ' * hop}↑ {name} ({file_}" + (f":{ls}" if ls else "") + ")"
                if entry not in callers:
                    callers.append(entry)
            if src in seed_ids:
                name = (nodes.get(tgt, {}).get("label") or tgt.split("::")[-1])
                file_ = nodes.get(tgt, {}).get("file", tgt.split("::")[0])
                ls = nodes.get(tgt, {}).get("line_start", "")
                entry = f"  {'  ' * hop}↓ {name} ({file_}" + (f":{ls}" if ls else "") + ")"
                if entry not in callees:
                    callees.append(entry)

    lines = [f"Call chain for '{function_name}' (depth={depth}):"]
    if callers:
        lines.append("CALLERS (who calls this function):")
        lines.extend(callers[:15])
    else:
        lines.append("CALLERS: none found")
    if callees:
        lines.append("CALLEES (what this function calls):")
        lines.extend(callees[:15])
    else:
        lines.append("CALLEES: none found")
    return _cap("\n".join(lines))


@tool
def get_business_rules_for(function_name: str) -> str:
    """
    Get business rules and constraints linked to a function from the
    knowledge graph. Faster than grep — queries pre-extracted rules.

    Args:
        function_name: Function or file name to look up business rules for
    """
    data_dir = getattr(_tls, 'data_dir', None)
    repo_name = getattr(_tls, 'repo_name', '')
    if not repo_name:
        return "ERROR: repo context not set"

    from pathlib import Path as _Path
    import json as _json
    base = data_dir or _Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
    rules_path = base / repo_name / "business_rules.json"
    if not rules_path.exists():
        return f"No business_rules.json found for repo '{repo_name}'."

    try:
        all_rules = _json.loads(rules_path.read_text())
    except Exception as e:
        return f"ERROR reading business rules: {e}"

    fn_lower = function_name.lower()
    matches = [
        r for r in all_rules
        if fn_lower in r.get("file", "").lower()
        or fn_lower in r.get("function_id", "").lower()
        or fn_lower in r.get("description", "").lower()
    ]

    if not matches:
        return f"No business rules found for '{function_name}'. This may be a safe area to modify."

    lines = [f"Business rules for '{function_name}' ({len(matches)} found):"]
    for r in matches[:10]:
        sev = r.get("severity", "unknown").upper()
        desc = r.get("description", "")[:200]
        src = r.get("source", "")
        flag = " ⚠ DO NOT VIOLATE" if sev in ("CRITICAL", "HIGH") else ""
        lines.append(f"  [{sev}]{flag} {desc}")
        if src:
            lines.append(f"    Source: {src}")
    return _cap("\n".join(lines))


@tool
def get_failure_history(function_name: str) -> str:
    """
    Get past incidents and failure records linked to a function.
    Shows what went wrong here before — critical for understanding risk.

    Args:
        function_name: Function or file name to look up failure history for
    """
    data_dir = getattr(_tls, 'data_dir', None)
    repo_name = getattr(_tls, 'repo_name', '')
    if not repo_name:
        return "ERROR: repo context not set"

    # Try Neo4j first
    try:
        from graph.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            rows = neo4j_client.run(
                "MATCH (fr:FailureRecord)-[:RESULTED_IN_CHANGE]->(n) "
                "WHERE (n:Function OR n:File) "
                "  AND (n.name CONTAINS $fn OR n.path CONTAINS $fn) "
                "  AND fr.repo = $repo "
                "RETURN fr.message AS message, fr.date AS date, fr.issue_ref AS ref, "
                "       fr.severity_hint AS sev "
                "ORDER BY fr.date DESC LIMIT 10",
                {"fn": function_name, "repo": repo_name},
            )
            if rows:
                lines = [f"Failure history for '{function_name}' ({len(rows)} records):"]
                for row in rows:
                    ref = f" ({row['ref']})" if row.get("ref") else ""
                    sev = f"[{row['sev'].upper()}] " if row.get("sev") else ""
                    lines.append(f"  {sev}[{row.get('date', '?')}]{ref} {row.get('message', '')[:200]}")
                return _cap("\n".join(lines))
    except Exception:
        pass

    return f"No failure history found for '{function_name}' (Neo4j not connected or no records)."


@tool
def get_blast_radius(function_name: str) -> str:
    """
    Find all files that directly call or import a function. Use this before
    making changes to understand the impact — critical for avoiding regressions.

    Args:
        function_name: Function name to find callers for
    """
    from agent.graph_utils import load_graph_data
    repo_name = getattr(_tls, 'repo_name', '')
    if not repo_name:
        return "ERROR: repo context not set"

    graph_data, _ = load_graph_data(repo_name)
    nodes = {n.get("id", ""): n for n in graph_data.get("nodes", [])}
    edges = graph_data.get("edges", [])

    fn_lower = function_name.lower()
    seed_ids = {
        nid for nid, n in nodes.items()
        if fn_lower in (n.get("label") or nid.split("::")[-1]).lower()
    }

    callers: dict[str, int] = {}
    for edge in edges:
        if edge.get("type") not in ("CALLS", "IMPORTS"):
            continue
        if edge.get("target", "") in seed_ids:
            src_file = edge.get("source", "").split("::")[0]
            if src_file:
                callers[src_file] = callers.get(src_file, 0) + 1

    if not callers:
        return f"No callers found for '{function_name}'. Change has LOW blast radius."

    sorted_callers = sorted(callers.items(), key=lambda x: x[1], reverse=True)
    risk = "CRITICAL" if len(callers) > 8 else "HIGH" if len(callers) > 3 else "MEDIUM"
    lines = [
        f"Blast radius for '{function_name}': {len(callers)} callers — Risk: {risk}",
        "Files that call this function:",
    ]
    for f, count in sorted_callers[:15]:
        lines.append(f"  - {f} ({count} call{'s' if count > 1 else ''})")
    if len(callers) > 15:
        lines.append(f"  ... {len(callers) - 15} more files")
    lines.append("\nConsider whether your change will break any of these callers.")
    return _cap("\n".join(lines))


@tool
def screen_files(file_paths: str, bug_description: str) -> str:
    """
    Screen multiple files for relevance to a bug using fast parallel LLM calls
    (Minions pattern). Each file is independently rated for relevance.
    Use when you have 5+ candidate files and need to narrow down quickly.

    Args:
        file_paths: Comma-separated list of relative file paths to screen
        bug_description: Short description of the bug you're investigating
    """
    repo_path = getattr(_tls, 'repo_path', None)
    if not repo_path:
        return "ERROR: repo path not set"

    files = [f.strip() for f in file_paths.split(",") if f.strip()]
    if not files:
        return "ERROR: no files provided"
    if len(files) > 20:
        files = files[:20]

    import concurrent.futures
    from agent.llm import structured_call as _structured_call, INTAKE_MODEL
    from pydantic import BaseModel

    class FileRelevance(BaseModel):
        relevant: bool
        confidence: float  # 0.0-1.0
        reason: str        # one sentence

    def _screen_one(fpath: str) -> tuple[str, bool, float, str]:
        try:
            resolved = repo_path / fpath
            if not resolved.exists():
                return fpath, False, 0.0, "file not found"
            content = resolved.read_text(encoding="utf-8", errors="replace")[:3000]
            prompt = (
                f"Bug: {bug_description}\n\n"
                f"File: {fpath}\n```\n{content}\n```\n\n"
                "Is this file likely relevant to the bug? "
                "Answer with relevant=true/false, confidence 0.0-1.0, and one-sentence reason."
            )
            result = _structured_call(INTAKE_MODEL, 200, FileRelevance, prompt)
            return fpath, result.relevant, result.confidence, result.reason
        except Exception as e:
            return fpath, False, 0.0, f"error: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_screen_one, f): f for f in files}
        results = []
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    # Sort: relevant first, then by confidence
    results.sort(key=lambda x: (not x[1], -x[2]))

    lines = [f"File screening results ({len(files)} files screened for: '{bug_description[:60]}'):"]
    lines.append("")
    relevant_count = sum(1 for r in results if r[1])
    lines.append(f"RELEVANT ({relevant_count} files):")
    for fpath, rel, conf, reason in results:
        if rel:
            lines.append(f"  ✓ {fpath} (confidence: {conf:.0%}) — {reason}")
    if relevant_count == 0:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"NOT RELEVANT ({len(results) - relevant_count} files):")
    for fpath, rel, conf, reason in results:
        if not rel:
            lines.append(f"  ✗ {fpath} — {reason}")
    return _cap("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool 14 — search_subagent
# ---------------------------------------------------------------------------

@tool
def search_subagent(query: str, context: str = "") -> str:
    """Search the codebase using an isolated search specialist (Haiku). The search runs in its own context — your conversation stays clean. Returns file:line-range spans relevant to your query. Use this instead of multiple grep_repo calls when you need to find where something lives.

    Args:
        query: What to search for (function name, concept, pattern, etc.)
        context: Bug description or additional context about what you're looking for
    """
    # Lazy imports to avoid circular dependencies
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    except ImportError as e:
        return f"ERROR: langchain_anthropic not available — {e}. Use grep_repo instead."

    # Build the subset of tools available to the search subagent
    search_tools = [grep_repo, read_file, read_function, list_files]

    try:
        haiku = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
        ).bind_tools(search_tools)
    except Exception as e:
        return f"ERROR: Could not initialise Haiku model — {e}. Use grep_repo instead."

    # Build focused system prompt
    context_clause = f"\nContext: {context}" if context else ""
    system_prompt = (
        f"You are a search specialist. Find code relevant to: {query}.{context_clause}\n"
        "Return ONLY file:line-range spans. Be terse.\n"
        "Use grep_repo to find patterns, read_file/read_function to confirm, list_files to explore.\n"
        "When you have enough spans, output them as plain text like:\n"
        "  path/to/file.py:10-45\n"
        "  path/to/other.py:120-135\n"
        "Do not explain — just return the spans."
    )

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Find code spans for: {query}"),
    ]

    # Build a tool executor map
    tool_map = {t.name: t for t in search_tools}

    max_iterations = 8
    iteration = 0
    final_answer = ""

    try:
        while iteration < max_iterations:
            iteration += 1
            response = haiku.invoke(messages)
            messages.append(response)

            # Check if Haiku wants to call tools
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # No tool calls — this is the final answer
                final_answer = response.content if isinstance(response.content, str) else str(response.content)
                break

            # Execute each tool call and append results
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", f"call_{iteration}")

                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    tool_result = f"ERROR: unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                    except Exception as te:
                        tool_result = f"ERROR running {tool_name}: {te}"

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                )

        else:
            # Hit iteration limit — ask for final answer without tools
            messages.append(
                HumanMessage(content="You've used 8 tool calls. Return your final file:line-range spans now.")
            )
            final_response = haiku.invoke(messages)
            final_answer = (
                final_response.content
                if isinstance(final_response.content, str)
                else str(final_response.content)
            )

    except Exception as e:
        return f"Search subagent failed: {e}. Use grep_repo directly."

    if not final_answer or not final_answer.strip():
        return f"Search subagent found no results for: {query}"

    return f"Search results for '{query}':\n{final_answer.strip()}"


# ---------------------------------------------------------------------------
# Execution flow & change-impact tools
# ---------------------------------------------------------------------------


def _load_flows_data(repo_name: str) -> dict:
    """Load flows.json for a repo, returning empty dict on failure."""
    data_dir = getattr(_tls, "data_dir", Path(os.environ.get("DATA_DIR", "/tmp/context_builder")))
    flows_path = data_dir / repo_name / "flows.json"
    if flows_path.exists():
        return _load_json_cached(flows_path) or {}
    return {}


@tool
def get_execution_flows(filter_file: str = "", top_n: int = 5) -> str:
    """
    Get the most critical execution flows in the codebase. Shows how code is
    reached at runtime — API routes, CLI commands, background tasks, etc.
    Use this to understand which entry points touch the area you're investigating.

    Args:
        filter_file: Optional file path substring to filter flows (e.g. "auth" or "api/routes")
        top_n: Number of top flows to return (default: 5)
    """
    repo_name = getattr(_tls, "repo_name", "")
    if not repo_name:
        return "ERROR: repo context not set"

    flows_data = _load_flows_data(repo_name)
    flows = flows_data.get("flows", [])
    if not flows:
        return "No execution flows found. Run `cli.py build` to generate flows.json."

    if filter_file:
        fl = filter_file.lower()
        flows = [f for f in flows if any(fl in fp.lower() for fp in f.get("files", []))]

    flows = flows[:top_n]

    if not flows:
        return f"No flows matching '{filter_file}'."

    lines = [f"Top {len(flows)} execution flows (by criticality):"]
    for i, flow in enumerate(flows, 1):
        lines.append(
            f"\n{i}. {flow['name']}  (criticality: {flow['criticality']:.2f})"
        )
        lines.append(f"   Entry: {flow['entry_point']}")
        lines.append(f"   Depth: {flow['depth']}, Nodes: {flow['node_count']}, Files: {flow['file_count']}")
        files = flow.get("files", [])[:5]
        if files:
            lines.append(f"   Files: {', '.join(files)}")
            if len(flow.get("files", [])) > 5:
                lines.append(f"   ... +{len(flow['files']) - 5} more files")
        path_preview = flow.get("path", [])[:6]
        if path_preview:
            labels = [p.rsplit("::", 1)[-1] if "::" in p else p for p in path_preview]
            trail = " → ".join(labels)
            if len(flow.get("path", [])) > 6:
                trail += " → ..."
            lines.append(f"   Path: {trail}")

    lines.append(f"\nTotal: {flows_data.get('flow_count', '?')} flows, "
                 f"{flows_data.get('entry_point_count', '?')} entry points")
    return _cap("\n".join(lines))


@tool
def get_change_impact(base_ref: str = "HEAD~1") -> str:
    """
    Analyze current git changes and show which functions are affected, their
    risk scores, and which execution flows are impacted. Use after editing
    code to understand the blast radius of your changes.

    Args:
        base_ref: Git ref to diff against (default: HEAD~1)
    """
    repo_name = getattr(_tls, "repo_name", "")
    repo_path = getattr(_tls, "repo_path", None)
    if not repo_name or not repo_path:
        return "ERROR: repo context not set"

    from agent.graph_utils import load_graph_data
    from analyzer.changes import detect_changes

    graph_data, _ = load_graph_data(repo_name)
    if not graph_data:
        return "ERROR: graph.json not found. Run `cli.py build` first."

    flows_data = _load_flows_data(repo_name)

    result = detect_changes(
        repo_path=repo_path,
        graph_data=graph_data,
        flows_data=flows_data,
        base_ref=base_ref,
    )

    lines = [result["risk_summary"]["summary"]]

    scored = result.get("changed_nodes", [])
    if scored:
        lines.append("\nChanged functions (by risk):")
        for n in sorted(scored, key=lambda x: x["risk_score"], reverse=True)[:10]:
            lines.append(f"  {n['label']} ({n['file']}) — risk: {n['risk_score']:.2f}")

    affected = result.get("affected_flows", [])
    if affected:
        lines.append(f"\nAffected flows ({len(affected)}):")
        for f in affected[:5]:
            lines.append(f"  - {f['name']} (criticality: {f.get('criticality', 0):.2f})")

    gaps = result.get("test_gaps", [])
    if gaps:
        lines.append(f"\nTest gaps ({len(gaps)}):")
        for g in gaps[:5]:
            lines.append(f"  - {g['label']} ({g['file']})")

    return _cap("\n".join(lines))


@tool
def get_dead_code(file_path: str = "") -> str:
    """
    Find unreachable code — functions with no callers, no tests, and not entry
    points. Use during refactors to identify safe deletion candidates.

    Args:
        file_path: Optional file path substring to filter results (e.g. "utils" or "api/")
    """
    repo_name = getattr(_tls, "repo_name", "")
    if not repo_name:
        return "ERROR: repo context not set"

    flows_data = _load_flows_data(repo_name)
    dead_code = flows_data.get("dead_code", [])

    if not dead_code:
        if not flows_data:
            return "No flows.json found. Run `cli.py build` to generate it."
        return "No dead code detected — all functions are either called, tested, or entry points."

    if file_path:
        fp = file_path.lower()
        dead_code = [d for d in dead_code if fp in d.get("file", "").lower()]

    if not dead_code:
        return f"No dead code matching '{file_path}'."

    lines = [f"Dead code: {len(dead_code)} unreachable function(s)"]
    for d in dead_code[:20]:
        lines.append(f"  - {d['label']} ({d['file']}) [{d['type']}]")
    if len(dead_code) > 20:
        lines.append(f"  ... +{len(dead_code) - 20} more")
    lines.append("\nThese functions have no callers, no tests, and are not entry points.")
    return _cap("\n".join(lines))


# Exploration tools — READ-ONLY. The agent uses these to understand the codebase.
# string_replace and check_syntax are intentionally excluded: direct edits during
# exploration leave the main repo dirty, which blocks git worktree creation in the
# test node. Patching happens exclusively in the repair stage via the sandbox worktree.
ALL_TOOLS = [
    grep_repo,
    read_file,
    read_function,
    list_files,
    get_function_info,
    get_file_structure,
    get_blast_radius,
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
