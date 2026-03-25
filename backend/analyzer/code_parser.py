"""
code_parser.py — AST-based source analysis using tree-sitter-python.

Primary engine: tree-sitter (fast, incremental).
Fallback: stdlib ast module for docstrings / decorators that tree-sitter may miss.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language singleton
# ---------------------------------------------------------------------------
PY_LANGUAGE = Language(tspython.language())

# ---------------------------------------------------------------------------
# TypedDict-style plain dicts (no runtime dependency on typing_extensions)
# ---------------------------------------------------------------------------
# ImportInfo
#   module: str          — e.g. "os.path" or "collections"
#   names: list[str]     — imported names; empty for bare `import foo`
#   alias: str | None    — `import numpy as np` → alias = "np"
#   is_from: bool        — True for `from x import y`
#
# MethodInfo / FunctionInfo (identical shape)
#   name, line_start, line_end, params, return_type, docstring, decorators
#
# ClassInfo
#   name, line_start, line_end, methods, bases, docstring, decorators
#
# ParsedFile
#   path, abs_path, language, classes, functions, imports, docstring, loc


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_text(node: Optional[Node], source: bytes) -> Optional[str]:
    if node is None:
        return None
    return _text(node, source)


# ---------------------------------------------------------------------------
# Docstring helpers
# ---------------------------------------------------------------------------

def _extract_docstring_ts(body_node: Node, source: bytes) -> Optional[str]:
    """Return the docstring from a body block using tree-sitter nodes."""
    if body_node is None:
        return None
    for child in body_node.children:
        if child.type == "expression_statement":
            inner = child.children[0] if child.children else None
            if inner and inner.type == "string":
                raw = _text(inner, source)
                return _clean_docstring(raw)
        # Skip decorators / comments at the top, stop at first real statement
        if child.type not in (
            "comment", "decorator", "expression_statement", "string"
        ):
            break
    return None


def _clean_docstring(raw: str) -> str:
    """Strip surrounding triple/single quotes and dedent."""
    import textwrap
    s = raw.strip()
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            s = s[len(q): len(s) - len(q)]
            break
    return textwrap.dedent(s).strip()


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------

def _parse_imports(tree_root: Node, source: bytes) -> list[dict]:
    imports: list[dict] = []

    def walk(node: Node) -> None:
        if node.type == "import_statement":
            # import foo, import foo as bar, import a.b.c
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(
                        {
                            "module": _text(child, source),
                            "names": [],
                            "alias": None,
                            "is_from": False,
                        }
                    )
                elif child.type == "aliased_import":
                    # aliased_import: dotted_name "as" identifier
                    parts = [c for c in child.children if c.type in ("dotted_name", "identifier")]
                    if len(parts) >= 2:
                        imports.append(
                            {
                                "module": _text(parts[0], source),
                                "names": [],
                                "alias": _text(parts[-1], source),
                                "is_from": False,
                            }
                        )
                    elif len(parts) == 1:
                        imports.append(
                            {
                                "module": _text(parts[0], source),
                                "names": [],
                                "alias": None,
                                "is_from": False,
                            }
                        )

        elif node.type == "import_from_statement":
            # from foo import bar, baz
            module = ""
            names: list[str] = []
            alias: Optional[str] = None
            past_import_keyword = False

            children = node.children
            # structure: "from" module_name "import" name1, name2, ...
            idx = 0
            while idx < len(children):
                c = children[idx]
                if c.type == "import":
                    past_import_keyword = True
                elif not past_import_keyword and c.type in ("dotted_name", "relative_import"):
                    # Before "import" keyword: this is the source module
                    module = _text(c, source)
                elif past_import_keyword and c.type in ("dotted_name", "identifier"):
                    # After "import" keyword: these are imported names
                    names.append(_text(c, source))
                elif c.type == "wildcard_import":
                    names.append("*")
                elif c.type == "aliased_import":
                    parts = [ch for ch in c.children if ch.type in ("dotted_name", "identifier")]
                    if len(parts) >= 2:
                        names.append(_text(parts[0], source))
                        alias = _text(parts[-1], source)
                    elif parts:
                        names.append(_text(parts[0], source))
                idx += 1

            imports.append(
                {
                    "module": module,
                    "names": names,
                    "alias": alias,
                    "is_from": True,
                }
            )

        # Recurse into module-level nodes only (avoid descending into function bodies)
        if node.type in ("module", "block") or node == tree_root:
            for child in node.children:
                if child.type not in (
                    "function_definition",
                    "class_definition",
                    "decorated_definition",
                ):
                    walk(child)

    walk(tree_root)
    return imports


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def _extract_decorators(node: Node, source: bytes) -> list[str]:
    """
    Given a function_definition or class_definition node, look at the
    preceding siblings in the parent for decorator nodes.  tree-sitter wraps
    decorated definitions in a `decorated_definition` node.
    """
    parent = node.parent
    if parent is None:
        return []
    decorators: list[str] = []
    if parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                decorators.append(_text(child, source).lstrip("@").strip())
    return decorators


# ---------------------------------------------------------------------------
# Parameter / return-type helpers
# ---------------------------------------------------------------------------

def _extract_params(params_node: Optional[Node], source: bytes) -> list[str]:
    if params_node is None:
        return []
    params: list[str] = []
    for child in params_node.children:
        if child.type in (
            "identifier",
            "typed_parameter",
            "default_parameter",
            "typed_default_parameter",
            "list_splat_pattern",   # *args
            "dictionary_splat_pattern",  # **kwargs
        ):
            params.append(_text(child, source))
        elif child.type in ("*", "**", ",", "(", ")"):
            continue
    return params


def _extract_return_type(func_node: Node, source: bytes) -> Optional[str]:
    """Return annotation if present (→ after def)."""
    # tree-sitter represents `-> ReturnType` as a `type` child preceded by `->`
    found_arrow = False
    for child in func_node.children:
        if child.type == "->":
            found_arrow = True
        elif found_arrow and child.type == "type":
            return _text(child, source)
    return None


# ---------------------------------------------------------------------------
# Conditionals & complexity extraction
# ---------------------------------------------------------------------------

_CONSTANT_RE = re.compile(
    r"^[A-Z][A-Z0-9_]*_(?:LIMIT|MAX|MIN|TIMEOUT|RATE|THRESHOLD|CAP|QUOTA|"
    r"WINDOW|PERIOD|SIZE|COUNT|RETRIES|ATTEMPTS|DELAY|INTERVAL|AGE|DAYS|HOURS)$"
)


def _walk_descendants(node: Node):
    """Yield all descendants of a tree-sitter node (depth-first)."""
    for child in node.children:
        yield child
        yield from _walk_descendants(child)


def _extract_conditionals(body_node: Optional[Node], source: bytes) -> list[dict]:
    """Extract if/elif conditionals from a function body using tree-sitter AST."""
    if body_node is None:
        return []
    conditionals: list[dict] = []

    for node in _walk_descendants(body_node):
        if node.type != "if_statement":
            continue

        # Extract the condition expression (first child after 'if' keyword)
        condition_text = ""
        for child in node.children:
            if child.type in (
                "comparison_operator", "boolean_operator", "not_operator",
                "identifier", "attribute", "call", "parenthesized_expression",
                "binary_operator", "unary_operator",
            ):
                condition_text = _text(child, source)
                break

        if not condition_text:
            # Fallback: grab text between 'if' and ':'
            full = _text(node, source)
            if_idx = full.find("if ")
            colon_idx = full.find(":")
            if if_idx != -1 and colon_idx > if_idx:
                condition_text = full[if_idx + 3:colon_idx].strip()

        # Count branches (elif + else)
        branch_count = 1  # the if itself
        for child in node.children:
            if child.type == "elif_clause":
                branch_count += 1
            elif child.type == "else_clause":
                branch_count += 1

        # Check if condition references a known constant pattern
        refs_constant = bool(_CONSTANT_RE.search(condition_text))

        conditionals.append({
            "line": node.start_point[0] + 1,
            "condition_text": condition_text[:200],  # cap length
            "branch_count": branch_count,
            "references_constant": refs_constant,
        })

    return conditionals


def _compute_complexity(body_node: Optional[Node]) -> int:
    """Simple cyclomatic complexity approximation: count if/elif/for/while/except + 1."""
    if body_node is None:
        return 1
    branches = 0
    for node in _walk_descendants(body_node):
        if node.type in (
            "if_statement", "elif_clause", "for_statement", "while_statement",
            "except_clause", "with_statement", "assert_statement",
        ):
            branches += 1
    return branches + 1


# ---------------------------------------------------------------------------
# Method / function parser
# ---------------------------------------------------------------------------

def _parse_function(func_node: Node, source: bytes) -> dict:
    name = ""
    params_node: Optional[Node] = None

    for child in func_node.children:
        if child.type == "identifier" and not name:
            name = _text(child, source)
        elif child.type == "parameters":
            params_node = child

    body_node: Optional[Node] = None
    for child in func_node.children:
        if child.type == "block":
            body_node = child
            break

    docstring = _extract_docstring_ts(body_node, source) if body_node else None
    params = _extract_params(params_node, source)
    return_type = _extract_return_type(func_node, source)
    decorators = _extract_decorators(func_node, source)
    conditionals = _extract_conditionals(body_node, source)
    complexity = _compute_complexity(body_node)

    return {
        "name": name,
        "line_start": func_node.start_point[0] + 1,
        "line_end": func_node.end_point[0] + 1,
        "params": params,
        "return_type": return_type,
        "docstring": docstring,
        "decorators": decorators,
        "conditionals": conditionals,
        "complexity": complexity,
    }


# ---------------------------------------------------------------------------
# Class parser
# ---------------------------------------------------------------------------

def _extract_base_classes(class_node: Node, source: bytes) -> list[str]:
    bases: list[str] = []
    for child in class_node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type in ("identifier", "dotted_name", "attribute"):
                    bases.append(_text(arg, source))
    return bases


def _parse_class(class_node: Node, source: bytes) -> dict:
    name = ""
    for child in class_node.children:
        if child.type == "identifier" and not name:
            name = _text(child, source)

    bases = _extract_base_classes(class_node, source)
    decorators = _extract_decorators(class_node, source)

    body_node: Optional[Node] = None
    for child in class_node.children:
        if child.type == "block":
            body_node = child
            break

    docstring = _extract_docstring_ts(body_node, source) if body_node else None
    methods: list[dict] = []

    if body_node:
        for child in body_node.children:
            actual = child
            if child.type == "decorated_definition":
                # unwrap to find the function_definition
                for sub in child.children:
                    if sub.type == "function_definition":
                        actual = sub
                        break
            if actual.type == "function_definition":
                methods.append(_parse_function(actual, source))

    return {
        "name": name,
        "line_start": class_node.start_point[0] + 1,
        "line_end": class_node.end_point[0] + 1,
        "methods": methods,
        "bases": bases,
        "docstring": docstring,
        "decorators": decorators,
    }


# ---------------------------------------------------------------------------
# AST stdlib fallback
# ---------------------------------------------------------------------------

def _ast_fallback_docstring(source_code: str, node_type: str, name: str) -> Optional[str]:
    """
    Use stdlib ast to find the docstring for a module / class / function by name.
    node_type: "module" | "class" | "function"
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    if node_type == "module":
        return ast.get_docstring(tree)

    for node in ast.walk(tree):
        if node_type == "class" and isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_docstring(node)
        if node_type == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node)
    return None


def _ast_fallback_decorators(source_code: str, node_type: str, name: str) -> list[str]:
    """Use stdlib ast to extract decorator names."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if node_type == "class" and isinstance(node, ast.ClassDef) and node.name == name:
            return [ast.unparse(d) for d in node.decorator_list]
        if node_type == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return [ast.unparse(d) for d in node.decorator_list]
    return []


# ---------------------------------------------------------------------------
# Main CodeParser
# ---------------------------------------------------------------------------

class CodeParser:
    """
    Parse all Python files under *repo_path* using tree-sitter-python.
    Falls back to stdlib ast for docstrings / decorators when tree-sitter
    returns None.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._parser = Parser(PY_LANGUAGE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    MAX_FILE_SIZE_BYTES = 512 * 1024  # skip files > 512KB
    SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build", ".tox", "migrations", "alembic"}

    JS_TS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

    def parse_all(self) -> list[dict]:
        results: list[dict] = []

        # Collect all candidate files
        candidates: list[Path] = []
        for f in sorted(self.repo_path.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix not in (".py", *self.JS_TS_EXTENSIONS):
                continue
            if any(part in self.SKIP_DIRS for part in f.parts):
                continue
            try:
                if f.stat().st_size > self.MAX_FILE_SIZE_BYTES:
                    logger.debug("Skipping large file: %s", f)
                    continue
            except OSError:
                continue
            candidates.append(f)

        for src_file in candidates:
            try:
                if src_file.suffix == ".py":
                    parsed = self._parse_file(src_file)
                else:
                    parsed = self._parse_js_ts_file(src_file)
                if parsed is not None:
                    results.append(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse %s: %s", src_file, exc)
        return results

    # ------------------------------------------------------------------
    # JS/TS lightweight regex parser
    # ------------------------------------------------------------------

    _JS_FUNC_RE = __import__("re").compile(
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    )
    _JS_ARROW_RE = __import__("re").compile(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+?)?\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=>\n]+?)?\s*=>",
    )
    _JS_CLASS_RE = __import__("re").compile(
        r"(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
    )
    _JS_METHOD_RE = __import__("re").compile(
        r"^\s+(?:async\s+)?(\w+)\s*\(([^)]*)\)",
    )
    _JS_CONTROL_FLOW = frozenset({
        "if", "for", "while", "switch", "catch", "constructor",
        "else", "try", "finally", "do", "return", "throw",
        "case", "default", "new", "typeof", "import", "export",
    })
    _JS_IMPORT_RE = __import__("re").compile(
        r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
    )
    _JS_JSDOC_RE = __import__("re").compile(
        r"/\*\*\s*(.*?)\s*\*/", __import__("re").DOTALL,
    )

    def _parse_js_ts_file(self, abs_path: Path) -> Optional[dict]:
        import re
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        rel_path = str(abs_path.relative_to(self.repo_path))
        lines = source.splitlines()
        loc = len(lines)

        # Extract imports
        imports = []
        for m in self._JS_IMPORT_RE.finditer(source):
            imports.append({"module": m.group(1), "names": [], "alias": None})

        # Extract JSDoc comments by line number
        jsdocs: dict[int, str] = {}
        for m in self._JS_JSDOC_RE.finditer(source):
            end_line = source[:m.end()].count("\n") + 1
            text = m.group(1).strip()
            # Clean up JSDoc: remove leading * from each line
            cleaned = "\n".join(
                line.strip().lstrip("* ").strip()
                for line in text.split("\n")
            ).strip()
            jsdocs[end_line] = cleaned

        def _find_jsdoc(lineno: int) -> Optional[str]:
            # JSDoc appears on the line(s) just before the definition
            for offset in range(1, 5):
                if (lineno - offset) in jsdocs:
                    return jsdocs[lineno - offset]
            return None

        # Extract functions
        functions: list[dict] = []
        for pattern in (self._JS_FUNC_RE, self._JS_ARROW_RE):
            for m in pattern.finditer(source):
                lineno = source[:m.start()].count("\n") + 1
                name = m.group(1)
                params = [p.strip().split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
                functions.append({
                    "name": name,
                    "id": f"{rel_path}::{name}",
                    "params": params,
                    "return_type": None,
                    "docstring": _find_jsdoc(lineno),
                    "lineno": lineno,
                    "decorators": [],
                    "calls": [],
                })

        # Extract classes
        classes: list[dict] = []
        for m in self._JS_CLASS_RE.finditer(source):
            lineno = source[:m.start()].count("\n") + 1
            cls_name = m.group(1)
            base = m.group(2)
            # Find methods within the class body (rough heuristic)
            methods: list[dict] = []
            # Look for method definitions in lines after class declaration
            brace_count = 0
            in_class = False
            for i, line in enumerate(lines[lineno - 1:], start=lineno):
                if "{" in line:
                    brace_count += line.count("{")
                    in_class = True
                if "}" in line:
                    brace_count -= line.count("}")
                if in_class and brace_count <= 0:
                    break
                method_match = self._JS_METHOD_RE.match(line)
                if method_match and method_match.group(1) not in self._JS_CONTROL_FLOW:
                    mname = method_match.group(1)
                    mparams = [p.strip().split(":")[0].strip() for p in method_match.group(2).split(",") if p.strip()]
                    methods.append({
                        "name": mname,
                        "id": f"{rel_path}::{cls_name}.{mname}",
                        "params": mparams,
                        "return_type": None,
                        "docstring": _find_jsdoc(i),
                        "lineno": i,
                        "decorators": [],
                        "calls": [],
                    })
            classes.append({
                "name": cls_name,
                "id": f"{rel_path}::{cls_name}",
                "bases": [base] if base else [],
                "docstring": _find_jsdoc(lineno),
                "lineno": lineno,
                "decorators": [],
                "methods": methods,
            })

        # Module-level docstring: first JSDoc or first comment block
        module_doc = jsdocs.get(1) or jsdocs.get(2) or None

        file_id = f"{rel_path}"
        return {
            "id": file_id,
            "path": rel_path,
            "abs_path": str(abs_path),
            "language": "typescript" if abs_path.suffix in (".ts", ".tsx") else "javascript",
            "loc": loc,
            "docstring": module_doc,
            "imports": imports,
            "classes": classes,
            "functions": functions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_file(self, abs_path: Path) -> Optional[dict]:
        try:
            source_bytes = abs_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", abs_path, exc)
            return None

        source_str = source_bytes.decode("utf-8", errors="replace")

        try:
            tree = self._parser.parse(source_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tree-sitter failed on %s: %s", abs_path, exc)
            return None

        root = tree.root_node
        if root.has_error:
            logger.debug("tree-sitter reported parse errors in %s (continuing)", abs_path)

        rel_path = str(abs_path.relative_to(self.repo_path))

        # --- module docstring ---
        module_docstring: Optional[str] = None
        for child in root.children:
            if child.type == "expression_statement":
                inner = child.children[0] if child.children else None
                if inner and inner.type == "string":
                    module_docstring = _clean_docstring(_text(inner, source_bytes))
                    break
            if child.type not in ("comment", "expression_statement", "string"):
                break

        # Fallback for module docstring
        if module_docstring is None:
            module_docstring = _ast_fallback_docstring(source_str, "module", "")

        # --- imports ---
        imports = _parse_imports(root, source_bytes)

        # --- top-level classes and functions ---
        classes: list[dict] = []
        functions: list[dict] = []

        for child in root.children:
            actual = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type in ("function_definition", "class_definition"):
                        actual = sub
                        break

            if actual.type == "class_definition":
                cls_info = _parse_class(actual, source_bytes)
                # fallback docstring
                if cls_info["docstring"] is None:
                    cls_info["docstring"] = _ast_fallback_docstring(
                        source_str, "class", cls_info["name"]
                    )
                # fallback decorators
                if not cls_info["decorators"]:
                    cls_info["decorators"] = _ast_fallback_decorators(
                        source_str, "class", cls_info["name"]
                    )
                # fallback for method docstrings / decorators
                for method in cls_info["methods"]:
                    if method["docstring"] is None:
                        method["docstring"] = _ast_fallback_docstring(
                            source_str, "function", method["name"]
                        )
                    if not method["decorators"]:
                        method["decorators"] = _ast_fallback_decorators(
                            source_str, "function", method["name"]
                        )
                classes.append(cls_info)

            elif actual.type == "function_definition":
                fn_info = _parse_function(actual, source_bytes)
                if fn_info["docstring"] is None:
                    fn_info["docstring"] = _ast_fallback_docstring(
                        source_str, "function", fn_info["name"]
                    )
                if not fn_info["decorators"]:
                    fn_info["decorators"] = _ast_fallback_decorators(
                        source_str, "function", fn_info["name"]
                    )
                functions.append(fn_info)

        loc = len(source_str.splitlines())

        return {
            "path": rel_path,
            "abs_path": str(abs_path),
            "language": "python",
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "docstring": module_docstring,
            "loc": loc,
        }
