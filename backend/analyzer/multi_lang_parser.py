"""
multi_lang_parser.py — Tree-sitter based parsing for JavaScript, TypeScript, Go, and Java.

Provides AST-level parsing when the corresponding tree-sitter language packages
are installed. Falls back gracefully to the regex-based parser in code_parser.py
when packages are missing.

Install language packages as needed:
    pip install tree-sitter-javascript tree-sitter-typescript
    pip install tree-sitter-go tree-sitter-java

These parsers feed into the same enriched node pipeline as the Python parser,
enabling the knowledge graph to cover multi-language repos.

JS/TS feature parity with Python parser (code_parser.py):
  - Decorators (TypeScript/NestJS: @Controller, @Get, @Injectable, etc.)
  - Return types (string or None, e.g. "Promise<User>", "string")
  - Param types (full typed params, e.g. ["id: number", "name: string"])
  - Conditionals ([{line, condition_text, branch_count, references_constant}])
  - Cyclomatic complexity count
  - Accurate line_start / line_end via tree-sitter end_point
  - Import symbols (names: ["UserService", "AuthGuard"])
  - Import aliases (alias: "US" for `import { UserService as US }`)
  - Implements clause (TypeScript `implements Interface1, Interface2`)
  - Async flag (is_async: True/False)
  - JSDoc (prev_named_sibling and parent's prev sibling for export-wrapped nodes)
  - Export tracking (is_exported: True/False)
  - React components (is_react_component: True if uppercase name + returns JSX)
  - Hooks (is_hook: True if name starts with "use")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy language loading — only imports what's installed
# ---------------------------------------------------------------------------

_LANGUAGE_CACHE: dict[str, object] = {}
_PARSER_CACHE: dict[str, object] = {}
_AVAILABLE_LANGUAGES: dict[str, bool] = {}


def _get_language(lang_key: str):
    """Load a tree-sitter Language object, or return None if not installed."""
    if lang_key in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_key]

    from tree_sitter import Language

    module_map = {
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "go": "tree_sitter_go",
        "java": "tree_sitter_java",
    }
    module_name = module_map.get(lang_key)
    if not module_name:
        _LANGUAGE_CACHE[lang_key] = None
        return None

    try:
        mod = __import__(module_name)
        # typescript module exposes language_typescript() and language_tsx()
        if lang_key == "typescript" and hasattr(mod, "language_typescript"):
            lang = Language(mod.language_typescript())
        else:
            lang = Language(mod.language())
        _LANGUAGE_CACHE[lang_key] = lang
        _AVAILABLE_LANGUAGES[lang_key] = True
        logger.info("Loaded tree-sitter language: %s", lang_key)
        return lang
    except (ImportError, Exception) as e:
        logger.debug("tree-sitter-%s not available: %s", lang_key, e)
        _LANGUAGE_CACHE[lang_key] = None
        _AVAILABLE_LANGUAGES[lang_key] = False
        return None


def _get_parser(lang_key: str):
    """Get or create a tree-sitter Parser for the given language."""
    if lang_key in _PARSER_CACHE:
        return _PARSER_CACHE[lang_key]

    lang = _get_language(lang_key)
    if lang is None:
        return None

    from tree_sitter import Parser
    parser = Parser(lang)
    _PARSER_CACHE[lang_key] = parser
    return parser


def get_available_languages() -> list[str]:
    """Return list of languages with tree-sitter parsers installed."""
    for lang in ("javascript", "typescript", "go", "java"):
        if lang not in _AVAILABLE_LANGUAGES:
            _get_language(lang)
    return [k for k, v in _AVAILABLE_LANGUAGES.items() if v]


# ---------------------------------------------------------------------------
# Suffix → language mapping
# ---------------------------------------------------------------------------

EXTENSION_TO_LANG: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
}

# Per-extension human-readable language label (finer than EXTENSION_TO_LANG)
EXTENSION_TO_LABEL: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
}

# Additional extensions the CodeParser should collect
SUPPORTED_EXTENSIONS = set(EXTENSION_TO_LANG.keys()) | {".py"}

# Directories to skip during parsing
SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "dist", "build", ".tox", "migrations", "alembic",
    "vendor", "target", "bin", "obj",
})


# ---------------------------------------------------------------------------
# Generic node text helper
# ---------------------------------------------------------------------------

def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_by_type(node, type_name: str):
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _child_by_field(node, field_name: str):
    return node.child_by_field_name(field_name)


def _children_by_type(node, type_name: str) -> list:
    return [c for c in node.children if c.type == type_name]


def _walk_descendants(node):
    """Yield all descendants of a node, depth-first."""
    for child in node.children:
        yield child
        yield from _walk_descendants(child)


# ---------------------------------------------------------------------------
# Constant reference pattern (same as Python parser)
# ---------------------------------------------------------------------------

_CONSTANT_RE = re.compile(
    r"[A-Z][A-Z0-9_]*_(?:LIMIT|MAX|MIN|TIMEOUT|RATE|THRESHOLD|CAP|QUOTA|"
    r"WINDOW|PERIOD|SIZE|COUNT|RETRIES|ATTEMPTS|DELAY|INTERVAL|AGE|DAYS|HOURS)"
)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript parser
# ---------------------------------------------------------------------------

def parse_js_ts_treesitter(abs_path: Path, repo_path: Path) -> Optional[dict]:
    """Parse a JS/TS file using tree-sitter. Returns ParsedFile dict or None."""
    lang_key = EXTENSION_TO_LANG.get(abs_path.suffix)
    if not lang_key:
        return None

    parser = _get_parser(lang_key)
    if parser is None:
        return None

    try:
        source = abs_path.read_bytes()
    except OSError:
        return None

    source_str = source.decode("utf-8", errors="replace")
    rel_path = str(abs_path.relative_to(repo_path))

    try:
        tree = parser.parse(source)
    except Exception as e:
        logger.debug("tree-sitter parse failed for %s: %s", rel_path, e)
        return None

    root = tree.root_node
    functions: list[dict] = []
    classes: list[dict] = []
    imports: list[dict] = []

    # ------------------------------------------------------------------
    # JSDoc extraction
    # ------------------------------------------------------------------

    def _extract_jsdoc(node) -> Optional[str]:
        """Extract JSDoc comment preceding a node.

        Checks:
          1. node.prev_named_sibling (direct sibling — most common)
          2. node.parent.prev_named_sibling (for export-wrapped nodes like
             `export function foo()` where the JSDoc sits before `export`)
        """
        candidates = [node.prev_named_sibling]
        if node.parent is not None:
            candidates.append(node.parent.prev_named_sibling)

        for prev in candidates:
            if prev is None:
                continue
            if prev.type == "comment":
                text = _text(prev, source)
                if text.startswith("/**"):
                    cleaned = re.sub(r"^/\*\*|\*/$", "", text)
                    cleaned = re.sub(r"^\s*\*\s?", "", cleaned, flags=re.MULTILINE)
                    return cleaned.strip()
        return None

    # ------------------------------------------------------------------
    # Parameter extraction — full typed params
    # ------------------------------------------------------------------

    def _extract_params(params_node) -> list[str]:
        """Extract parameters preserving type annotations.

        Returns strings like:
          ["id: number", "name: string", "opts?: Options", "...rest: T[]"]
        instead of just bare names.
        """
        if not params_node:
            return []
        result: list[str] = []
        for child in params_node.children:
            if child.type in (",", "(", ")"):
                continue
            if child.type in (
                "required_parameter",
                "optional_parameter",
                "rest_parameter",
                "assignment_pattern",
            ):
                # Full text preserves `: type` annotation
                result.append(_text(child, source))
            elif child.type == "identifier":
                result.append(_text(child, source))
            # object/array destructuring patterns — keep full text
            elif child.type in ("object_pattern", "array_pattern"):
                result.append(_text(child, source))
        return result

    # ------------------------------------------------------------------
    # Return type extraction
    # ------------------------------------------------------------------

    def _extract_return_type(func_node) -> Optional[str]:
        """Extract TypeScript return type annotation (`: ReturnType`)."""
        # tree-sitter-typescript: return type is in a `type_annotation` child
        # after the parameter list.  field name is "return_type".
        rt = _child_by_field(func_node, "return_type")
        if rt:
            # type_annotation node text is `: string` — strip leading colon
            text = _text(rt, source).lstrip(":").strip()
            return text if text else None
        return None

    # ------------------------------------------------------------------
    # Decorator extraction (TypeScript/NestJS)
    # ------------------------------------------------------------------

    def _extract_decorators(node) -> list[str]:
        """Collect decorator nodes preceding a class or function.

        In tree-sitter-typescript, decorators appear as siblings BEFORE the
        class/function node inside the same parent, or as children of an
        `export_statement` that also contains the decorated node.
        """
        decorators: list[str] = []
        parent = node.parent
        if parent is None:
            return decorators

        # Walk backwards through parent's children to find decorator siblings
        found_self = False
        for child in reversed(list(parent.children)):
            if child is node:
                found_self = True
                continue
            if not found_self:
                continue
            if child.type == "decorator":
                decorators.insert(0, _text(child, source))
            else:
                # Stop at the first non-decorator before the node
                break

        return decorators

    # ------------------------------------------------------------------
    # Async flag
    # ------------------------------------------------------------------

    def _is_async(func_node) -> bool:
        """Return True if the function/arrow has an `async` keyword child."""
        for child in func_node.children:
            if child.type == "async":
                return True
        return False

    # ------------------------------------------------------------------
    # Export flag
    # ------------------------------------------------------------------

    def _is_exported(node) -> bool:
        """Return True if the node's immediate parent is an export_statement."""
        parent = node.parent
        if parent is None:
            return False
        return parent.type in ("export_statement",)

    # ------------------------------------------------------------------
    # React component / hook detection
    # ------------------------------------------------------------------

    def _is_react_component(name: str, func_node) -> bool:
        """Heuristic: uppercase name + body contains JSX element."""
        if not name or not name[0].isupper():
            return False
        # Look for jsx_element or jsx_self_closing_element in the body
        body = _child_by_field(func_node, "body")
        if body is None:
            return False
        for desc in _walk_descendants(body):
            if desc.type in ("jsx_element", "jsx_self_closing_element"):
                return True
        return False

    def _is_hook(name: str) -> bool:
        return bool(name) and name.startswith("use") and len(name) > 3 and name[3].isupper()

    # ------------------------------------------------------------------
    # Conditionals extraction
    # ------------------------------------------------------------------

    def _extract_conditionals(body_node) -> list[dict]:
        """Extract if/switch/ternary conditionals from a function body."""
        if body_node is None:
            return []
        conditionals: list[dict] = []

        for node in _walk_descendants(body_node):
            if node.type == "if_statement":
                condition_node = _child_by_field(node, "condition")
                condition_text = _text(condition_node, source) if condition_node else ""
                # Strip outer parentheses if present
                if condition_text.startswith("(") and condition_text.endswith(")"):
                    condition_text = condition_text[1:-1].strip()

                # Count branches: if + elif(else if) + else
                branch_count = 1
                alt = _child_by_field(node, "alternative")
                while alt is not None:
                    branch_count += 1
                    # else-if chain: alternative is an if_statement
                    if alt.type == "else_clause":
                        inner = None
                        for c in alt.children:
                            if c.type == "if_statement":
                                inner = c
                                break
                        if inner:
                            alt = _child_by_field(inner, "alternative")
                        else:
                            break
                    else:
                        break

                refs_constant = bool(_CONSTANT_RE.search(condition_text))
                conditionals.append({
                    "line": node.start_point[0] + 1,
                    "condition_text": condition_text[:200],
                    "branch_count": branch_count,
                    "references_constant": refs_constant,
                })

            elif node.type == "switch_statement":
                condition_node = _child_by_field(node, "value")
                condition_text = _text(condition_node, source) if condition_node else ""
                # Count case/default clauses
                body = _child_by_field(node, "body")
                branch_count = 0
                if body:
                    for c in body.children:
                        if c.type in ("switch_case", "switch_default"):
                            branch_count += 1
                branch_count = max(branch_count, 1)
                refs_constant = bool(_CONSTANT_RE.search(condition_text))
                conditionals.append({
                    "line": node.start_point[0] + 1,
                    "condition_text": condition_text[:200],
                    "branch_count": branch_count,
                    "references_constant": refs_constant,
                })

            elif node.type == "ternary_expression":
                condition_node = _child_by_field(node, "condition")
                if condition_node is None:
                    # fallback: first child before `?`
                    for c in node.children:
                        if c.type == "?":
                            break
                        condition_node = c
                condition_text = _text(condition_node, source) if condition_node else ""
                refs_constant = bool(_CONSTANT_RE.search(condition_text))
                conditionals.append({
                    "line": node.start_point[0] + 1,
                    "condition_text": condition_text[:200],
                    "branch_count": 2,  # ternary always has two branches
                    "references_constant": refs_constant,
                })

        return conditionals

    # ------------------------------------------------------------------
    # Cyclomatic complexity
    # ------------------------------------------------------------------

    def _compute_complexity(body_node) -> int:
        """Cyclomatic complexity: count decision points + 1."""
        if body_node is None:
            return 1
        count = 0
        for node in _walk_descendants(body_node):
            if node.type in (
                "if_statement",
                "else_clause",           # each else adds a branch
                "for_statement",
                "for_in_statement",
                "while_statement",
                "do_statement",
                "switch_case",
                "catch_clause",
                "ternary_expression",
                "logical_expression",    # && / || each add a branch
            ):
                count += 1
        return count + 1

    # ------------------------------------------------------------------
    # Import parsing
    # ------------------------------------------------------------------

    def _parse_import(node) -> list[dict]:
        """Parse an import_statement node into ImportInfo dicts.

        Handles:
          import defaultExport from 'module'
          import { named } from 'module'
          import { named as alias } from 'module'
          import * as ns from 'module'
          import defaultExport, { named } from 'module'
        """
        source_node = _child_by_field(node, "source")
        if not source_node:
            return []

        module = _text(source_node, source).strip("'\"")
        result: list[dict] = []

        # Collect import clauses
        for child in node.children:
            if child.type == "import_clause":
                names: list[str] = []
                aliases: dict[str, str] = {}

                for item in child.children:
                    if item.type == "identifier":
                        # default import: `import Foo from ...`
                        names.append(_text(item, source))

                    elif item.type == "namespace_import":
                        # `import * as ns from ...`
                        alias_node = None
                        for c in item.children:
                            if c.type == "identifier":
                                alias_node = c
                        if alias_node:
                            ns_name = _text(alias_node, source)
                            names.append("*")
                            aliases["*"] = ns_name

                    elif item.type == "named_imports":
                        # `import { Foo, Bar as B } from ...`
                        for spec in item.children:
                            if spec.type == "import_specifier":
                                # name [as alias]
                                specifier_children = [
                                    c for c in spec.children
                                    if c.type == "identifier"
                                ]
                                if len(specifier_children) >= 2:
                                    orig = _text(specifier_children[0], source)
                                    alias_name = _text(specifier_children[-1], source)
                                    names.append(orig)
                                    if orig != alias_name:
                                        aliases[orig] = alias_name
                                elif len(specifier_children) == 1:
                                    names.append(_text(specifier_children[0], source))

                # Build single alias string for backward compat
                # (first alias if there's exactly one, else None)
                alias_val: Optional[str] = None
                if len(aliases) == 1:
                    alias_val = next(iter(aliases.values()))

                result.append({
                    "module": module,
                    "names": names,
                    "alias": alias_val,
                    "aliases": aliases,
                    "is_from": True,
                })

        # bare `import 'module'` with no clause
        if not result:
            result.append({
                "module": module,
                "names": [],
                "alias": None,
                "aliases": {},
                "is_from": False,
            })

        return result

    # ------------------------------------------------------------------
    # Function info builder
    # ------------------------------------------------------------------

    def _build_function(func_node, name: str, decl_node=None, in_class: str = "") -> dict:
        """Build a function/method info dict from a tree-sitter node.

        func_node    — the actual function/arrow_function/method_definition node
        name         — resolved name string
        decl_node    — outer declaration node (lexical_declaration etc.) used for
                       JSDoc lookup and export detection (may be None)
        in_class     — enclosing class name, if any
        """
        outer = decl_node or func_node

        params_node = _child_by_field(func_node, "parameters")
        body_node = _child_by_field(func_node, "body")
        return_type = _extract_return_type(func_node)
        is_async = _is_async(func_node)
        jsdoc = _extract_jsdoc(outer)
        decorators = _extract_decorators(outer)
        exported = _is_exported(outer)
        params = _extract_params(params_node)
        conditionals = _extract_conditionals(body_node)
        complexity = _compute_complexity(body_node)

        react_component = _is_react_component(name, func_node)
        hook = _is_hook(name)

        prefix = f"{in_class}." if in_class else ""
        return {
            "name": name,
            "id": f"{rel_path}::{prefix}{name}",
            "params": params,
            "return_type": return_type,
            "docstring": jsdoc,
            "lineno": func_node.start_point[0] + 1,
            "line_start": func_node.start_point[0] + 1,
            "line_end": func_node.end_point[0] + 1,
            "decorators": decorators,
            "calls": [],
            "is_async": is_async,
            "is_exported": exported,
            "is_react_component": react_component,
            "is_hook": hook,
            "conditionals": conditionals,
            "complexity": complexity,
        }

    # ------------------------------------------------------------------
    # Class info builder
    # ------------------------------------------------------------------

    def _build_class(class_node, decl_node=None) -> dict:
        """Build a class info dict from a tree-sitter class_declaration node."""
        outer = decl_node or class_node
        name_node = _child_by_field(class_node, "name")
        if not name_node:
            return None

        cls_name = _text(name_node, source)
        jsdoc = _extract_jsdoc(outer)
        decorators = _extract_decorators(outer)
        exported = _is_exported(outer)

        # Base classes (extends)
        heritage = _child_by_type(class_node, "class_heritage")
        bases: list[str] = []
        implements: list[str] = []
        if heritage:
            for h in heritage.children:
                if h.type == "extends_clause":
                    for c in h.children:
                        if c.type in ("identifier", "member_expression"):
                            bases.append(_text(c, source))
                elif h.type == "implements_clause":
                    # TypeScript: `implements Interface1, Interface2`
                    for c in h.children:
                        if c.type in ("identifier", "type_identifier", "generic_type"):
                            implements.append(_text(c, source))

        # Methods
        body = _child_by_field(class_node, "body")
        methods: list[dict] = []
        if body:
            pending_decorators: list[str] = []
            for member in body.children:
                if member.type == "decorator":
                    pending_decorators.append(_text(member, source))
                    continue

                if member.type in ("method_definition", "public_field_definition"):
                    if member.type == "method_definition":
                        mname_node = _child_by_field(member, "name")
                        mparams_node = _child_by_field(member, "parameters")
                        mbody_node = _child_by_field(member, "body")
                        if mname_node:
                            mname = _text(mname_node, source)
                            mreturn_type = _extract_return_type(member)
                            m_is_async = _is_async(member)
                            m_decorators = list(pending_decorators) + _extract_decorators(member)
                            m_jsdoc = _extract_jsdoc(member)
                            m_conds = _extract_conditionals(mbody_node)
                            m_complexity = _compute_complexity(mbody_node)
                            methods.append({
                                "name": mname,
                                "id": f"{rel_path}::{cls_name}.{mname}",
                                "params": _extract_params(mparams_node),
                                "return_type": mreturn_type,
                                "docstring": m_jsdoc,
                                "lineno": member.start_point[0] + 1,
                                "line_start": member.start_point[0] + 1,
                                "line_end": member.end_point[0] + 1,
                                "decorators": m_decorators,
                                "calls": [],
                                "is_async": m_is_async,
                                "is_exported": False,   # methods aren't independently exported
                                "is_react_component": False,
                                "is_hook": _is_hook(mname),
                                "conditionals": m_conds,
                                "complexity": m_complexity,
                            })
                    pending_decorators = []
                else:
                    pending_decorators = []

        return {
            "name": cls_name,
            "id": f"{rel_path}::{cls_name}",
            "bases": bases,
            "implements": implements,
            "docstring": jsdoc,
            "lineno": class_node.start_point[0] + 1,
            "line_start": class_node.start_point[0] + 1,
            "line_end": class_node.end_point[0] + 1,
            "decorators": decorators,
            "methods": methods,
            "is_exported": exported,
        }

    # ------------------------------------------------------------------
    # Top-level walker
    # ------------------------------------------------------------------

    def _walk(node, in_class: str = ""):
        """Walk AST nodes at the current level and populate functions/classes/imports."""
        for child in node.children:

            # ---- import statements ----
            if child.type in ("import_statement", "import_declaration"):
                imports.extend(_parse_import(child))

            # ---- function declarations ----
            elif child.type in ("function_declaration", "generator_function_declaration"):
                name_node = _child_by_field(child, "name")
                if name_node:
                    fn = _build_function(child, _text(name_node, source))
                    functions.append(fn)

            # ---- async function declarations (tree-sitter may wrap) ----
            # Some grammars use `async` as a modifier child rather than a separate node type

            # ---- arrow/function expression in variable declarations ----
            elif child.type in ("lexical_declaration", "variable_declaration"):
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name_node = _child_by_field(decl, "name")
                        value_node = _child_by_field(decl, "value")
                        if name_node and value_node and value_node.type in (
                            "arrow_function", "function", "generator_function"
                        ):
                            name = _text(name_node, source)
                            fn = _build_function(value_node, name, decl_node=child)
                            functions.append(fn)

            # ---- export statements ----
            elif child.type == "export_statement":
                # Recurse — the inner node (function_declaration, class_declaration,
                # lexical_declaration, etc.) will be picked up on the next level.
                _walk(child, in_class)

            # ---- class declarations ----
            elif child.type == "class_declaration":
                cls = _build_class(child)
                if cls:
                    classes.append(cls)

            # ---- expression statements (e.g. IIFE, module.exports) ----
            # Skip — no top-level function/class info to extract here

    _walk(root)

    # Module-level JSDoc: look for leading block comment in the file
    module_doc: Optional[str] = None
    for child in root.children:
        if child.type == "comment":
            text = _text(child, source)
            if text.startswith("/**") or text.startswith("/*!"):
                cleaned = re.sub(r"^/\*[*!]|\*/$", "", text)
                cleaned = re.sub(r"^\s*\*\s?", "", cleaned, flags=re.MULTILINE)
                module_doc = cleaned.strip()
                break
        elif child.type not in ("comment",):
            break

    language_label = EXTENSION_TO_LABEL.get(abs_path.suffix, lang_key)

    return {
        "id": rel_path,
        "path": rel_path,
        "abs_path": str(abs_path),
        "language": language_label,
        "loc": len(source_str.splitlines()),
        "docstring": module_doc,
        "imports": imports,
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Go parser
# ---------------------------------------------------------------------------

def parse_go_treesitter(abs_path: Path, repo_path: Path) -> Optional[dict]:
    """Parse a Go file using tree-sitter. Returns ParsedFile dict or None."""
    parser = _get_parser("go")
    if parser is None:
        return None

    try:
        source = abs_path.read_bytes()
    except OSError:
        return None

    source_str = source.decode("utf-8", errors="replace")
    rel_path = str(abs_path.relative_to(repo_path))

    try:
        tree = parser.parse(source)
    except Exception:
        return None

    root = tree.root_node
    functions: list[dict] = []
    classes: list[dict] = []  # Go structs as "classes"
    imports: list[dict] = []

    for child in root.children:
        # Functions and methods
        if child.type in ("function_declaration", "method_declaration"):
            name_node = _child_by_field(child, "name")
            params_node = _child_by_field(child, "parameters")

            if name_node:
                name = _text(name_node, source)
                params = []
                if params_node:
                    for p in params_node.children:
                        if p.type == "parameter_declaration":
                            pname = _child_by_type(p, "identifier")
                            if pname:
                                params.append(_text(pname, source))

                # Check for receiver (method)
                receiver = _child_by_field(child, "receiver")
                receiver_type = ""
                if receiver:
                    for r in receiver.children:
                        if r.type == "parameter_declaration":
                            type_node = _child_by_type(r, "type_identifier") or _child_by_type(r, "pointer_type")
                            if type_node:
                                receiver_type = _text(type_node, source).strip("*")

                # Extract doc comment
                docstring = None
                prev = child.prev_named_sibling
                if prev and prev.type == "comment":
                    docstring = _text(prev, source).lstrip("/ ").strip()

                fn_id = f"{rel_path}::{receiver_type + '.' if receiver_type else ''}{name}"
                functions.append({
                    "name": name,
                    "id": fn_id,
                    "params": params,
                    "return_type": None,
                    "docstring": docstring,
                    "lineno": child.start_point[0] + 1,
                    "line_start": child.start_point[0] + 1,
                    "line_end": child.end_point[0] + 1,
                    "decorators": [],
                    "calls": [],
                })

        # Struct types (as "classes")
        elif child.type == "type_declaration":
            for spec in child.children:
                if spec.type == "type_spec":
                    name_node = _child_by_field(spec, "name")
                    type_node = _child_by_field(spec, "type")
                    if name_node and type_node and type_node.type == "struct_type":
                        struct_name = _text(name_node, source)
                        docstring = None
                        prev = child.prev_named_sibling
                        if prev and prev.type == "comment":
                            docstring = _text(prev, source).lstrip("/ ").strip()

                        classes.append({
                            "name": struct_name,
                            "id": f"{rel_path}::{struct_name}",
                            "bases": [],
                            "docstring": docstring,
                            "lineno": child.start_point[0] + 1,
                            "line_start": child.start_point[0] + 1,
                            "line_end": child.end_point[0] + 1,
                            "decorators": [],
                            "methods": [],
                        })

        # Imports
        elif child.type == "import_declaration":
            for spec in child.children:
                if spec.type == "import_spec_list":
                    for imp in spec.children:
                        if imp.type == "import_spec":
                            path_node = _child_by_field(imp, "path")
                            if path_node:
                                imports.append({
                                    "module": _text(path_node, source).strip('"'),
                                    "names": [],
                                    "alias": None,
                                    "aliases": {},
                                    "is_from": False,
                                })
                elif spec.type == "import_spec":
                    path_node = _child_by_field(spec, "path")
                    if path_node:
                        imports.append({
                            "module": _text(path_node, source).strip('"'),
                            "names": [],
                            "alias": None,
                            "aliases": {},
                            "is_from": False,
                        })

    # Link methods to structs
    struct_map = {c["name"]: c for c in classes}
    for fn in list(functions):
        if "." in fn["name"]:
            continue
        # Check if function ID has a receiver prefix
        parts = fn["id"].split("::")[-1].split(".")
        if len(parts) == 2 and parts[0] in struct_map:
            struct_map[parts[0]]["methods"].append(fn)

    return {
        "id": rel_path,
        "path": rel_path,
        "abs_path": str(abs_path),
        "language": "go",
        "loc": len(source_str.splitlines()),
        "docstring": None,
        "imports": imports,
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Java parser
# ---------------------------------------------------------------------------

def parse_java_treesitter(abs_path: Path, repo_path: Path) -> Optional[dict]:
    """Parse a Java file using tree-sitter. Returns ParsedFile dict or None."""
    parser = _get_parser("java")
    if parser is None:
        return None

    try:
        source = abs_path.read_bytes()
    except OSError:
        return None

    source_str = source.decode("utf-8", errors="replace")
    rel_path = str(abs_path.relative_to(repo_path))

    try:
        tree = parser.parse(source)
    except Exception:
        return None

    root = tree.root_node
    functions: list[dict] = []
    classes: list[dict] = []
    imports: list[dict] = []

    def _extract_javadoc(node) -> Optional[str]:
        prev = node.prev_named_sibling
        if prev and prev.type == "block_comment":
            text = _text(prev, source)
            if text.startswith("/**"):
                cleaned = re.sub(r"^/\*\*|\*/$", "", text)
                cleaned = re.sub(r"^\s*\*\s?", "", cleaned, flags=re.MULTILINE)
                return cleaned.strip()
        return None

    def _parse_class_body(class_node, class_name: str):
        methods = []
        body = _child_by_field(class_node, "body")
        if not body:
            return methods

        for member in body.children:
            if member.type == "method_declaration":
                name_node = _child_by_field(member, "name")
                params_node = _child_by_field(member, "parameters")
                if name_node:
                    mname = _text(name_node, source)
                    params = []
                    if params_node:
                        for p in params_node.children:
                            if p.type == "formal_parameter":
                                pname = _child_by_field(p, "name")
                                if pname:
                                    params.append(_text(pname, source))
                    methods.append({
                        "name": mname,
                        "id": f"{rel_path}::{class_name}::{mname}",
                        "params": params,
                        "return_type": None,
                        "docstring": _extract_javadoc(member),
                        "lineno": member.start_point[0] + 1,
                        "line_start": member.start_point[0] + 1,
                        "line_end": member.end_point[0] + 1,
                        "decorators": [],
                        "calls": [],
                    })

            # Constructor
            elif member.type == "constructor_declaration":
                name_node = _child_by_field(member, "name")
                if name_node:
                    methods.append({
                        "name": _text(name_node, source),
                        "id": f"{rel_path}::{class_name}::__init__",
                        "params": [],
                        "return_type": None,
                        "docstring": _extract_javadoc(member),
                        "lineno": member.start_point[0] + 1,
                        "line_start": member.start_point[0] + 1,
                        "line_end": member.end_point[0] + 1,
                        "decorators": [],
                        "calls": [],
                    })

        return methods

    for child in root.children:
        if child.type == "class_declaration":
            name_node = _child_by_field(child, "name")
            if name_node:
                cls_name = _text(name_node, source)
                # Extract superclass
                superclass = _child_by_field(child, "superclass")
                bases = []
                if superclass:
                    bases.append(_text(superclass, source))

                methods = _parse_class_body(child, cls_name)
                classes.append({
                    "name": cls_name,
                    "id": f"{rel_path}::{cls_name}",
                    "bases": bases,
                    "docstring": _extract_javadoc(child),
                    "lineno": child.start_point[0] + 1,
                    "line_start": child.start_point[0] + 1,
                    "line_end": child.end_point[0] + 1,
                    "decorators": [],
                    "methods": methods,
                })

        elif child.type == "import_declaration":
            # Java: import com.example.Foo;
            full_text = _text(child, source).strip().rstrip(";")
            module = full_text.replace("import ", "").replace("static ", "").strip()
            imports.append({
                "module": module,
                "names": [],
                "alias": None,
                "aliases": {},
                "is_from": False,
            })

        elif child.type == "package_declaration":
            pass  # Could extract package name if needed

    return {
        "id": rel_path,
        "path": rel_path,
        "abs_path": str(abs_path),
        "language": "java",
        "loc": len(source_str.splitlines()),
        "docstring": None,
        "imports": imports,
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Unified multi-language parse entry point
# ---------------------------------------------------------------------------

def parse_file_multi_lang(abs_path: Path, repo_path: Path) -> Optional[dict]:
    """Parse any supported file type. Returns ParsedFile dict or None.

    Uses tree-sitter when the language parser is installed, otherwise returns None
    (caller should fall back to regex-based parsing).
    """
    suffix = abs_path.suffix
    lang = EXTENSION_TO_LANG.get(suffix)
    if not lang:
        return None

    if lang in ("javascript", "typescript"):
        return parse_js_ts_treesitter(abs_path, repo_path)
    elif lang == "go":
        return parse_go_treesitter(abs_path, repo_path)
    elif lang == "java":
        return parse_java_treesitter(abs_path, repo_path)
    return None
