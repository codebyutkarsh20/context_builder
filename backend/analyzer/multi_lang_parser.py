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


# ---------------------------------------------------------------------------
# JavaScript / TypeScript parser
# ---------------------------------------------------------------------------

def parse_js_ts_treesitter(abs_path: Path, repo_path: Path) -> Optional[dict]:
    """Parse a JS/TS file using tree-sitter. Returns ParsedFile dict or None."""
    lang_key = EXTENSION_TO_LANG.get(abs_path.suffix)
    if not lang_key:
        return None

    # For JSX/TSX, try JS/TS parser (they handle JSX syntax)
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

    def _extract_jsdoc(node) -> Optional[str]:
        """Extract JSDoc comment preceding a node."""
        prev = node.prev_named_sibling
        if prev and prev.type == "comment":
            text = _text(prev, source)
            if text.startswith("/**"):
                cleaned = re.sub(r"^/\*\*|\*/$", "", text)
                cleaned = re.sub(r"^\s*\*\s?", "", cleaned, flags=re.MULTILINE)
                return cleaned.strip()
        return None

    def _extract_params(params_node) -> list[str]:
        """Extract parameter names from a formal_parameters node."""
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type in ("identifier", "required_parameter", "optional_parameter",
                              "rest_parameter", "assignment_pattern"):
                name_node = _child_by_field(child, "pattern") or _child_by_field(child, "name") or child
                if name_node.type == "identifier":
                    result.append(_text(name_node, source))
                elif name_node.type == "rest_parameter":
                    inner = _child_by_type(name_node, "identifier")
                    if inner:
                        result.append(f"...{_text(inner, source)}")
                else:
                    result.append(_text(name_node, source).split(":")[0].strip())
        return result

    def _walk(node, in_class: str = ""):
        for child in node.children:
            # Function declarations
            if child.type in ("function_declaration", "generator_function_declaration"):
                name_node = _child_by_field(child, "name")
                params_node = _child_by_field(child, "parameters")
                if name_node:
                    name = _text(name_node, source)
                    fn = {
                        "name": name,
                        "id": f"{rel_path}::{in_class + '.' if in_class else ''}{name}",
                        "params": _extract_params(params_node),
                        "return_type": None,
                        "docstring": _extract_jsdoc(child),
                        "lineno": child.start_point[0] + 1,
                        "decorators": [],
                        "calls": [],
                    }
                    if in_class:
                        return fn  # Return to caller for method collection
                    functions.append(fn)

            # Arrow / function expressions in variable declarations
            elif child.type in ("lexical_declaration", "variable_declaration"):
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name_node = _child_by_field(decl, "name")
                        value_node = _child_by_field(decl, "value")
                        if name_node and value_node and value_node.type in ("arrow_function", "function"):
                            name = _text(name_node, source)
                            params_node = _child_by_field(value_node, "parameters")
                            functions.append({
                                "name": name,
                                "id": f"{rel_path}::{name}",
                                "params": _extract_params(params_node),
                                "return_type": None,
                                "docstring": _extract_jsdoc(child),
                                "lineno": child.start_point[0] + 1,
                                "decorators": [],
                                "calls": [],
                            })

            # Export statements
            elif child.type == "export_statement":
                _walk(child, in_class)

            # Class declarations
            elif child.type == "class_declaration":
                name_node = _child_by_field(child, "name")
                if name_node:
                    cls_name = _text(name_node, source)
                    # Extract base class
                    heritage = _child_by_type(child, "class_heritage")
                    bases = []
                    if heritage:
                        for h in heritage.children:
                            if h.type == "identifier":
                                bases.append(_text(h, source))

                    # Extract methods
                    body = _child_by_field(child, "body")
                    methods = []
                    if body:
                        for member in body.children:
                            if member.type == "method_definition":
                                mname_node = _child_by_field(member, "name")
                                mparams_node = _child_by_field(member, "parameters")
                                if mname_node:
                                    mname = _text(mname_node, source)
                                    methods.append({
                                        "name": mname,
                                        "id": f"{rel_path}::{cls_name}::{mname}",
                                        "params": _extract_params(mparams_node),
                                        "return_type": None,
                                        "docstring": _extract_jsdoc(member),
                                        "lineno": member.start_point[0] + 1,
                                        "decorators": [],
                                        "calls": [],
                                    })

                    classes.append({
                        "name": cls_name,
                        "id": f"{rel_path}::{cls_name}",
                        "bases": bases,
                        "docstring": _extract_jsdoc(child),
                        "lineno": child.start_point[0] + 1,
                        "decorators": [],
                        "methods": methods,
                    })

            # Import statements
            elif child.type in ("import_statement", "import_declaration"):
                source_node = _child_by_field(child, "source")
                if source_node:
                    module = _text(source_node, source).strip("'\"")
                    imports.append({"module": module, "names": [], "alias": None})

    _walk(root)

    return {
        "id": rel_path,
        "path": rel_path,
        "abs_path": str(abs_path),
        "language": "typescript" if abs_path.suffix in (".ts", ".tsx") else "javascript",
        "loc": len(source_str.splitlines()),
        "docstring": None,
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
                                })
                elif spec.type == "import_spec":
                    path_node = _child_by_field(spec, "path")
                    if path_node:
                        imports.append({
                            "module": _text(path_node, source).strip('"'),
                            "names": [],
                            "alias": None,
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
                    "decorators": [],
                    "methods": methods,
                })

        elif child.type == "import_declaration":
            # Java: import com.example.Foo;
            full_text = _text(child, source).strip().rstrip(";")
            module = full_text.replace("import ", "").replace("static ", "").strip()
            imports.append({"module": module, "names": [], "alias": None})

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
