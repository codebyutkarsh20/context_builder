"""BusinessLogicExtractor: mines parsed source files for business rules and constraints."""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword patterns
# ---------------------------------------------------------------------------

_DOCSTRING_BUSINESS_KEYWORDS: re.Pattern = re.compile(
    r"\b(must|should|validates?|ensures?|required|business rule|constraint|policy)\b",
    re.IGNORECASE,
)

_TODO_FIXME_PATTERN: re.Pattern = re.compile(
    r"#\s*(TODO|FIXME)[:\s]+(.+)",
    re.IGNORECASE,
)

_CONSTANT_PATTERN: re.Pattern = re.compile(
    r"^[A-Z][A-Z0-9_]*_(?:LIMIT|MAX|MIN|TIMEOUT|RATE|THRESHOLD|CAP|QUOTA|"
    r"WINDOW|PERIOD|SIZE|COUNT|RETRIES|ATTEMPTS|DELAY|INTERVAL|AGE|DAYS|HOURS)$",
)

# Matches FastAPI/Flask/Django route decorators
# e.g. @app.get("/path"), @router.post("/path"), @app.route("/path", methods=["POST"])
_ROUTE_DECORATOR_PATTERN: re.Pattern = re.compile(
    r'@\w+\.(get|post|put|patch|delete|route|head|options|websocket)\s*\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------


@dataclass
class _BusinessRule:
    content: str
    source_file: str
    source_line: int
    rule_type: str  # "docstring" | "todo" | "constant"
    enforced_by: list[str] = field(default_factory=list)  # list of function IDs

    @property
    def rule_id(self) -> str:
        # Deterministic ID to keep extractions idempotent across re-runs
        raw = f"{self.source_file}:{self.source_line}:{self.rule_type}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def to_pipeline_dict(self) -> dict:
        """Return a dict matching the schema expected by the repair pipeline.

        Maps internal _BusinessRule fields to the schema used by
        api/knowledge.py and agent/pipeline.py._load_business_rules().
        """
        return {
            "id": self.rule_id,
            "description": self.content,
            "file": self.source_file,
            # Comma-separated to preserve all linked functions (not just first)
            "function_id": ",".join(self.enforced_by),
            "severity": "medium",
            "source": self.rule_type,
            "created_at": None,
        }


def persist_rules_to_file(rules: list["_BusinessRule"], out_path: Path) -> int:
    """Write auto-extracted rules to the pipeline-expected flat JSON file.

    Merges with any existing human-submitted rules (keyed by id).
    Returns the count of newly written rules (0 if all were already present).

    Parameters
    ----------
    rules:
        Auto-extracted _BusinessRule objects to persist.
    out_path:
        Path to the ``business_rules.json`` flat file for this repo.
    """
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            logger.warning(
                "business_rules.json at %s is corrupted — treating as empty", out_path
            )
            existing = []

    existing_ids = {r["id"] for r in existing if "id" in r}
    new_dicts = [r.to_pipeline_dict() for r in rules if r.rule_id not in existing_ids]

    out_path.write_text(json.dumps(existing + new_dicts, indent=2, default=str))
    logger.info(
        "persist_rules_to_file: wrote %d new rule(s) to %s (%d existing preserved)",
        len(new_dicts),
        out_path,
        len(existing),
    )
    return len(new_dicts)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class BusinessLogicExtractor:
    """Scan parsed repository files for business rules and write them to Neo4j.

    Parameters
    ----------
    repo_name:
        The name of the repository as stored in the ``Repo`` node.
    parsed:
        List of parsed-file records (dicts or objects with ``__dict__``).
        Each record is expected to have at minimum:
            - ``path``      — relative or absolute path of the source file
            - ``id``        — Neo4j node ID for the File node
            - ``classes``   — list of class dicts (``name``, ``docstring``, ``methods``, ``lineno``)
            - ``functions`` — list of function dicts (``id``, ``name``, ``docstring``, ``lineno``)
        The raw ``content`` or ``source`` field (full source text) is used when
        present; otherwise the file is re-read from disk via ``path``.
    """

    def __init__(self, repo_name: str, parsed: list[Any]) -> None:
        self.repo_name = repo_name
        self._parsed = [p if isinstance(p, dict) else vars(p) for p in parsed]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> int:
        """Extract business rules from all parsed files and persist to Neo4j.

        Returns
        -------
        int
            Total number of ``BusinessRule`` nodes created.
        """
        total_created = 0
        for file_record in self._parsed:
            rules = self._extract_from_file(file_record)
            for rule in rules:
                created = self._persist_rule(rule, file_record)
                if created:
                    total_created += 1

        logger.info(
            "BusinessLogicExtractor: created %d rule(s) for repo '%s'.",
            total_created,
            self.repo_name,
        )
        return total_created

    def extract_all(self) -> list[_BusinessRule]:
        """Extract all rules from parsed files and return them (no Neo4j persistence).

        Useful when running in --no-neo4j mode.

        Returns
        -------
        list[_BusinessRule]
            All extracted business rules.
        """
        all_rules: list[_BusinessRule] = []
        for file_record in self._parsed:
            all_rules.extend(self._extract_from_file(file_record))
        logger.info(
            "BusinessLogicExtractor: extracted %d rule(s) (in-memory) for repo '%s'.",
            len(all_rules),
            self.repo_name,
        )
        return all_rules

    # ------------------------------------------------------------------
    # Per-file extraction
    # ------------------------------------------------------------------

    # File-name patterns that identify test/config files.
    # Business rules extracted from test files are almost always noise
    # (e.g. "# TODO: must add more test cases" looks like a rule but isn't).
    _TEST_PATH_PATTERNS = re.compile(
        r"(^|[\\/])(tests?|conftest|test_[^/\\]+)([\\/]|$)|test_[^/\\]+\.py$",
        re.IGNORECASE,
    )

    def _extract_from_file(self, file_record: dict[str, Any]) -> list[_BusinessRule]:
        source_file: str = file_record.get("path", "unknown")

        # Skip test/fixture files — their TODOs and docstrings are not business rules.
        if self._TEST_PATH_PATTERNS.search(source_file):
            return []

        rules: list[_BusinessRule] = []

        # ---- 1. Docstring rules (classes + functions) --------------------
        rules.extend(self._scan_docstrings(file_record))

        # ---- 2. TODO/FIXME comments in source code ----------------------
        source = self._get_source(file_record)
        if source:
            rules.extend(self._scan_todo_fixme(source, source_file))
            rules.extend(self._scan_constants(source, source_file))
            rules.extend(self._scan_api_endpoints(file_record, source, source_file))

        return rules

    # ------------------------------------------------------------------
    # Docstring scanner
    # ------------------------------------------------------------------

    def _scan_docstrings(self, file_record: dict[str, Any]) -> list[_BusinessRule]:
        source_file: str = file_record.get("path", "unknown")
        rules: list[_BusinessRule] = []

        def _check_docstring(
            doc: str | None,
            lineno: int,
            enforced_by: list[str],
        ) -> None:
            if not doc:
                return
            for line in doc.splitlines():
                if _DOCSTRING_BUSINESS_KEYWORDS.search(line):
                    rules.append(
                        _BusinessRule(
                            content=line.strip(),
                            source_file=source_file,
                            source_line=lineno,
                            rule_type="docstring",
                            enforced_by=enforced_by,
                        )
                    )

        # File-level docstring
        file_doc: str | None = file_record.get("docstring")
        _check_docstring(file_doc, lineno=1, enforced_by=[])

        # Class docstrings
        for cls in file_record.get("classes") or []:
            cls_dict = cls if isinstance(cls, dict) else vars(cls)
            _check_docstring(
                cls_dict.get("docstring"),
                lineno=cls_dict.get("lineno", 0),
                enforced_by=[],
            )
            # Method docstrings within the class
            for method in cls_dict.get("methods") or []:
                m_dict = method if isinstance(method, dict) else vars(method)
                fn_id = m_dict.get("id") or ""
                _check_docstring(
                    m_dict.get("docstring"),
                    lineno=m_dict.get("lineno", 0),
                    enforced_by=[fn_id] if fn_id else [],
                )

        # Top-level function docstrings
        for fn in file_record.get("functions") or []:
            fn_dict = fn if isinstance(fn, dict) else vars(fn)
            fn_id = fn_dict.get("id") or ""
            _check_docstring(
                fn_dict.get("docstring"),
                lineno=fn_dict.get("lineno", 0),
                enforced_by=[fn_id] if fn_id else [],
            )

        return rules

    # ------------------------------------------------------------------
    # TODO/FIXME scanner
    # ------------------------------------------------------------------

    def _scan_todo_fixme(self, source: str, source_file: str) -> list[_BusinessRule]:
        rules: list[_BusinessRule] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            match = _TODO_FIXME_PATTERN.search(line)
            if match:
                comment_text = match.group(2).strip()
                # Only capture business-relevant TODOs
                if _DOCSTRING_BUSINESS_KEYWORDS.search(comment_text) or len(comment_text) > 20:
                    rules.append(
                        _BusinessRule(
                            content=f"{match.group(1).upper()}: {comment_text}",
                            source_file=source_file,
                            source_line=lineno,
                            rule_type="todo",
                        )
                    )
        return rules

    # ------------------------------------------------------------------
    # Constants scanner
    # ------------------------------------------------------------------

    def _scan_constants(self, source: str, source_file: str) -> list[_BusinessRule]:
        """Detect module-level constants whose names suggest business constraints."""
        rules: list[_BusinessRule] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return rules

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Only module-level assignments (depth-1 targets)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name: str = target.id
                if not _CONSTANT_PATTERN.match(name):
                    continue
                # Retrieve value representation
                try:
                    value = ast.literal_eval(node.value)
                    value_str = repr(value)
                except (ValueError, TypeError):
                    value_str = ast.unparse(node.value) if hasattr(ast, "unparse") else "?"

                rules.append(
                    _BusinessRule(
                        content=f"{name} = {value_str}",
                        source_file=source_file,
                        source_line=node.lineno,
                        rule_type="constant",
                    )
                )

        return rules

    # ------------------------------------------------------------------
    # API endpoint scanner
    # ------------------------------------------------------------------

    def _scan_api_endpoints(
        self, file_record: dict[str, Any], source: str, source_file: str
    ) -> list[_BusinessRule]:
        """Detect API route decorators and record them as business rules."""
        rules: list[_BusinessRule] = []
        lines = source.splitlines()
        for lineno, line in enumerate(lines, start=1):
            # Skip comment and string literal lines (e.g. docstrings, regex defs)
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith(("'", '"', "r'")):
                continue
            match = _ROUTE_DECORATOR_PATTERN.search(line)
            if not match:
                continue
            http_method = match.group(1).upper()
            path = match.group(2)
            # Validate: real URL paths start with /
            # Allow {param} (FastAPI), :param (Express), <param> (Flask)
            if not path.startswith("/"):
                continue
            # Try to find the function name on the next non-empty line
            fn_name = ""
            for next_line in lines[lineno:lineno + 3]:
                fn_match = re.match(r"\s*(?:async\s+)?def\s+(\w+)", next_line)
                if fn_match:
                    fn_name = fn_match.group(1)
                    break
            content = f"API Endpoint: {http_method} {path}"
            if fn_name:
                content += f" → handler: {fn_name}()"
            rules.append(
                _BusinessRule(
                    content=content,
                    source_file=source_file,
                    source_line=lineno,
                    rule_type="endpoint",
                )
            )
        return rules

    # ------------------------------------------------------------------
    # Neo4j persistence
    # ------------------------------------------------------------------

    def _persist_rule(
        self,
        rule: _BusinessRule,
        file_record: dict[str, Any],
    ) -> bool:
        """Write a BusinessRule node to Neo4j.  Returns True if newly created."""
        rule_id = rule.rule_id

        # Skip Neo4j if not connected
        if not neo4j_client.is_connected():
            return True

        # Check existence (idempotent)
        existing = neo4j_client.run(
            "MATCH (br:BusinessRule {id: $id}) RETURN br.id AS id",
            {"id": rule_id},
        )
        if existing:
            logger.debug("BusinessRule %s already exists — skipping.", rule_id)
            return False

        # Create the node
        neo4j_client.run(
            "CREATE (br:BusinessRule {"
            "  id: $id, "
            "  content: $content, "
            "  source_file: $source_file, "
            "  source_line: $source_line, "
            "  rule_type: $rule_type"
            "})",
            {
                "id": rule_id,
                "content": rule.content,
                "source_file": rule.source_file,
                "source_line": rule.source_line,
                "rule_type": rule.rule_type,
            },
        )

        # Link BusinessRule → File (FOUND_IN)
        file_id: str = file_record.get("id", "")
        if file_id:
            neo4j_client.run(
                "MATCH (br:BusinessRule {id: $rid}), (f:File {id: $fid}) "
                "MERGE (br)-[:FOUND_IN]->(f)",
                {"rid": rule_id, "fid": file_id},
            )

        # Link BusinessRule → Function (ENFORCED_BY)
        for fn_id in rule.enforced_by:
            if not fn_id:
                continue
            try:
                neo4j_client.run(
                    "MATCH (br:BusinessRule {id: $rid}), (fn:Function {id: $fnid}) "
                    "MERGE (br)-[:ENFORCED_BY]->(fn)",
                    {"rid": rule_id, "fnid": fn_id},
                )
            except Exception as exc:
                logger.debug(
                    "Could not create ENFORCED_BY edge for rule %s → %s: %s",
                    rule_id,
                    fn_id,
                    exc,
                )

        logger.debug(
            "Created BusinessRule '%s' (%s) at %s:%d",
            rule_id,
            rule.rule_type,
            rule.source_file,
            rule.source_line,
        )
        return True

    # ------------------------------------------------------------------
    # Source-code helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_source(file_record: dict[str, Any]) -> str | None:
        """Return the source code for a file record, reading from disk if needed."""
        # Prefer pre-loaded content stored on the record
        for key in ("content", "source", "raw_source"):
            content = file_record.get(key)
            if content and isinstance(content, str):
                return content

        # Try abs_path first (most reliable — set by CodeParser)
        for path_key in ("abs_path", "path"):
            path_str: str | None = file_record.get(path_key)
            if not path_str:
                continue
            path = Path(path_str)
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Could not read source file '%s': %s", path_str, exc)

        logger.debug("Source file not found on disk for record: %s", file_record.get("path"))
        return None
