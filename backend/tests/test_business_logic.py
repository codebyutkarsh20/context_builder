"""
Unit tests for enricher/business_logic.py — BusinessLogicExtractor.

Runs in --no-neo4j mode (uses extract_all() which skips Neo4j persistence).

Covers:
  - Docstring business keyword detection (must, should, validates, ensures, required,
    business rule, constraint, policy)
  - TODO/FIXME comment extraction
  - Module-level constant scanning (MAX/MIN/LIMIT/TIMEOUT/etc.)
  - API endpoint decorator scanning (FastAPI/Flask/Django routes)
  - Source loading from record (content key, abs_path)
  - Rule IDs are deterministic (idempotent)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enricher.business_logic import BusinessLogicExtractor, _BusinessRule


# ---------------------------------------------------------------------------
# Helper: build a minimal parsed file record
# ---------------------------------------------------------------------------

def _record(path="app.py", content="", classes=None, functions=None, docstring=None):
    return {
        "path": path,
        "id": f"f:{path}",
        "content": content,
        "docstring": docstring,
        "classes": classes or [],
        "functions": functions or [],
    }


def _make_extractor(records):
    with patch("enricher.business_logic.neo4j_client") as mock_neo4j:
        mock_neo4j.is_connected.return_value = False
        ext = BusinessLogicExtractor("test-repo", records)
    return ext


# ---------------------------------------------------------------------------
# Docstring extraction
# ---------------------------------------------------------------------------

class TestDocstringExtraction:

    def test_must_keyword_extracted(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::process",
                "name": "process",
                "docstring": "Email must be validated before saving.",
                "lineno": 5,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        contents = [r.content for r in rules]
        assert any("must" in c.lower() for c in contents)

    def test_validates_keyword_extracted(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::validate_email",
                "name": "validate_email",
                "docstring": "Validates email format per RFC 5322.",
                "lineno": 10,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("validates" in r.content.lower() for r in rules)

    def test_business_rule_keyword_extracted(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::calc",
                "name": "calc",
                "docstring": "Business rule: all prices must be positive.",
                "lineno": 15,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("business rule" in r.content.lower() for r in rules)

    def test_constraint_keyword_extracted(self):
        rec = _record(
            classes=[{
                "name": "UserAccount",
                "docstring": "Constraint: username must be unique.",
                "lineno": 3,
                "methods": [],
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("constraint" in r.content.lower() for r in rules)

    def test_policy_keyword_extracted(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::check_policy",
                "name": "check_policy",
                "docstring": "Enforces policy: no more than 3 failed login attempts.",
                "lineno": 20,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("policy" in r.content.lower() for r in rules)

    def test_rule_type_is_docstring(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::fn",
                "name": "fn",
                "docstring": "This must return a valid UUID.",
                "lineno": 5,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        doc_rules = [r for r in rules if r.rule_type == "docstring"]
        assert len(doc_rules) >= 1

    def test_no_keyword_docstring_skipped(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::helper",
                "name": "helper",
                "docstring": "Calculates the total sum of items.",
                "lineno": 5,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert len(rules) == 0

    def test_method_docstring_extracted(self):
        rec = _record(
            classes=[{
                "name": "PaymentService",
                "docstring": None,
                "lineno": 1,
                "methods": [{
                    "id": "f:app.py::PaymentService.charge",
                    "name": "charge",
                    "docstring": "Validates the card before charging.",
                    "lineno": 10,
                }],
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("validates" in r.content.lower() for r in rules)

    def test_file_level_docstring_extracted(self):
        rec = _record(
            docstring="This module ensures all payments are validated.",
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("ensures" in r.content.lower() for r in rules)

    def test_required_keyword_extracted(self):
        rec = _record(
            functions=[{
                "id": "f:app.py::fn",
                "name": "fn",
                "docstring": "User ID is required.",
                "lineno": 5,
            }]
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert any("required" in r.content.lower() for r in rules)


# ---------------------------------------------------------------------------
# TODO/FIXME extraction
# ---------------------------------------------------------------------------

class TestTodoFixmeExtraction:

    def test_todo_with_long_message_extracted(self):
        content = "# TODO: Add validation for null email addresses in user creation\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        todo_rules = [r for r in rules if r.rule_type == "todo"]
        assert len(todo_rules) >= 1

    def test_fixme_extracted(self):
        content = "# FIXME: This calculation violates the 5% rounding constraint\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        todo_rules = [r for r in rules if r.rule_type == "todo"]
        assert len(todo_rules) >= 1

    def test_todo_content_prefix(self):
        content = "# TODO: must validate before saving\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        todo_rules = [r for r in rules if r.rule_type == "todo"]
        assert todo_rules[0].content.startswith("TODO:")

    def test_short_todo_without_keyword_not_extracted(self):
        # Less than 20 chars and no business keyword → skipped
        content = "# TODO: fix it\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        todo_rules = [r for r in rules if r.rule_type == "todo"]
        assert len(todo_rules) == 0

    def test_todo_line_number_recorded(self):
        content = "x = 1\ny = 2\n# TODO: ensure email is not null before saving\nz = 3\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        todo_rules = [r for r in rules if r.rule_type == "todo"]
        assert todo_rules[0].source_line == 3


# ---------------------------------------------------------------------------
# Constant extraction
# ---------------------------------------------------------------------------

class TestConstantExtraction:

    def test_retry_count_constant(self):
        # _CONSTANT_PATTERN requires suffix: LIMIT|MAX|MIN|TIMEOUT|RATE|THRESHOLD|etc.
        # RETRY_COUNT ends with _COUNT → matches
        content = "RETRY_COUNT = 3\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert any("RETRY_COUNT" in r.content for r in const_rules)

    def test_rate_limit_constant(self):
        content = "RATE_LIMIT = 100\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert any("RATE_LIMIT" in r.content for r in const_rules)

    def test_timeout_constant(self):
        content = "REQUEST_TIMEOUT = 30\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert any("REQUEST_TIMEOUT" in r.content for r in const_rules)

    def test_non_matching_constant_skipped(self):
        # USER_ID doesn't end with LIMIT/MAX/MIN/TIMEOUT/etc. → not a constant rule
        content = "USER_ID = 42\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert len(const_rules) == 0

    def test_constant_value_in_content(self):
        # PAGE_SIZE ends with _SIZE → matches
        content = "PAGE_SIZE = 50\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert any("50" in r.content for r in const_rules)

    def test_syntax_error_source_handled_gracefully(self):
        content = "def foo(\n"  # broken syntax
        rec = _record(content=content)
        ext = _make_extractor([rec])
        # Should not raise
        rules = ext.extract_all()
        assert isinstance(rules, list)


# ---------------------------------------------------------------------------
# API endpoint extraction
# ---------------------------------------------------------------------------

class TestApiEndpointExtraction:

    def test_fastapi_get_route_extracted(self):
        content = (
            "@router.get('/users/{user_id}')\n"
            "async def get_user(user_id: int):\n"
            "    pass\n"
        )
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        endpoint_rules = [r for r in rules if r.rule_type == "endpoint"]
        assert len(endpoint_rules) >= 1
        assert any("/users/" in r.content for r in endpoint_rules)

    def test_fastapi_post_route_extracted(self):
        content = (
            "@app.post('/orders')\n"
            "async def create_order():\n"
            "    pass\n"
        )
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        endpoint_rules = [r for r in rules if r.rule_type == "endpoint"]
        assert any("POST" in r.content for r in endpoint_rules)

    def test_handler_name_included(self):
        content = (
            "@router.delete('/users/{id}')\n"
            "async def delete_user(id: int):\n"
            "    pass\n"
        )
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        endpoint_rules = [r for r in rules if r.rule_type == "endpoint"]
        assert any("delete_user" in r.content for r in endpoint_rules)

    def test_non_path_route_not_extracted(self):
        # path doesn't start with /
        content = '@app.get("no_slash_path")\nasync def fn(): pass\n'
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        endpoint_rules = [r for r in rules if r.rule_type == "endpoint"]
        assert len(endpoint_rules) == 0

    def test_comment_line_skipped(self):
        content = "# @app.get('/users') — example route\n"
        rec = _record(content=content)
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        endpoint_rules = [r for r in rules if r.rule_type == "endpoint"]
        assert len(endpoint_rules) == 0


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

class TestSourceLoading:

    def test_loads_from_content_key(self):
        # RETRY_COUNT ends with _COUNT → matches _CONSTANT_PATTERN
        content = "RETRY_COUNT = 3\n"
        rec = {"path": "app.py", "id": "f:app.py", "content": content,
               "docstring": None, "classes": [], "functions": []}
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert len(const_rules) >= 1

    def test_loads_from_abs_path(self, tmp_path):
        f = tmp_path / "mymodule.py"
        f.write_text("RATE_LIMIT = 100\n")
        rec = {"path": "mymodule.py", "id": "f:mymodule.py", "abs_path": str(f),
               "docstring": None, "classes": [], "functions": []}
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        const_rules = [r for r in rules if r.rule_type == "constant"]
        assert len(const_rules) >= 1

    def test_missing_source_skips_gracefully(self):
        rec = {"path": "/nonexistent/path.py", "id": "f:path.py",
               "docstring": None, "classes": [], "functions": []}
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert isinstance(rules, list)


# ---------------------------------------------------------------------------
# Rule ID determinism
# ---------------------------------------------------------------------------

class TestRuleIdDeterminism:

    def test_same_source_produces_same_id(self):
        rec = _record(
            path="app.py",
            functions=[{
                "id": "f:app.py::fn",
                "name": "fn",
                "docstring": "This must validate the input.",
                "lineno": 5,
            }]
        )
        ext1 = _make_extractor([rec])
        ext2 = _make_extractor([rec])
        rules1 = ext1.extract_all()
        rules2 = ext2.extract_all()
        ids1 = {r.rule_id for r in rules1}
        ids2 = {r.rule_id for r in rules2}
        assert ids1 == ids2

    def test_different_source_produces_different_id(self):
        rec_a = _record(path="a.py", functions=[{
            "id": "f:a.py::fn", "name": "fn",
            "docstring": "Must validate email.", "lineno": 5,
        }])
        rec_b = _record(path="b.py", functions=[{
            "id": "f:b.py::fn", "name": "fn",
            "docstring": "Must validate email.", "lineno": 5,
        }])
        rules_a = _make_extractor([rec_a]).extract_all()
        rules_b = _make_extractor([rec_b]).extract_all()
        ids_a = {r.rule_id for r in rules_a}
        ids_b = {r.rule_id for r in rules_b}
        assert ids_a != ids_b


# ---------------------------------------------------------------------------
# extract() return count
# ---------------------------------------------------------------------------

class TestExtractCount:

    def test_extract_all_returns_list(self):
        # RETRY_COUNT ends with _COUNT → matches constant pattern
        rec = _record(
            content="RETRY_COUNT = 3\n",
            functions=[{
                "id": "f:app.py::fn", "name": "fn",
                "docstring": "Must validate.", "lineno": 5,
            }],
        )
        ext = _make_extractor([rec])
        rules = ext.extract_all()
        assert isinstance(rules, list)
        assert len(rules) >= 2  # at least 1 constant + 1 docstring
