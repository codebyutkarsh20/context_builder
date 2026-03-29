"""Tests for _BusinessRule.to_pipeline_dict() and persist_rules_to_file()."""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enricher.business_logic import _BusinessRule, persist_rules_to_file


def _make_rule(
    content="email must be validated",
    source_file="api/users.py",
    source_line=10,
    rule_type="docstring",
    enforced_by=None,
) -> _BusinessRule:
    return _BusinessRule(
        content=content,
        source_file=source_file,
        source_line=source_line,
        rule_type=rule_type,
        enforced_by=enforced_by or [],
    )


class TestToPipelineDict:
    def test_description_maps_from_content(self):
        rule = _make_rule(content="price must be positive")
        d = rule.to_pipeline_dict()
        assert d["description"] == "price must be positive"

    def test_file_maps_from_source_file(self):
        rule = _make_rule(source_file="billing/invoice.py")
        d = rule.to_pipeline_dict()
        assert d["file"] == "billing/invoice.py"

    def test_source_maps_from_rule_type(self):
        rule = _make_rule(rule_type="constant")
        d = rule.to_pipeline_dict()
        assert d["source"] == "constant"

    def test_severity_is_always_medium(self):
        rule = _make_rule()
        assert rule.to_pipeline_dict()["severity"] == "medium"

    def test_function_id_with_single_enforced_by(self):
        rule = _make_rule(enforced_by=["f:api/users.py::validate_email"])
        d = rule.to_pipeline_dict()
        assert d["function_id"] == "f:api/users.py::validate_email"

    def test_function_id_with_multiple_enforced_by(self):
        rule = _make_rule(enforced_by=["f:a.py::foo", "f:b.py::bar"])
        d = rule.to_pipeline_dict()
        assert d["function_id"] == "f:a.py::foo,f:b.py::bar"

    def test_function_id_empty_when_no_enforced_by(self):
        rule = _make_rule(enforced_by=[])
        assert rule.to_pipeline_dict()["function_id"] == ""

    def test_id_is_rule_id(self):
        rule = _make_rule()
        assert rule.to_pipeline_dict()["id"] == rule.rule_id

    def test_id_is_stable_across_calls(self):
        rule = _make_rule()
        assert rule.to_pipeline_dict()["id"] == rule.to_pipeline_dict()["id"]


class TestPersistRulesToFile:
    def test_creates_file_when_absent(self, tmp_path):
        rule = _make_rule()
        out = tmp_path / "business_rules.json"
        count = persist_rules_to_file([rule], out)
        assert count == 1
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["description"] == rule.content

    def test_returns_zero_for_empty_rules(self, tmp_path):
        out = tmp_path / "business_rules.json"
        count = persist_rules_to_file([], out)
        assert count == 0
        # File created with empty list
        assert json.loads(out.read_text()) == []

    def test_merges_with_existing_human_rules(self, tmp_path):
        human = [{"id": "human-1", "description": "manual rule", "file": "x.py"}]
        out = tmp_path / "business_rules.json"
        out.write_text(json.dumps(human))
        rule = _make_rule()
        count = persist_rules_to_file([rule], out)
        assert count == 1
        merged = json.loads(out.read_text())
        assert len(merged) == 2
        assert any(r["id"] == "human-1" for r in merged)

    def test_deduplicates_by_rule_id(self, tmp_path):
        rule = _make_rule()
        out = tmp_path / "business_rules.json"
        persist_rules_to_file([rule], out)
        count2 = persist_rules_to_file([rule], out)
        assert count2 == 0
        data = json.loads(out.read_text())
        assert len(data) == 1

    def test_human_rules_not_overwritten_on_second_run(self, tmp_path):
        human = [{"id": "human-99", "description": "do not delete me"}]
        out = tmp_path / "business_rules.json"
        out.write_text(json.dumps(human))
        rule = _make_rule()
        persist_rules_to_file([rule], out)
        persist_rules_to_file([rule], out)  # second run
        data = json.loads(out.read_text())
        assert any(r["id"] == "human-99" for r in data)

    def test_corrupted_json_treated_as_empty(self, tmp_path):
        out = tmp_path / "business_rules.json"
        out.write_text("{{not valid json}}")
        rule = _make_rule()
        # Should not raise; corrupted file treated as empty
        count = persist_rules_to_file([rule], out)
        assert count == 1
        data = json.loads(out.read_text())
        assert len(data) == 1
