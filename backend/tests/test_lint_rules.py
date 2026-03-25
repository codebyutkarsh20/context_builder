"""
Unit tests for agent/lint_rules.py — LintViolation, check_file, run_lint_on_patches,
generate_default_rules.

Covers:
  - Built-in rules: no-sync-http-in-async, no-float-for-money, no-hardcoded-secrets,
    no-bare-except, no-print-in-production
  - Custom per-repo rules loaded from JSON
  - File-pattern filtering
  - Exclude-files filtering
  - Context-pattern and context-keyword checks
  - run_lint_on_patches (full pipeline with real files)
  - generate_default_rules (FastAPI, SQLAlchemy, Redis)
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.lint_rules as lr_module
from agent.lint_rules import (
    LintViolation,
    check_file,
    generate_default_rules,
    run_lint_on_patches,
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lr_module, "DATA_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# LintViolation NamedTuple
# ---------------------------------------------------------------------------

class TestLintViolation:

    def test_fields(self):
        v = LintViolation(
            rule_id="no-bare-except",
            file="app.py",
            line=42,
            message="Bare except",
            severity="warning",
            fix_hint="Use except Exception:",
        )
        assert v.rule_id == "no-bare-except"
        assert v.line == 42
        assert v.severity == "warning"


# ---------------------------------------------------------------------------
# check_file — no-bare-except
# ---------------------------------------------------------------------------

class TestNoBareExcept:

    def test_bare_except_detected(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-bare-except" in rule_ids

    def test_except_exception_not_flagged(self):
        code = "try:\n    x = 1\nexcept Exception:\n    pass\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-bare-except" not in rule_ids

    def test_except_valueerror_not_flagged(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    pass\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-bare-except" not in rule_ids

    def test_violation_has_correct_line_number(self):
        code = "x = 1\ntry:\n    pass\nexcept:\n    pass\n"
        violations = check_file(Path("app.py"), code)
        bare = [v for v in violations if v.rule_id == "no-bare-except"]
        assert bare[0].line == 4


# ---------------------------------------------------------------------------
# check_file — no-print-in-production
# ---------------------------------------------------------------------------

class TestNoPrintInProduction:

    def test_print_detected_in_production(self):
        code = "def foo():\n    print('debug')\n    return 1\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-print-in-production" in rule_ids

    def test_print_not_flagged_in_test_file(self):
        code = "def test_foo():\n    print('testing')\n"
        violations = check_file(Path("test_app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-print-in-production" not in rule_ids

    def test_print_not_flagged_in_conftest(self):
        code = "print('fixture setup')\n"
        violations = check_file(Path("conftest.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-print-in-production" not in rule_ids

    def test_logging_not_flagged(self):
        code = "logger.info('starting')\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-print-in-production" not in rule_ids


# ---------------------------------------------------------------------------
# check_file — no-hardcoded-secrets
# ---------------------------------------------------------------------------

class TestNoHardcodedSecrets:

    def test_api_key_assignment_detected(self):
        code = "api_key = 'sk-live-abcdefghijklmnopqrstuvwx'\n"
        violations = check_file(Path("config.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-hardcoded-secrets" in rule_ids

    def test_password_assignment_detected(self):
        code = "password = 'MySuperSecretPassword1234'\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-hardcoded-secrets" in rule_ids

    def test_env_var_reference_not_flagged(self):
        # pattern excludes os.environ, settings., config.
        code = "api_key = os.environ.get('API_KEY')\n"
        violations = check_file(Path("app.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-hardcoded-secrets" not in rule_ids

    def test_non_py_file_not_checked(self):
        code = "api_key = 'sk-live-abcdefghijklmnopqrstuvwx'\n"
        violations = check_file(Path("config.yaml"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-hardcoded-secrets" not in rule_ids


# ---------------------------------------------------------------------------
# check_file — no-float-for-money
# ---------------------------------------------------------------------------

class TestNoFloatForMoney:

    def test_float_with_price_context_detected(self):
        code = "total_price = float(raw_amount)\n"
        violations = check_file(Path("billing.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-float-for-money" in rule_ids

    def test_float_without_money_context_not_flagged(self):
        code = "angle = float(degrees)\n"
        violations = check_file(Path("math_utils.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-float-for-money" not in rule_ids

    def test_decimal_not_flagged(self):
        code = "amount = Decimal(str(value))\ntotal_price = amount\n"
        violations = check_file(Path("billing.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-float-for-money" not in rule_ids


# ---------------------------------------------------------------------------
# check_file — no-sync-http-in-async
# ---------------------------------------------------------------------------

class TestNoSyncHttpInAsync:

    def test_requests_in_async_function_detected(self):
        code = (
            "import requests\n"
            "\n"
            "async def fetch_data(url):\n"
            "    response = requests.get(url)\n"
            "    return response.json()\n"
        )
        violations = check_file(Path("api.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-sync-http-in-async" in rule_ids

    def test_requests_in_sync_function_not_flagged(self):
        code = (
            "import requests\n"
            "\n"
            "def fetch_data(url):\n"
            "    response = requests.get(url)\n"
            "    return response.json()\n"
        )
        violations = check_file(Path("api.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-sync-http-in-async" not in rule_ids

    def test_httpx_not_flagged(self):
        code = (
            "async def fetch(url):\n"
            "    async with httpx.AsyncClient() as client:\n"
            "        return await client.get(url)\n"
        )
        violations = check_file(Path("api.py"), code)
        rule_ids = [v.rule_id for v in violations]
        assert "no-sync-http-in-async" not in rule_ids


# ---------------------------------------------------------------------------
# check_file — custom repo rules
# ---------------------------------------------------------------------------

class TestCustomRepoRules:

    def test_custom_rule_applied(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        custom_rules = [
            {
                "id": "no-print-stmt",
                "pattern": r"print\(",
                "file_pattern": r"\.py$",
                "message": "No print in prod",
                "severity": "error",
                "fix_hint": "Use logging",
            }
        ]
        (repo_dir / "lint_rules.json").write_text(json.dumps(custom_rules))

        code = "print('hello')\n"
        violations = check_file(Path("app.py"), code, repo_name="my-repo")
        rule_ids = [v.rule_id for v in violations]
        assert "no-print-stmt" in rule_ids

    def test_custom_rule_with_wrong_file_pattern_skipped(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        custom_rules = [
            {
                "id": "js-only",
                "pattern": r"console\.log",
                "file_pattern": r"\.js$",
                "message": "No console.log",
                "severity": "warning",
                "fix_hint": "",
            }
        ]
        (repo_dir / "lint_rules.json").write_text(json.dumps(custom_rules))

        code = "console.log('test')\n"
        violations = check_file(Path("app.py"), code, repo_name="my-repo")
        rule_ids = [v.rule_id for v in violations]
        assert "js-only" not in rule_ids


# ---------------------------------------------------------------------------
# run_lint_on_patches
# ---------------------------------------------------------------------------

class TestRunLintOnPatches:

    def test_runs_on_existing_file(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text(
            "def fetch(url):\n"
            "    pass\n\n"
            "try:\n"
            "    x = 1\n"
            "except:\n"
            "    pass\n"
        )
        patches = [{"file_path": "app.py"}]
        violations = run_lint_on_patches(patches, repo)
        rule_ids = [v["rule_id"] for v in violations]
        assert "no-bare-except" in rule_ids

    def test_skips_missing_file(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        patches = [{"file_path": "nonexistent.py"}]
        violations = run_lint_on_patches(patches, repo)
        assert violations == []

    def test_skips_patch_without_file_path(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        violations = run_lint_on_patches([{}], repo)
        assert violations == []

    def test_returns_list_of_dicts(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("try:\n    x=1\nexcept:\n    pass\n")
        violations = run_lint_on_patches([{"file_path": "app.py"}], repo)
        assert isinstance(violations, list)
        if violations:
            v = violations[0]
            assert "rule_id" in v
            assert "file" in v
            assert "line" in v
            assert "message" in v
            assert "severity" in v
            assert "fix_hint" in v

    def test_multiple_patches(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("try:\n    x=1\nexcept:\n    pass\n")
        (repo / "b.py").write_text("API_KEY = 'sk-live-abcdefghijklmnopqrst'\n")
        patches = [{"file_path": "a.py"}, {"file_path": "b.py"}]
        violations = run_lint_on_patches(patches, repo)
        rule_ids = {v["rule_id"] for v in violations}
        assert "no-bare-except" in rule_ids
        assert "no-hardcoded-secrets" in rule_ids


# ---------------------------------------------------------------------------
# generate_default_rules
# ---------------------------------------------------------------------------

class TestGenerateDefaultRules:

    def _write_graph(self, tmp_path, repo_name, tech_stack):
        repo_dir = tmp_path / repo_name
        repo_dir.mkdir(exist_ok=True)
        graph = {"stats": {"tech_stack": tech_stack}}
        (repo_dir / "graph.json").write_text(json.dumps(graph))

    def test_no_graph_returns_empty(self, tmp_path):
        rules = generate_default_rules("no-graph-repo")
        assert rules == []

    def test_fastapi_generates_async_rule(self, tmp_path):
        self._write_graph(tmp_path, "myrepo", ["FastAPI", "Python"])
        rules = generate_default_rules("myrepo")
        rule_ids = [r["id"] for r in rules]
        assert "async-consistency" in rule_ids

    def test_sqlalchemy_generates_no_raw_sql_rule(self, tmp_path):
        self._write_graph(tmp_path, "myrepo", ["Python", "SQLAlchemy"])
        rules = generate_default_rules("myrepo")
        rule_ids = [r["id"] for r in rules]
        assert "no-raw-sql" in rule_ids

    def test_redis_generates_key_prefix_rule(self, tmp_path):
        self._write_graph(tmp_path, "myrepo", ["Python", "Redis"])
        rules = generate_default_rules("myrepo")
        rule_ids = [r["id"] for r in rules]
        assert "redis-key-prefix" in rule_ids

    def test_no_special_tech_returns_empty(self, tmp_path):
        self._write_graph(tmp_path, "myrepo", ["Go", "Docker"])
        rules = generate_default_rules("myrepo")
        assert rules == []

    def test_rules_saved_to_disk(self, tmp_path):
        self._write_graph(tmp_path, "myrepo", ["FastAPI"])
        generate_default_rules("myrepo")
        rules_path = tmp_path / "myrepo" / "lint_rules.json"
        assert rules_path.exists()
        saved = json.loads(rules_path.read_text())
        assert len(saved) >= 1

    def test_corrupt_graph_json_returns_empty(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "graph.json").write_text("NOT JSON")
        rules = generate_default_rules("myrepo")
        assert rules == []
