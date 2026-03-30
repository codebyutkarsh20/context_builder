"""
Tests for Phase 2B multi-file coordination — PLAN.md Test Plan.

Covers:
  - multi_file_coordinator_node: patches caller files when signatures change
  - Reviewer caller completeness check: surfaces unpatched callers
  - deduplicate_patches after retry merge: same-file conflict → best patch wins
  - _find_callers_from_graph: test files filtered from blast radius
  - _load_graph_data: no caching (documents current behavior for Phase 2B-4)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import (
    multi_file_coordinator_node,
    _find_callers_from_graph,
    _load_graph_data,
)
from agent.patch_utils import deduplicate_patches


# ---------------------------------------------------------------------------
# 1. test_multi_file_patch_applies — coordinator patches caller files
# ---------------------------------------------------------------------------

class TestMultiFilePatchApplies:
    """multi_file_coordinator_node should add caller patches when the fix
    changes a public interface (e.g., renamed function)."""

    def _make_state(self, patches, caller_files, caller_source_map, repo_path):
        return {
            "repair": {
                "patches": patches,
                "explanation": "Renamed process_order to handle_order",
            },
            "caller_files": caller_files,
            "localization": {"fault_files": ["services/order.py"]},
            "work_order": {
                "ticket_id": "TEST-MF-1",
                "repo_name": "test-repo",
                "repo_path": str(repo_path),
            },
        }

    def test_coordinator_adds_caller_patch(self, tmp_path):
        """When a fix renames a function, the coordinator should patch callers."""
        # Setup: fault file already patched, caller file needs updating
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "services").mkdir()
        (repo / "services" / "order.py").write_text(
            "def handle_order(order_id):\n    return db.get(order_id)\n"
        )
        (repo / "api").mkdir()
        (repo / "api" / "views.py").write_text(
            "from services.order import process_order\n\n"
            "def create_order(request):\n"
            "    return process_order(request.order_id)\n"
        )

        patches = [{"file_path": "services/order.py", "original_code": "process_order", "patched_code": "handle_order"}]
        caller_files = ["api/views.py"]

        state = self._make_state(patches, caller_files, {}, repo)

        # Mock the LLM call to return a caller patch
        mock_repair_result = MagicMock()
        mock_repair_result.patches = [
            {
                "file_path": "api/views.py",
                "original_code": "from services.order import process_order",
                "patched_code": "from services.order import handle_order",
            }
        ]

        with patch("agent.pipeline._structured_call", return_value=mock_repair_result):
            with patch("agent.pipeline._find_file_in_repo", side_effect=lambda rp, cp: rp / cp):
                result = multi_file_coordinator_node(state)

        # Verify caller patch was merged into repair patches
        result_patches = result["repair"]["patches"]
        patched_files = [p["file_path"] for p in result_patches]
        assert "api/views.py" in patched_files, (
            f"Caller file api/views.py should be patched. Got: {patched_files}"
        )
        assert len(result_patches) == 2  # original + caller

    def test_coordinator_skips_when_no_callers(self):
        """No caller_files → coordinator returns state unchanged."""
        state = {
            "repair": {"patches": [{"file_path": "a.py", "original_code": "x", "patched_code": "y"}]},
            "caller_files": [],
        }
        result = multi_file_coordinator_node(state)
        assert result["repair"]["patches"] == state["repair"]["patches"]

    def test_coordinator_skips_when_callers_already_patched(self):
        """All callers already have patches → no additional patches needed."""
        state = {
            "repair": {"patches": [
                {"file_path": "services/order.py", "original_code": "x", "patched_code": "y"},
                {"file_path": "api/views.py", "original_code": "a", "patched_code": "b"},
            ]},
            "caller_files": ["api/views.py"],
            "localization": {"fault_files": ["services/order.py"]},
        }
        result = multi_file_coordinator_node(state)
        assert len(result["repair"]["patches"]) == 2  # unchanged

    def test_coordinator_rejects_patch_failing_source_verification(self, tmp_path):
        """Coordinator patch that doesn't match caller source is rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "caller.py").write_text("import something_else\n")

        state = {
            "repair": {
                "patches": [{"file_path": "fault.py", "original_code": "x", "patched_code": "y"}],
                "explanation": "fix",
            },
            "caller_files": ["caller.py"],
            "localization": {"fault_files": ["fault.py"]},
            "work_order": {"ticket_id": "T-1", "repo_name": "r", "repo_path": str(repo)},
        }

        mock_result = MagicMock()
        mock_result.patches = [
            {"file_path": "caller.py", "original_code": "NONEXISTENT CODE", "patched_code": "fixed"}
        ]

        with patch("agent.pipeline._structured_call", return_value=mock_result):
            with patch("agent.pipeline._find_file_in_repo", side_effect=lambda rp, cp: rp / cp):
                result = multi_file_coordinator_node(state)

        # Original patch only — caller patch rejected because original_code not in source
        assert len(result["repair"]["patches"]) == 1


# ---------------------------------------------------------------------------
# 2. test_caller_completeness_check — reviewer surfaces unpatched callers
# ---------------------------------------------------------------------------

class TestCallerCompletenessCheck:
    """Reviewer prompt includes CALLER FILES NOT PATCHED warning when
    callers exist but aren't covered by patches (PLAN.md Phase 2B-3)."""

    def test_unpatched_caller_appears_in_reviewer_prompt(self):
        """When caller_files has files not in patches, the reviewer gets a warning."""
        # Simulate the logic from review_node lines 2148-2159
        caller_files = ["api/views.py", "api/router.py"]
        patches = [{"file_path": "services/order.py"}]

        patched_file_set = {p.get("file_path", "") for p in patches}
        unpatched = [c for c in caller_files if c not in patched_file_set]

        assert unpatched == ["api/views.py", "api/router.py"]
        assert len(unpatched) == 2

    def test_all_callers_patched_no_warning(self):
        """When all callers are patched, no warning is generated."""
        caller_files = ["api/views.py"]
        patches = [
            {"file_path": "services/order.py"},
            {"file_path": "api/views.py"},
        ]

        patched_file_set = {p.get("file_path", "") for p in patches}
        unpatched = [c for c in caller_files if c not in patched_file_set]

        assert unpatched == []

    def test_empty_caller_files_no_warning(self):
        """No caller_files → no warning."""
        caller_files = []
        patches = [{"file_path": "services/order.py"}]

        patched_file_set = {p.get("file_path", "") for p in patches}
        unpatched = [c for c in caller_files if c not in patched_file_set]

        assert unpatched == []


# ---------------------------------------------------------------------------
# 3. test_dedup_after_retry_merge — same-file conflict resolved
# ---------------------------------------------------------------------------

class TestDedupAfterRetryMerge:
    """deduplicate_patches removes same-file duplicate patches after retry
    merge (PLAN.md Phase 2A-2, pipeline.py:1860-1862)."""

    def test_dedup_keeps_last_for_same_file_same_original(self):
        """When retry produces a second patch for the same (file, original_code),
        dedup keeps the last one (dict overwrite semantics)."""
        patches = [
            {"file_path": "app.py", "original_code": "def foo():\n    return 1", "patched_code": "def foo():\n    return 2"},
            {"file_path": "app.py", "original_code": "def foo():\n    return 1", "patched_code": "def foo():\n    return 3"},
        ]
        result = deduplicate_patches(patches)
        assert len(result) == 1
        assert result[0]["patched_code"] == "def foo():\n    return 3"

    def test_dedup_preserves_different_files(self):
        """Patches for different files are all kept."""
        patches = [
            {"file_path": "a.py", "original_code": "x", "patched_code": "y"},
            {"file_path": "b.py", "original_code": "x", "patched_code": "z"},
        ]
        result = deduplicate_patches(patches)
        assert len(result) == 2

    def test_dedup_preserves_different_original_code_same_file(self):
        """Two patches for the same file but different original_code are both kept."""
        patches = [
            {"file_path": "app.py", "original_code": "def foo(): pass", "patched_code": "def foo(): return 1"},
            {"file_path": "app.py", "original_code": "def bar(): pass", "patched_code": "def bar(): return 2"},
        ]
        result = deduplicate_patches(patches)
        assert len(result) == 2

    def test_dedup_uses_first_200_chars(self):
        """Key uses first 200 chars of original_code — patches differing only
        after char 200 are considered duplicates."""
        base = "x" * 200
        patches = [
            {"file_path": "a.py", "original_code": base + "AAA", "patched_code": "v1"},
            {"file_path": "a.py", "original_code": base + "BBB", "patched_code": "v2"},
        ]
        result = deduplicate_patches(patches)
        # Both have same key (first 200 chars identical) → last one wins
        assert len(result) == 1
        assert result[0]["patched_code"] == "v2"

    def test_dedup_empty_list(self):
        """Empty input returns empty output."""
        assert deduplicate_patches([]) == []


# ---------------------------------------------------------------------------
# 4. test_test_file_not_in_callers — full multi-file flow
# ---------------------------------------------------------------------------

class TestTestFileNotInCallersMultiFile:
    """_find_callers_from_graph filters test/conftest files from blast radius
    results, ensuring test files don't get patched by the coordinator.
    Extends the Phase 2A tests to cover the multi-file flow."""

    def _make_graph(self, edges):
        return {"edges": [{"type": "CALLS", "source": s, "target": t} for s, t in edges]}

    def test_test_files_excluded_from_callers(self):
        """test_ prefixed files and /tests/ directory files are filtered out."""
        graph = self._make_graph([
            ("api/views.py", "services/order.py"),
            ("test_order.py", "services/order.py"),
            ("tests/test_integration.py", "services/order.py"),
        ])
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert "api/views.py" in callers
        assert "test_order.py" not in callers
        assert "tests/test_integration.py" not in callers

    def test_conftest_excluded(self):
        graph = self._make_graph([
            ("conftest.py", "services/order.py"),
            ("api/router.py", "services/order.py"),
        ])
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert "conftest.py" not in callers
        assert "api/router.py" in callers

    def test_pycache_excluded(self):
        graph = self._make_graph([
            ("app/__pycache__/order.cpython-311.pyc", "services/order.py"),
        ])
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert len(callers) == 0

    def test_fault_file_not_its_own_caller(self):
        """The fault file itself must not appear in caller list."""
        graph = self._make_graph([
            ("services/order.py", "services/order.py"),
            ("api/views.py", "services/order.py"),
        ])
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert "services/order.py" not in callers
        assert "api/views.py" in callers

    def test_max_8_callers_returned(self):
        """At most 8 callers are returned (sorted by path)."""
        edges = [(f"caller_{i:02d}.py", "fault.py") for i in range(15)]
        graph = self._make_graph(edges)
        callers = _find_callers_from_graph(graph, ["fault.py"], [])
        assert len(callers) <= 8

    def test_imports_edge_type_included(self):
        """IMPORTS edges are also followed, not just CALLS."""
        graph = {"edges": [
            {"type": "IMPORTS", "source": "api/views.py", "target": "services/order.py"},
        ]}
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert "api/views.py" in callers

    def test_other_edge_types_ignored(self):
        """Edges that are neither CALLS nor IMPORTS are ignored."""
        graph = {"edges": [
            {"type": "CONTAINS", "source": "api/views.py", "target": "services/order.py"},
        ]}
        callers = _find_callers_from_graph(graph, ["services/order.py"], [])
        assert len(callers) == 0


# ---------------------------------------------------------------------------
# 5. test_graph_data_cached — documents current behavior (no cache)
# ---------------------------------------------------------------------------

class TestGraphDataCaching:
    """_load_graph_data currently reads from disk on every call.
    These tests document this behavior — Phase 2B-4 should add caching."""

    def test_load_reads_graph_json(self, tmp_path):
        """_load_graph_data reads graph.json from DATA_DIR/{repo}/."""
        repo_dir = tmp_path / "test-repo"
        repo_dir.mkdir()
        graph = {"nodes": [{"id": "f:a.py"}], "edges": [], "stats": {"files": 1}}
        (repo_dir / "graph.json").write_text(json.dumps(graph))

        with patch("agent.pipeline.DATA_DIR", tmp_path):
            result_graph, result_enriched = _load_graph_data("test-repo")

        assert result_graph["nodes"][0]["id"] == "f:a.py"
        assert result_enriched == {}  # no enriched_nodes.json

    def test_load_reads_enriched_nodes(self, tmp_path):
        """_load_graph_data also reads enriched_nodes.json if present."""
        repo_dir = tmp_path / "test-repo"
        repo_dir.mkdir()
        (repo_dir / "graph.json").write_text('{"nodes": [], "edges": []}')
        enriched = {"business_rules": [{"id": "BR-1", "text": "No negative prices"}]}
        (repo_dir / "enriched_nodes.json").write_text(json.dumps(enriched))

        with patch("agent.pipeline.DATA_DIR", tmp_path):
            _, result_enriched = _load_graph_data("test-repo")

        assert result_enriched["business_rules"][0]["id"] == "BR-1"

    def test_load_returns_empty_on_missing_files(self, tmp_path):
        """Missing graph files return empty dicts (no crash)."""
        repo_dir = tmp_path / "test-repo"
        repo_dir.mkdir()

        with patch("agent.pipeline.DATA_DIR", tmp_path):
            graph_data, enriched = _load_graph_data("test-repo")

        assert graph_data == {}
        assert enriched == {}

    def test_consecutive_loads_both_read_disk(self, tmp_path):
        """Two calls to _load_graph_data both read from disk (no cache).
        This test documents the current behavior; Phase 2B-4 should add
        mtime-based caching."""
        repo_dir = tmp_path / "test-repo"
        repo_dir.mkdir()
        graph = {"nodes": [], "edges": []}
        (repo_dir / "graph.json").write_text(json.dumps(graph))

        with patch("agent.pipeline.DATA_DIR", tmp_path):
            with patch("agent.pipeline.json.loads", wraps=json.loads) as mock_loads:
                _load_graph_data("test-repo")
                _load_graph_data("test-repo")

        # json.loads called at least twice (once per call) — proves no cache
        assert mock_loads.call_count >= 2, (
            f"Expected >=2 json.loads calls (no cache), got {mock_loads.call_count}"
        )
