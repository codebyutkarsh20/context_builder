"""Unit tests for analyzer.changes — change detection and risk scoring."""

from analyzer.changes import (
    _parse_unified_diff,
    compute_change_risk,
    find_affected_flows,
    map_changes_to_nodes,
)
from analyzer.flows import _build_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(nodes, edges):
    return {"nodes": nodes, "edges": edges, "hotspots": []}


def _fn(id, label=None, line_start=0, line_end=0, **kw):
    n = {
        "id": id,
        "label": label or id.rsplit("::", 1)[-1],
        "type": "function",
        "file": id.split("::")[0],
        "line_start": line_start,
        "line_end": line_end,
    }
    n.update(kw)
    return n


def _call(src, tgt):
    return {"source": src, "target": tgt, "type": "CALLS"}


def _tested_by(src, tgt):
    return {"source": src, "target": tgt, "type": "TESTED_BY"}


# ---------------------------------------------------------------------------
# parse_unified_diff
# ---------------------------------------------------------------------------

class TestParseUnifiedDiff:
    def test_single_hunk(self):
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,3 +10,5 @@ def bar():\n"
        )
        result = _parse_unified_diff(diff)
        assert "foo.py" in result
        assert result["foo.py"] == [(10, 14)]

    def test_multiple_hunks(self):
        diff = (
            "+++ b/foo.py\n"
            "@@ -5,2 +5,3 @@\n"
            "@@ -20,1 +21,4 @@\n"
        )
        result = _parse_unified_diff(diff)
        assert result["foo.py"] == [(5, 7), (21, 24)]

    def test_single_line_hunk(self):
        diff = (
            "+++ b/foo.py\n"
            "@@ -10,1 +10 @@\n"
        )
        result = _parse_unified_diff(diff)
        assert result["foo.py"] == [(10, 10)]

    def test_deletion_hunk(self):
        diff = (
            "+++ b/foo.py\n"
            "@@ -10,3 +10,0 @@\n"
        )
        result = _parse_unified_diff(diff)
        assert result["foo.py"] == [(10, 10)]

    def test_multiple_files(self):
        diff = (
            "+++ b/a.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+++ b/b.py\n"
            "@@ -5,1 +5,3 @@\n"
        )
        result = _parse_unified_diff(diff)
        assert "a.py" in result
        assert "b.py" in result

    def test_empty_diff(self):
        assert _parse_unified_diff("") == {}


# ---------------------------------------------------------------------------
# map_changes_to_nodes
# ---------------------------------------------------------------------------

class TestMapChangesToNodes:
    def test_overlap_detected(self):
        g = _make_graph(
            [_fn("a.py::foo", line_start=10, line_end=20)],
            [],
        )
        result = map_changes_to_nodes(g, {"a.py": [(15, 15)]})
        assert len(result) == 1
        assert result[0]["id"] == "a.py::foo"

    def test_no_overlap(self):
        g = _make_graph(
            [_fn("a.py::foo", line_start=10, line_end=20)],
            [],
        )
        result = map_changes_to_nodes(g, {"a.py": [(25, 30)]})
        assert len(result) == 0

    def test_boundary_overlap(self):
        g = _make_graph(
            [_fn("a.py::foo", line_start=10, line_end=20)],
            [],
        )
        # Change at line 20 (last line of function)
        result = map_changes_to_nodes(g, {"a.py": [(20, 20)]})
        assert len(result) == 1

    def test_suffix_path_match(self):
        g = _make_graph(
            [_fn("src/a.py::foo", label="foo", line_start=5, line_end=15)],
            [],
        )
        # Diff uses relative path without src/ prefix
        result = map_changes_to_nodes(g, {"a.py": [(10, 10)]})
        # suffix match: "src/a.py" ends with "a.py"
        assert len(result) == 1

    def test_nodes_without_lines_skipped(self):
        g = _make_graph(
            [_fn("a.py::foo", line_start=0, line_end=0)],
            [],
        )
        result = map_changes_to_nodes(g, {"a.py": [(1, 100)]})
        assert len(result) == 0

    def test_deduplication(self):
        g = _make_graph(
            [_fn("a.py::foo", line_start=10, line_end=20)],
            [],
        )
        # Multiple ranges overlap the same function
        result = map_changes_to_nodes(g, {"a.py": [(12, 12), (18, 18)]})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# compute_change_risk
# ---------------------------------------------------------------------------

class TestComputeChangeRisk:
    def test_untested_gets_higher_risk(self):
        g = _make_graph(
            [_fn("a.py::foo"), _fn("a.py::bar")],
            [_tested_by("test.py::test_foo", "a.py::foo")],
        )
        nodes_by_id, calls_forward, called_ids, tested_by, calls_reverse = _build_indices(g)
        risk_tested = compute_change_risk(
            g["nodes"][0], nodes_by_id, calls_forward, called_ids, tested_by,
            calls_reverse=calls_reverse,
        )
        risk_untested = compute_change_risk(
            g["nodes"][1], nodes_by_id, calls_forward, called_ids, tested_by,
            calls_reverse=calls_reverse,
        )
        assert risk_untested > risk_tested

    def test_security_keyword_boost(self):
        g = _make_graph(
            [_fn("auth.py::validate_token"), _fn("utils.py::format_string")],
            [],
        )
        nodes_by_id, calls_forward, called_ids, tested_by, calls_reverse = _build_indices(g)
        risk_auth = compute_change_risk(
            g["nodes"][0], nodes_by_id, calls_forward, called_ids, tested_by,
            calls_reverse=calls_reverse,
        )
        risk_util = compute_change_risk(
            g["nodes"][1], nodes_by_id, calls_forward, called_ids, tested_by,
            calls_reverse=calls_reverse,
        )
        assert risk_auth > risk_util

    def test_flow_participation_boost(self):
        g = _make_graph([_fn("a.py::foo")], [])
        nodes_by_id, calls_forward, called_ids, tested_by, calls_reverse = _build_indices(g)
        flows = [{"path": ["a.py::foo", "b.py::bar"]}]
        risk_in_flow = compute_change_risk(
            g["nodes"][0], nodes_by_id, calls_forward, called_ids, tested_by, flows,
            calls_reverse=calls_reverse,
        )
        risk_no_flow = compute_change_risk(
            g["nodes"][0], nodes_by_id, calls_forward, called_ids, tested_by, [],
            calls_reverse=calls_reverse,
        )
        assert risk_in_flow > risk_no_flow

    def test_many_callers_boost(self):
        callers = [_fn(f"c.py::caller{i}") for i in range(10)]
        target = _fn("a.py::popular")
        edges = [_call(c["id"], "a.py::popular") for c in callers]
        g = _make_graph([target] + callers, edges)
        nodes_by_id, calls_forward, called_ids, tested_by, calls_reverse = _build_indices(g)
        risk = compute_change_risk(
            target, nodes_by_id, calls_forward, called_ids, tested_by,
            calls_reverse=calls_reverse,
        )
        # Should have caller_count contribution
        assert risk > 0.20  # at least untested (0.20) + some caller score

    def test_score_capped_at_1(self):
        # Worst case: security keyword, many callers, untested, in many flows
        node = _fn("auth.py::validate_token")
        callers = [_fn(f"c.py::caller{i}") for i in range(30)]
        edges = [_call(c["id"], "auth.py::validate_token") for c in callers]
        g = _make_graph([node] + callers, edges)
        nodes_by_id, calls_forward, called_ids, tested_by, calls_reverse = _build_indices(g)
        flows = [{"path": ["auth.py::validate_token"]} for _ in range(20)]
        risk = compute_change_risk(
            node, nodes_by_id, calls_forward, called_ids, tested_by, flows,
            calls_reverse=calls_reverse,
        )
        assert risk <= 1.0


# ---------------------------------------------------------------------------
# find_affected_flows
# ---------------------------------------------------------------------------

class TestFindAffectedFlows:
    def test_finds_matching_flow(self):
        flows = [
            {"name": "login", "path": ["a.py::login", "a.py::validate"], "criticality": 0.8},
            {"name": "signup", "path": ["b.py::signup"], "criticality": 0.3},
        ]
        affected = find_affected_flows({"a.py::validate"}, flows)
        assert len(affected) == 1
        assert affected[0]["name"] == "login"

    def test_no_match(self):
        flows = [{"name": "login", "path": ["a.py::login"], "criticality": 0.5}]
        affected = find_affected_flows({"b.py::unrelated"}, flows)
        assert len(affected) == 0

    def test_sorted_by_criticality(self):
        flows = [
            {"name": "low", "path": ["a.py::foo"], "criticality": 0.2},
            {"name": "high", "path": ["a.py::foo"], "criticality": 0.9},
        ]
        affected = find_affected_flows({"a.py::foo"}, flows)
        assert affected[0]["name"] == "high"

    def test_empty_flows(self):
        assert find_affected_flows({"a.py::foo"}, []) == []
