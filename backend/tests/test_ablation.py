"""Tests for the component ablation framework.

These are pure-logic tests — no API calls, no eval runs. They cover the
ablation-flag state machine, the runner's arm→component mapping, and the
report-building / formatting functions that turn per-arm summaries into a
ranked contribution table.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ablation_flags
from agent.eval.ablation import (
    build_ablation_report,
    format_ablation_table,
    to_markdown,
)


# ── ablation_flags state machine ────────────────────────────────────────

def test_flags_default_nothing_disabled():
    ablation_flags.clear()
    for comp in ablation_flags.COMPONENTS:
        assert not ablation_flags.is_disabled(comp)


def test_flags_set_and_clear():
    ablation_flags.set_disabled({"scout", "verifier"})
    assert ablation_flags.is_disabled("scout")
    assert ablation_flags.is_disabled("verifier")
    assert not ablation_flags.is_disabled("graph")
    ablation_flags.clear()
    assert not ablation_flags.is_disabled("scout")


def test_flags_ignore_unknown_components():
    ablation_flags.set_disabled({"scout", "not_a_real_component"})
    assert ablation_flags.is_disabled("scout")
    assert ablation_flags.disabled_set() == {"scout"}
    ablation_flags.clear()


def test_flags_are_thread_local():
    """Flags set on one thread must not leak into another."""
    ablation_flags.set_disabled({"scout"})
    seen = {}

    def worker():
        seen["disabled_in_worker"] = ablation_flags.is_disabled("scout")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen["disabled_in_worker"] is False  # worker thread has its own state
    assert ablation_flags.is_disabled("scout")   # main thread unchanged
    ablation_flags.clear()


def test_label_known_and_unknown():
    assert ablation_flags.label("verifier") == "Forked verifier"
    assert ablation_flags.label("mystery") == "mystery"


# ── runner arm → component mapping ──────────────────────────────────────

def test_runner_ablation_pipeline_mapping():
    from agent.eval.runner import EvalRunner
    mapping = EvalRunner.ABLATION_PIPELINES
    # Every component has exactly one single-component arm.
    for comp in ablation_flags.COMPONENTS:
        assert mapping[f"react_no_{comp}"] == {comp}
    # The reference pipeline is not an ablation arm.
    assert "react" not in mapping


# ── report building ─────────────────────────────────────────────────────

def _sample_summary():
    """Reference passes 50%. Removing scout costs 30pp; verifier 10pp;
    lessons nothing; graph is net-negative (arm scores *higher*)."""
    return {
        "react":             {"pass_rate": 0.50, "test_pass_rate": 0.60, "avg_cost_usd": 0.55, "avg_tool_calls": 16},
        "react_no_scout":    {"pass_rate": 0.20, "test_pass_rate": 0.30, "avg_cost_usd": 0.40, "avg_tool_calls": 22},
        "react_no_verifier": {"pass_rate": 0.40, "test_pass_rate": 0.45, "avg_cost_usd": 0.48, "avg_tool_calls": 14},
        "react_no_lessons":  {"pass_rate": 0.50, "test_pass_rate": 0.60, "avg_cost_usd": 0.55, "avg_tool_calls": 16},
        "react_no_graph":    {"pass_rate": 0.55, "test_pass_rate": 0.62, "avg_cost_usd": 0.42, "avg_tool_calls": 18},
    }


def test_build_report_contributions():
    components = ["scout", "verifier", "lessons", "graph"]
    report = build_ablation_report(_sample_summary(), components)

    assert report["reference_pass_rate"] == 0.50
    assert report["components"]["scout"]["contribution"] == 0.30
    assert report["components"]["verifier"]["contribution"] == 0.10
    assert report["components"]["lessons"]["contribution"] == 0.0
    # Graph arm scored higher than reference → negative contribution.
    assert report["components"]["graph"]["contribution"] == -0.05


def test_build_report_ranking_is_descending_by_contribution():
    components = ["scout", "verifier", "lessons", "graph"]
    report = build_ablation_report(_sample_summary(), components)
    assert report["ranking"] == ["scout", "verifier", "lessons", "graph"]


def test_build_report_per_metric_deltas():
    report = build_ablation_report(_sample_summary(), ["scout"])
    cost = report["components"]["scout"]["metrics"]["avg_cost_usd"]
    assert cost["reference"] == 0.55
    assert cost["ablated"] == 0.40
    assert round(cost["delta"], 2) == 0.15


def test_build_report_handles_missing_arm_gracefully():
    # No arm data for "brt" → treated as 0 pass rate, contribution = full.
    report = build_ablation_report({"react": {"pass_rate": 0.5}}, ["brt"])
    assert report["components"]["brt"]["contribution"] == 0.5
    assert report["components"]["brt"]["pass_rate_ablated"] == 0.0


def test_build_report_empty_summary():
    report = build_ablation_report({}, ["scout"])
    assert report["reference_pass_rate"] == 0.0
    assert report["components"]["scout"]["contribution"] == 0.0


# ── formatting ──────────────────────────────────────────────────────────

def test_format_table_contains_labels_and_ranking():
    report = build_ablation_report(_sample_summary(), ["scout", "verifier", "lessons", "graph"])
    table = format_ablation_table(report)
    assert "COMPONENT ABLATION" in table
    assert "Scout localization" in table
    assert "50.0%" in table          # reference pass rate
    assert "+30.0%" in table         # scout contribution, signed
    # Scout (highest contribution) appears before graph (lowest) in the table.
    assert table.index("Scout localization") < table.index("Knowledge graph context")


def test_to_markdown_is_a_table():
    report = build_ablation_report(_sample_summary(), ["scout", "verifier"])
    md = to_markdown(report)
    assert md.startswith("# Component Ablation Report")
    assert "| Component | Pass rate w/o it | Contribution |" in md
    assert "Scout localization" in md
