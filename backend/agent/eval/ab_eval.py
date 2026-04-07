"""
ab_eval.py -- A/B evaluation runner.

Runs the eval suite twice:
  A) Full pipeline (scout + BRT enabled)
  B) Baseline (scout + BRT disabled)

Produces a comparison report showing per-bug and aggregate metrics.

Usage:
  python -m agent.eval.ab_eval [--bugs BUG_ID,...] [--build-graph]
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from agent.eval.runner import EvalRunner, EvalRunReport, RESULTS_DIR
from agent.eval.scoring import build_summary

logger = logging.getLogger(__name__)

# Metrics used for the comparison table
COMPARISON_METRICS = [
    ("pass_rate", "Pass Rate", "%"),
    ("localization_accuracy", "Localization", "%"),
    ("fix_rate", "Fix Rate", "%"),
    ("approval_rate", "Approval Rate", "%"),
    ("test_pass_rate", "Test Pass Rate", "%"),
    ("avg_cost_usd", "Avg Cost ($)", "$"),
    ("avg_tool_calls", "Avg Tool Calls", "#"),
    ("avg_duration_seconds", "Avg Duration (s)", "s"),
]


def run_ab_eval(
    dataset_path: str | Path = "eval/bugs.json",
    bug_filter: str | None = None,
    sentinel: bool = False,
    timeout_per_case: int = 600,
    results_dir: str | Path = RESULTS_DIR,
    build_graph: bool = False,
    progress_cb: Callable[[str, str, int, int], None] | None = None,
) -> dict:
    """Run A/B comparison: full pipeline vs baseline (no scout, no BRT).

    Parameters
    ----------
    dataset_path : str or Path
        Path to the eval bugs JSON file.
    bug_filter : str or None
        If set, run only the bug with this ticket_id.
    sentinel : bool
        If True, run only the first 5 bugs.
    timeout_per_case : int
        Per-case timeout in seconds.
    results_dir : str or Path
        Directory for output files.
    build_graph : bool
        If True, build the knowledge graph before running the agent.
    progress_cb : callable or None
        Called with (arm_label, ticket_id, current_index, total) after each bug.

    Returns
    -------
    dict
        Comparison report with per-bug results, aggregate metrics, and deltas.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    # ------------------------------------------------------------------
    # Arm A: Full pipeline (scout + BRT enabled) -- uses "react" pipeline
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("A/B Eval — Arm A: Full pipeline (scout + BRT enabled)")
    logger.info("=" * 70)

    runner_full = EvalRunner(
        dataset_path=dataset_path,
        pipelines=["react"],
        timeout_per_case=timeout_per_case,
        results_dir=results_dir,
        build_graph=build_graph,
    )

    def _progress_a(tid: str, cur: int, total: int) -> None:
        if progress_cb:
            progress_cb("full", tid, cur, total)

    report_full = runner_full.run(
        bug_filter=bug_filter,
        sentinel=sentinel,
        progress_cb=_progress_a,
    )

    # Save arm-A results
    full_path = results_dir / f"ab_full_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(report_full.to_dict(), f, indent=2, default=str)
    logger.info("Arm A results saved to %s", full_path)

    # ------------------------------------------------------------------
    # Arm B: Baseline (scout + BRT disabled) -- uses "react_v2" pipeline
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("A/B Eval — Arm B: Baseline (scout + BRT disabled)")
    logger.info("=" * 70)

    runner_baseline = EvalRunner(
        dataset_path=dataset_path,
        pipelines=["react_v2"],
        timeout_per_case=timeout_per_case,
        results_dir=results_dir,
        build_graph=build_graph,
    )

    def _progress_b(tid: str, cur: int, total: int) -> None:
        if progress_cb:
            progress_cb("baseline", tid, cur, total)

    report_baseline = runner_baseline.run(
        bug_filter=bug_filter,
        sentinel=sentinel,
        progress_cb=_progress_b,
    )

    # Save arm-B results
    baseline_path = results_dir / f"ab_baseline_{ts}.json"
    with open(baseline_path, "w") as f:
        json.dump(report_baseline.to_dict(), f, indent=2, default=str)
    logger.info("Arm B results saved to %s", baseline_path)

    # ------------------------------------------------------------------
    # Build comparison
    # ------------------------------------------------------------------
    comparison = _build_ab_comparison(report_full, report_baseline)
    comparison["metadata"] = {
        "timestamp": ts,
        "dataset_path": str(dataset_path),
        "bug_filter": bug_filter,
        "sentinel": sentinel,
        "build_graph": build_graph,
        "full_run_id": report_full.run_id,
        "baseline_run_id": report_baseline.run_id,
        "full_results_path": str(full_path),
        "baseline_results_path": str(baseline_path),
    }

    # Save comparison report
    comparison_path = results_dir / f"ab_comparison_{ts}.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison report saved to %s", comparison_path)

    return comparison


def _build_ab_comparison(
    report_full: EvalRunReport,
    report_baseline: EvalRunReport,
) -> dict:
    """Build a structured comparison between full and baseline runs.

    Returns
    -------
    dict
        Keys: aggregate, per_bug, winner.
    """
    summary_full = report_full.summary.get("react", {})
    summary_baseline = report_baseline.summary.get("react_v2", {})

    # Aggregate metric deltas
    aggregate: dict[str, dict] = {}
    for metric_key, label, unit in COMPARISON_METRICS:
        val_full = summary_full.get(metric_key, 0)
        val_baseline = summary_baseline.get(metric_key, 0)
        delta = round(val_full - val_baseline, 4) if isinstance(val_full, (int, float)) and isinstance(val_baseline, (int, float)) else None
        aggregate[metric_key] = {
            "label": label,
            "unit": unit,
            "full": val_full,
            "baseline": val_baseline,
            "delta": delta,
        }

    # Per-bug comparison
    full_by_bug: dict[str, dict] = {}
    for r in report_full.results:
        full_by_bug[r.ticket_id] = {
            "pass": r.score.get("full_pass", False),
            "localization_hit": r.score.get("localization_hit", False),
            "fix_generated": r.score.get("fix_generated", False),
            "cost_usd": r.cost_usd,
            "tool_calls": r.score.get("tool_call_count", 0),
            "duration_seconds": r.duration_seconds,
        }

    baseline_by_bug: dict[str, dict] = {}
    for r in report_baseline.results:
        baseline_by_bug[r.ticket_id] = {
            "pass": r.score.get("full_pass", False),
            "localization_hit": r.score.get("localization_hit", False),
            "fix_generated": r.score.get("fix_generated", False),
            "cost_usd": r.cost_usd,
            "tool_calls": r.score.get("tool_call_count", 0),
            "duration_seconds": r.duration_seconds,
        }

    all_bugs = sorted(set(full_by_bug.keys()) | set(baseline_by_bug.keys()))
    per_bug: dict[str, dict] = {}
    for bug_id in all_bugs:
        fb = full_by_bug.get(bug_id, {})
        bb = baseline_by_bug.get(bug_id, {})
        full_pass = fb.get("pass", False)
        base_pass = bb.get("pass", False)

        if full_pass and not base_pass:
            verdict = "full_only"
        elif base_pass and not full_pass:
            verdict = "baseline_only"
        elif full_pass and base_pass:
            verdict = "both_pass"
        else:
            verdict = "both_fail"

        per_bug[bug_id] = {
            "full": fb,
            "baseline": bb,
            "verdict": verdict,
        }

    # Overall winner determination
    full_pass_rate = summary_full.get("pass_rate", 0)
    baseline_pass_rate = summary_baseline.get("pass_rate", 0)
    if full_pass_rate > baseline_pass_rate:
        winner = "full"
        winner_reason = f"full pipeline pass rate ({full_pass_rate:.0%}) > baseline ({baseline_pass_rate:.0%})"
    elif baseline_pass_rate > full_pass_rate:
        winner = "baseline"
        winner_reason = f"baseline pass rate ({baseline_pass_rate:.0%}) > full ({full_pass_rate:.0%})"
    else:
        full_cost = summary_full.get("avg_cost_usd", 0)
        base_cost = summary_baseline.get("avg_cost_usd", 0)
        if full_cost <= base_cost:
            winner = "full"
            winner_reason = f"tied on pass rate ({full_pass_rate:.0%}), full cheaper (${full_cost:.2f} vs ${base_cost:.2f})"
        else:
            winner = "baseline"
            winner_reason = f"tied on pass rate ({baseline_pass_rate:.0%}), baseline cheaper (${base_cost:.2f} vs ${full_cost:.2f})"

    return {
        "aggregate": aggregate,
        "per_bug": per_bug,
        "winner": winner,
        "winner_reason": winner_reason,
    }


def format_comparison_table(comparison: dict) -> str:
    """Format the comparison as a human-readable table string.

    Returns a string suitable for console printing.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  A/B EVAL COMPARISON: Full Pipeline vs Baseline")
    lines.append("=" * 72)

    # Aggregate metrics table
    lines.append("")
    lines.append(f"  {'Metric':<22} {'Full':>10} {'Baseline':>10} {'Delta':>10}")
    lines.append(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10}")

    aggregate = comparison.get("aggregate", {})
    for metric_key, label, unit in COMPARISON_METRICS:
        entry = aggregate.get(metric_key, {})
        val_full = entry.get("full", 0)
        val_baseline = entry.get("baseline", 0)
        delta = entry.get("delta")

        if unit == "%":
            full_str = f"{val_full:.1%}" if isinstance(val_full, float) else str(val_full)
            base_str = f"{val_baseline:.1%}" if isinstance(val_baseline, float) else str(val_baseline)
            delta_str = f"{delta:+.1%}" if delta is not None else "N/A"
        elif unit == "$":
            full_str = f"${val_full:.2f}"
            base_str = f"${val_baseline:.2f}"
            delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
        else:
            full_str = f"{val_full:.1f}" if isinstance(val_full, float) else str(val_full)
            base_str = f"{val_baseline:.1f}" if isinstance(val_baseline, float) else str(val_baseline)
            delta_str = f"{delta:+.1f}" if delta is not None else "N/A"

        lines.append(f"  {label:<22} {full_str:>10} {base_str:>10} {delta_str:>10}")

    # Per-bug verdicts
    per_bug = comparison.get("per_bug", {})
    if per_bug:
        lines.append("")
        lines.append(f"  {'Bug ID':<20} {'Full':>8} {'Base':>8} {'Verdict':<15}")
        lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 15}")

        for bug_id, data in per_bug.items():
            full_pass = "PASS" if data.get("full", {}).get("pass") else "FAIL"
            base_pass = "PASS" if data.get("baseline", {}).get("pass") else "FAIL"
            verdict = data.get("verdict", "unknown")
            lines.append(f"  {bug_id:<20} {full_pass:>8} {base_pass:>8} {verdict:<15}")

    # Verdict counts
    verdicts = [d["verdict"] for d in per_bug.values()]
    full_only = verdicts.count("full_only")
    baseline_only = verdicts.count("baseline_only")
    both_pass = verdicts.count("both_pass")
    both_fail = verdicts.count("both_fail")

    lines.append("")
    lines.append(f"  Summary: {both_pass} both pass, {both_fail} both fail, "
                 f"{full_only} full-only, {baseline_only} baseline-only")

    # Winner
    lines.append("")
    winner = comparison.get("winner", "unknown")
    reason = comparison.get("winner_reason", "")
    lines.append(f"  Winner: {winner.upper()}")
    lines.append(f"  Reason: {reason}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point (python -m agent.eval.ab_eval)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run A/B evaluation: full pipeline vs baseline (no scout, no BRT)."
    )
    parser.add_argument(
        "--bugs", type=str, default=None,
        help="Comma-separated bug IDs to run (default: all).",
    )
    parser.add_argument(
        "--dataset", type=str, default="eval/bugs.json",
        help="Path to eval bugs JSON file.",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-case timeout in seconds.",
    )
    parser.add_argument(
        "--sentinel", action="store_true",
        help="Run only the first 5 bugs.",
    )
    parser.add_argument(
        "--build-graph", action="store_true",
        help="Build knowledge graph before running agent.",
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR),
        help="Results output directory.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # If --bugs is specified, run one at a time (the runner only supports single bug filter)
    bug_ids = [b.strip() for b in args.bugs.split(",")] if args.bugs else [None]

    for bug_id in bug_ids:
        comparison = run_ab_eval(
            dataset_path=args.dataset,
            bug_filter=bug_id,
            sentinel=args.sentinel,
            timeout_per_case=args.timeout,
            results_dir=args.output,
            build_graph=args.build_graph,
        )
        print(format_comparison_table(comparison))


if __name__ == "__main__":
    _main()
