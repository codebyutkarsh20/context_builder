"""ablation.py — component ablation eval.

Measures *what each part of the harness is worth*. The agent's pass rate comes
from the LLM plus a system of components — Scout localization, the knowledge
graph, the per-repo learning loop, BRT generation, and the forked verifier.
This runs the eval once with everything on (the reference arm) and once per
component with that component disabled, then attributes the pass-rate drop to
the component:

    contribution(component) = pass_rate(full) - pass_rate(full minus component)

A large positive contribution means the component is pulling its weight; a
near-zero or negative one means it costs tokens without earning pass rate.

The disabling is done by `EvalRunner`'s ablation arms (`react_no_<component>`),
which thread a disabled-set through `run_ticket_react` into `ablation_flags`.

Usage:
  python -m agent.eval.ablation --sentinel
  python -m agent.eval.ablation --components scout,verifier --dataset eval/swebench_50.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from agent import ablation_flags
from agent.eval.runner import EvalRunner, RESULTS_DIR

logger = logging.getLogger(__name__)

REFERENCE_PIPELINE = "react"

# Metrics reported per arm. (summary_key, label, unit)
ABLATION_METRICS = [
    ("pass_rate", "Pass Rate", "%"),
    ("test_pass_rate", "Test Pass", "%"),
    ("localization_accuracy", "Localization", "%"),
    ("avg_cost_usd", "Avg Cost", "$"),
    ("avg_tool_calls", "Tool Calls", "#"),
]


def build_ablation_report(summary: dict, components: list[str]) -> dict:
    """Attribute pass-rate contribution to each ablated component.

    Parameters
    ----------
    summary : dict
        Per-pipeline summary as produced by ``EvalRunner`` — keyed by pipeline
        name (``react`` plus ``react_no_<component>`` arms), each value a dict of
        metric → number.
    components : list[str]
        Components that were ablated (each must have a ``react_no_<c>`` arm).

    Returns
    -------
    dict
        ``reference`` (the full-arm metrics), ``components`` (per-component arm
        metrics + deltas), and ``ranking`` (components sorted by pass-rate
        contribution, descending). Pure function — no I/O.
    """
    reference = summary.get(REFERENCE_PIPELINE, {})
    full_pass = _num(reference.get("pass_rate"))

    comp_report: dict[str, dict] = {}
    for comp in components:
        arm = summary.get(f"react_no_{comp}", {})
        arm_pass = _num(arm.get("pass_rate"))
        # Contribution = how much pass rate is lost when this component is removed.
        contribution = round(full_pass - arm_pass, 4)

        metrics: dict[str, dict] = {}
        for key, label, unit in ABLATION_METRICS:
            ref_val = _num(reference.get(key))
            arm_val = _num(arm.get(key))
            metrics[key] = {
                "label": label,
                "unit": unit,
                "reference": ref_val,
                "ablated": arm_val,
                "delta": round(ref_val - arm_val, 4),
            }

        comp_report[comp] = {
            "label": ablation_flags.label(comp),
            "arm_pipeline": f"react_no_{comp}",
            "pass_rate_ablated": arm_pass,
            "contribution": contribution,
            "metrics": metrics,
        }

    ranking = sorted(
        components,
        key=lambda c: comp_report[c]["contribution"],
        reverse=True,
    )

    return {
        "reference": dict(reference),
        "reference_pass_rate": full_pass,
        "components": comp_report,
        "ranking": ranking,
    }


def format_ablation_table(report: dict) -> str:
    """Render the ablation report as a human-readable console table."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  COMPONENT ABLATION — contribution to pass rate")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Reference (all components on): {_pct(report.get('reference_pass_rate', 0))} pass rate")
    lines.append("")
    lines.append(f"  {'Component':<26} {'Pass w/o':>10} {'Contribution':>14}")
    lines.append(f"  {'-' * 26} {'-' * 10} {'-' * 14}")

    components = report.get("components", {})
    for comp in report.get("ranking", []):
        entry = components.get(comp, {})
        label = entry.get("label", comp)
        arm_pass = entry.get("pass_rate_ablated", 0)
        contribution = entry.get("contribution", 0)
        lines.append(
            f"  {label:<26} {_pct(arm_pass):>10} {_signed_pct(contribution):>14}"
        )

    lines.append("")
    lines.append("  Contribution = reference pass rate − pass rate with the component removed.")
    lines.append("  Higher is more valuable. Near-zero or negative = not earning its tokens.")
    lines.append("")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def to_markdown(report: dict) -> str:
    """Render the ablation report as a committable Markdown table."""
    lines: list[str] = []
    lines.append("# Component Ablation Report")
    lines.append("")
    lines.append(
        f"Reference pass rate (all components enabled): "
        f"**{_pct(report.get('reference_pass_rate', 0))}**"
    )
    lines.append("")
    lines.append("| Component | Pass rate w/o it | Contribution |")
    lines.append("|---|---|---|")
    components = report.get("components", {})
    for comp in report.get("ranking", []):
        entry = components.get(comp, {})
        lines.append(
            f"| {entry.get('label', comp)} | {_pct(entry.get('pass_rate_ablated', 0))} "
            f"| {_signed_pct(entry.get('contribution', 0))} |"
        )
    lines.append("")
    lines.append(
        "_Contribution = reference pass rate − pass rate with the component removed._"
    )
    lines.append("")
    return "\n".join(lines)


def run_ablation(
    dataset_path: str | Path = "eval/bugs.json",
    components: list[str] | None = None,
    bug_filter: str | None = None,
    sentinel: bool = False,
    timeout_per_case: int = 600,
    results_dir: str | Path = RESULTS_DIR,
    build_graph: bool = True,
    progress_cb: Callable[[str, str, int, int], None] | None = None,
) -> dict:
    """Run the full ablation: reference arm + one arm per component.

    ``components`` defaults to all of ``ablation_flags.COMPONENTS``. The
    reference arm and every ablation arm run on the same bugs in a single
    ``EvalRunner`` pass. ``build_graph`` defaults to True so the reference arm
    actually has a knowledge graph to use (the ``graph`` arm disables reading it).
    """
    components = list(components) if components else list(ablation_flags.COMPONENTS)
    unknown = [c for c in components if c not in ablation_flags.COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown ablation components {unknown}. "
            f"Valid: {', '.join(ablation_flags.COMPONENTS)}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    pipelines = [REFERENCE_PIPELINE] + [f"react_no_{c}" for c in components]
    logger.info("Ablation: reference + %d arms (%s)", len(components), ", ".join(components))

    runner = EvalRunner(
        dataset_path=dataset_path,
        pipelines=pipelines,
        timeout_per_case=timeout_per_case,
        results_dir=results_dir,
        build_graph=build_graph,
    )

    def _progress(tid: str, cur: int, total: int) -> None:
        if progress_cb:
            progress_cb("ablation", tid, cur, total)

    eval_report = runner.run(bug_filter=bug_filter, sentinel=sentinel, progress_cb=_progress)

    report = build_ablation_report(eval_report.summary, components)
    report["metadata"] = {
        "timestamp": ts,
        "dataset_path": str(dataset_path),
        "components": components,
        "bug_filter": bug_filter,
        "sentinel": sentinel,
        "build_graph": build_graph,
        "run_id": eval_report.run_id,
        "pipelines": pipelines,
    }

    out_path = results_dir / f"ablation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Ablation report saved to %s", out_path)
    report["metadata"]["report_path"] = str(out_path)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(value) -> float:
    """Coerce a metric to float, treating None/non-numeric as 0.0."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _pct(value) -> str:
    return f"{_num(value):.1%}"


def _signed_pct(value) -> str:
    return f"{_num(value):+.1%}"


# ---------------------------------------------------------------------------
# CLI entry point (python -m agent.eval.ablation)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Component ablation: measure each harness component's contribution to pass rate."
    )
    parser.add_argument("--dataset", default="eval/bugs.json", help="Path to eval bugs JSON.")
    parser.add_argument(
        "--components", default=None,
        help=f"Comma-separated components to ablate (default: all — {', '.join(ablation_flags.COMPONENTS)}).",
    )
    parser.add_argument("--bug", default=None, help="Run only this ticket_id.")
    parser.add_argument("--sentinel", action="store_true", help="Run only the first 5 bugs.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-case timeout (s).")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Results output directory.")
    parser.add_argument("--no-build-graph", action="store_true", help="Skip knowledge-graph build.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    components = (
        [c.strip() for c in args.components.split(",")] if args.components else None
    )

    report = run_ablation(
        dataset_path=args.dataset,
        components=components,
        bug_filter=args.bug,
        sentinel=args.sentinel,
        timeout_per_case=args.timeout,
        results_dir=args.output,
        build_graph=not args.no_build_graph,
    )
    print(format_ablation_table(report))


if __name__ == "__main__":
    _main()
