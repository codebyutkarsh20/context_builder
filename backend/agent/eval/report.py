"""
report.py — Markdown and JSON report generation for eval runs.

Produces:
  - Aggregate summary table per pipeline
  - A/B comparison table (the centerpiece)
  - Per-bug breakdown
  - Failure analysis
  - Cost analysis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_json_report(report: Any) -> str:
    """Serialize an EvalRunReport to JSON string."""
    return json.dumps(report.to_dict(), indent=2, default=str)


def generate_markdown_report(report: Any) -> str:
    """Generate a comprehensive markdown eval report.

    Parameters
    ----------
    report : EvalRunReport
        Complete eval run report with results and summaries.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    sections = [
        _header_section(report),
        _summary_section(report),
        _ab_comparison_section(report),
        _per_bug_section(report),
        _failure_analysis_section(report),
        _cost_section(report),
    ]
    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header_section(report: Any) -> str:
    ts = datetime.fromtimestamp(report.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pipelines = ", ".join(report.pipelines)
    mode = "Graph-less (exploration tools only)" if report.graph_less else "With knowledge graph"

    lines = [
        f"# Eval Report — {report.run_id}",
        "",
        f"**Date**: {ts}",
        f"**Dataset**: {report.dataset_path} ({report.total_bugs} bugs)",
        f"**Pipelines**: {pipelines}",
        f"**Mode**: {mode}",
    ]
    return "\n".join(lines)


def _summary_section(report: Any) -> str:
    if not report.summary:
        return ""

    lines = ["## Summary", ""]

    for pipeline, summary in report.summary.items():
        total = summary.get("total", 0)
        if total == 0:
            continue

        pr = summary.get("pass_rate", 0)
        target_icon = "+" if pr >= 0.8 else "-"

        lines.extend([
            f"### {pipeline.title()} Pipeline",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Pass rate | **{pr:.0%}** ({int(pr * total)}/{total}) [{target_icon} 80% target] |",
            f"| Localization accuracy | {summary.get('localization_accuracy', 0):.0%} |",
            f"| Fix rate | {summary.get('fix_rate', 0):.0%} |",
            f"| Approval rate | {summary.get('approval_rate', 0):.0%} |",
            f"| Patch correctness (file-level) | {summary.get('patch_correctness_avg', 0):.0%} |",
            f"| Multi-file complete | {summary.get('multi_file_complete_rate', 0):.0%} |",
            f"| Test pass rate | {summary.get('test_pass_rate', 0):.0%} |",
            f"| Avg confidence | {summary.get('avg_confidence', 0):.2f} |",
            f"| Avg cost | ${summary.get('avg_cost_usd', 0):.2f} |",
            f"| Avg duration | {summary.get('avg_duration_seconds', 0):.0f}s |",
            f"| Avg tool calls | {summary.get('avg_tool_calls', 0):.0f} |",
            "",
        ])

    return "\n".join(lines)


def _ab_comparison_section(report: Any) -> str:
    comp = report.comparison
    if not comp or not comp.get("per_bug"):
        return ""

    p1, p2 = comp["pipelines"]
    per_bug = comp["per_bug"]
    winner = comp.get("overall_winner", "tie")

    lines = [
        "## A/B Comparison",
        "",
        f"**Overall winner**: **{winner}**",
        "",
        f"| Bug | {p1.title()} | {p2.title()} | Winner |",
        f"|-----|{'------|' * 3}",
    ]

    for tid, per_pipeline in per_bug.items():
        r1 = per_pipeline.get(p1, {})
        r2 = per_pipeline.get(p2, {})

        def _fmt(r: dict) -> str:
            if r.get("pass"):
                return f"PASS (${r.get('cost', 0):.2f}, {r.get('duration', 0):.0f}s)"
            return "FAIL"

        if r1.get("pass") and not r2.get("pass"):
            winner = p1
        elif r2.get("pass") and not r1.get("pass"):
            winner = p2
        else:
            winner = "tie"
        lines.append(f"| {tid} | {_fmt(r1)} | {_fmt(r2)} | {winner} |")

    # Delta summary
    deltas = comp.get("deltas", {})
    if deltas:
        lines.extend(["", "### Metric Deltas", ""])
        lines.append(f"| Metric | {p1.title()} | {p2.title()} | Delta |")
        lines.append(f"|--------|{'------|' * 3}")
        for metric, vals in deltas.items():
            v1 = vals.get(p1, 0)
            v2 = vals.get(p2, 0)
            delta = vals.get("delta", 0)
            sign = "+" if delta > 0 else ""

            # Format based on metric type
            if "cost" in metric:
                lines.append(f"| {metric} | ${v1:.2f} | ${v2:.2f} | {sign}${delta:.2f} |")
            elif "duration" in metric or "seconds" in metric:
                lines.append(f"| {metric} | {v1:.0f}s | {v2:.0f}s | {sign}{delta:.0f}s |")
            elif "tool_calls" in metric or "calls" in metric:
                lines.append(f"| {metric} | {v1:.1f} | {v2:.1f} | {sign}{delta:.1f} |")
            else:
                lines.append(f"| {metric} | {v1:.0%} | {v2:.0%} | {sign}{delta:.0%} |")

    return "\n".join(lines)


def _per_bug_section(report: Any) -> str:
    if not report.results:
        return ""

    lines = ["## Per-Bug Breakdown", ""]

    # Group by ticket_id
    bugs: dict[str, list] = {}
    for r in report.results:
        bugs.setdefault(r.ticket_id, []).append(r)

    for tid, results in bugs.items():
        lines.append(f"### {tid}")
        for r in results:
            s = r.score
            icon = "PASS" if s.get("full_pass") else "FAIL"
            lines.append(
                f"- **{r.pipeline}**: [{icon}] "
                f"loc={'HIT' if s.get('localization_hit') else 'MISS'} "
                f"fix={'YES' if s.get('fix_generated') else 'NO'} "
                f"review={s.get('review_approved', False)} "
                f"conf={s.get('confidence', 0):.2f} "
                f"cost=${r.cost_usd:.2f} "
                f"time={r.duration_seconds:.0f}s"
            )
            if r.error:
                lines.append(f"  - Error: {r.error[:120]}")
        lines.append("")

    return "\n".join(lines)


def _failure_analysis_section(report: Any) -> str:
    failures = [r for r in report.results if not r.score.get("full_pass")]
    if not failures:
        return "## Failure Analysis\n\nNo failures."

    # Group by failure mode
    modes: dict[str, list[str]] = {}
    for r in failures:
        s = r.score
        if not s.get("localization_hit"):
            mode = "Localization miss"
        elif not s.get("fix_generated"):
            mode = "No fix generated"
        elif not s.get("review_approved"):
            mode = "Review rejected"
        elif s.get("error"):
            mode = "Error/crash"
        else:
            mode = "Unknown"
        modes.setdefault(mode, []).append(f"{r.ticket_id}/{r.pipeline}")

    lines = ["## Failure Analysis", ""]
    for mode, tickets in sorted(modes.items(), key=lambda x: -len(x[1])):
        lines.append(f"- **{mode}** ({len(tickets)}): {', '.join(tickets)}")

    return "\n".join(lines)


def _cost_section(report: Any) -> str:
    if not report.results:
        return ""

    total_cost = sum(r.cost_usd for r in report.results)
    total_time = sum(r.duration_seconds for r in report.results)
    count = len(report.results)

    lines = [
        "## Cost & Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cases | {count} |",
        f"| Total cost | ${total_cost:.2f} |",
        f"| Total time | {total_time:.0f}s ({total_time / 60:.1f}min) |",
        f"| Avg cost/case | ${total_cost / count:.2f} |" if count else "",
        f"| Avg time/case | {total_time / count:.0f}s |" if count else "",
    ]

    # Projected cost at 100/day
    if count:
        daily_cost = (total_cost / count) * 100
        lines.extend([
            "",
            f"**Projected at 100 changes/day**: ${daily_cost:.0f}/day (${daily_cost * 30:.0f}/month)",
        ])

    return "\n".join(lines)
