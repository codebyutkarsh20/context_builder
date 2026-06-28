"""
Context Builder CLI
Usage:
    python cli.py build /path/to/any/repo
    python cli.py build /path/to/any/repo --name my-project --summaries
    python cli.py query my-project "What does the payment module do?"
    python cli.py list
"""
import os
import sys
from pathlib import Path

# Load .env from project root so DATA_DIR, ANTHROPIC_API_KEY, GH_TOKEN etc.
# are consistent between CLI runs and the backend container. Without this,
# CLI eval writes lessons to /tmp/context_builder (default) while the backend
# writes to /data — splitting learnings across two locations.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv is optional; env vars may come from the shell

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(help="Build and query knowledge graphs for any code repository.")
console = Console()

# ---------------------------------------------------------------------------
# Eval subcommand group
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluation suite for the AI Deploy Agent.")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    bug: str = typer.Option(None, "--bug", "-b", help="Run only this ticket_id"),
    pipeline: str = typer.Option("react", "--pipeline", "-p",
                                 help="Pipeline(s) to run: 'react' (v3.0), 'react_v2' (v2.0 baseline), or 'react,react_v2' for A/B"),
    dataset: str = typer.Option("eval/bugs.json", "--dataset", "-d", help="Path to bugs JSON"),
    timeout: int = typer.Option(600, "--timeout", help="Per-case timeout in seconds"),
    sentinel: bool = typer.Option(False, "--sentinel", help="Run only first 5 bugs (fast regression check)"),
    create_prs: bool = typer.Option(False, "--create-prs", help="Create real PRs (default: dry-run)"),
    output: str = typer.Option("eval/results", "--output", "-o", help="Results output directory"),
    build_graph: bool = typer.Option(False, "--build-graph", "-g",
                                     help="Build knowledge graph before running agent (full system test)"),
    natural_lang: bool = typer.Option(False, "--natural-lang", "--nl",
                                      help="Use nl_description (business-language) instead of technical description"),
):
    """
    Run the evaluation suite on the ReAct pipeline.

    Examples:
        python cli.py eval run                                     # All bugs (react v3.0)
        python cli.py eval run --bug CB-001                        # Single bug
        python cli.py eval run --pipeline react,react_v2           # A/B: v3 vs v2 baseline
        python cli.py eval run --build-graph                       # Full system (with knowledge graph)
        python cli.py eval run --sentinel                          # Fast 3-bug regression check
        python cli.py eval run --natural-lang                      # Use business-language descriptions
        python cli.py eval run --nl --pipeline react,react_v2      # A/B: technical vs natural-language
        python cli.py eval run --timeout 300                       # 5min timeout per case
    """
    from agent.eval.runner import EvalRunner

    # Support comma-separated pipelines for A/B comparison
    pipelines = [p.strip() for p in pipeline.split(",")]

    nl_label = "[yellow]Business-language (nl_description)[/yellow]" if natural_lang else "[green]Technical (description)[/green]"
    console.print(Panel(
        f"[bold cyan]Dataset:[/bold cyan]   {dataset}\n"
        f"[bold cyan]Pipelines:[/bold cyan] {', '.join(pipelines)}\n"
        f"[bold cyan]Mode:[/bold cyan]      {'[green]With knowledge graph[/green]' if build_graph else '[yellow]Graph-less (exploration tools only)[/yellow]'}\n"
        f"[bold cyan]Descriptions:[/bold cyan] {nl_label}\n"
        f"[bold cyan]Sentinel:[/bold cyan]  {sentinel}\n"
        f"[bold cyan]Dry run:[/bold cyan]   {not create_prs}\n"
        f"[bold cyan]Timeout:[/bold cyan]   {timeout}s per case",
        title="[bold]Eval Suite[/bold]",
        border_style="cyan",
    ))

    runner = EvalRunner(
        dataset_path=Path(dataset),
        pipelines=pipelines,
        timeout_per_case=timeout,
        create_prs=create_prs,
        results_dir=Path(output),
        build_graph=build_graph,
        natural_lang=natural_lang,
    )

    def _progress(tid: str, current: int, total: int) -> None:
        console.print(f"  [dim]Completed {current}/{total}: {tid}[/dim]")

    report = runner.run(bug_filter=bug, sentinel=sentinel, progress_cb=_progress)

    # Print summary
    from agent.eval.report import generate_markdown_report
    console.print()
    console.print(generate_markdown_report(report))


@eval_app.command("curate")
def eval_curate(
    swe_bench_path: str = typer.Argument(..., help="Path to SWE-bench-lite dataset JSON"),
    output: str = typer.Option("eval/bugs.json", "--output", "-o", help="Output bugs.json path"),
    max_bugs: int = typer.Option(25, "--max", help="Maximum number of bugs to include"),
):
    """
    Curate eval bugs from a SWE-bench-lite dataset.

    Download the SWE-bench-lite dataset from HuggingFace first, then run:
        python cli.py eval curate /path/to/swe-bench-lite.json
    """
    from agent.eval.dataset import curate_from_swe_bench

    bugs = curate_from_swe_bench(swe_bench_path, output, max_bugs=max_bugs)
    console.print(f"[green]Curated {len(bugs)} bugs → {output}[/green]")

    # Show distribution
    single = sum(1 for b in bugs if b.get("difficulty") == "single-file")
    multi = len(bugs) - single
    console.print(f"  Single-file: {single} | Multi-file: {multi}")

    categories = {}
    for b in bugs:
        cat = b.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        console.print(f"  {cat}: {count}")


@eval_app.command("report")
def eval_report(
    results_dir: str = typer.Option("eval/results", "--dir", "-d", help="Results directory"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown or json"),
):
    """
    View the latest evaluation report.
    """
    from agent.eval.regression import load_previous_report
    from agent.eval.report import generate_markdown_report, generate_json_report

    report = load_previous_report(results_dir)
    if not report:
        console.print("[yellow]No eval results found.[/yellow]")
        raise typer.Exit(1)

    if format == "json":
        import json
        console.print(json.dumps(report._data, indent=2, default=str))
    else:
        # Build a lightweight report proxy for markdown generation
        console.print(generate_markdown_report(report))


@eval_app.command("gate")
def eval_gate(
    results_file: str = typer.Argument("eval/results/latest.json", help="Path to eval results JSON"),
    min_pass_rate: float = typer.Option(0.75, "--min-pass-rate", help="Minimum pass rate (0-1)"),
    max_regression: float = typer.Option(0.05, "--max-regression", help="Max allowed regression (0-1)"),
):
    """
    CI regression gate. Exit 0 if passing, exit 1 if regressed.

    Usage in CI:
        python cli.py eval gate eval/results/latest.json --min-pass-rate 0.75
    """
    from agent.eval.regression import check_regression_gate, load_previous_report

    current = load_previous_report(Path(results_file).parent)
    if not current:
        console.print("[red]No eval results found at {results_file}[/red]")
        raise typer.Exit(1)

    passed, reason = check_regression_gate(
        current, None, min_pass_rate=min_pass_rate, max_regression=max_regression,
    )

    if passed:
        console.print(f"[bold green]GATE PASSED[/bold green]: {reason}")
        raise typer.Exit(0)
    else:
        console.print(f"[bold red]GATE FAILED[/bold red]: {reason}")
        raise typer.Exit(1)


@eval_app.command("track-prs")
def eval_track_prs(
    poll: bool = typer.Option(True, "--poll/--no-poll", help="Poll GitHub for updates"),
):
    """
    Track PR review outcomes for the 80% approval target.
    """
    from agent.eval.pr_tracker import PRTracker

    tracker = PRTracker()

    if poll:
        console.print("[dim]Polling GitHub for PR review updates...[/dim]")
        tracker.poll_all()

    metrics = tracker.compute_approval_rate()
    tracked = tracker.list_tracked()

    if not tracked:
        console.print("[yellow]No PRs tracked yet. Run eval with --create-prs first.[/yellow]")
        return

    table = Table(title="Tracked PRs", show_header=True)
    table.add_column("Ticket", style="cyan")
    table.add_column("Pipeline", style="green")
    table.add_column("State")
    table.add_column("Review")
    table.add_column("PR URL", style="dim")

    for pr in tracked:
        review = pr.get("review_decision") or "pending"
        state = pr.get("state", "?")
        table.add_row(
            pr["ticket_id"], pr["pipeline"], state, review, pr["pr_url"]
        )

    console.print(table)
    console.print()
    console.print(Panel(
        f"[bold]Approval rate:[/bold] {metrics['approval_rate']:.0%} "
        f"({metrics['approved']}/{metrics['reviewed']} reviewed)\n"
        f"[bold]Target met:[/bold] {'YES' if metrics['target_met'] else 'NO'} (80%)",
        title="80% Target",
        border_style="green" if metrics["target_met"] else "red",
    ))


@eval_app.command("trace")
def eval_trace(
    bug: str = typer.Option(None, "--bug", "-b", help="Ticket ID to show (e.g. CB-003)"),
    pipeline: str = typer.Option(None, "--pipeline", "-p", help="Pipeline to show (react or react_v2)"),
    run_id: str = typer.Option(None, "--run", "-r", help="Run ID (default: latest)"),
    results_dir: str = typer.Option("eval/results", "--dir", "-d", help="Results directory"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full tool output (not just preview)"),
):
    """
    Show step-by-step agent trace for a bug run.

    Examples:
        python cli.py eval trace --bug CB-003
        python cli.py eval trace --bug CB-003 --pipeline react
        python cli.py eval trace --bug CB-003 --run d52c4a80 --full
        python cli.py eval trace                          # list available traces
    """
    import json
    from rich.rule import Rule
    from rich.text import Text

    traces_dir = Path(results_dir) / "traces"
    if not traces_dir.exists():
        console.print("[yellow]No traces found. Run eval first.[/yellow]")
        raise typer.Exit(1)

    # Resolve run_id — default to latest
    if run_id is None:
        runs = sorted(traces_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            console.print("[yellow]No trace runs found.[/yellow]")
            raise typer.Exit(1)
        run_dir = runs[0]
        run_id = run_dir.name
    else:
        run_dir = traces_dir / run_id
        if not run_dir.exists():
            console.print(f"[red]Run '{run_id}' not found in {traces_dir}[/red]")
            raise typer.Exit(1)

    # Find matching trace files
    trace_files = sorted(run_dir.glob("*.json"))
    if bug:
        trace_files = [f for f in trace_files if f.stem.startswith(bug)]
    if pipeline:
        trace_files = [f for f in trace_files if f.stem.endswith(f"_{pipeline}")]

    if not trace_files:
        available = [f.stem for f in sorted(run_dir.glob("*.json"))]
        console.print(f"[yellow]No matching traces in run '{run_id}'.[/yellow]")
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    # If multiple matches and no specific selection, list them
    if len(trace_files) > 1 and not (bug and pipeline):
        console.print(f"\n[bold]Run:[/bold] {run_id}  |  [bold]Traces:[/bold]")
        for f in trace_files:
            data = json.loads(f.read_text())
            outcome = data.get("run_outcome", {})
            summary = data.get("summary", {})
            status = outcome.get("outcome", "?")
            cost = summary.get("total_cost_usd", 0)
            calls = summary.get("total_tool_calls", 0)
            color = "green" if status == "submitted" else "red" if status == "escalated" else "yellow"
            console.print(
                f"  [cyan]{f.stem}[/cyan]  [{color}]{status}[/{color}]  "
                f"{calls} calls  ${cost:.2f}"
            )
        console.print("\n[dim]Use --bug and --pipeline to drill into a specific trace.[/dim]")
        return

    # Print each matching trace
    for trace_file in trace_files:
        _print_trace(console, trace_file, run_id, full)


def _print_trace(console: "Console", trace_file: "Path", run_id: str, full: bool) -> None:
    """Pretty-print a single agent trace file."""
    import json
    from rich.rule import Rule
    from rich.text import Text
    from rich.padding import Padding

    data = json.loads(trace_file.read_text())
    ticket_id, pipeline_name = trace_file.stem.rsplit("_", 1)

    # ── Header ──────────────────────────────────────────────────────────────
    outcome = data.get("run_outcome", {})
    summary = data.get("summary", {})
    status = outcome.get("outcome", "?")
    cost = summary.get("total_cost_usd", 0)
    calls = summary.get("total_tool_calls", 0)
    duration = outcome.get("elapsed_seconds", 0)
    tokens = summary.get("total_tokens", 0)

    status_color = "green" if status == "submitted" else "red" if status == "escalated" else "yellow"
    tests_ok = outcome.get("tests_passed", False)
    tests_skip = outcome.get("tests_skipped", False)
    test_str = "[green]tests PASS[/green]" if tests_ok else (
        "[yellow]tests SKIPPED[/yellow]" if tests_skip else "[red]tests FAIL[/red]"
    )

    console.print()
    console.print(Panel(
        f"[bold cyan]{ticket_id}[/bold cyan]  |  [bold]{pipeline_name}[/bold] pipeline  |  "
        f"[{status_color}]{status.upper()}[/{status_color}]  |  "
        f"{calls} tool calls  ${cost:.2f}  {duration:.0f}s  {tokens:,} tokens  |  {test_str}",
        title=f"[bold]Agent Trace[/bold] — run {run_id}",
        border_style=status_color,
    ))

    # ── Stage timings ────────────────────────────────────────────────────────
    stage_timings = data.get("stage_timings", {})
    timing_parts = []
    for stage, t in stage_timings.items():
        dur = t.get("duration_ms", 0) / 1000
        if dur > 0.1:
            timing_parts.append(f"{stage}={dur:.1f}s")
    if timing_parts:
        console.print(f"[dim]Stage timings: {' → '.join(timing_parts)}[/dim]")

    # ── Phase breakdown ──────────────────────────────────────────────────────
    phases = data.get("phase_breakdown", {})
    if phases:
        phase_parts = []
        for ph, info in phases.items():
            tc = info.get("tool_calls", 0)
            tools = ", ".join(info.get("tools_used", []))
            phase_parts.append(f"[bold]{ph}[/bold]({tc})")
        console.print(f"Phases: {' → '.join(phase_parts)}")

    # ── Events ──────────────────────────────────────────────────────────────
    events = data.get("events", [])
    console.print()

    # Track current phase for transition markers
    current_phase = "explore"
    call_num = 0
    # Pair tool_calls with their following tool_result
    # Build index: call_number → (tool_call_event, tool_result_event)
    tool_pairs: dict[int, dict] = {}
    transitions: dict[int, dict] = {}  # call_number → transition

    for e in events:
        et = e["event_type"]
        d = e.get("data", {})
        if et == "tool_call":
            n = d.get("call_number", 0)
            tool_pairs.setdefault(n, {})["call"] = e
        elif et == "tool_result":
            # tool_result doesn't have call_number, match by proximity
            # find the last tool_call without a result
            pass
        elif et == "state_transition":
            n = d.get("at_call", 0)
            transitions[n] = d

    # Re-pair results sequentially
    result_queue = [e for e in events if e["event_type"] == "tool_result"]
    result_idx = 0
    for n in sorted(tool_pairs.keys()):
        if result_idx < len(result_queue):
            tool_pairs[n]["result"] = result_queue[result_idx]
            result_idx += 1

    # Print each tool call
    PHASE_COLORS = {
        "explore": "cyan", "edit": "yellow", "test": "blue",
        "review": "magenta", "submit": "green", "other": "dim",
    }

    for n in sorted(tool_pairs.keys()):
        pair = tool_pairs[n]
        call_e = pair.get("call")
        result_e = pair.get("result")
        if not call_e:
            continue

        cd = call_e["data"]
        ts = call_e["timestamp"]
        tool_name = cd.get("tool_name", "?")
        phase = cd.get("phase", "other")
        args = cd.get("args", {})
        reasoning = cd.get("reasoning", "")
        phase_color = PHASE_COLORS.get(phase, "white")

        # Print transition marker before the call that triggered it
        if n in transitions:
            t = transitions[n]
            console.print(
                f"\n  [dim]── phase: [bold]{t['from_phase']}[/bold] → "
                f"[bold]{t['to_phase']}[/bold]  "
                f"(call #{t['at_call']}, ${t.get('cost_usd_at_transition', 0):.3f}) ──[/dim]"
            )

        # Format args as a short string
        args_str = _fmt_args(tool_name, args, full)

        # Call line
        console.print(
            f"  [dim]{n:>2}[/dim]  [{phase_color}]{phase:<8}[/{phase_color}]  "
            f"[bold white]{tool_name}[/bold white]  [dim]{args_str}[/dim]  "
            f"[dim]{ts:.1f}s[/dim]"
        )

        # Reasoning (if present)
        if reasoning:
            short = reasoning[:180].replace("\n", " ")
            console.print(f"       [italic dim]↳ {short}[/italic dim]")

        # Result
        if result_e:
            rd = result_e["data"]
            preview = rd.get("result_preview", rd.get("result", str(rd)))
            if not full:
                preview = str(preview)[:200].replace("\n", " ↵ ")
            ok = not any(x in str(preview).lower() for x in ("error:", "not found", "failed", "usage error"))
            result_icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            dur_ms = rd.get("duration_ms", 0)
            console.print(f"       {result_icon} [{dur_ms}ms] [dim]{preview}[/dim]")

    # ── Final outcome ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(style="dim"))
    escalate_reason = outcome.get("escalate_reason", "")
    loc_found = outcome.get("localization_found", False)
    sandbox_ok = outcome.get("sandbox_created", False)

    console.print(
        f"[bold]Outcome:[/bold] [{status_color}]{status}[/{status_color}]  "
        f"loc={'[green]HIT[/green]' if loc_found else '[red]MISS[/red]'}  "
        f"sandbox={'[green]YES[/green]' if sandbox_ok else '[dim]NO[/dim]'}  "
        f"tests={'[green]PASS[/green]' if tests_ok else ('[yellow]SKIP[/yellow]' if tests_skip else '[red]FAIL[/red]')}"
    )
    if escalate_reason:
        console.print(f"[red]Escalate reason:[/red] {escalate_reason[:200]}")

    # ── Wasted calls ─────────────────────────────────────────────────────────
    wasted = data.get("wasted_calls", {})
    if wasted.get("max_grep_streak", 0) > 2:
        console.print(f"[yellow]⚠ grep streak: {wasted['max_grep_streak']} consecutive greps[/yellow]")
    if wasted.get("test_attempts", 0) > 2:
        console.print(f"[yellow]⚠ test attempts: {wasted['test_attempts']} (retry loop?)[/yellow]")


def _fmt_args(tool_name: str, args: dict, full: bool) -> str:
    """Format tool args into a short readable string."""
    if not args:
        return ""
    # Tool-specific formatting
    if tool_name in ("read_function", "read_file"):
        fp = args.get("file_path", "")
        fn = args.get("function_name", "")
        return f"{fp}::{fn}" if fn else fp
    if tool_name == "string_replace":
        fp = args.get("file_path", "")
        old = args.get("old_string", "")[:60].replace("\n", "↵")
        new = args.get("new_string", "")[:60].replace("\n", "↵")
        return f"{fp}  [{old}] → [{new}]" if not full else f"{fp}"
    if tool_name == "grep_repo":
        return f"pattern={args.get('pattern', '')!r}"
    if tool_name == "run_tests":
        return args.get("test_path", "") or args.get("test_command", "")
    if tool_name == "record_localization":
        files = args.get("fault_files", [])
        return ", ".join(files[:3])
    if tool_name in ("request_review", "submit_fix"):
        expl = args.get("explanation", "")[:80].replace("\n", " ")
        return expl
    if tool_name == "list_files":
        return args.get("directory", "")
    # Default: join all values
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and v:
            parts.append(f"{v[:50]}")
        elif isinstance(v, list):
            parts.append(str(v)[:50])
    return "  ".join(parts)[:120]


@eval_app.command("ab")
def eval_ab(
    bug: str = typer.Option(None, "--bug", "-b", help="Run only this ticket_id"),
    dataset: str = typer.Option("eval/bugs.json", "--dataset", "-d", help="Path to bugs JSON"),
    timeout: int = typer.Option(600, "--timeout", help="Per-case timeout in seconds"),
    sentinel: bool = typer.Option(False, "--sentinel", help="Run only first 5 bugs (fast regression check)"),
    output: str = typer.Option("eval/results", "--output", "-o", help="Results output directory"),
    build_graph: bool = typer.Option(False, "--build-graph", "-g",
                                     help="Build knowledge graph before running agent"),
):
    """
    Run A/B comparison: full pipeline (scout+BRT) vs baseline (no scout, no BRT).

    Runs the eval suite twice — once with all features enabled, once with
    scout and BRT disabled — then prints a comparison table.

    Examples:
        python cli.py eval ab                                  # Full A/B on all bugs
        python cli.py eval ab --bug CB-001                     # Single bug comparison
        python cli.py eval ab --sentinel                       # Quick 5-bug comparison
        python cli.py eval ab --build-graph                    # With knowledge graph
    """
    from agent.eval.ab_eval import run_ab_eval, format_comparison_table

    console.print(Panel(
        f"[bold cyan]Dataset:[/bold cyan]   {dataset}\n"
        f"[bold cyan]Arms:[/bold cyan]      A=full (scout+BRT)  vs  B=baseline (no scout, no BRT)\n"
        f"[bold cyan]Mode:[/bold cyan]      {'[green]With knowledge graph[/green]' if build_graph else '[yellow]Graph-less (exploration tools only)[/yellow]'}\n"
        f"[bold cyan]Sentinel:[/bold cyan]  {sentinel}\n"
        f"[bold cyan]Timeout:[/bold cyan]   {timeout}s per case",
        title="[bold]A/B Eval Comparison[/bold]",
        border_style="magenta",
    ))

    def _progress(arm: str, tid: str, current: int, total: int) -> None:
        arm_label = "[green]FULL[/green]" if arm == "full" else "[yellow]BASE[/yellow]"
        console.print(f"  {arm_label} Completed {current}/{total}: {tid}")

    comparison = run_ab_eval(
        dataset_path=Path(dataset),
        bug_filter=bug,
        sentinel=sentinel,
        timeout_per_case=timeout,
        results_dir=Path(output),
        build_graph=build_graph,
        progress_cb=_progress,
    )

    # Print formatted comparison
    console.print(format_comparison_table(comparison))


@eval_app.command("ablate")
def eval_ablate(
    bug: str = typer.Option(None, "--bug", "-b", help="Run only this ticket_id"),
    dataset: str = typer.Option("eval/bugs.json", "--dataset", "-d", help="Path to bugs JSON"),
    components: str = typer.Option(None, "--components", "-c",
                                   help="Comma-separated components to ablate (default: all)"),
    timeout: int = typer.Option(600, "--timeout", help="Per-case timeout in seconds"),
    sentinel: bool = typer.Option(False, "--sentinel", help="Run only first 5 bugs (fast check)"),
    output: str = typer.Option("eval/results", "--output", "-o", help="Results output directory"),
    no_build_graph: bool = typer.Option(False, "--no-build-graph",
                                        help="Skip knowledge-graph build (graph arm becomes meaningless)"),
):
    """
    Component ablation: measure each harness component's contribution to pass rate.

    Runs the eval once with everything enabled (reference) and once per component
    with that component disabled, then ranks components by how much pass rate they
    contribute (reference − ablated). Components: scout, brt, graph, lessons, verifier.

    Examples:
        python cli.py eval ablate --sentinel                       # all components, 5 bugs
        python cli.py eval ablate --components scout,verifier      # just these two
        python cli.py eval ablate --dataset eval/swebench_50.json  # bigger set
    """
    from agent import ablation_flags
    from agent.eval.ablation import run_ablation, format_ablation_table

    comp_list = [c.strip() for c in components.split(",")] if components else list(ablation_flags.COMPONENTS)

    console.print(Panel(
        f"[bold cyan]Dataset:[/bold cyan]     {dataset}\n"
        f"[bold cyan]Components:[/bold cyan]  {', '.join(comp_list)}\n"
        f"[bold cyan]Arms:[/bold cyan]        reference + {len(comp_list)} ablation arm(s)\n"
        f"[bold cyan]Graph:[/bold cyan]       {'[yellow]disabled[/yellow]' if no_build_graph else '[green]built[/green]'}\n"
        f"[bold cyan]Sentinel:[/bold cyan]    {sentinel}\n"
        f"[bold cyan]Timeout:[/bold cyan]     {timeout}s per case",
        title="[bold]Component Ablation[/bold]",
        border_style="magenta",
    ))

    def _progress(arm: str, tid: str, current: int, total: int) -> None:
        console.print(f"  [cyan]ABLATE[/cyan] Completed {current}/{total}: {tid}")

    report = run_ablation(
        dataset_path=Path(dataset),
        components=comp_list,
        bug_filter=bug,
        sentinel=sentinel,
        timeout_per_case=timeout,
        results_dir=Path(output),
        build_graph=not no_build_graph,
        progress_cb=_progress,
    )

    console.print(format_ablation_table(report))


@eval_app.command("baseline")
def eval_baseline(
    bug: str = typer.Option(None, "--bug", "-b", help="Run only this ticket_id"),
    dataset: str = typer.Option("eval/bugs.json", "--dataset", "-d", help="Path to bugs JSON"),
):
    """
    Run the dumb-loop baseline diagnostic.

    Single-shot Claude API call per bug. No ReAct loop, no tools, no retries,
    no graph context. Measures the fix-rate floor to quantify infrastructure value.

    Examples:
        python cli.py eval baseline                    # All bugs
        python cli.py eval baseline --bug CB-001       # Single bug
    """
    from agent.eval.baseline import run_baseline

    console.print(Panel(
        f"[bold cyan]Dataset:[/bold cyan]   {dataset}\n"
        f"[bold cyan]Model:[/bold cyan]     claude-sonnet-4-6 (temp=0, single-shot)\n"
        f"[bold cyan]Mode:[/bold cyan]      [yellow]Dumb loop — no tools, no retries, no graph[/yellow]",
        title="[bold]Baseline Diagnostic[/bold]",
        border_style="yellow",
    ))

    run_baseline(dataset_path=Path(dataset), bug_filter=bug)


@app.command()
def build(
    repo_path: str = typer.Argument(..., help="Path to any local Git repository"),
    name: str = typer.Option(None, "--name", "-n", help="Override repo name"),
    summaries: bool = typer.Option(False, "--summaries", "-s", help="Generate LLM business summaries (needs ANTHROPIC_API_KEY)"),
    no_neo4j: bool = typer.Option(False, "--no-neo4j", help="Skip Neo4j, only generate context.md files"),
):
    """
    Analyze any repository and build its knowledge graph + context document.

    Examples:
        python cli.py build ~/projects/my-api
        python cli.py build /path/to/django-app --name django-app --summaries
        python cli.py build https://... (clone first, then pass local path)
    """
    from analyzer.structure import StructureAnalyzer
    from analyzer.code_parser import CodeParser
    from analyzer.call_graph import CallGraphBuilder
    from analyzer.git_analyzer import GitAnalyzer
    from compiler.context_doc import ContextCompiler

    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)

    repo_name = name or path.name

    console.print(Panel(
        f"[bold cyan]Analyzing repo:[/bold cyan] {path}\n"
        f"[bold cyan]Repo name:[/bold cyan]    {repo_name}\n"
        f"[bold cyan]LLM summaries:[/bold cyan] {'Yes' if summaries else 'No (pass --summaries to enable)'}",
        title="[bold]Context Builder[/bold]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=100)

        progress.update(task, description="[cyan]Analyzing structure...", completed=10)
        structure = StructureAnalyzer(path).analyze()
        _print_structure_summary(structure)

        progress.update(task, description="[cyan]Parsing code (Tree-sitter)...", completed=30)
        parser = CodeParser(path)
        parsed = parser.parse_all()
        console.print(f"  [green]✓[/green] Parsed {len(parsed)} files "
                      f"({sum(f['loc'] for f in parsed):,} lines of code)")

        progress.update(task, description="[cyan]Building call graph + PageRank...", completed=50)
        cg = CallGraphBuilder(parsed)
        graph_data = cg.build()
        console.print(f"  [green]✓[/green] {len(graph_data['nodes'])} nodes, "
                      f"{len(graph_data['edges'])} edges, "
                      f"top hotspot: [bold]{graph_data['hotspots'][0]['label'] if graph_data['hotspots'] else 'n/a'}[/bold]")
        # Leiden community detection — clusters codebase into named modules
        # Enables Scout tier: ticket → community in 1 Haiku call instead of full-codebase search
        progress.update(task, description="[cyan]Running Leiden community detection...", completed=53)
        try:
            from graph.community import build_communities, annotate_graph_with_communities
            communities = build_communities(graph_data)
            graph_data = annotate_graph_with_communities(graph_data, communities)
            community_names = [c["name"] for c in communities]
            console.print(f"  [green]✓[/green] {len(communities)} communities detected: {', '.join(community_names[:6])}"
                          + (f" +{len(communities)-6} more" if len(communities) > 6 else ""))
            # Store communities alongside graph.json for community classifier
            import json as _json_c
            _comm_out = Path(f"/tmp/context_builder/{repo_name}")
            _comm_out.mkdir(parents=True, exist_ok=True)
            (_comm_out / "communities.json").write_text(_json_c.dumps(communities, default=str))
        except Exception as e:
            console.print(f"  [dim]  Community detection skipped: {e}[/dim]")
            communities = []

        # Write graph.json for agent kickstart and dashboard fallback
        _graph_out = Path(f"/tmp/context_builder/{repo_name}")
        _graph_out.mkdir(parents=True, exist_ok=True)
        import json as _json2; (_graph_out / "graph.json").write_text(_json2.dumps(graph_data, default=str))

        # Execution flow detection — entry points, BFS flow tracing, dead code
        progress.update(task, description="[cyan]Detecting execution flows...", completed=48)
        try:
            from analyzer.flows import build_flows
            flow_result = build_flows(graph_data, communities=communities)
            import json as _json_f
            (_graph_out / "flows.json").write_text(_json_f.dumps(flow_result, default=str))
            console.print(
                f"  [green]✓[/green] Flows: {flow_result['flow_count']} execution flows, "
                f"{flow_result['entry_point_count']} entry points, "
                f"{flow_result['dead_code_count']} dead-code nodes"
            )
        except Exception as e:
            console.print(f"  [dim]  Flow detection skipped: {e}[/dim]")

        progress.update(task, description="[cyan]Analyzing git history...", completed=50)
        git_analyzer = GitAnalyzer(path)
        git_data = git_analyzer.analyze()
        if git_data.get("hotspot_files"):
            console.print(f"  [green]✓[/green] Git: {len(git_data['hotspot_files'])} change hotspots found")

        # Detect data access patterns (reads/writes)
        progress.update(task, description="[cyan]Detecting data access patterns...", completed=55)
        from analyzer.data_access import detect_data_access
        data_access = detect_data_access(parsed)
        if data_access:
            console.print(f"  [green]✓[/green] Data access: {len(data_access)} functions with I/O detected")
            # Annotate graph_data nodes with reads_from / writes_to
            node_map = {n["id"]: n for n in graph_data["nodes"]}
            for func_id, access in data_access.items():
                if func_id in node_map:
                    node_map[func_id]["reads_from"] = access["reads_from"]
                    node_map[func_id]["writes_to"] = access["writes_to"]

        # Extract decision points
        progress.update(task, description="[cyan]Extracting decision points...", completed=58)
        from enricher.decision_points import extract_decision_points
        decision_points = extract_decision_points(parsed)
        console.print(f"  [green]✓[/green] {len(decision_points)} decision points extracted")

        # Extract domain concepts
        progress.update(task, description="[cyan]Extracting domain concepts...", completed=60)
        from enricher.domain_concepts import extract_domain_concepts
        domain_concepts = extract_domain_concepts(parsed)
        console.print(f"  [green]✓[/green] {len(domain_concepts)} domain concepts identified")

        # Mine git decision context
        git_decisions = git_analyzer.extract_decision_context()
        if git_decisions:
            console.print(f"  [green]✓[/green] Git decisions: {len(git_decisions)} decision-laden commits found")

        if summaries:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                console.print("  [yellow]⚠[/yellow]  ANTHROPIC_API_KEY not set — skipping summaries")
            else:
                # LLM-enhanced decision points + domain concepts (before Neo4j write)
                progress.update(task, description="[cyan]Enhancing decision points (Claude)...", completed=62)
                from enricher.decision_points import enhance_with_llm as enhance_decisions
                enhanced_dp = enhance_decisions(decision_points, api_key)
                if enhanced_dp:
                    console.print(f"  [green]✓[/green] {enhanced_dp} decision points enhanced with LLM")

                progress.update(task, description="[cyan]Enriching domain concepts (Claude)...", completed=63)
                from enricher.domain_concepts import enhance_with_llm as enhance_concepts
                enhanced_dc = enhance_concepts(domain_concepts, parsed, api_key)
                if enhanced_dc:
                    console.print(f"  [green]✓[/green] {enhanced_dc} domain concepts enriched with LLM")

        if not no_neo4j:
            try:
                progress.update(task, description="[cyan]Writing to Neo4j...", completed=65)
                from graph.neo4j_client import neo4j_client
                from graph.builder import GraphBuilder
                neo4j_client.connect()
                neo4j_client.ensure_constraints()
                builder = GraphBuilder(repo_name, path)
                builder.ingest(
                    structure, parsed, graph_data,
                    decision_points=decision_points,
                    domain_concepts=domain_concepts,
                )
                console.print(f"  [green]✓[/green] Knowledge graph written to Neo4j")
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow]  Neo4j unavailable ({e}) — skipping graph DB. Use --no-neo4j to suppress.")

        if summaries and os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            progress.update(task, description="[cyan]Generating LLM file summaries (Claude)...", completed=70)
            from enricher.summarizer import Summarizer
            summarizer = Summarizer(repo_name)
            summarizer.enrich()
            console.print(f"  [green]✓[/green] LLM file summaries generated")

            progress.update(task, description="[cyan]Generating LLM function summaries (Claude)...", completed=73)
            fn_count = summarizer.enrich_functions()
            if fn_count:
                console.print(f"  [green]✓[/green] {fn_count} function summaries generated")

        progress.update(task, description="[cyan]Extracting business rules...", completed=80)
        from enricher.business_logic import BusinessLogicExtractor, persist_rules_to_file
        extractor = BusinessLogicExtractor(repo_name, parsed)
        if no_neo4j:
            rules = extractor.extract_all()
            rules_count = len(rules)
        else:
            rules_count = extractor.extract()
            rules = extractor.extract_all()  # get rules for file persistence
        # Always write to business_rules.json so the pipeline can read them
        out_dir_br = Path(f"/tmp/context_builder/{repo_name}")
        out_dir_br.mkdir(parents=True, exist_ok=True)
        persist_rules_to_file(rules, out_dir_br / "business_rules.json")
        console.print(f"  [green]✓[/green] {rules_count} business rules extracted")

        # Mine failure records from git history
        progress.update(task, description="[cyan]Mining failure records...", completed=82)
        from graph.business.failure_records import mine_failure_records, persist_failure_records
        fr = mine_failure_records(path)
        if fr:
            if not no_neo4j:
                persist_failure_records(fr, repo_name)
            console.print(f"  [green]✓[/green] {len(fr)} failure records mined from git history")
        else:
            console.print(f"  [dim]  ℹ Failure records: 0 (set ENABLE_FAILURE_RECORDS=true to enable)[/dim]")

        # Save enriched nodes cache
        progress.update(task, description="[cyan]Building enriched node cache...", completed=88)
        from embeddings.embedder import build_enriched_nodes
        enriched = build_enriched_nodes(parsed, graph_data, decision_points, domain_concepts, rules if no_neo4j else [])
        import json as _json
        enriched_path = Path(f"/tmp/context_builder/{repo_name}/enriched_nodes.json")
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_path.write_text(_json.dumps(enriched, default=str))
        console.print(f"  [green]✓[/green] Enriched node cache: {len(enriched)} nodes")

        progress.update(task, description="[cyan]Compiling context document...", completed=92)
        if no_neo4j:
            _compile_without_neo4j(
                repo_name, structure, parsed, graph_data, rules,
                repo_path=path,
                decision_points=decision_points,
                domain_concepts=domain_concepts,
                git_decisions=git_decisions,
            )
        else:
            compiler = ContextCompiler(repo_name, repo_path=path)
            compiler.compile()

        progress.update(task, description="[green]Done!", completed=100)

    out = Path(f"/tmp/context_builder/{repo_name}")
    console.print(Panel(
        f"[bold green]Context built successfully![/bold green]\n\n"
        f"[cyan]context.md[/cyan]  → {out}/context.md\n"
        f"[cyan]summary.md[/cyan] → {out}/summary.md\n\n"
        f"[dim]Open the dashboard at http://localhost:5173 to explore visually.[/dim]\n"
        f"[dim]Or query from CLI: python cli.py query {repo_name} \"your question\"[/dim]",
        border_style="green",
    ))

    _print_hotspots(graph_data["hotspots"][:5])


@app.command()
def query(
    repo_name: str = typer.Argument(..., help="Name of an already-analyzed repo"),
    question: str = typer.Argument(..., help="Natural language question about the repo"),
):
    """
    Ask a question about a repo using its generated context + Claude.

    Example:
        python cli.py query my-api "How does user authentication work?"
        python cli.py query django-shop "What happens when a payment fails?"
    """
    import anthropic
    out = Path(f"/tmp/context_builder/{repo_name}")
    context_path = out / "context.md"

    if not context_path.exists():
        console.print(f"[red]No context found for '{repo_name}'.[/red] Run: python cli.py build /path/to/repo --name {repo_name}")
        raise typer.Exit(1)

    context = context_path.read_text()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    client = anthropic.Anthropic(api_key=api_key)
    with console.status("[cyan]Thinking..."):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=(
                "You are a senior software engineer. You have been given the complete context "
                "document for a code repository. Answer questions about the codebase accurately "
                "and in detail, citing specific files, functions, and business rules from the context. "
                "Do not say 'I don't have access to the code' — you have the full context."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"<context>\n{context[:80000]}\n</context>\n\nQuestion: {question}",
                }
            ],
        )

    answer = response.content[0].text
    console.print(Panel(answer, title="[bold]Answer[/bold]", border_style="cyan"))


@app.command("list")
def list_repos():
    """List all repos that have been analyzed."""
    base = Path("/tmp/context_builder")
    if not base.exists() or not any(base.iterdir()):
        console.print("[yellow]No repos analyzed yet.[/yellow] Run: python cli.py build /path/to/repo")
        return

    table = Table(title="Analyzed Repositories")
    table.add_column("Repo", style="cyan")
    table.add_column("context.md", style="green")
    table.add_column("summary.md", style="green")
    table.add_column("Size", style="dim")

    for d in sorted(base.iterdir()):
        if d.is_dir():
            ctx = d / "context.md"
            summ = d / "summary.md"
            size = f"{ctx.stat().st_size // 1024}KB" if ctx.exists() else "—"
            table.add_row(
                d.name,
                "✓" if ctx.exists() else "✗",
                "✓" if summ.exists() else "✗",
                size,
            )
    console.print(table)


def _compile_without_neo4j(
    repo_name: str, structure: dict, parsed: list, graph_data: dict, rules: list,
    repo_path: str | Path | None = None,
    decision_points: list | None = None,
    domain_concepts: list | None = None,
    git_decisions: list | None = None,
    out_dir: Path | None = None,
):
    """Write context.md and summary.md directly from in-memory data (no Neo4j)."""
    from datetime import datetime, timezone
    out = Path(out_dir) if out_dir else Path(f"/tmp/context_builder/{repo_name}")
    out.mkdir(parents=True, exist_ok=True)

    tech = ", ".join(structure.get("tech_stack", []))
    stats = structure.get("file_stats", {})
    entries = structure.get("entry_points", [])
    readme = structure.get("readme_content", "")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    total_loc = sum(pf.get("loc", 0) for pf in parsed)
    lines = [
        f"# Repository Context: {repo_name}\n",
        f"> Generated by Context Builder on {generated_at}\n",
        f"> **Layers:** Structure | File Index | Symbol Map | Hotspots | Business Rules | Call Flow\n\n",
        "## 1. Repository Structure\n\n",
        f"**Tech Stack:** {tech}\n\n",
        f"**{stats.get('total_files',0)} files** | **{total_loc:,} lines of code**\n\n",
        f"**Entry Points:** {', '.join(entries) or '—'}\n\n",
    ]

    if readme:
        lines += [f"### README\n\n```\n{readme[:2000]}\n```\n\n"]

    # ---- Layer 2: File Profiles (rich per-file metadata) --------------------
    # Build a lookup of rules per file for inline association
    rules_by_file: dict[str, list] = {}
    for r in rules:
        src = getattr(r, "source_file", "") or ""
        rules_by_file.setdefault(src, []).append(r)

    lines += ["## 2. File Profiles\n\n"]
    lines += ["_Detailed breakdown of every file: what it does, what it contains, and how it connects._\n\n"]

    for pf in parsed:
        fpath = pf.get("path", "unknown")
        doc = (pf.get("docstring") or "").strip()
        loc = pf.get("loc", 0)
        classes = pf.get("classes", [])
        functions = pf.get("functions", [])
        imports = pf.get("imports", [])

        lines.append(f"### `{fpath}`\n\n")

        # Purpose line from module docstring
        if doc:
            first_lines = " ".join(ln.strip() for ln in doc.splitlines()[:3] if ln.strip())
            lines.append(f"**Purpose:** {first_lines}\n\n")
        else:
            # Infer purpose from file name and contents
            fname = Path(fpath).stem
            if classes and not functions:
                lines.append(f"**Purpose:** Defines {', '.join(c['name'] for c in classes[:3])} class{'es' if len(classes) > 1 else ''}.\n\n")
            elif functions and not classes:
                lines.append(f"**Purpose:** Utility module with {len(functions)} function{'s' if len(functions) > 1 else ''}.\n\n")
            elif classes and functions:
                lines.append(f"**Purpose:** Contains {len(classes)} class{'es' if len(classes) > 1 else ''} and {len(functions)} function{'s' if len(functions) > 1 else ''}.\n\n")
            else:
                lines.append(f"**Purpose:** {fname} module ({loc} lines).\n\n")

        lines.append(f"**Lines of Code:** {loc}\n\n")

        # Imports
        if imports:
            import_names = []
            for imp in imports[:10]:
                if isinstance(imp, dict):
                    mod = imp.get("module", "")
                    names = imp.get("names", [])
                    if names:
                        import_names.append(f"`{mod}` ({', '.join(names[:3])})")
                    elif mod:
                        import_names.append(f"`{mod}`")
                elif isinstance(imp, str):
                    import_names.append(f"`{imp}`")
            if import_names:
                lines.append(f"**Imports:** {', '.join(import_names)}")
                if len(imports) > 10:
                    lines.append(f" +{len(imports) - 10} more")
                lines.append("\n\n")

        # Classes with their methods and docstrings
        if classes:
            lines.append("**Classes:**\n\n")
            for cls in classes:
                bases = f"({', '.join(cls.get('bases', []))})" if cls.get("bases") else ""
                cls_doc = (cls.get("docstring") or "").split("\n")[0][:100]
                cls_doc_str = f" — {cls_doc}" if cls_doc else ""
                lines.append(f"- **`{cls['name']}{bases}`**{cls_doc_str}\n")
                methods = cls.get("methods", [])
                if methods:
                    for m in methods[:15]:
                        params = ", ".join(m.get("params", []))
                        ret = f" → `{m.get('return_type')}`" if m.get("return_type") else ""
                        m_doc = (m.get("docstring") or "").split("\n")[0][:60]
                        m_doc_str = f" — {m_doc}" if m_doc else ""
                        lines.append(f"  - `{m['name']}({params})`{ret}{m_doc_str}\n")
                    if len(methods) > 15:
                        lines.append(f"  - ... +{len(methods) - 15} more methods\n")
            lines.append("\n")

        # Functions with signatures and docstrings
        if functions:
            lines.append("**Functions:**\n\n")
            for fn in functions[:20]:
                params = ", ".join(fn.get("params", []))
                ret = f" → `{fn.get('return_type')}`" if fn.get("return_type") else ""
                fn_doc = (fn.get("docstring") or "").split("\n")[0][:80]
                fn_doc_str = f" — {fn_doc}" if fn_doc else ""
                lines.append(f"- `{fn['name']}({params})`{ret}{fn_doc_str}\n")
            if len(functions) > 20:
                lines.append(f"- ... +{len(functions) - 20} more functions\n")
            lines.append("\n")

        # Associated business rules
        file_rules = rules_by_file.get(fpath, [])
        if file_rules:
            lines.append("**Business Rules:**\n\n")
            for r in file_rules[:5]:
                content = getattr(r, "content", str(r))
                rtype = getattr(r, "rule_type", "")
                lines.append(f"- [{rtype}] {content}\n")
            if len(file_rules) > 5:
                lines.append(f"- ... +{len(file_rules) - 5} more rules\n")
            lines.append("\n")

        lines.append("---\n\n")

    lines += ["\n## 4. Call Graph Hotspots\n\n| Rank | Symbol | Type | PageRank | File |\n|------|--------|------|----------|------|\n"]
    for i, h in enumerate(graph_data.get("hotspots", [])[:20], 1):
        file_label = h.get("file", "—")
        label = h.get("label", h.get("name", "unknown"))
        htype = h.get("type", "unknown")
        pr = h.get("pagerank", 0.0)
        try:
            pr_str = f"{float(pr):.4f}"
        except (ValueError, TypeError):
            pr_str = str(pr)
        lines.append(f"| {i} | `{label}` | {htype} | {pr_str} | `{file_label}` |\n")

    # Business rules grouped by type
    lines += ["\n## 5. Business Rules & Constraints\n\n"]
    by_type: dict = {}
    for r in rules:
        rt = getattr(r, "rule_type", "other")
        by_type.setdefault(rt, []).append(r)

    type_labels = {
        "endpoint": "API Endpoints",
        "constant": "Business Constants & Limits",
        "docstring": "Rules from Docstrings",
        "todo": "Pending Issues (TODOs)",
    }
    for rt, label in type_labels.items():
        items = by_type.get(rt, [])
        if not items:
            continue
        lines.append(f"### {label}\n\n")
        for r in items:
            content = getattr(r, "content", str(r))
            src = getattr(r, "source_file", "")
            lineno = getattr(r, "source_line", "")
            src_ref = f"  (`{src}:{lineno}`)" if src else ""
            lines.append(f"- {content}{src_ref}\n")
        lines.append("\n")

    if not rules:
        lines.append("_No rules extracted._\n")

    # ---- Layer 6: Decision Points & Business Decisions ----------------------
    dp_list = decision_points or []
    dc_list = domain_concepts or []
    gd_list = git_decisions or []

    if dp_list or dc_list or gd_list:
        lines += ["\n## 6. Decision Points & Business Decisions\n\n"]
        lines += ["_Code locations where business logic is encoded as conditionals, thresholds, or role checks._\n\n"]

        if dc_list:
            lines += ["### Domain Concepts\n\n"]
            lines += ["| Concept | Type | Related Classes |\n|---------|------|----------------|\n"]
            for dc in dc_list[:20]:
                classes_str = ", ".join(f"`{c}`" for c in dc.get("related_classes", [])[:5])
                desc = dc.get("description") or ""
                lines.append(f"| **{dc['name']}** | {dc.get('type', '')} | {classes_str} |\n")
                if desc:
                    lines.append(f"\n  _{desc}_\n\n")
            lines.append("\n")

        # Group decision points by type
        dp_by_type: dict[str, list] = {}
        for dp in dp_list:
            dp_by_type.setdefault(dp.get("condition_type", "other"), []).append(dp)

        type_labels = {
            "threshold": "⚠️ Threshold Decisions (magic numbers, limits)",
            "role_check": "🔐 Role & Permission Checks",
            "status_check": "📋 Status Checks",
            "feature_flag": "🚩 Feature Flags",
            "error_guard": "🛡️ Error Guards",
            "logic_branch": "🔀 Logic Branches",
        }

        for dp_type in ("threshold", "role_check", "status_check", "feature_flag"):
            items = dp_by_type.get(dp_type, [])
            if not items:
                continue
            label = type_labels.get(dp_type, dp_type)
            lines.append(f"### {label}\n\n")
            for dp in items[:15]:
                func_short = dp.get("function_id", "").split("::")[-1]
                explanation = dp.get("explanation") or ""
                question = dp.get("question_for_human") or ""
                lines.append(f"- **`{func_short}`** line {dp.get('line', '?')}: `{dp.get('condition', '')}`\n")
                if explanation:
                    lines.append(f"  - _{explanation}_\n")
                if question:
                    lines.append(f"  - ❓ {question}\n")
            if len(items) > 15:
                lines.append(f"- ... +{len(items) - 15} more\n")
            lines.append("\n")

        if gd_list:
            lines += ["### Git-Mined Decisions\n\n"]
            lines += ["_Commits containing business decision context._\n\n"]
            for gd in gd_list[:15]:
                date = gd.get("date", "")[:10]
                msg = gd.get("message", "")[:120]
                dtype = gd.get("decision_type", "")
                files = ", ".join(f"`{f}`" for f in gd.get("affected_files", [])[:3])
                lines.append(f"- [{date}] **{dtype}**: {msg}\n")
                if files:
                    lines.append(f"  - Files: {files}\n")
            lines.append("\n")

    # Call flow: import graph
    edges = graph_data.get("edges", [])
    import_edges = [(e.get("source", "?"), e.get("target", "?")) for e in edges if e.get("type") == "IMPORTS"]
    call_edges = [(e.get("source", "?"), e.get("target", "?")) for e in edges if e.get("type") == "CALLS"]

    lines += ["\n## 6. Call Flow & Module Dependencies\n\n"]
    if import_edges:
        lines.append("### Module Import Graph\n\n| Importer | Imports |\n|----------|---------|\n")
        for src, tgt in import_edges[:30]:
            lines.append(f"| `{src}` | `{tgt}` |\n")
        lines.append("\n")
    if call_edges:
        lines.append("### Key Call Relationships\n\n| Caller | Callee |\n|--------|--------|\n")
        for src, tgt in call_edges[:30]:
            lines.append(f"| `{src}` | `{tgt}` |\n")

    # ---- Layer 7: Source Code ------------------------------------------------
    lines += ["\n## 7. Source Code\n\n"]
    lines += ["_Complete source code of all indexed files._\n\n"]

    _MAX_FILE_SIZE = 50_000  # skip files larger than 50KB
    _MAX_TOTAL_CODE = 500_000  # cap total source code at 500KB
    total_code_chars = 0

    for pf in parsed:
        if total_code_chars >= _MAX_TOTAL_CODE:
            lines.append(f"\n> ⚠ Source code truncated at {_MAX_TOTAL_CODE // 1000}KB to stay within context limits.\n")
            break

        file_path_str = pf.get("path", "")
        if not file_path_str:
            continue

        # Resolve full path
        if repo_path:
            full_path = Path(repo_path) / file_path_str
        else:
            full_path = Path(file_path_str)

        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if len(content) > _MAX_FILE_SIZE:
            content = content[:_MAX_FILE_SIZE] + f"\n... [truncated at {_MAX_FILE_SIZE // 1000}KB]"

        # Detect language for syntax highlighting
        ext = full_path.suffix.lower()
        lang_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".go": "go",
            ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c",
            ".cpp": "cpp", ".h": "c", ".css": "css", ".html": "html",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".sh": "bash", ".sql": "sql", ".md": "markdown",
        }
        lang = lang_map.get(ext, "")

        lines.append(f"### `{file_path_str}`\n\n")
        lines.append(f"```{lang}\n{content}\n```\n\n")
        total_code_chars += len(content)

    context_md = "".join(lines)
    (out / "context.md").write_text(context_md)

    # Compact summary (~3k tokens)
    summary = (
        f"# {repo_name} — Summary\n\n"
        f"**Stack:** {tech}\n"
        f"**Files:** {stats.get('total_files',0)} | **LOC:** ~{total_loc:,}\n"
        f"**Entry Points:** {', '.join(entries) or '—'}\n\n"
    )
    summary += "## Top Hotspots\n" + "\n".join(
        f"{i+1}. `{h['label']}` ({h['type']}) pagerank={h['pagerank']:.4f}"
        for i, h in enumerate(graph_data.get("hotspots", [])[:10])
    )
    if rules:
        summary += "\n\n## Business Rules\n" + "\n".join(
            f"- [{getattr(r, 'rule_type', '')}] {getattr(r, 'content', '')}"
            for r in rules[:20]
        )
    (out / "summary.md").write_text(summary)


@app.command()
def fix(
    ticket_id: str = typer.Argument(..., help="Bug ticket ID (e.g., PROJ-1234)"),
    title: str = typer.Option("", "--title", "-t", help="Bug title"),
    description: str = typer.Option("", "--desc", "-d", help="Bug description"),
    repo_path: str = typer.Option("", "--repo", "-r", help="Path to local repo"),
    repo_name: str = typer.Option("", "--name", "-n", help="Repo name (for graph lookup)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Skip PR creation"),
    best_of_n: int = typer.Option(1, "--best-of-n", "-N", help="Run N parallel instances, pick best (1=off, 3=recommended)"),
):
    """
    Run the AI agent to fix a bug.

    Examples:
        python cli.py fix PROJ-1234 --title "Bug title" --desc "description" --repo /path/to/repo
    """
    from agent.trace import RunTrace

    work_order = {
        "ticket_id": ticket_id,
        "title": title or ticket_id,
        "description": description or f"Fix bug {ticket_id}",
        "repo_name": repo_name or (Path(repo_path).name if repo_path else ticket_id.lower()),
        "repo_path": repo_path,
        "priority": "medium",
        "comments": [],
    }

    trace = RunTrace(job_id=ticket_id, enabled=True)

    console.print(Panel(
        f"[bold cyan]Ticket:[/bold cyan]    {ticket_id}\n"
        f"[bold cyan]Dry run:[/bold cyan]   {dry_run}\n"
        f"[bold cyan]Best-of-N:[/bold cyan] {best_of_n}\n"
        f"[bold cyan]Repo:[/bold cyan]      {repo_path or '(auto-detect)'}",
        title="[bold]AI Deploy Agent[/bold]",
        border_style="cyan",
    ))

    from agent.react_pipeline import run_ticket_react
    result = run_ticket_react(work_order, trace=trace, dry_run=dry_run, best_of_n=best_of_n)

    status = result.get("status", "unknown")
    pr_url = result.get("pr_url", "")
    error = result.get("error", "")

    if status == "done":
        console.print(f"\n[bold green]SUCCESS[/bold green] — {pr_url or 'Fix generated'}")
    elif status == "escalated":
        reason = result.get("escalate_reason", result.get("error", "unknown"))
        console.print(f"\n[bold yellow]ESCALATED[/bold yellow] — {reason}")
    else:
        console.print(f"\n[bold red]FAILED[/bold red] — {error or status}")

    # Print best-of-N stats if applicable
    bon_stats = result.get("best_of_n_stats")
    if bon_stats:
        console.print(
            f"[dim]Best-of-{bon_stats['n']}: {bon_stats['submitted']}/{bon_stats['n']} submitted, "
            f"{bon_stats['test_pass']}/{bon_stats['n']} tests passed[/dim]"
        )

    # Print trace summary
    report = trace.to_report()
    console.print(f"\n[dim]Duration: {report.get('total_duration_seconds', 0):.0f}s | "
                  f"Tool calls: {result.get('tool_call_count', 'N/A')} | "
                  f"Cost: ${result.get('cost_usd', 0):.4f}[/dim]")


def _print_structure_summary(structure: dict):
    techs = ", ".join(structure.get("tech_stack", [])) or "unknown"
    stats = structure.get("file_stats", {})
    console.print(f"  [green]✓[/green] Tech stack: [bold]{techs}[/bold] | "
                  f"{stats.get('total_files', 0)} files | "
                  f"~{stats.get('total_lines', 0):,} lines")


def _print_hotspots(hotspots: list):
    if not hotspots:
        return
    table = Table(title="Top Hotspots (by PageRank)", show_header=True)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("PageRank", style="yellow")
    for i, h in enumerate(hotspots, 1):
        table.add_row(str(i), h["label"], h["type"], f"{h['pagerank']:.4f}")
    console.print(table)


@app.command()
def update(
    repo_path: str = typer.Argument(..., help="Path to the repo to incrementally update"),
    name: str = typer.Option(None, "--name", "-n", help="Override repo name"),
    since: str = typer.Option("", "--since", help="Only re-index files changed since this git ref (e.g. HEAD~1)"),
):
    """
    Incrementally update a repo's knowledge graph — only re-parses changed files.

    Uses SHA-256 content hashing (code-review-graph pattern): compares current file
    hashes against the stored graph, re-parses only files that changed.
    <2s re-index for a single-file change vs. 30s+ for a full rebuild.

    Examples:
        python cli.py update ~/projects/my-api          # Hash-based incremental
        python cli.py update ~/projects/my-api --since HEAD~3  # Git diff range
    """
    import hashlib
    import json as _json
    from analyzer.code_parser import CodeParser
    from analyzer.call_graph import CallGraphBuilder

    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)

    repo_name = name or path.name
    data_dir = Path(f"/tmp/context_builder/{repo_name}")

    graph_path = data_dir / "graph.json"
    if not graph_path.exists():
        console.print(f"[yellow]No existing graph for '{repo_name}'. Run 'build' first.[/yellow]")
        raise typer.Exit(1)

    existing_graph = _json.loads(graph_path.read_text())

    # Build hash index of existing file nodes
    old_hashes: dict[str, str] = {}
    for node in existing_graph.get("nodes", []):
        if node.get("type") == "file":
            fpath = node.get("file") or node.get("id", "")
            if fpath and node.get("content_hash"):
                old_hashes[fpath] = node["content_hash"]

    # Find changed files
    changed_files: list[str] = []

    if since:
        # Use git diff for explicit range
        try:
            import subprocess as _sp
            result = _sp.run(
                ["git", "diff", "--name-only", since, "HEAD"],
                cwd=str(path), capture_output=True, text=True, timeout=15,
            )
            changed_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except Exception as e:
            console.print(f"[yellow]git diff failed: {e} — falling back to hash comparison[/yellow]")

    if not changed_files:
        # Hash-based detection: compare current SHA-256 of each source file
        src_extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs"}
        for fpath in old_hashes:
            full = path / fpath
            if not full.exists():
                changed_files.append(fpath)  # deleted
                continue
            try:
                content = full.read_bytes()
                current_hash = hashlib.sha256(content).hexdigest()[:16]
                if old_hashes[fpath] != current_hash:
                    changed_files.append(fpath)
            except Exception:
                pass
        # Also detect genuinely new files
        for ext in src_extensions:
            for f in path.rglob(f"*{ext}"):
                try:
                    rel = str(f.relative_to(path))
                except ValueError:
                    continue
                if rel not in old_hashes:
                    _skip = ("__pycache__", "node_modules", ".venv", "venv", "dist", "build")
                    if not any(s in rel for s in _skip):
                        changed_files.append(rel)

    if not changed_files:
        console.print(f"[green]No changes detected for '{repo_name}'. Graph is up to date.[/green]")
        return

    console.print(f"[cyan]Re-indexing {len(changed_files)} changed file(s) for '{repo_name}'...[/cyan]")
    for f in changed_files[:10]:
        console.print(f"  [dim]  {f}[/dim]")
    if len(changed_files) > 10:
        console.print(f"  [dim]  ... +{len(changed_files) - 10} more[/dim]")

    # Re-parse only changed files
    parser = CodeParser(path)
    reparsed = []
    for fpath in changed_files:
        full = path / fpath
        if full.exists():
            try:
                parsed = parser.parse_file(full)
                if parsed:
                    reparsed.append(parsed)
            except Exception as e:
                console.print(f"  [yellow]⚠ Parse error in {fpath}: {e}[/yellow]")

    if not reparsed:
        console.print("[yellow]No files successfully re-parsed.[/yellow]")
        return

    # Rebuild partial call graph for changed files
    cg = CallGraphBuilder(reparsed)
    new_graph_data = cg.build()

    # Merge: remove old nodes/edges for changed files, add new ones
    changed_set = set(changed_files)
    kept_nodes = [
        n for n in existing_graph.get("nodes", [])
        if (n.get("file") or n.get("id", "").split("::")[0]) not in changed_set
    ]
    kept_edges = [
        e for e in existing_graph.get("edges", [])
        if (e.get("source", "").split("::")[0]) not in changed_set
        and (e.get("target", "").split("::")[0]) not in changed_set
    ]

    # Add new content hashes to new nodes
    for node in new_graph_data.get("nodes", []):
        if node.get("type") == "file":
            fpath = node.get("file") or node.get("id", "")
            full = path / fpath
            if full.exists():
                try:
                    content = full.read_bytes()
                    node["content_hash"] = hashlib.sha256(content).hexdigest()[:16]
                except Exception:
                    pass

    merged_graph = {
        **existing_graph,
        "nodes": kept_nodes + new_graph_data.get("nodes", []),
        "edges": kept_edges + new_graph_data.get("edges", []),
    }
    merged_graph["stats"] = {
        **existing_graph.get("stats", {}),
        "last_updated": str(Path(repo_path).stat().st_mtime),
        "files_reindexed": len(changed_files),
    }

    graph_path.write_text(_json.dumps(merged_graph, default=str))
    console.print(
        f"[green]✓[/green] Updated: {len(reparsed)} files re-indexed | "
        f"{len(merged_graph['nodes'])} nodes | {len(merged_graph['edges'])} edges"
    )



if __name__ == "__main__":
    app()
