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
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(help="Build and query knowledge graphs for any code repository.")
console = Console()


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

        if summaries:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                console.print("  [yellow]⚠[/yellow]  ANTHROPIC_API_KEY not set — skipping summaries")
            else:
                progress.update(task, description="[cyan]Generating LLM file summaries (Claude)...", completed=70)
                from enricher.summarizer import Summarizer
                summarizer = Summarizer(repo_name)
                summarizer.enrich()
                console.print(f"  [green]✓[/green] LLM file summaries generated")

                progress.update(task, description="[cyan]Generating LLM function summaries (Claude)...", completed=73)
                fn_count = summarizer.enrich_functions()
                if fn_count:
                    console.print(f"  [green]✓[/green] {fn_count} function summaries generated")

                # LLM-enhanced decision points
                progress.update(task, description="[cyan]Enhancing decision points (Claude)...", completed=76)
                from enricher.decision_points import enhance_with_llm as enhance_decisions
                enhanced_dp = enhance_decisions(decision_points, api_key)
                if enhanced_dp:
                    console.print(f"  [green]✓[/green] {enhanced_dp} decision points enhanced with LLM")

                # LLM-enhanced domain concepts
                progress.update(task, description="[cyan]Enriching domain concepts (Claude)...", completed=78)
                from enricher.domain_concepts import enhance_with_llm as enhance_concepts
                enhanced_dc = enhance_concepts(domain_concepts, parsed, api_key)
                if enhanced_dc:
                    console.print(f"  [green]✓[/green] {enhanced_dc} domain concepts enriched with LLM")

        progress.update(task, description="[cyan]Extracting business rules...", completed=80)
        from enricher.business_logic import BusinessLogicExtractor
        extractor = BusinessLogicExtractor(repo_name, parsed)
        if no_neo4j:
            rules = extractor.extract_all()
            rules_count = len(rules)
        else:
            rules_count = extractor.extract()
            rules = []
        console.print(f"  [green]✓[/green] {rules_count} business rules extracted")

        # Save enriched nodes + build embeddings
        progress.update(task, description="[cyan]Building enriched node cache...", completed=85)
        from embeddings.embedder import build_enriched_nodes, NodeEmbedder
        enriched = build_enriched_nodes(parsed, graph_data, decision_points, domain_concepts, rules if no_neo4j else [])
        import json as _json
        enriched_path = Path(f"/tmp/context_builder/{repo_name}/enriched_nodes.json")
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_path.write_text(_json.dumps(enriched, default=str))
        console.print(f"  [green]✓[/green] Enriched node cache: {len(enriched)} nodes")

        progress.update(task, description="[cyan]Building vector embeddings (ChromaDB)...", completed=88)
        try:
            embedder = NodeEmbedder(repo_name, Path("/tmp/context_builder"))
            embed_count = embedder.build_embeddings(enriched)
            console.print(f"  [green]✓[/green] Embedded {embed_count} nodes into ChromaDB")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow]  ChromaDB embedding failed: {e}")

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
):
    """Write context.md and summary.md directly from in-memory data (no Neo4j)."""
    from datetime import datetime, timezone
    out = Path(f"/tmp/context_builder/{repo_name}")
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


if __name__ == "__main__":
    app()
