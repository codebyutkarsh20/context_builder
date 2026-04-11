"""
graph_utils.py — Knowledge graph query utilities extracted from pipeline.py.

Loads graph data, finds callers/importers, builds reviewer context,
loads business rules, and assembles kickstart orientation context.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def load_graph_data(repo_name: str) -> tuple[dict, dict]:
    """Load graph.json and enriched_nodes.json for a repo."""
    graph_data: dict = {}
    enriched: dict = {}
    try:
        graph_path = DATA_DIR / repo_name / "graph.json"
        if graph_path.exists():
            graph_data = json.loads(graph_path.read_text())
    except Exception as e:
        logger.warning("Failed to load graph.json: %s", e)
    try:
        enriched_path = DATA_DIR / repo_name / "enriched_nodes.json"
        if enriched_path.exists():
            enriched = json.loads(enriched_path.read_text())
    except Exception as e:
        logger.warning("Failed to load enriched_nodes.json: %s", e)
    return graph_data, enriched


def find_callers_from_graph(
    graph_data: dict, fault_files: list[str], fault_functions: list[str],
) -> list[str]:
    """Use the knowledge graph CALLS/IMPORTS edges to find caller files."""
    edges = graph_data.get("edges", [])
    if not edges:
        return []

    target_ids: set[str] = set()
    for f in fault_files:
        target_ids.add(f)
        stem = Path(f).stem
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if stem in tgt:
                target_ids.add(tgt)
            if stem in src:
                target_ids.add(src)

    for fn in fault_functions:
        target_ids.add(fn)

    caller_files: set[str] = set()
    fault_file_set = set(fault_files)
    _caller_noise = ("test_", "conftest", "/tests/", "/test/", "/__pycache__/")

    for edge in edges:
        etype = edge.get("type", "")
        if etype not in ("CALLS", "IMPORTS"):
            continue
        target = edge.get("target", "")
        source = edge.get("source", "")

        if target in target_ids or any(t in target for t in target_ids):
            src_file = source.split("::")[0] if "::" in source else source
            if src_file and src_file not in fault_file_set:
                if not any(pat in src_file.lower() for pat in _caller_noise):
                    caller_files.add(src_file)

    return sorted(caller_files)[:8]


def find_callers_via_grep(repo_path: Path, fault_files: list[str]) -> list[str]:
    """Fallback: grep for files that import the fault files."""
    caller_paths: list[str] = []
    seen: set[str] = set()
    _caller_noise = ("test_", "conftest", "/tests/", "/test/", "/__pycache__/")

    for rel_path in fault_files:
        stem = Path(rel_path).stem
        parts = Path(rel_path).with_suffix("").parts
        search_terms = [f"import {stem}", f"from {stem}"]
        if len(parts) > 1:
            for i in range(len(parts) - 1):
                search_terms.append(f"from {'.'.join(parts[i:])}")
        try:
            for term in search_terms:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", term, str(repo_path)],
                    capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    p = Path(line)
                    if p.exists() and str(p) not in seen:
                        if any(pat in str(p).lower() for pat in _caller_noise):
                            continue
                        rel = str(p.relative_to(repo_path))
                        if rel not in set(fault_files):
                            seen.add(str(p))
                            caller_paths.append(rel)
                            if len(caller_paths) >= 5:
                                return caller_paths
        except Exception:
            pass
    return caller_paths


def build_reviewer_context(repo_name: str, modified_files: list[str]) -> str:
    """Build independent reviewer context from graph data."""
    graph_data, enriched = load_graph_data(repo_name)
    sections = []

    rules = []
    for nid, node in enriched.items():
        ntype = node.get("type", "")
        if ntype not in ("business_rule", "decision_point"):
            continue
        node_file = node.get("file", "") or node.get("function_id", "")
        if any(f in node_file for f in modified_files):
            if ntype == "business_rule":
                rules.append(f"  [{node.get('rule_type', 'policy')}] {node.get('content', node.get('name', ''))[:200]}")
            else:
                q = node.get("question_for_human", "")
                if q:
                    rules.append(f"  [decision] {node.get('name', '')}: {q[:200]}")

    if rules:
        sections.append("BUSINESS RULES & DECISION POINTS (for modified files):")
        sections.extend(rules[:15])

    callers = find_callers_from_graph(graph_data, modified_files, [])
    if callers:
        sections.append("\nBLAST RADIUS (files that call/import the modified code):")
        for c in callers[:10]:
            sections.append(f"  - {c}")
        risk = "CRITICAL" if len(callers) > 5 else "HIGH" if len(callers) > 2 else "MEDIUM"
        sections.append(f"  Risk level: {risk}")
    else:
        sections.append("\nBLAST RADIUS: No downstream consumers detected. Risk: LOW")

    return "\n".join(sections) if sections else "No business rules or blast radius data available."


def load_business_rules(repo_name: str, fault_files: list[str]) -> str:
    """Load stored business rules + failure history relevant to the fault files."""
    from graph.neo4j_client import neo4j_client

    fault_functions = [Path(f).stem for f in fault_files]
    sections: list[str] = []

    rules_path = DATA_DIR / repo_name / "business_rules.json"
    relevant_rules: list[str] = []
    if rules_path.exists():
        try:
            all_rules = json.loads(rules_path.read_text())
            for rule in all_rules:
                rule_file = rule.get("file", "")
                rule_func = rule.get("function_id", "")
                if any(f in rule_file or f in rule_func for f in fault_files):
                    severity = rule.get("severity", "medium").upper()
                    marker = "DO NOT VIOLATE" if severity in ("CRITICAL", "HIGH") else ""
                    relevant_rules.append(
                        f"  [{severity}] {rule.get('description', '')[:300]} {marker}\n"
                        f"    Source: {rule.get('source', 'unknown')} | File: {rule_file}"
                    )
        except Exception:
            pass

    if relevant_rules:
        sections.append(
            "\n\nBUSINESS RULES (verified knowledge base — DO NOT VIOLATE):\n"
            + "\n".join(relevant_rules)
        )

    try:
        if neo4j_client.is_connected():
            for file_path in fault_files:
                rows = neo4j_client.run(
                    "MATCH (fr:FailureRecord)-[:RESULTED_IN_CHANGE]->(n) "
                    "WHERE (n:Function OR n:File) "
                    "  AND (n.path ENDS WITH $file OR n.name IN $funcs) "
                    "  AND fr.repo = $repo "
                    "RETURN fr.message AS message, fr.date AS date, "
                    "       fr.issue_ref AS issue_ref, fr.severity_hint AS severity "
                    "ORDER BY fr.date DESC LIMIT 5",
                    {"file": file_path, "funcs": fault_functions, "repo": repo_name},
                )
                if rows:
                    failure_lines = []
                    for row in rows:
                        ref = f" ({row['issue_ref']})" if row.get("issue_ref") else ""
                        failure_lines.append(
                            f"  [{row.get('date', '?')}]{ref} {row.get('message', '')[:200]}"
                        )
                    sections.append(
                        f"\n\nPAST FAILURES touching {file_path}:\n"
                        + "\n".join(failure_lines)
                    )
    except Exception as exc:
        logger.debug("FailureRecord query failed (non-fatal): %s", exc)

    if not sections:
        fault_desc = ", ".join(fault_files[:3]) or "target function"
        return (
            "\n\nWARNING: No business rules or failure history found for "
            f"{fault_desc}. Treat as high-risk — do not remove validation "
            "logic without explicit confirmation."
        )

    return "".join(sections)


def _build_repo_map(graph_data: dict) -> str:
    """Build directory-grouped file tree from graph.json file nodes. Cap at 60 lines."""
    from collections import defaultdict

    file_nodes = [n for n in graph_data.get("nodes", []) if n.get("type") == "file"]
    if not file_nodes:
        return ""

    _skip = frozenset({"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".eggs"})
    dirs: dict[str, list[str]] = defaultdict(list)
    for node in file_nodes:
        fpath = node.get("file") or node.get("id", "")
        if not fpath:
            continue
        parent = str(Path(fpath).parent)
        if any(s in parent for s in _skip):
            continue
        dirs[parent].append(fpath)

    lines: list[str] = []
    for dir_path in sorted(dirs.keys()):
        if any(s in dir_path for s in _skip):
            continue
        files = sorted(dirs[dir_path])
        dir_label = dir_path if dir_path != "." else "(root)"
        lines.append(f"  {dir_label}/")
        for f in files:
            lines.append(f"    {Path(f).name}")
        if len(lines) > 55:
            remaining = sum(len(v) for v in dirs.values()) - len(lines)
            if remaining > 0:
                lines.append(f"    ... ({remaining} more files)")
            break

    if not lines:
        return ""
    return "REPO MAP (directory structure from graph):\n" + "\n".join(lines[:60])


def _build_function_locator(
    graph_data: dict, hint_functions: list[str], hint_files: list[str],
) -> str:
    """Map hint function names → file:line from graph nodes. Expands 1 hop via CALLS edges."""
    if not hint_functions and not hint_files:
        return ""

    func_nodes = [n for n in graph_data.get("nodes", []) if n.get("type") == "function"]
    edges = graph_data.get("edges", [])
    hint_fn_lower = {f.lower() for f in hint_functions}
    hint_file_set = set(hint_files)

    # Direct matches: name match OR file match
    matched: dict[str, dict] = {}
    for node in func_nodes:
        nid = node.get("id", "")
        name = node.get("label") or nid.split("::")[-1]
        if name.lower() in hint_fn_lower or node.get("file") in hint_file_set:
            matched[nid] = node

    # 1-hop expansion via CALLS edges
    matched_ids = set(matched.keys())
    func_by_id = {n.get("id", ""): n for n in func_nodes}
    hop1: dict[str, dict] = {}
    for edge in edges:
        if edge.get("type") != "CALLS":
            continue
        src, tgt = edge.get("source", ""), edge.get("target", "")
        for a, b in ((src, tgt), (tgt, src)):
            if a in matched_ids and b not in matched_ids and b in func_by_id:
                hop1[b] = func_by_id[b]

    def _fmt(node: dict, tag: str = "") -> str:
        nid = node.get("id", "")
        name = node.get("label") or nid.split("::")[-1]
        fpath = node.get("file", "?")
        ls, le = node.get("line_start", ""), node.get("line_end", "")
        loc = f" (line {ls}-{le})" if ls and le else ""
        suffix = f"  [{tag}]" if tag else ""
        return f"  {name} → {fpath}{loc}{suffix}"

    entries: list[str] = [_fmt(n) for n in list(matched.values())[:15]]
    entries += [_fmt(n, "1-hop") for n in list(hop1.values())[:5]]

    if not entries:
        return ""
    return "FUNCTION LOCATOR (hint functions + 1-hop neighbors):\n" + "\n".join(entries[:20])


def _build_call_subgraph(
    graph_data: dict, hint_files: list[str], hint_functions: list[str],
) -> str:
    """Pre-filter call graph to 2-hop neighborhood of hint area. Cap at 25 edges."""
    if not hint_files and not hint_functions:
        return ""

    edges = graph_data.get("edges", [])
    hint_file_set = set(hint_files)
    hint_fn_lower = {f.lower() for f in hint_functions}

    # Build set of node ids that belong to the hint area
    hint_ids: set[str] = set()
    for node in graph_data.get("nodes", []):
        nid = node.get("id", "")
        name = node.get("label") or nid.split("::")[-1]
        if node.get("file") in hint_file_set or name.lower() in hint_fn_lower:
            hint_ids.add(nid)
    # Also include bare file ids
    hint_ids.update(hint_file_set)

    _test_markers = ("test_", "/tests/", "/test/", "conftest")

    inbound: list[str] = []
    outbound: list[str] = []
    test_links: list[str] = []
    seen: set[str] = set()

    for edge in edges:
        etype = edge.get("type", "")
        if etype not in ("CALLS", "IMPORTS"):
            continue
        src, tgt = edge.get("source", ""), edge.get("target", "")
        src_file = src.split("::")[0]
        tgt_file = tgt.split("::")[0]

        is_hint_src = src_file in hint_file_set or src in hint_ids
        is_hint_tgt = tgt_file in hint_file_set or tgt in hint_ids
        if not is_hint_src and not is_hint_tgt:
            continue

        key = f"{src}→{tgt}"
        if key in seen:
            continue
        seen.add(key)

        src_label = src.split("::")[-1] if "::" in src else src_file
        tgt_label = tgt.split("::")[-1] if "::" in tgt else tgt_file
        line = f"  {src_file}::{src_label} —[{etype}]→ {tgt_file}::{tgt_label}"

        is_test = any(m in src.lower() or m in tgt.lower() for m in _test_markers)
        if is_test:
            test_links.append(line)
        elif is_hint_tgt and not is_hint_src:
            inbound.append(line)
        elif is_hint_src and not is_hint_tgt:
            outbound.append(line)

    parts: list[str] = []
    if inbound:
        parts.append("  INBOUND (callers of the hint area):")
        parts.extend(inbound[:10])
    if outbound:
        parts.append("  OUTBOUND (what the hint area calls):")
        parts.extend(outbound[:10])
    if test_links:
        parts.append("  TEST COVERAGE (test files touching hint area):")
        parts.extend(test_links[:5])

    if not parts:
        return ""
    return "CALL SUBGRAPH (2-hop around hint area):\n" + "\n".join(parts[:25])


def _build_flow_context(
    repo_name: str, hint_files: list[str], data_dir: Path,
) -> str:
    """Build execution flow context for hint files from flows.json."""
    flows_path = data_dir / repo_name / "flows.json"
    if not flows_path.exists():
        return ""
    try:
        flows_data = json.loads(flows_path.read_text())
    except Exception:
        return ""

    flows = flows_data.get("flows", [])
    if not flows or not hint_files:
        return ""

    # Find flows that touch any hint file.
    hint_set = set(hint_files)
    relevant: list[dict] = []
    for flow in flows:
        flow_files = set(flow.get("files", []))
        if flow_files & hint_set:
            relevant.append(flow)

    if not relevant:
        return ""

    # Sort by criticality, take top 5.
    relevant.sort(key=lambda f: f.get("criticality", 0), reverse=True)
    relevant = relevant[:5]

    parts = ["EXECUTION FLOWS touching hint area:"]
    for flow in relevant:
        path_preview = flow.get("path", [])[:6]
        labels = [p.rsplit("::", 1)[-1] if "::" in p else p for p in path_preview]
        trail = " → ".join(labels)
        if len(flow.get("path", [])) > 6:
            trail += " → ..."
        parts.append(
            f"  [{flow.get('criticality', 0):.2f}] {flow['name']}: {trail}"
        )

    return "\n".join(parts)


def build_kickstart_context(
    repo_name: str, repo_path: str | None, intent: dict, data_dir: Path,
) -> str:
    """Build orientation context for exploration from graph + failure signals."""
    from graph.neo4j_client import neo4j_client

    sections: list[str] = []
    hint_files = [f for f in intent.get("likely_affected_modules", [])[:5] if f]
    hint_functions = [f for f in intent.get("likely_affected_functions", [])[:5] if f]
    bug_query = " ".join(filter(None, [
        intent.get("actual_behavior", ""),
        intent.get("expected_behavior", ""),
    ]))

    # 0. Pre-built structural context (from stored graph — zero tool calls needed)
    try:
        graph_data_for_map, _ = load_graph_data(repo_name)
        repo_map = _build_repo_map(graph_data_for_map)
        func_locator = _build_function_locator(graph_data_for_map, hint_functions, hint_files)
        call_subgraph = _build_call_subgraph(graph_data_for_map, hint_files, hint_functions)
        flow_context = _build_flow_context(repo_name, hint_files, data_dir)
        orientation_parts = [p for p in [repo_map, func_locator, call_subgraph, flow_context] if p]
        if orientation_parts:
            sections.append(
                "REPO ORIENTATION (pre-built from stored graph — use this before calling list_files or grep):\n\n"
                + "\n\n".join(orientation_parts)
            )
    except Exception as e:
        logger.debug("Repo orientation build failed (non-fatal): %s", e)

    # 1. Graph neighbors
    try:
        neighbors: list[str] = []
        queried_via_neo4j = False
        if neo4j_client.is_connected() and (hint_files or hint_functions):
            try:
                rows = neo4j_client.run(
                    "MATCH (a)-[r:CALLS|IMPORTS]->(b) "
                    "WHERE (a.path IN $files OR b.path IN $files "
                    "       OR a.name IN $funcs OR b.name IN $funcs) "
                    "RETURN a.path AS src_file, a.name AS src_name, "
                    "       b.path AS tgt_file, b.name AS tgt_name, type(r) AS rel "
                    "LIMIT 15",
                    {"files": hint_files, "funcs": hint_functions},
                )
                if rows:
                    queried_via_neo4j = True
                    for row in rows:
                        src = f"{row.get('src_file', '?')}::{row.get('src_name', '?')}"
                        tgt = f"{row.get('tgt_file', '?')}::{row.get('tgt_name', '?')}"
                        neighbors.append(f"  - {src} —[{row.get('rel','CALLS')}]→ {tgt}")
            except Exception:
                pass
        if not queried_via_neo4j:
            graph_data, _ = load_graph_data(repo_name)
            for f in find_callers_from_graph(graph_data, hint_files, hint_functions)[:8]:
                neighbors.append(f"  - {f}")
        if neighbors:
            sections.append(
                "GRAPH NEIGHBORS of hint area (callers / callees):\n" + "\n".join(neighbors[:12])
            )
    except Exception:
        pass

    # 2. Past failures
    try:
        if neo4j_client.is_connected() and hint_files:
            failure_lines: list[str] = []
            for file_path in hint_files[:3]:
                rows = neo4j_client.run(
                    "MATCH (fr:FailureRecord)-[:RESULTED_IN_CHANGE]->(n) "
                    "WHERE (n:Function OR n:File) "
                    "  AND n.path ENDS WITH $file AND fr.repo = $repo "
                    "RETURN fr.message AS message, fr.date AS date, fr.issue_ref AS ref "
                    "ORDER BY fr.date DESC LIMIT 3",
                    {"file": file_path, "repo": repo_name},
                )
                for row in rows:
                    ref = f" ({row['ref']})" if row.get("ref") else ""
                    failure_lines.append(
                        f"  - [{row.get('date', '?')}]{ref} {row.get('message', '')[:120]}"
                    )
            if failure_lines:
                sections.append("PAST FAILURES in hint area:\n" + "\n".join(failure_lines))
    except Exception:
        pass

    # 3. Business rules
    try:
        rules_path = data_dir / repo_name / "business_rules.json"
        if rules_path.exists():
            all_rules = json.loads(rules_path.read_text())
            relevant = [
                r for r in all_rules
                if any(
                    h in r.get("file", "") or h in r.get("function_id", "")
                    for h in hint_files + hint_functions
                )
            ][:8]
            if relevant:
                lines = [
                    f"  - [{r.get('severity', '?').upper()}] {r.get('description', '')[:120]}"
                    for r in relevant
                ]
                sections.append(
                    "BUSINESS RULES (linked to hint area — keep in mind while exploring):\n"
                    + "\n".join(lines)
                )
    except Exception:
        pass

    # 4. PageRank hotspots
    try:
        graph_data, _ = load_graph_data(repo_name)
        hotspots = sorted(
            [n for n in graph_data.get("nodes", [])
             if n.get("type") == "function" and n.get("pagerank", 0) > 0],
            key=lambda n: n.get("pagerank", 0),
            reverse=True,
        )[:8]
        if hotspots:
            lines = [
                f"  - {h.get('id', '?')} (rank: {h.get('pagerank', 0):.3f})"
                for h in hotspots
            ]
            sections.append(
                "REPO HOTSPOTS (most central functions — not necessarily the bug):\n"
                + "\n".join(lines)
            )
    except Exception:
        pass

    if not sections:
        return ""
    return (
        "\n\nORIENTATION (starting map — explore freely, don't be constrained by this):\n\n"
        + "\n\n".join(sections)
    )
