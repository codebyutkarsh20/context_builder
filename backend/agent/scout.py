"""
scout.py — Scout tier: 3-agent Fault Localisation (FL) pipeline.

Runs before the main ReAct fix loop to answer: "Where exactly is the bug?"

Pipeline (LLM4FL-inspired):
  Agent 1 — Context Extractor  (Haiku, cheap):   bug desc + snippets → key entities
  Agent 2 — Graph-RAG Debugger (Sonnet):          entities + graph data → top-5 suspects
  Agent 3 — Verbal RL Re-ranker (Opus):           top-5 suspects → ranked + annotated report

Each agent's output feeds directly into the next.
Total target cost: < $0.50 per bug.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model aliases (sourced from llm.py constants)
# ---------------------------------------------------------------------------
_EXTRACTOR_MODEL = "claude-haiku-4-5-20251001"
_DEBUGGER_MODEL = "claude-sonnet-4-6"
_RERANKER_MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Pydantic output schemas for structured LLM calls
# ---------------------------------------------------------------------------


class ExtractedContext(BaseModel):
    """Agent 1 output: key entities extracted from the bug description."""

    function_names: list[str] = Field(
        default_factory=list,
        description="Function or method names mentioned or implied by the bug",
    )
    error_types: list[str] = Field(
        default_factory=list,
        description="Exception types, error codes, or failure modes described",
    )
    data_structures: list[str] = Field(
        default_factory=list,
        description="Key data structures, models, or variables involved",
    )
    module_hints: list[str] = Field(
        default_factory=list,
        description="Module, file, or package names that are likely involved",
    )
    bug_summary: str = Field(
        default="",
        description="One-sentence plain-English summary of what is broken",
    )


class SuspectLocation(BaseModel):
    """A single suspicious code location identified by Agent 2."""

    file: str = Field(description="Relative file path, e.g. agent/pipeline.py")
    function: str = Field(
        default="",
        description="Function or method name within the file (empty if unknown)",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial confidence score (0.0–1.0)",
    )
    reason: str = Field(
        default="",
        description="One-sentence rationale for why this location is suspect",
    )


class GraphDebuggerOutput(BaseModel):
    """Agent 2 output: top-5 suspicious locations from graph traversal."""

    suspects: list[SuspectLocation] = Field(
        default_factory=list,
        description="Up to 5 suspicious locations, ordered by descending confidence",
    )
    blast_radius_files: list[str] = Field(
        default_factory=list,
        description="Other files likely impacted if the bug is in the suspect locations",
    )
    relevant_business_rule_ids: list[str] = Field(
        default_factory=list,
        description="IDs or descriptions of business rules relevant to the suspect area",
    )


class RankedLocation(BaseModel):
    """A single re-ranked location with enriched reasoning."""

    file: str = Field(description="Relative file path")
    function: str = Field(default="", description="Function or method name")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Final confidence after re-ranking (0.0–1.0)",
    )
    reason: str = Field(
        default="",
        description="Full reasoning: why this location, what to look for, what risk it carries",
    )


class RerankerOutput(BaseModel):
    """Agent 3 output: final ranked localisation report."""

    ranked_locations: list[RankedLocation] = Field(
        default_factory=list,
        description="Top locations ordered by final confidence (highest first)",
    )
    relevant_failure_records: list[str] = Field(
        default_factory=list,
        description="Past failure messages or incident refs relevant to this bug",
    )
    additional_blast_radius: list[str] = Field(
        default_factory=list,
        description="Extra files the re-ranker identified as blast radius beyond Agent 2's list",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_business_rules(data_dir: Path, repo_name: str) -> list[dict]:
    """Load business_rules.json; return [] on any failure."""
    path = data_dir / repo_name / "business_rules.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("scout: failed to load business_rules.json: %s", exc)
        return []


def _summarise_graph(
    graph_data: dict,
    enriched: dict,
    entities: ExtractedContext,
) -> str:
    """
    Build a compact graph summary (< 1200 chars) relevant to the extracted entities.

    Strategy:
      1. Find nodes whose id/label/file overlaps with entity hints.
      2. For each matched node, collect its direct CALLS/IMPORTS edges (1-hop).
      3. Include PageRank for hotspot signalling.
    """
    nodes: list[dict] = graph_data.get("nodes", [])
    edges: list[dict] = graph_data.get("edges", [])

    # Build lookup sets from entities
    fn_hints = {n.lower() for n in entities.function_names}
    mod_hints = {m.lower() for m in entities.module_hints}
    all_hints = fn_hints | mod_hints

    # Score nodes
    scored: list[tuple[float, dict]] = []
    for node in nodes:
        nid: str = node.get("id", "")
        label: str = (node.get("label") or nid.split("::")[-1]).lower()
        file_: str = (node.get("file") or "").lower()
        score = 0.0
        if label in fn_hints:
            score += 3.0
        if any(h in file_ for h in mod_hints):
            score += 2.0
        if any(h in label for h in all_hints):
            score += 1.0
        if score > 0:
            score += node.get("pagerank", 0.0) * 10  # boost hot nodes
            scored.append((score, node))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_nodes = [n for _, n in scored[:8]]

    if not top_nodes:
        # Fall back: just the top PageRank nodes
        ranked_all = sorted(nodes, key=lambda n: n.get("pagerank", 0.0), reverse=True)
        top_nodes = ranked_all[:5]

    # Build edge map for top nodes
    top_ids = {n.get("id", "") for n in top_nodes}
    relevant_edges: list[str] = []
    seen_edges: set[str] = set()
    for edge in edges:
        if edge.get("type") not in ("CALLS", "IMPORTS"):
            continue
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src not in top_ids and tgt not in top_ids:
            continue
        key = f"{src}→{tgt}"
        if key in seen_edges:
            continue
        seen_edges.add(key)
        src_label = src.split("::")[-1] if "::" in src else src
        tgt_label = tgt.split("::")[-1] if "::" in tgt else tgt
        relevant_edges.append(f"  {src_label} →[{edge['type']}]→ {tgt_label}")
        if len(relevant_edges) >= 15:
            break

    # Render
    node_lines: list[str] = []
    for node in top_nodes:
        nid = node.get("id", "?")
        file_ = node.get("file", "?")
        pr = node.get("pagerank", 0.0)
        # Include docstring snippet from enriched if present
        doc = ""
        enriched_node = enriched.get(nid, {})
        if enriched_node:
            raw_doc = enriched_node.get("docstring") or ""
            doc = f" — {raw_doc[:60]}" if raw_doc else ""
        node_lines.append(f"  {nid} (file: {file_}, rank: {pr:.3f}){doc}")

    parts: list[str] = []
    if node_lines:
        parts.append("TOP MATCHED NODES:\n" + "\n".join(node_lines))
    if relevant_edges:
        parts.append("RELEVANT EDGES:\n" + "\n".join(relevant_edges))

    summary = "\n\n".join(parts)
    # Hard-cap at 1200 chars to keep prompt tight
    return summary[:1200] if summary else "(no graph data available)"


def _summarise_business_rules(
    rules: list[dict],
    entities: ExtractedContext,
) -> str:
    """Return a compact string of relevant business rules (<= 800 chars)."""
    if not rules:
        return "(no business rules available)"

    fn_hints = {f.lower() for f in entities.function_names}
    mod_hints = {m.lower() for m in entities.module_hints}

    relevant: list[dict] = []
    for rule in rules:
        rule_file = (rule.get("file") or "").lower()
        rule_func = (rule.get("function_id") or "").lower()
        if any(h in rule_file or h in rule_func for h in fn_hints | mod_hints):
            relevant.append(rule)

    if not relevant:
        # Return the top-severity rules as a fallback
        relevant = sorted(rules, key=lambda r: r.get("severity", "low"), reverse=True)[:5]

    lines: list[str] = []
    for r in relevant[:6]:
        sev = r.get("severity", "?").upper()
        desc = r.get("description", "")[:120]
        src_file = r.get("file", "")
        lines.append(f"  [{sev}] {desc} (file: {src_file})")

    result = "\n".join(lines)
    return result[:800] if result else "(no relevant rules found)"


def _extract_failure_records(
    graph_data: dict,
    entities: ExtractedContext,
) -> list[str]:
    """
    Pull FailureRecord-style notes from graph nodes (type == 'failure_record').
    These are optionally embedded in graph.json by the builder pipeline.
    """
    records: list[str] = []
    fn_hints = {f.lower() for f in entities.function_names}
    mod_hints = {m.lower() for m in entities.module_hints}

    for node in graph_data.get("nodes", []):
        if node.get("type") != "failure_record":
            continue
        nid = node.get("id", "").lower()
        content = node.get("content") or node.get("label") or node.get("message") or ""
        file_ = (node.get("file") or "").lower()
        if any(h in nid or h in file_ for h in fn_hints | mod_hints):
            records.append(content[:200])

    return records[:5]


# ---------------------------------------------------------------------------
# Agent 1 — Context Extractor (Haiku)
# ---------------------------------------------------------------------------


def _run_extractor(work_order: dict, intent: dict) -> ExtractedContext:
    """
    Agent 1: Extract key entities from the bug description.

    Input:  work_order dict + intent dict (from intake_node)
    Output: ExtractedContext Pydantic model
    """
    from agent.llm import structured_call

    title = work_order.get("title", "")
    description = work_order.get("description", "")[:600]
    component = work_order.get("affected_component") or ""
    actual = intent.get("actual_behavior", "")[:200]
    expected = intent.get("expected_behavior", "")[:200]
    hint_fns = ", ".join(intent.get("likely_affected_functions", [])[:5])
    hint_mods = ", ".join(intent.get("likely_affected_modules", [])[:5])

    prompt = f"""You are a code analyst. Extract structured fault-localisation entities from this bug report.

BUG TITLE: {title}
COMPONENT: {component}
DESCRIPTION: {description}
ACTUAL BEHAVIOUR: {actual}
EXPECTED BEHAVIOUR: {expected}
HINT FUNCTIONS: {hint_fns or '(none)'}
HINT MODULES: {hint_mods or '(none)'}

Return:
- function_names: function/method names mentioned or strongly implied
- error_types: exception classes or error codes referenced
- data_structures: key variables, models, or data types involved
- module_hints: module/file names that are likely involved
- bug_summary: one sentence describing what is broken

Keep each list short (≤ 6 items). Only include what you are confident about."""

    return structured_call(_EXTRACTOR_MODEL, 512, ExtractedContext, prompt)


# ---------------------------------------------------------------------------
# Agent 2 — Graph-RAG Debugger (Sonnet)
# ---------------------------------------------------------------------------


def _run_debugger(
    extracted: ExtractedContext,
    graph_summary: str,
    rules_summary: str,
    work_order: dict,
) -> GraphDebuggerOutput:
    """
    Agent 2: Traverse graph context to identify top-5 suspicious locations.

    Input:  ExtractedContext + compact graph/rules summaries
    Output: GraphDebuggerOutput Pydantic model
    """
    from agent.llm import structured_call

    bug_summary = extracted.bug_summary or work_order.get("title", "")
    fn_list = ", ".join(extracted.function_names[:6]) or "(none)"
    err_list = ", ".join(extracted.error_types[:4]) or "(none)"
    mod_list = ", ".join(extracted.module_hints[:5]) or "(none)"

    prompt = f"""You are a fault localisation debugger. Given extracted entities and graph context, identify the top-5 most suspicious code locations.

BUG: {bug_summary}
KEY FUNCTIONS: {fn_list}
ERROR TYPES: {err_list}
MODULES: {mod_list}

GRAPH CONTEXT:
{graph_summary}

BUSINESS RULES:
{rules_summary}

Instructions:
1. Identify up to 5 file+function pairs most likely to contain the bug.
2. For each suspect: set file (relative path), function name, confidence (0.0–1.0), and a one-sentence reason.
3. List blast_radius_files: other files whose behaviour would change if you fix the suspect locations.
4. List relevant_business_rule_ids: IDs or short descriptions of rules that constrain this area.

Rank suspects by confidence descending. Be specific — prefer exact function names over generic files."""

    return structured_call(_DEBUGGER_MODEL, 800, GraphDebuggerOutput, prompt)


# ---------------------------------------------------------------------------
# Agent 3 — Verbal RL Re-ranker (Opus)
# ---------------------------------------------------------------------------


def _run_reranker(
    suspects: GraphDebuggerOutput,
    extracted: ExtractedContext,
    rules: list[dict],
    failure_records: list[str],
    graph_data: dict,
    work_order: dict,
) -> RerankerOutput:
    """
    Agent 3: Re-rank the top-5 suspects using blast radius, business rules, past failures,
    and PageRank hotspot data.

    Input:  GraphDebuggerOutput + enriched context
    Output: RerankerOutput Pydantic model
    """
    from agent.llm import structured_call

    # Build compact suspect list
    suspect_lines: list[str] = []
    for i, s in enumerate(suspects.suspects, 1):
        suspect_lines.append(
            f"  {i}. {s.file}::{s.function or '?'} (confidence={s.confidence:.2f}) — {s.reason}"
        )
    suspects_text = "\n".join(suspect_lines) or "  (none)"

    # Build compact failure records string
    failure_text = (
        "\n".join(f"  - {r}" for r in failure_records[:5])
        if failure_records
        else "  (none)"
    )

    # Build hotspot signal from graph PageRank
    hotspot_lines: list[str] = []
    for node in sorted(
        graph_data.get("nodes", []),
        key=lambda n: n.get("pagerank", 0.0),
        reverse=True,
    )[:6]:
        hotspot_lines.append(
            f"  {node.get('id', '?')} (rank: {node.get('pagerank', 0.0):.3f})"
        )
    hotspot_text = "\n".join(hotspot_lines) or "  (none)"

    # Relevant rules — top severity first, capped at 5
    relevant_rules: list[dict] = sorted(
        rules, key=lambda r: r.get("severity", "low"), reverse=True
    )[:5]
    rule_lines = [
        f"  [{r.get('severity','?').upper()}] {r.get('description','')[:100]}"
        for r in relevant_rules
    ]
    rule_text = "\n".join(rule_lines) or "  (none)"

    blast_files = ", ".join(suspects.blast_radius_files[:6]) or "(none)"
    bug_summary = extracted.bug_summary or work_order.get("title", "")

    prompt = f"""You are a senior engineer re-ranking fault localisation candidates.

BUG: {bug_summary}

CANDIDATE LOCATIONS (from graph analysis):
{suspects_text}

BUSINESS RULES (high-severity):
{rule_text}

PAST FAILURES in this codebase:
{failure_text}

CODE HOTSPOTS (most-referenced functions by PageRank):
{hotspot_text}

INITIAL BLAST RADIUS: {blast_files}

Re-rank the candidates using this reasoning process:
1. Does any candidate directly match a past failure location? If yes, boost confidence.
2. Is any candidate a hotspot? High PageRank means widespread impact — handle with care.
3. Do business rules constrain any candidate? Critical/High rules raise the stakes.
4. Adjust confidence based on blast radius — a suspect touching many callers is high-risk.
5. Assign a final confidence (0.0–1.0) and a full reasoning note for each.

Also list:
- relevant_failure_records: exact failure messages or refs that apply here
- additional_blast_radius: any extra files you'd add to the blast radius

Return all candidates ranked highest-confidence first."""

    return structured_call(_RERANKER_MODEL, 1024, RerankerOutput, prompt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scout_localize(
    repo_name: str,
    work_order: dict,
    intent: dict,
    data_dir: Path,
    community_name: str | None = None,
) -> dict:
    """
    Run the 3-agent FL pipeline to produce a Localisation Report.

    Agents run sequentially; each failure falls back gracefully and passes
    partial results to the next stage rather than aborting.

    Returns:
        {
            "top_locations": [{"file": str, "function": str, "confidence": float, "reason": str}],
            "blast_radius_files": [str],
            "relevant_business_rules": [str],
            "relevant_failure_records": [str],
            "scout_cost_usd": float,
        }
    """
    from agent.graph_utils import load_graph_data
    from agent.llm import estimate_cost

    logger.info(
        "scout: starting FL pipeline — repo=%s ticket=%s",
        repo_name,
        work_order.get("ticket_id", "?"),
    )

    # -----------------------------------------------------------------------
    # Load graph data once (shared by all agents)
    # -----------------------------------------------------------------------
    graph_data: dict = {}
    enriched: dict = {}
    try:
        graph_data, enriched = load_graph_data(repo_name)
    except Exception as exc:
        logger.warning("scout: failed to load graph data: %s", exc)

    business_rules = _load_business_rules(data_dir, repo_name)

    # Accumulate cost (token-based estimate; we don't have exact counts from structured_call)
    total_cost = 0.0

    # -----------------------------------------------------------------------
    # Agent 1 — Context Extractor (Haiku)
    # -----------------------------------------------------------------------
    extracted: ExtractedContext = ExtractedContext(
        function_names=intent.get("likely_affected_functions", [])[:6],
        module_hints=intent.get("likely_affected_modules", [])[:6],
        bug_summary=intent.get("actual_behavior", work_order.get("title", "")),
    )
    try:
        extracted = _run_extractor(work_order, intent)
        # Approximate cost: ~400 input tokens, ~150 output tokens for Haiku
        total_cost += estimate_cost(_EXTRACTOR_MODEL, 400, 150)
        logger.info(
            "scout[1/3]: extracted %d functions, %d modules",
            len(extracted.function_names),
            len(extracted.module_hints),
        )
    except Exception as exc:
        logger.error("scout[1/3]: Context Extractor failed (using fallback): %s", exc)
        # extracted already holds the intent-seeded fallback above

    # -----------------------------------------------------------------------
    # Agent 2 — Graph-RAG Debugger (Sonnet)
    # -----------------------------------------------------------------------
    graph_summary = _summarise_graph(graph_data, enriched, extracted)
    rules_summary = _summarise_business_rules(business_rules, extracted)

    debugger_output: GraphDebuggerOutput = GraphDebuggerOutput()
    try:
        debugger_output = _run_debugger(extracted, graph_summary, rules_summary, work_order)
        # Approximate cost: ~900 input tokens, ~300 output tokens for Sonnet
        total_cost += estimate_cost(_DEBUGGER_MODEL, 900, 300)
        logger.info(
            "scout[2/3]: Graph-RAG Debugger found %d suspects",
            len(debugger_output.suspects),
        )
    except Exception as exc:
        logger.error("scout[2/3]: Graph-RAG Debugger failed (using fallback): %s", exc)
        # Build a minimal fallback from extracted entities + graph nodes
        fallback_suspects: list[SuspectLocation] = []
        file_set: set[str] = set()
        for fn in extracted.function_names[:3]:
            for node in graph_data.get("nodes", []):
                node_label = (node.get("label") or node.get("id", "").split("::")[-1]).lower()
                if fn.lower() in node_label and node.get("file"):
                    f = node["file"]
                    if f not in file_set:
                        file_set.add(f)
                        fallback_suspects.append(
                            SuspectLocation(
                                file=f,
                                function=fn,
                                confidence=0.4,
                                reason=f"Graph node match for function '{fn}'",
                            )
                        )
        for mod in extracted.module_hints[:2]:
            for node in graph_data.get("nodes", []):
                if node.get("type") == "file" and mod.lower() in (node.get("file") or "").lower():
                    f = node["file"]
                    if f not in file_set:
                        file_set.add(f)
                        fallback_suspects.append(
                            SuspectLocation(
                                file=f,
                                function="",
                                confidence=0.3,
                                reason=f"Module hint match for '{mod}'",
                            )
                        )
        debugger_output = GraphDebuggerOutput(suspects=fallback_suspects[:5])

    # -----------------------------------------------------------------------
    # Agent 3 — Verbal RL Re-ranker (Opus)
    # -----------------------------------------------------------------------
    failure_records = _extract_failure_records(graph_data, extracted)

    reranker_output: RerankerOutput = RerankerOutput()
    if debugger_output.suspects:
        try:
            reranker_output = _run_reranker(
                debugger_output,
                extracted,
                business_rules,
                failure_records,
                graph_data,
                work_order,
            )
            # Approximate cost: ~1000 input tokens, ~400 output tokens for Opus
            total_cost += estimate_cost(_RERANKER_MODEL, 1000, 400)
            logger.info(
                "scout[3/3]: Re-ranker produced %d ranked locations",
                len(reranker_output.ranked_locations),
            )
        except Exception as exc:
            logger.error("scout[3/3]: Re-ranker failed (using debugger output as-is): %s", exc)
            # Demote to ranked from debugger output unchanged
            reranker_output = RerankerOutput(
                ranked_locations=[
                    RankedLocation(
                        file=s.file,
                        function=s.function,
                        confidence=s.confidence,
                        reason=s.reason,
                    )
                    for s in sorted(
                        debugger_output.suspects,
                        key=lambda s: s.confidence,
                        reverse=True,
                    )
                ],
                relevant_failure_records=failure_records,
                additional_blast_radius=[],
            )
    else:
        logger.warning("scout: no suspects from Agent 2 — skipping re-ranker")
        reranker_output = RerankerOutput(
            relevant_failure_records=failure_records,
        )

    # -----------------------------------------------------------------------
    # Assemble final report
    # -----------------------------------------------------------------------
    top_locations: list[dict[str, Any]] = [
        {
            "file": loc.file,
            "function": loc.function,
            "confidence": round(loc.confidence, 3),
            "reason": loc.reason,
        }
        for loc in reranker_output.ranked_locations[:5]
    ]

    # Merge blast radius from Debugger + Re-ranker, deduplicate, preserve order
    blast_set: dict[str, None] = {}  # ordered set via dict keys
    for f in debugger_output.blast_radius_files:
        blast_set[f] = None
    for f in reranker_output.additional_blast_radius:
        blast_set[f] = None
    blast_radius_files = list(blast_set.keys())[:10]

    # Relevant business rules: merge from both agent outputs
    rule_set: dict[str, None] = {}
    for r in debugger_output.relevant_business_rule_ids:
        rule_set[r] = None
    # Also add any high-severity rules touching the top location files
    top_files = {loc["file"] for loc in top_locations}
    for rule in business_rules:
        if rule.get("severity", "").lower() in ("critical", "high"):
            rule_file = rule.get("file", "")
            if any(tf in rule_file or rule_file in tf for tf in top_files):
                desc = rule.get("description", "")[:120]
                if desc:
                    rule_set[desc] = None
    relevant_business_rules = list(rule_set.keys())[:8]

    # Failure records
    relevant_failure_records = (reranker_output.relevant_failure_records or failure_records)[:5]

    report: dict[str, Any] = {
        "top_locations": top_locations,
        "blast_radius_files": blast_radius_files,
        "relevant_business_rules": relevant_business_rules,
        "relevant_failure_records": relevant_failure_records,
        "scout_cost_usd": round(total_cost, 6),
    }

    logger.info(
        "scout: done — top_locations=%d blast_radius=%d cost=$%.4f",
        len(top_locations),
        len(blast_radius_files),
        total_cost,
    )
    return report
