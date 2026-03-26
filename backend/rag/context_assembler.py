"""
context_assembler.py — Build targeted context from retrieved nodes for Claude.

Assembles a ~10-20K token context document from the relevant subgraph,
instead of dumping the entire 308K token context.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return len(text) // 4


class ContextAssembler:
    """Build targeted context from retrieval results."""

    def __init__(self, repo_name: str, data_dir: Path) -> None:
        self.repo_name = repo_name
        self.data_dir = Path(data_dir)
        self._enriched: dict[str, dict] | None = None

    def assemble(
        self,
        primary_ids: list[str],
        expanded_ids: list[str],
        edges: list[dict],
        scores: dict[str, float] | None = None,
        token_budget: int = 15000,
    ) -> str:
        """
        Assemble targeted context from retrieved nodes.

        Priority order:
        1. Primary matches — full detail (name, docstring, params, etc.)
        2. Expanded neighbors — name + summary only
        3. Edge relationships — compact table
        4. Business rules + decision points for matched files
        """
        enriched = self._load_enriched()
        if not enriched:
            return "No enriched node data available. Run analysis first."

        sections: list[str] = []
        used_tokens = 0

        # Header
        header = (
            f"# Context for: {self.repo_name}\n"
            f"> {len(primary_ids)} primary matches + {len(expanded_ids)} related nodes via Graph RAG\n"
        )
        sections.append(header)
        used_tokens += _estimate_tokens(header)

        # Section 1: Primary matches (full detail)
        primary_section = self._render_primary(primary_ids, enriched, scores or {})
        primary_tokens = _estimate_tokens(primary_section)
        if used_tokens + primary_tokens < token_budget:
            sections.append(primary_section)
            used_tokens += primary_tokens
        else:
            # Primary alone exceeds budget: truncate it to fit rather than
            # skipping it entirely, which would produce context with no primary
            # matches at all.
            # Clamp to 0 so that a negative budget yields an empty string
            # instead of a tail-truncated substring (Python negative slice bug).
            chars_left = max(0, (token_budget - used_tokens) * 4)
            truncated = primary_section[:chars_left]
            sections.append(truncated)
            used_tokens += _estimate_tokens(truncated)

        # Section 2: Related code (compact)
        if used_tokens < token_budget * 0.7:
            related_section = self._render_related(expanded_ids, enriched, primary_ids)
            related_tokens = _estimate_tokens(related_section)
            budget_left = token_budget - used_tokens
            if related_tokens > budget_left:
                # Truncate to stay within budget (budget_left * 4 converts tokens → chars).
                # Clamp to 0 to avoid negative slice indices.
                related_section = related_section[:max(0, budget_left * 4)]
            if related_section:
                sections.append(related_section)
                used_tokens += _estimate_tokens(related_section)

        # Section 3: Call relationships
        if edges and used_tokens < token_budget * 0.85:
            edge_section = self._render_edges(edges)
            sections.append(edge_section)
            used_tokens += _estimate_tokens(edge_section)

        # Section 4: Business rules + decision points for matched files
        if used_tokens < token_budget * 0.95:
            matched_files = set()
            for nid in primary_ids + expanded_ids:
                node = enriched.get(nid, {})
                if node.get("file"):
                    matched_files.add(node["file"])
            biz_section = self._render_business_context(matched_files, enriched)
            sections.append(biz_section)

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_primary(self, node_ids: list[str], enriched: dict, scores: dict) -> str:
        """Render primary matches with full detail."""
        lines = ["## Primary Matches\n"]

        for nid in node_ids:
            node = enriched.get(nid)
            if not node:
                continue

            ntype = node.get("type", "")
            if ntype == "file":
                lines.append(self._render_file_detail(node))
            elif ntype == "function":
                lines.append(self._render_function_detail(node))
            elif ntype == "class":
                lines.append(self._render_class_detail(node))
            elif ntype == "domain_concept":
                name = node.get("name", "")
                desc = node.get("description") or ""
                classes = ", ".join(node.get("related_classes", [])[:5])
                lines.append(f"**Domain: {name}** — {desc}")
                if classes:
                    lines.append(f"  Related: {classes}")
                lines.append("")

        return "\n".join(lines)

    def _render_file_detail(self, node: dict) -> str:
        doc = node.get("docstring", "")
        classes = ", ".join(node.get("classes", [])[:8])
        funcs = ", ".join(node.get("functions", [])[:8])
        imports = ", ".join(node.get("imports", [])[:8])

        parts = [f"### `{node.get('file', '')}`"]
        if doc:
            parts.append(f"**Purpose:** {doc[:300]}")
        if imports:
            parts.append(f"**Imports:** {imports}")
        if classes:
            parts.append(f"**Classes:** {classes}")
        if funcs:
            parts.append(f"**Functions:** {funcs}")
        parts.append("")
        return "\n".join(parts)

    def _render_function_detail(self, node: dict) -> str:
        params = ", ".join(node.get("params", []))
        ret = f" -> {node['return_type']}" if node.get("return_type") else ""
        doc = node.get("docstring", "")

        parts = [f"### `{node.get('file', '')}::{node.get('name', '')}({params}){ret}`"]
        if doc:
            parts.append(f"**Purpose:** {doc[:300]}")
        if node.get("complexity", 1) > 5:
            parts.append(f"**Complexity:** {node['complexity']}")
        parts.append("")
        return "\n".join(parts)

    def _render_class_detail(self, node: dict) -> str:
        bases = ", ".join(node.get("bases", []))
        methods = ", ".join(node.get("methods", [])[:10])
        doc = node.get("docstring", "")

        parts = [f"### `{node.get('file', '')}::{node.get('name', '')}`"]
        if bases:
            parts.append(f"**Inherits:** {bases}")
        if doc:
            parts.append(f"**Purpose:** {doc[:300]}")
        if methods:
            parts.append(f"**Methods:** {methods}")
        parts.append("")
        return "\n".join(parts)

    def _render_related(self, node_ids: list[str], enriched: dict, exclude: list[str]) -> str:
        """Render expanded neighbors as compact list."""
        exclude_set = set(exclude)
        lines = ["## Related Code\n"]

        for nid in node_ids:
            if nid in exclude_set:
                continue
            node = enriched.get(nid)
            if not node:
                continue

            ntype = node.get("type", "")
            name = node.get("name", "")
            doc = (node.get("docstring") or "")[:80]
            file_path = node.get("file", "")

            if ntype == "function":
                params = ", ".join(node.get("params", []))
                ret = f" -> {node['return_type']}" if node.get("return_type") else ""
                desc = f" — {doc}" if doc else ""
                lines.append(f"- `{name}({params}){ret}` ({file_path}){desc}")
            elif ntype == "class":
                desc = f" — {doc}" if doc else ""
                lines.append(f"- class `{name}` ({file_path}){desc}")
            elif ntype == "file":
                desc = f" — {doc}" if doc else ""
                lines.append(f"- `{file_path}`{desc}")

        if len(lines) == 1:
            return ""
        lines.append("")
        return "\n".join(lines)

    def _render_edges(self, edges: list[dict]) -> str:
        """Render edges as a compact relationship table."""
        if not edges:
            return ""

        # Group by type
        by_type: dict[str, list] = {}
        for e in edges:
            etype = e.get("type", "RELATED")
            by_type.setdefault(etype, []).append(e)

        lines = ["## Relationships\n"]
        for etype in ("CALLS", "IMPORTS", "INHERITS", "CONTAINS"):
            items = by_type.get(etype, [])
            if not items:
                continue
            lines.append(f"**{etype}:**")
            for e in items[:15]:
                src = e.get("source", "").split("::")[-1]
                tgt = e.get("target", "").split("::")[-1]
                lines.append(f"- `{src}` → `{tgt}`")
            if len(items) > 15:
                lines.append(f"- ... +{len(items) - 15} more")
            lines.append("")

        return "\n".join(lines)

    def _render_business_context(self, file_ids: set[str], enriched: dict) -> str:
        """Render business rules and decision points for matched files."""
        lines = []
        rules = []
        decisions = []

        for nid, node in enriched.items():
            if node.get("type") == "business_rule" and node.get("file") in file_ids:
                rules.append(node)
            if node.get("type") == "decision_point" and node.get("file") in file_ids:
                decisions.append(node)

        if rules:
            lines.append("## Business Rules\n")
            for r in rules[:10]:
                lines.append(f"- [{r.get('rule_type', '')}] {r.get('content', r.get('name', ''))}")
            lines.append("")

        if decisions:
            lines.append("## Decision Points\n")
            for dp in decisions[:10]:
                func = (dp.get("function_id") or "").split("::")[-1]
                lines.append(f"- `{func}`: `{dp.get('name', '')}` [{dp.get('condition_type', '')}]")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_enriched(self) -> dict[str, dict]:
        if self._enriched is not None:
            return self._enriched
        path = self.data_dir / self.repo_name / "enriched_nodes.json"
        if path.exists():
            self._enriched = json.loads(path.read_text())
        else:
            self._enriched = {}
        return self._enriched
