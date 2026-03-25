"""Summarizer: enriches File and Function nodes in Neo4j with AI-generated business summaries."""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

_MODEL = os.environ.get("SUMMARIZER_MODEL", "claude-haiku-4-5-20251001")
_MAX_TOKENS = 400

_PROMPT_TEMPLATE = """\
You are a senior software architect analyzing a Python module for a codebase knowledge graph.

Module path: {path}

Module docstring:
{docstring}

Classes defined: {classes}

Functions/methods defined: {functions}

Write a concise but complete business-purpose summary (3-5 sentences) covering:
1. WHY this module exists and what business problem it solves
2. What key operations it performs (without implementation detail)
3. What other parts of the system depend on or interact with it
4. Any important constraints, rules, or side effects a developer must know

Do NOT describe language constructs (classes, functions). Focus purely on the BUSINESS PURPOSE and DOMAIN BEHAVIOR.
Respond with ONLY the summary text, no preamble or labels."""


_FUNC_PROMPT_TEMPLATE = """\
You are a senior software architect. For each function listed below (from the same file),
write ONE sentence describing its BUSINESS purpose (not implementation details).

File: {path}

Functions:
{function_list}

Respond in EXACTLY this format (one line per function, no extra text):
function_name: One sentence business purpose.
"""


class Summarizer:
    """Generate and persist AI business-purpose summaries for File and Function nodes.

    Parameters
    ----------
    repo_name:
        The name of the repository as stored in the ``Repo`` node.
    """

    def __init__(self, repo_name: str) -> None:
        self.repo_name = repo_name
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before running the Summarizer."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self) -> int:
        """Generate and store summaries for all un-summarized File nodes.

        Idempotent: files that already have a ``summary`` property are skipped.

        Returns
        -------
        int
            Number of files newly summarized in this run.
        """
        files = self._fetch_unsummarized_files()

        if not files:
            logger.info(
                "All files in '%s' already have summaries — nothing to do.",
                self.repo_name,
            )
            return 0

        logger.info(
            "Summarizing %d file(s) in repo '%s'.",
            len(files),
            self.repo_name,
        )

        summarized = 0
        for file_record in files:
            file_id: str = file_record["id"]
            try:
                summary = self._generate_summary(file_record)
                self._persist_summary(file_id, summary)
                summarized += 1
                logger.debug("Summarized '%s'.", file_record.get("path", file_id))
            except anthropic.APIError as exc:
                logger.error(
                    "Anthropic API error while summarizing '%s': %s",
                    file_record.get("path", file_id),
                    exc,
                )
            except Exception as exc:
                logger.error(
                    "Unexpected error while summarizing '%s': %s",
                    file_record.get("path", file_id),
                    exc,
                )

        logger.info(
            "Summarization complete for repo '%s': %d/%d files enriched.",
            self.repo_name,
            summarized,
            len(files),
        )
        return summarized

    def enrich_functions(self) -> int:
        """Generate and store summaries for all un-summarized Function nodes.

        Groups functions by parent file and sends one LLM call per file
        to minimize API usage. Idempotent.

        Returns the number of functions newly summarized.
        """
        file_groups = self._fetch_unsummarized_functions()

        if not file_groups:
            logger.info("All functions in '%s' already have summaries.", self.repo_name)
            return 0

        total_funcs = sum(len(fns) for fns in file_groups.values())
        logger.info("Summarizing %d function(s) across %d file(s) in '%s'.",
                     total_funcs, len(file_groups), self.repo_name)

        summarized = 0
        for file_path, functions in file_groups.items():
            try:
                summaries = self._generate_function_summaries(file_path, functions)
                for fn_id, summary in summaries.items():
                    self._persist_function_summary(fn_id, summary)
                    summarized += 1
            except anthropic.APIError as exc:
                logger.error("API error summarizing functions in '%s': %s", file_path, exc)
            except Exception as exc:
                logger.error("Error summarizing functions in '%s': %s", file_path, exc)

        logger.info("Function summarization complete: %d/%d enriched.", summarized, total_funcs)
        return summarized

    # ------------------------------------------------------------------
    # Neo4j helpers
    # ------------------------------------------------------------------

    def _fetch_unsummarized_functions(self) -> dict[str, list[dict[str, Any]]]:
        """Return Function nodes grouped by file that don't have summaries yet."""
        rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(fn:Function) "
            "WHERE fn.summary IS NULL AND fn.name IS NOT NULL "
            "RETURN fn.id AS id, fn.name AS name, fn.file AS file, "
            "       fn.params AS params, fn.return_type AS return_type, "
            "       fn.docstring AS docstring "
            "ORDER BY fn.file, fn.name "
            "LIMIT 500",
            {"repo": self.repo_name},
        )

        # Group by file
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            file_path = row.get("file", "unknown")
            groups.setdefault(file_path, []).append(row)

        return groups

    def _generate_function_summaries(
        self, file_path: str, functions: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Call Claude once per file to summarize all its functions."""
        func_lines = []
        for fn in functions:
            params = ", ".join(fn.get("params") or [])
            ret = f" -> {fn['return_type']}" if fn.get("return_type") else ""
            doc = (fn.get("docstring") or "")[:80]
            doc_str = f"  # {doc}" if doc else ""
            func_lines.append(f"- {fn['name']}({params}){ret}{doc_str}")

        prompt = _FUNC_PROMPT_TEMPLATE.format(
            path=file_path,
            function_list="\n".join(func_lines),
        )

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS * 2,  # more room for multiple functions
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text = block.text.strip()
                break

        # Parse "function_name: description" lines
        results: dict[str, str] = {}
        fn_id_map = {fn["name"]: fn["id"] for fn in functions}

        for line in text.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            name, _, desc = line.partition(":")
            name = name.strip().rstrip("()")  # handle "func_name():" format
            desc = desc.strip()
            if name in fn_id_map and desc:
                results[fn_id_map[name]] = desc[:300]

        return results

    def _persist_function_summary(self, fn_id: str, summary: str) -> None:
        """Write summary onto a Function node."""
        neo4j_client.run(
            "MATCH (fn:Function {id: $fid}) SET fn.summary = $summary",
            {"fid": fn_id, "summary": summary},
        )

    def _fetch_unsummarized_files(self) -> list[dict[str, Any]]:
        """Return File nodes that do not yet have a summary property."""
        rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File) "
            "WHERE f.summary IS NULL "
            "RETURN f.id AS id, f.path AS path, f.docstring AS docstring "
            "ORDER BY f.path",
            {"repo": self.repo_name},
        )

        # Enrich with classes and functions in a second pass
        enriched: list[dict[str, Any]] = []
        for row in rows:
            file_id = row["id"]

            classes = neo4j_client.run(
                "MATCH (f:File {id: $fid})-[:CONTAINS]->(c:Class) "
                "RETURN c.name AS name",
                {"fid": file_id},
            )
            functions = neo4j_client.run(
                "MATCH (f:File {id: $fid})-[:CONTAINS]->(fn:Function) "
                "WHERE NOT (:Class)-[:CONTAINS]->(fn) "
                "RETURN fn.name AS name",
                {"fid": file_id},
            )

            enriched.append(
                {
                    "id": file_id,
                    "path": row.get("path", ""),
                    "docstring": row.get("docstring") or "",
                    "classes": [r["name"] for r in classes if r.get("name")],
                    "functions": [r["name"] for r in functions if r.get("name")],
                }
            )

        return enriched

    def _persist_summary(self, file_id: str, summary: str) -> None:
        """Write the summary onto the File node and update the repo summary_count."""
        neo4j_client.run(
            "MATCH (f:File {id: $fid}) "
            "SET f.summary = $summary",
            {"fid": file_id, "summary": summary},
        )

        # Increment summary_count on the parent Repo node
        neo4j_client.run(
            "MATCH (r:Repo {name: $repo}) "
            "SET r.summary_count = coalesce(r.summary_count, 0) + 1",
            {"repo": self.repo_name},
        )

    # ------------------------------------------------------------------
    # AI helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, file_record: dict[str, Any]) -> str:
        path = file_record.get("path", "unknown")
        docstring = file_record.get("docstring") or "No module docstring."
        classes: list[str] = file_record.get("classes") or []
        functions: list[str] = file_record.get("functions") or []

        classes_str = ", ".join(classes) if classes else "none"
        functions_str = ", ".join(functions) if functions else "none"

        return _PROMPT_TEMPLATE.format(
            path=path,
            docstring=docstring.strip(),
            classes=classes_str,
            functions=functions_str,
        )

    def _generate_summary(self, file_record: dict[str, Any]) -> str:
        """Call the Claude API and return the generated summary text."""
        prompt = self._build_prompt(file_record)

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # Extract text from the first content block
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()

        raise ValueError(
            f"Claude returned no text content for file '{file_record.get('path', '?')}'"
        )
