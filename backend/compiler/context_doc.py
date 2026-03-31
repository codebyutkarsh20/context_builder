"""ContextCompiler: assembles context.md and summary.md from the Neo4j graph."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Badge helpers
# ---------------------------------------------------------------------------

_TECH_BADGE_MAP: dict[str, str] = {
    "python": "![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)",
    "javascript": "![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)",
    "typescript": "![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)",
    "java": "![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=openjdk&logoColor=white)",
    "go": "![Go](https://img.shields.io/badge/Go-00ADD8?style=flat&logo=go&logoColor=white)",
    "rust": "![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)",
    "c": "![C](https://img.shields.io/badge/C-A8B9CC?style=flat&logo=c&logoColor=black)",
    "cpp": "![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white)",
    "ruby": "![Ruby](https://img.shields.io/badge/Ruby-CC342D?style=flat&logo=ruby&logoColor=white)",
    "php": "![PHP](https://img.shields.io/badge/PHP-777BB4?style=flat&logo=php&logoColor=white)",
    "swift": "![Swift](https://img.shields.io/badge/Swift-FA7343?style=flat&logo=swift&logoColor=white)",
    "kotlin": "![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=flat&logo=kotlin&logoColor=white)",
    "scala": "![Scala](https://img.shields.io/badge/Scala-DC322F?style=flat&logo=scala&logoColor=white)",
    "haskell": "![Haskell](https://img.shields.io/badge/Haskell-5D4F85?style=flat&logo=haskell&logoColor=white)",
    "docker": "![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)",
    "neo4j": "![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)",
    "fastapi": "![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)",
    "django": "![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white)",
    "flask": "![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)",
    "react": "![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)",
    "vue": "![Vue.js](https://img.shields.io/badge/Vue.js-4FC08D?style=flat&logo=vuedotjs&logoColor=white)",
    "angular": "![Angular](https://img.shields.io/badge/Angular-DD0031?style=flat&logo=angular&logoColor=white)",
}


def _tech_badge(tech: str) -> str:
    """Return a Markdown badge for *tech*, falling back to a plain label."""
    key = tech.lower().replace(" ", "").replace(".", "").replace("+", "p")
    return _TECH_BADGE_MAP.get(key, f"![{tech}](https://img.shields.io/badge/{tech.replace(' ', '_')}-grey?style=flat)")


# ---------------------------------------------------------------------------
# Tree builder (lightweight — no filesystem access required)
# ---------------------------------------------------------------------------


def _build_tree(paths: list[str]) -> str:
    """Convert a flat list of file paths into an ASCII tree string."""
    if not paths:
        return "_No files indexed._"

    tree: dict = {}
    for p in sorted(paths):
        parts = Path(p).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def _render(node: dict, prefix: str = "") -> None:
        children = list(node.items())
        for i, (name, subtree) in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.append(f"{prefix}{connector}{name}")
            extension = "    " if i == len(children) - 1 else "│   "
            if subtree:
                _render(subtree, prefix + extension)

    _render(tree)
    return "```\n" + "\n".join(lines) + "\n```"


# ---------------------------------------------------------------------------
# Docstring helpers
# ---------------------------------------------------------------------------


def _first_line(text: str | None) -> str:
    """Return the first non-empty line of *text*, or '—'."""
    if not text:
        return "—"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "—"


def _first_n_lines(text: str | None, n: int = 3) -> str:
    """Return the first *n* non-empty lines of *text* joined by a space."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines[:n])


# ---------------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------------


class ContextCompiler:
    """Compile a full context document from a repository's Neo4j graph.

    Parameters
    ----------
    repo_name:
        The name of the repository as stored in the ``Repo`` node.
    """

    def __init__(self, repo_name: str, repo_path: str | Path | None = None) -> None:
        self.repo_name = repo_name
        self.repo_path = Path(repo_path) if repo_path else None
        self._out_dir = Path(f"/tmp/context_builder/{repo_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self) -> tuple[Path, Path]:
        """Build context.md and summary.md for the repository.

        Returns
        -------
        (context_path, summary_path)
            Absolute paths of the two generated files.
        """
        os.makedirs(self._out_dir, exist_ok=True)

        # ---- Fetch all data up-front ----------------------------------------
        repo_info = self._fetch_repo_info()
        files = self._fetch_files()
        classes = self._fetch_classes()
        functions = self._fetch_functions()
        hotspots = self._fetch_hotspots(top_n=20)
        business_summaries = self._fetch_business_summaries()
        business_rules = self._fetch_business_rules()
        call_edges = self._fetch_call_edges(limit=40)
        import_edges = self._fetch_import_edges(limit=30)
        readme = self._fetch_repo_readme()
        decision_points = self._fetch_decision_points()
        domain_concepts = self._fetch_domain_concepts()
        # Batch-fetch imports/exports for all files (replaces N+1 per-file queries)
        all_ie = self._fetch_all_imports_exports()
        # Extract project conventions (linters, CI, coding standards)
        project_conventions = self._extract_project_conventions()

        # ---- Render sections -------------------------------------------------
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        context_md = self._render_context(
            repo_info=repo_info,
            files=files,
            classes=classes,
            functions=functions,
            hotspots=hotspots,
            business_summaries=business_summaries,
            business_rules=business_rules,
            call_edges=call_edges,
            import_edges=import_edges,
            decision_points=decision_points,
            domain_concepts=domain_concepts,
            readme=readme,
            generated_at=generated_at,
            all_ie=all_ie,
            project_conventions=project_conventions,
        )

        summary_md = self._render_summary(
            repo_info=repo_info,
            hotspots=hotspots[:10],
            business_summaries=business_summaries,
            generated_at=generated_at,
        )

        context_path = self._out_dir / "context.md"
        summary_path = self._out_dir / "summary.md"

        context_path.write_text(context_md, encoding="utf-8")
        summary_path.write_text(summary_md, encoding="utf-8")

        # Persist project conventions as JSON for pipeline use
        if project_conventions and any(project_conventions.get(k) for k in ("linters", "formatters", "pre_commit_hooks")):
            import json as _json
            conventions_path = self._out_dir / "project_conventions.json"
            conventions_path.write_text(_json.dumps(project_conventions, indent=2))
            logger.info("Saved project conventions → %s", conventions_path)

        logger.info(
            "Compiled context docs for '%s' → %s",
            self.repo_name,
            self._out_dir,
        )
        return context_path, summary_path

    # ------------------------------------------------------------------
    # Project conventions extraction
    # ------------------------------------------------------------------

    def _extract_project_conventions(self) -> dict[str, Any]:
        """Extract linting, CI, and coding standards from repo config files.

        Reads .pre-commit-config.yaml, pyproject.toml, setup.cfg, .flake8,
        tox.ini, .github/workflows/, and other convention-defining files.
        Returns a dict describing the project's quality gates.
        """
        if not self.repo_path or not self.repo_path.exists():
            return {}

        conventions: dict[str, Any] = {
            "linters": [],
            "formatters": [],
            "ci_checks": [],
            "test_framework": None,
            "python_version": None,
            "line_length": None,
            "import_sorting": None,
            "type_checking": None,
            "pre_commit_hooks": [],
            "raw_configs": {},
        }

        rp = self.repo_path

        # --- Pre-commit config ---
        precommit = rp / ".pre-commit-config.yaml"
        if precommit.exists():
            try:
                import yaml
                cfg = yaml.safe_load(precommit.read_text())
                for repo in (cfg or {}).get("repos", []):
                    for hook in repo.get("hooks", []):
                        hook_id = hook.get("id", "")
                        conventions["pre_commit_hooks"].append(hook_id)
                        if hook_id in ("flake8", "ruff", "pylint", "pyflakes"):
                            conventions["linters"].append(hook_id)
                        elif hook_id in ("black", "autopep8", "yapf", "ruff-format"):
                            conventions["formatters"].append(hook_id)
                        elif hook_id in ("isort",):
                            conventions["import_sorting"] = "isort"
                        elif hook_id in ("mypy", "pyright", "pytype"):
                            conventions["type_checking"] = hook_id
            except Exception:
                # yaml not available or parse error — read raw
                raw = precommit.read_text()[:2000]
                conventions["raw_configs"]["pre-commit"] = raw
                # Basic extraction via string matching
                for linter in ("flake8", "ruff", "pylint", "black", "isort", "mypy"):
                    if linter in raw:
                        conventions["linters" if linter in ("flake8", "ruff", "pylint") else "formatters"].append(linter)

        # --- pyproject.toml ---
        pyproject = rp / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    tomllib = None  # type: ignore

            if tomllib:
                try:
                    with open(pyproject, "rb") as f:
                        cfg = tomllib.load(f)
                    tool = cfg.get("tool", {})
                    # Ruff config
                    if "ruff" in tool:
                        conventions["linters"].append("ruff")
                        ruff_cfg = tool["ruff"]
                        if "line-length" in ruff_cfg:
                            conventions["line_length"] = ruff_cfg["line-length"]
                    # Black config
                    if "black" in tool:
                        conventions["formatters"].append("black")
                        if "line-length" in tool["black"]:
                            conventions["line_length"] = tool["black"]["line-length"]
                    # isort config
                    if "isort" in tool:
                        conventions["import_sorting"] = "isort"
                    # mypy config
                    if "mypy" in tool:
                        conventions["type_checking"] = "mypy"
                    # pytest config
                    if "pytest" in tool:
                        conventions["test_framework"] = "pytest"
                    # Python version
                    requires = cfg.get("project", {}).get("requires-python", "")
                    if requires:
                        conventions["python_version"] = requires
                except Exception:
                    pass

        # --- setup.cfg ---
        setup_cfg = rp / "setup.cfg"
        if setup_cfg.exists():
            try:
                import configparser
                cfg = configparser.ConfigParser()
                cfg.read(setup_cfg)
                if cfg.has_section("flake8"):
                    conventions["linters"].append("flake8")
                    ml = cfg.get("flake8", "max-line-length", fallback=None)
                    if ml:
                        conventions["line_length"] = int(ml)
                if cfg.has_section("isort"):
                    conventions["import_sorting"] = "isort"
                if cfg.has_section("mypy"):
                    conventions["type_checking"] = "mypy"
                if cfg.has_section("tool:pytest.ini_options"):
                    conventions["test_framework"] = "pytest"
            except Exception:
                pass

        # --- .flake8 ---
        flake8_cfg = rp / ".flake8"
        if flake8_cfg.exists():
            conventions["linters"].append("flake8")

        # --- tox.ini ---
        tox_ini = rp / "tox.ini"
        if tox_ini.exists():
            try:
                content = tox_ini.read_text()[:2000]
                if "flake8" in content:
                    conventions["linters"].append("flake8")
                if "pytest" in content:
                    conventions["test_framework"] = "pytest"
                if "mypy" in content:
                    conventions["type_checking"] = "mypy"
            except Exception:
                pass

        # --- CI workflows ---
        gh_workflows = rp / ".github" / "workflows"
        if gh_workflows.exists():
            for wf in gh_workflows.glob("*.yml"):
                try:
                    content = wf.read_text()[:3000]
                    conventions["ci_checks"].append(wf.name)
                except Exception:
                    pass

        # --- Test framework detection ---
        if not conventions["test_framework"]:
            if (rp / "pytest.ini").exists() or (rp / "conftest.py").exists():
                conventions["test_framework"] = "pytest"
            elif (rp / "tests").exists():
                conventions["test_framework"] = "unknown"

        # Deduplicate
        conventions["linters"] = sorted(set(conventions["linters"]))
        conventions["formatters"] = sorted(set(conventions["formatters"]))

        return conventions

    def _render_project_conventions(self, conventions: dict[str, Any]) -> str:
        """Render Layer 0 — Project Conventions."""
        if not conventions or not any(conventions.get(k) for k in ("linters", "formatters", "pre_commit_hooks", "ci_checks")):
            return ""

        lines = ["## 0. Project Conventions & Quality Gates\n"]
        lines.append("> **IMPORTANT: All generated code MUST comply with these rules.**\n")

        if conventions.get("linters"):
            lines.append(f"**Linters:** {', '.join(conventions['linters'])}")
        if conventions.get("formatters"):
            lines.append(f"**Formatters:** {', '.join(conventions['formatters'])}")
        if conventions.get("line_length"):
            lines.append(f"**Max line length:** {conventions['line_length']}")
        if conventions.get("import_sorting"):
            lines.append(f"**Import sorting:** {conventions['import_sorting']}")
        if conventions.get("type_checking"):
            lines.append(f"**Type checking:** {conventions['type_checking']}")
        if conventions.get("test_framework"):
            lines.append(f"**Test framework:** {conventions['test_framework']}")
        if conventions.get("python_version"):
            lines.append(f"**Python version:** {conventions['python_version']}")

        if conventions.get("pre_commit_hooks"):
            lines.append(f"\n**Pre-commit hooks (all must pass):** {', '.join(conventions['pre_commit_hooks'])}")

        if conventions.get("ci_checks"):
            lines.append(f"\n**CI workflows:** {', '.join(conventions['ci_checks'])}")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Neo4j data fetchers
    # ------------------------------------------------------------------

    def _fetch_repo_info(self) -> dict[str, Any]:
        rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo}) "
            "RETURN r.name AS name, r.path AS path, "
            "       r.tech_stack AS tech_stack, "
            "       r.entry_points AS entry_points, "
            "       r.file_count AS file_count",
            {"repo": self.repo_name},
        )
        return rows[0] if rows else {}

    def _fetch_files(self) -> list[dict[str, Any]]:
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File) "
            "RETURN f.id AS id, f.path AS path, f.language AS language, "
            "       f.loc AS loc, f.docstring AS docstring, "
            "       f.summary AS summary "
            "ORDER BY f.path",
            {"repo": self.repo_name},
        )

    def _fetch_classes(self) -> list[dict[str, Any]]:
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File)-[:CONTAINS]->(c:Class) "
            "OPTIONAL MATCH (c)-[:CONTAINS]->(m:Function) "
            "WITH f, c, collect({name: m.name, params: m.params, return_type: m.return_type}) AS methods "
            "RETURN f.path AS file_path, c.id AS id, c.name AS name, "
            "       c.bases AS bases, c.docstring AS docstring, methods "
            "ORDER BY f.path, c.name",
            {"repo": self.repo_name},
        )

    def _fetch_functions(self) -> list[dict[str, Any]]:
        """Fetch top-level functions (not inside a class)."""
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File)-[:CONTAINS]->(fn:Function) "
            "WHERE NOT (:Class)-[:CONTAINS]->(fn) "
            "RETURN f.path AS file_path, fn.id AS id, fn.name AS name, "
            "       fn.params AS params, fn.return_type AS return_type, "
            "       fn.docstring AS docstring "
            "ORDER BY f.path, fn.name",
            {"repo": self.repo_name},
        )

    def _fetch_hotspots(self, top_n: int = 20) -> list[dict[str, Any]]:
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
            "WHERE n.pagerank IS NOT NULL "
            "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
            "       n.path AS path, n.pagerank AS pagerank "
            "ORDER BY n.pagerank DESC "
            "LIMIT $top_n",
            {"repo": self.repo_name, "top_n": top_n},
        )

    def _fetch_file_imports_exports(self, file_id: str) -> dict[str, list[str]]:
        """Return {imports: [...], exports: [...]} for a single file."""
        imports_rows = neo4j_client.run(
            "MATCH (f:File {id: $fid})-[:IMPORTS]->(t) "
            "RETURN t.name AS name",
            {"fid": file_id},
        )
        exports_rows = neo4j_client.run(
            "MATCH (f:File {id: $fid})-[:CONTAINS]->(n) "
            "WHERE n:Class OR n:Function "
            "RETURN n.name AS name",
            {"fid": file_id},
        )
        return {
            "imports": [r["name"] for r in imports_rows if r.get("name")],
            "exports": [r["name"] for r in exports_rows if r.get("name")],
        }

    def _fetch_all_imports_exports(self) -> dict[str, dict[str, list[str]]]:
        """Batch-fetch imports and exports for ALL files in the repo (2 queries total)."""
        result: dict[str, dict[str, list[str]]] = {}

        # All imports
        import_rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File)-[:IMPORTS]->(t) "
            "RETURN f.id AS file_id, t.name AS name",
            {"repo": self.repo_name},
        )
        for row in import_rows:
            fid = row.get("file_id", "")
            if fid:
                result.setdefault(fid, {"imports": [], "exports": []})
                if row.get("name"):
                    result[fid]["imports"].append(row["name"])

        # All exports
        export_rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File)-[:CONTAINS]->(n) "
            "WHERE n:Class OR n:Function "
            "RETURN f.id AS file_id, n.name AS name",
            {"repo": self.repo_name},
        )
        for row in export_rows:
            fid = row.get("file_id", "")
            if fid:
                result.setdefault(fid, {"imports": [], "exports": []})
                if row.get("name"):
                    result[fid]["exports"].append(row["name"])

        return result

    def _fetch_business_summaries(self) -> list[dict[str, Any]]:
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File) "
            "WHERE f.summary IS NOT NULL "
            "RETURN f.path AS path, f.summary AS summary, f.docstring AS docstring "
            "ORDER BY f.path",
            {"repo": self.repo_name},
        )

    def _fetch_business_rules(self) -> list[dict[str, Any]]:
        """Fetch extracted BusinessRule nodes linked to this repo's files."""
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(f:File) "
            "MATCH (br:BusinessRule)-[:FOUND_IN]->(f) "
            "RETURN br.content AS content, br.rule_type AS rule_type, "
            "       br.source_file AS source_file, br.source_line AS source_line "
            "ORDER BY br.rule_type, br.source_file",
            {"repo": self.repo_name},
        )

    def _fetch_call_edges(self, limit: int = 40) -> list[dict[str, Any]]:
        """Fetch top CALLS relationships for the call flow section."""
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(a) "
            "MATCH (a)-[:CALLS]->(b) "
            "WHERE a.pagerank IS NOT NULL "
            "RETURN a.name AS caller, a.file AS caller_file, "
            "       b.name AS callee, b.file AS callee_file "
            "ORDER BY a.pagerank DESC "
            "LIMIT $limit",
            {"repo": self.repo_name, "limit": limit},
        )

    def _fetch_import_edges(self, limit: int = 30) -> list[dict[str, Any]]:
        """Fetch IMPORTS edges to show module dependency graph."""
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(a:File) "
            "MATCH (a)-[:IMPORTS]->(b:File) "
            "RETURN a.path AS importer, b.path AS imported "
            "ORDER BY a.path "
            "LIMIT $limit",
            {"repo": self.repo_name, "limit": limit},
        )

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _fetch_repo_readme(self) -> str:
        """Fetch README content from Neo4j if stored, else return empty string."""
        rows = neo4j_client.run(
            "MATCH (r:Repo {name: $repo}) RETURN r.readme AS readme",
            {"repo": self.repo_name},
        )
        return (rows[0].get("readme") or "") if rows else ""

    def _render_layer1(self, repo_info: dict[str, Any], files: list[dict[str, Any]], readme: str = "") -> str:
        """Layer 1 — Repository Structure."""
        tech_stack: list[str] = repo_info.get("tech_stack") or []
        entry_points: list[str] = repo_info.get("entry_points") or []
        file_count: int = repo_info.get("file_count") or len(files)

        # Tech-stack badges
        if tech_stack:
            badges = " ".join(_tech_badge(t) for t in tech_stack)
        else:
            badges = "_No tech stack detected._"

        # Directory tree from file paths
        paths = [f["path"] for f in files if f.get("path")]
        tree = _build_tree(paths)

        # Entry points
        if entry_points:
            ep_list = "\n".join(f"- `{ep}`" for ep in entry_points)
        else:
            ep_list = "_No entry points detected._"

        # Stats line
        total_loc = sum(f.get("loc") or 0 for f in files)
        stats_line = f"**{file_count} files** | **{total_loc:,} lines of code**\n"

        # README section
        readme_section = ""
        if readme and readme.strip():
            readme_section = f"\n### README\n\n```\n{readme[:2000]}\n```\n"

        return (
            "## 1. Repository Structure\n\n"
            "### Tech Stack\n\n"
            f"{badges}\n\n"
            f"{stats_line}\n"
            "### Directory Overview\n\n"
            f"{tree}\n\n"
            "### Entry Points\n\n"
            f"{ep_list}\n"
            f"{readme_section}"
        )

    def _render_layer2(
        self,
        files: list[dict[str, Any]],
        all_ie: dict[str, dict[str, list[str]]] | None = None,
    ) -> str:
        """Layer 2 — File Index."""
        rows = ["| File | Purpose | Imports | Exports |", "|------|---------|---------|---------|"]
        for f in files:
            file_id = f.get("id", "")
            path = f.get("path", "—")
            purpose = _first_line(f.get("docstring"))

            ie = (all_ie or {}).get(file_id, {"imports": [], "exports": []})
            imports_str = ", ".join(ie["imports"][:5]) or "—"
            exports_str = ", ".join(ie["exports"][:5]) or "—"
            if len(ie["imports"]) > 5:
                imports_str += f", +{len(ie['imports']) - 5} more"
            if len(ie["exports"]) > 5:
                exports_str += f", +{len(ie['exports']) - 5} more"

            rows.append(f"| `{path}` | {purpose} | {imports_str} | {exports_str} |")

        return "## 2. File Index\n\n" + "\n".join(rows) + "\n"

    def _render_layer3(
        self,
        classes: list[dict[str, Any]],
        functions: list[dict[str, Any]],
    ) -> str:
        """Layer 3 — Symbol Map."""
        lines: list[str] = ["## 3. Symbol Map\n", "### Classes\n"]

        if classes:
            for cls in classes:
                name = cls.get("name", "?")
                bases = cls.get("bases") or []
                bases_str = f"({', '.join(bases)})" if bases else ""
                doc = _first_line(cls.get("docstring"))
                lines.append(f"**{name}{bases_str}** — {doc}")

                methods: list[dict] = cls.get("methods") or []
                valid_methods = [m for m in methods if m.get("name")]
                if valid_methods:
                    method_sigs = []
                    for m in valid_methods:
                        params = m.get("params") or []
                        if isinstance(params, list):
                            params_str = ", ".join(str(p) for p in params)
                        else:
                            params_str = str(params)
                        method_sigs.append(f"`{m['name']}({params_str})`")
                    lines.append(f"  Methods: {', '.join(method_sigs)}")
                lines.append("")
        else:
            lines.append("_No classes found._\n")

        lines.append("### Functions\n")

        if functions:
            for fn in functions:
                name = fn.get("name", "?")
                params = fn.get("params") or []
                if isinstance(params, list):
                    params_str = ", ".join(str(p) for p in params)
                else:
                    params_str = str(params)
                ret = fn.get("return_type") or "Any"
                doc = _first_line(fn.get("docstring"))
                doc_str = f" — {doc}" if doc and doc != "—" else ""
                lines.append(f"- `{name}({params_str})` → `{ret}`{doc_str}")
        else:
            lines.append("_No top-level functions found._")

        return "\n".join(lines) + "\n"

    def _render_layer4(self, hotspots: list[dict[str, Any]]) -> str:
        """Layer 4 — Call Graph Hotspots."""
        lines: list[str] = [
            "## 4. Call Graph Hotspots\n",
            "Top files and functions by PageRank importance:\n",
            "| Rank | Symbol | Type | PageRank | File |",
            "|------|--------|------|----------|------|",
        ]

        for i, node in enumerate(hotspots, start=1):
            name = node.get("name", "?")
            labels: list[str] = node.get("labels") or []
            node_type = next((lb for lb in labels if lb not in ("File",)), labels[0] if labels else "?")
            pagerank = node.get("pagerank")
            pr_str = f"{pagerank:.6f}" if isinstance(pagerank, float) else str(pagerank or "—")
            path = node.get("path") or "—"
            lines.append(f"| {i} | `{name}` | {node_type} | {pr_str} | `{path}` |")

        if not hotspots:
            lines.append("| — | _No PageRank data available_ | — | — | — |")

        return "\n".join(lines) + "\n"

    def _render_layer5(self, classes: list[dict[str, Any]]) -> str:
        """Layer 5 — Data Models (all fields and methods)."""
        lines: list[str] = ["## 5. Data Models\n"]

        if not classes:
            lines.append("_No classes found._\n")
            return "\n".join(lines)

        for cls in classes:
            name = cls.get("name", "?")
            bases = cls.get("bases") or []
            bases_str = f"({', '.join(bases)})" if bases else ""
            doc = _first_line(cls.get("docstring"))
            file_path = cls.get("file_path", "?")

            lines.append(f"### `{name}{bases_str}`")
            lines.append(f"*File: `{file_path}`*")
            if doc and doc != "—":
                lines.append(f"> {doc}")
            lines.append("")

            methods: list[dict] = cls.get("methods") or []
            valid_methods = [m for m in methods if m.get("name")]
            if valid_methods:
                lines.append("**Methods:**")
                for m in valid_methods:
                    params = m.get("params") or []
                    if isinstance(params, list):
                        params_str = ", ".join(str(p) for p in params)
                    else:
                        params_str = str(params)
                    ret = m.get("return_type") or ""
                    ret_str = f" → `{ret}`" if ret else ""
                    lines.append(f"- `{m['name']}({params_str})`{ret_str}")
            lines.append("")

        return "\n".join(lines)

    def _render_layer6(self, business_summaries: list[dict[str, Any]]) -> str:
        """Layer 6 — Business Logic Summaries."""
        lines: list[str] = ["## 6. Business Logic Summaries\n"]

        if not business_summaries:
            lines.append("_No business logic summaries available. Run the enricher to generate them._\n")
            return "\n".join(lines)

        for entry in business_summaries:
            path = entry.get("path", "unknown")
            summary = entry.get("summary") or ""
            docstring = entry.get("docstring") or ""

            filename = Path(path).name
            lines.append(f"### {filename}")
            lines.append(f"*`{path}`*\n")

            if summary:
                lines.append(summary)
            elif docstring:
                lines.append(_first_n_lines(docstring, 3))
            else:
                lines.append("_No summary available._")

            lines.append("")

        return "\n".join(lines)

    def _render_business_rules(self, rules: list[dict[str, Any]]) -> str:
        """Layer 7 — Extracted Business Rules & Constraints."""
        lines: list[str] = ["## 7. Extracted Business Rules & Constraints\n"]

        if not rules:
            lines.append("_No business rules extracted yet._\n")
            return "\n".join(lines)

        # Group by rule_type
        by_type: dict[str, list[dict]] = {}
        for r in rules:
            rt = r.get("rule_type", "other")
            by_type.setdefault(rt, []).append(r)

        type_labels = {
            "constant": "Business Constants & Limits",
            "docstring": "Rules from Docstrings",
            "todo": "Pending / Known Issues (TODOs)",
        }

        for rule_type, label in type_labels.items():
            items = by_type.get(rule_type, [])
            if not items:
                continue
            lines.append(f"### {label}\n")
            for r in items:
                content = r.get("content", "")
                src = r.get("source_file", "")
                lineno = r.get("source_line", "")
                src_ref = f"`{src}:{lineno}`" if src else ""
                lines.append(f"- {content}  {src_ref}")
            lines.append("")

        # Any remaining types
        for rule_type, items in by_type.items():
            if rule_type in type_labels:
                continue
            lines.append(f"### {rule_type.title()}\n")
            for r in items:
                content = r.get("content", "")
                src = r.get("source_file", "")
                lines.append(f"- {content}  (`{src}`)")
            lines.append("")

        return "\n".join(lines)

    def _render_source_code(self, files: list[dict[str, Any]]) -> str:
        """Layer 9 — Full source code of all indexed files."""
        lines: list[str] = [
            "## 9. Source Code\n",
            "_Complete source code of all indexed files._\n",
        ]

        if not self.repo_path:
            # Try to get repo path from Neo4j
            rows = neo4j_client.run(
                "MATCH (r:Repo {name: $repo}) RETURN r.path AS path",
                {"repo": self.repo_name},
            )
            if rows and rows[0].get("path"):
                self.repo_path = Path(rows[0]["path"])
            else:
                lines.append("_Source code unavailable — repo path not found._\n")
                return "\n".join(lines)

        _MAX_FILE_SIZE = 50_000
        _MAX_TOTAL_CODE = 500_000
        total_code_chars = 0

        _LANG_MAP = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".go": "go",
            ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c",
            ".cpp": "cpp", ".h": "c", ".css": "css", ".html": "html",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".sh": "bash", ".sql": "sql", ".md": "markdown",
        }

        for f in files:
            if total_code_chars >= _MAX_TOTAL_CODE:
                lines.append(
                    f"\n> Source code truncated at {_MAX_TOTAL_CODE // 1000}KB "
                    "to stay within context limits.\n"
                )
                break

            file_path_str = f.get("path", "")
            if not file_path_str:
                continue

            full_path = self.repo_path / file_path_str
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if len(content) > _MAX_FILE_SIZE:
                content = content[:_MAX_FILE_SIZE] + f"\n... [truncated at {_MAX_FILE_SIZE // 1000}KB]"

            ext = full_path.suffix.lower()
            lang = _LANG_MAP.get(ext, "")

            lines.append(f"### `{file_path_str}`\n")
            lines.append(f"```{lang}\n{content}\n```\n")
            total_code_chars += len(content)

        return "\n".join(lines)

    def _render_call_flow(self, call_edges: list[dict[str, Any]], import_edges: list[dict[str, Any]]) -> str:
        """Layer 8 — Call Flow & Module Dependencies."""
        lines: list[str] = ["## 8. Call Flow & Module Dependencies\n"]

        if import_edges:
            lines.append("### Module Import Graph\n")
            lines.append("| Importer | Imports |")
            lines.append("|----------|---------|")
            seen_pairs: set = set()
            for e in import_edges:
                pair = (e.get("importer", ""), e.get("imported", ""))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                lines.append(f"| `{pair[0]}` | `{pair[1]}` |")
            lines.append("")

        if call_edges:
            lines.append("### Key Call Relationships (top hotspots)\n")
            lines.append("| Caller | Callee | Caller File |")
            lines.append("|--------|--------|------------|")
            seen_calls: set = set()
            for e in call_edges:
                caller = e.get("caller", "")
                callee = e.get("callee", "")
                caller_file = e.get("caller_file", "")
                key = (caller, callee)
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                lines.append(f"| `{caller}` | `{callee}` | `{caller_file}` |")
            lines.append("")

        if not import_edges and not call_edges:
            lines.append("_No call/import relationships found. Build the call graph first._\n")

        return "\n".join(lines)

    def _render_decision_points(self, decision_points: list[dict[str, Any]], domain_concepts: list[dict[str, Any]]) -> str:
        """Layer 9 — Decision Points & Business Decisions."""
        lines: list[str] = ["## 9. Decision Points & Business Decisions\n"]
        lines.append("_Code locations where business logic is encoded as conditionals, thresholds, or role checks._\n")

        if domain_concepts:
            lines.append("### Domain Concepts\n")
            lines.append("| Concept | Type | Related Classes |")
            lines.append("|---------|------|----------------|")
            for dc in domain_concepts[:20]:
                classes_str = ", ".join(f"`{c}`" for c in (dc.get("related_classes") or [])[:5])
                lines.append(f"| **{dc.get('name', '')}** | {dc.get('type', '')} | {classes_str} |")
            lines.append("")

        # Group decision points by type
        dp_by_type: dict[str, list] = {}
        for dp in decision_points:
            dp_by_type.setdefault(dp.get("condition_type", "other"), []).append(dp)

        type_labels = {
            "threshold": "Threshold Decisions (magic numbers, limits)",
            "role_check": "Role & Permission Checks",
            "status_check": "Status Checks",
            "feature_flag": "Feature Flags",
        }

        for dp_type in ("threshold", "role_check", "status_check", "feature_flag"):
            items = dp_by_type.get(dp_type, [])
            if not items:
                continue
            lines.append(f"### {type_labels.get(dp_type, dp_type)}\n")
            for dp in items[:15]:
                func_name = (dp.get("function_name") or dp.get("function_id") or "").split("::")[-1]
                condition = dp.get("condition", "")
                explanation = dp.get("explanation") or ""
                lines.append(f"- **`{func_name}`** line {dp.get('line', '?')}: `{condition}`")
                if explanation:
                    lines.append(f"  - _{explanation}_")
            if len(items) > 15:
                lines.append(f"- ... +{len(items) - 15} more")
            lines.append("")

        if not decision_points and not domain_concepts:
            lines.append("_No decision points detected. Run the enricher to extract them._\n")

        return "\n".join(lines)

    def _fetch_decision_points(self) -> list[dict[str, Any]]:
        """Fetch DecisionPoint nodes from Neo4j."""
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(fn:Function)-[:HAS_DECISION]->(dp:DecisionPoint) "
            "RETURN dp.id AS id, dp.line AS line, dp.condition AS condition, "
            "       dp.condition_type AS condition_type, dp.explanation AS explanation, "
            "       dp.file AS file, fn.name AS function_name "
            "ORDER BY dp.condition_type, dp.file "
            "LIMIT 200",
            {"repo": self.repo_name},
        )

    def _fetch_domain_concepts(self) -> list[dict[str, Any]]:
        """Fetch DomainConcept nodes from Neo4j."""
        rows = neo4j_client.run(
            "MATCH (dc:DomainConcept) "
            "OPTIONAL MATCH (dc)-[:REPRESENTS]->(c:Class) "
            "WITH dc, collect(c.name) AS classes "
            "RETURN dc.id AS id, dc.name AS name, dc.type AS type, "
            "       dc.description AS description, classes AS related_classes "
            "ORDER BY dc.name "
            "LIMIT 50",
        )
        return rows

    # ------------------------------------------------------------------
    # Full document renderers
    # ------------------------------------------------------------------

    def _render_context(
        self,
        *,
        repo_info: dict[str, Any],
        files: list[dict[str, Any]],
        classes: list[dict[str, Any]],
        functions: list[dict[str, Any]],
        hotspots: list[dict[str, Any]],
        business_summaries: list[dict[str, Any]],
        business_rules: list[dict[str, Any]] | None = None,
        call_edges: list[dict[str, Any]] | None = None,
        import_edges: list[dict[str, Any]] | None = None,
        decision_points: list[dict[str, Any]] | None = None,
        domain_concepts: list[dict[str, Any]] | None = None,
        readme: str = "",
        generated_at: str,
        all_ie: dict[str, dict[str, list[str]]] | None = None,
        project_conventions: dict[str, Any] | None = None,
    ) -> str:
        sections = [
            f"# Repository Context: {self.repo_name}\n",
            f"> Generated by Context Builder on {generated_at}\n",
            f"> **Layers:** Conventions | Structure | File Index | Symbol Map | Hotspots | Data Models | Business Summaries | Business Rules | Call Flow | Decision Points | Source Code\n",
            self._render_project_conventions(project_conventions or {}),
            self._render_layer1(repo_info, files, readme=readme),
            self._render_layer2(files, all_ie=all_ie),
            self._render_layer3(classes, functions),
            self._render_layer4(hotspots),
            self._render_layer5(classes),
            self._render_layer6(business_summaries),
            self._render_business_rules(business_rules or []),
            self._render_call_flow(call_edges or [], import_edges or []),
            self._render_decision_points(decision_points or [], domain_concepts or []),
            self._render_source_code(files),
        ]
        return "\n".join(sections)

    def _render_summary(
        self,
        *,
        repo_info: dict[str, Any],
        hotspots: list[dict[str, Any]],
        business_summaries: list[dict[str, Any]],
        generated_at: str,
    ) -> str:
        """Compressed summary (~3000 tokens): Layer 1 + top hotspots + business summaries."""
        # Fetch files list for Layer 1 tree
        files = self._fetch_files()

        layer1 = self._render_layer1(repo_info, files, readme="")

        # Compact hotspot table (top 10)
        hotspot_lines = [
            "## Top 10 Hotspots\n",
            "| Rank | Symbol | Type | PageRank | File |",
            "|------|--------|------|----------|------|",
        ]
        for i, node in enumerate(hotspots, start=1):
            name = node.get("name", "?")
            labels: list[str] = node.get("labels") or []
            node_type = next((lb for lb in labels if lb not in ("File",)), labels[0] if labels else "?")
            pagerank = node.get("pagerank")
            pr_str = f"{pagerank:.6f}" if isinstance(pagerank, float) else str(pagerank or "—")
            path = node.get("path") or "—"
            hotspot_lines.append(f"| {i} | `{name}` | {node_type} | {pr_str} | `{path}` |")

        if not hotspots:
            hotspot_lines.append("| — | _No PageRank data_ | — | — | — |")

        hotspot_section = "\n".join(hotspot_lines) + "\n"

        # Business summaries (compact)
        biz_lines: list[str] = ["## Business Logic Summaries\n"]
        if business_summaries:
            for entry in business_summaries:
                path = entry.get("path", "unknown")
                summary = entry.get("summary") or _first_n_lines(entry.get("docstring"), 2)
                filename = Path(path).name
                if summary:
                    biz_lines.append(f"**{filename}**: {summary}\n")
        else:
            biz_lines.append("_Run the enricher to generate summaries._\n")

        biz_section = "\n".join(biz_lines)

        header = (
            f"# Repository Summary: {self.repo_name}\n\n"
            f"> Generated by Context Builder on {generated_at}\n\n"
            "> This is a compressed summary. See `context.md` for full detail.\n"
        )

        return "\n".join([header, layer1, hotspot_section, biz_section])
