# TODOS

## Active



### P2 — Continue extracting utilities from pipeline.py
**What:** pipeline.py is still 3300 lines. Utility functions (_redact_secrets, _fuzzy_match_replace, should_iterate, etc.) are still imported by tests. Extract into focused modules: llm_utils.py, secrets.py, file_utils.py, analysis_utils.py, patch_utils_extended.py.
**Why:** Fixed pipeline is retired but the file remains as a utility library. 6-module extraction plan ready (see prior eng review).

### P2 — Validate ENFORCED_BY edge precision before production use
**What:** Run `BusinessLogicExtractor` on 20 manually annotated functions, measure precision/recall of ENFORCED_BY edges. Gate business context injection on >80% precision.
**Why:** The extractor links rules to functions by docstring keyword matching. If precision is low (<80%), repair prompts get noisy rule context that degrades fix quality (-3% per research findings).
**Context:** Open question #1 in design doc. Do this validation after Phase 1 ships, before Phase 3 is wired to the repair pipeline.
**Depends on:** Phase 1 shipped (ENFORCED_BY edges exist in Neo4j).

### P3 — FailureRecord severity classification (enhanced)
**What:** Basic severity classification exists (`_severity_hint` parses P0/Sev0/critical/hotfix keywords). Enhance with: fetch PR labels via GitHub API, or infer from Jira priority field for richer signal.
**Why:** Current classification is keyword-only from commit messages. PR labels and Jira priority give cleaner signal for distinguishing P0 incidents from cosmetic fixes.
**Context:** `mine_failure_records()` in `backend/graph/business/failure_records.py`. Basic version already shipped (v2.7). Enhancement is optional.

### P4 — Approach C: Forward-Looking Structured History (Lore protocol)
**What:** After 5-10 agent PRs are merged, implement agent-written git trailers on every PR: rules checked, blast radius, failure pattern ID. Git becomes the FailureRecords DB over time (arxiv 2603.15566).
**Why:** FailureRecords mined from git history are retrospective and limited by commit message quality. Agent-authored trailers create forward-looking, structured history with full agent context attached to each change.
**Context:** Explicitly deferred (Approach C) in design doc — cold start requires 5-10 real agent PRs to produce useful data.
**Depends on:** Wedge demo shipped + 5-10 real agent PRs merged via Phase 3.

### P5 — Extract external side effects behind interface (dependency inversion)
**What:** `pr_creation_node()` has multiple external side effects (git push, gh pr create, feature flags, _enrich_from_fix) each guarded by `if not dry_run`. Extract behind a thin interface and no-op in test/eval mode.
**Why:** The boolean-flag approach works for 3-4 calls but creates maintenance burden as new side effects are added. Forgetting a guard is a silent bug.
**Context:** Identified by outside voice during eng review (2026-03-30). Current dry_run guards: git push, feature flag creation, PR creation, enrichment.
**Depends on:** Production validation shipped (current sprint).

## Completed

### P1 — Concept-to-code mapping for business-language tickets
**Completed:** v3.2 (2026-04-13)
`graph_utils.query_concept_to_code()`: extracts keywords from ticket title + description, queries `BusinessRule` nodes in Neo4j (OPTIONAL MATCH `ENFORCED_BY` → Function), falls back to flat `business_rules.json` scan when Neo4j is unavailable.
Results merged into `intent.likely_affected_functions` + `intent.likely_affected_modules` in `intake_node()`, and a `## RELEVANT BUSINESS RULES` section is injected into the kickstart context.
19 unit tests in `tests/test_concept_to_code.py` cover keyword extraction, JSON fallback, Neo4j mock path, exception fallback to JSON, section formatting, and intent-merge logic.

### P1 — Django SWE-bench test infrastructure (3 infra bugs)
**Completed:** v3.1 (2026-04-12)
`repo_manager.py`: three Django eval infra fixes landed together.
1. **Missing INSTALLED_APPS**: detect minimal test_sqlite.py → write `swe_bench_django_settings.py` to venv site-packages with full INSTALLED_APPS + MIGRATION_MODULES (mirrors Django's runtests.py setup_collect_tests).
2. **InvalidBasesError**: `MIGRATION_MODULES = {'auth': None, 'contenttypes': None, 'sessions': None}` prevents migration state resolver from choking on `auth_tests.UserProxy → auth.User` inheritance.
3. **Duplicate-module PYTHONPATH error**: use relative `"tests"` (not absolute `{repo_dir}/tests`) so Python resolves against the sandbox worktree's cwd, making both the file path and the sys.path entry point to the same physical file.
New helpers: `_extract_django_test_apps()`, `_django_test_app_has_models()`, `_find_site_packages()`, `_write_full_django_settings()`.

### P1 — Expand eval dataset with multi-file bugs
**Completed:** v3.0 (2026-04-12)
`eval/bugs.json`: added 10 multi-file bugs from SWE-bench-Verified. Sourced via `datasets` lib — filtered for 2+ source-file patches in repos we support (django, pytest). All bugs validated against `dataset.py` schema. Dataset grows from 25 → 35 bugs (25 single-file + 10 multi-file). Multi-file scoring already handled by `_score_multi_file_complete()` and `multi_file_complete_rate` in `scoring.py`.

### P1 — Multi-candidate patch sampling
**Completed:** v2.9 (2026-04-09)
`react_pipeline.py`: `_run_best_of_n()` launches N parallel instances via `ProcessPoolExecutor`, picks best by test_pass → verifier APPROVE → review confidence → cost. `verifier_node` runs after each agent loop for fresh-context independent review. CLI: `--best-of-n N` flag on `fix` command. Max N=5 hard cap.

### P1 — Independent verifier subagent (cavekit speculative pattern)
**Completed:** v2.9 (2026-04-09)
`react_pipeline.py`: `verifier_node()` runs after `react_agent_node`, uses a fresh-context Haiku call with only diff + test results + bug description. Adds `verifier_verdict`, `verifier_confidence`, `epr_score` (BRT pass rate) to state. Blocks PR creation if verifier REJECTS with high confidence.

### P0 — Integration tests for react tool/guardrail/prompt contract
**Completed:** v0.3.1.0 (2026-04-02)
75 regression tests in `test_react_contracts.py`: guardrail gates, exit code chain (pytest 0-5 → sandbox prefix → guardrail state → submit gate), prompt-guardrail alignment, anti-pattern thresholds, terminal detection.

### P0 — Retire fixed LangGraph pipeline
**Completed:** v0.3.1.0 (2026-04-02)
Removed `--no-react` from CLI and eval. ReAct is the only supported runtime. `pipeline.py` remains as utility library (should_iterate, _redact_secrets, etc.) with `run_ticket()` and `build_agent_graph()` deprecated.

### P1 — Jira project prefix must be configurable
**Completed:** Already implemented (discovered 2026-03-30 during eng review)
`failure_records.py:37`: `JIRA_PROJECT_PREFIX` env var with default `PROJ`. Tests at `test_failure_records.py:54`.

### P0 — Bug: `repo_path` missing from `/api/repos` response
**Completed:** v0.1.1.0 (2026-03-29)
`api/repos.py`: initialize `repo_path: ""` in base entry and always assign from stats — field now present on all repos regardless of whether graph.json exists.

### P2 — Fix `TESTED_BY` edge `rstrip` bug
**Completed:** v2.8 (2026-04-08)
`call_graph.py:603`: `.rstrip(".py")` → `.removesuffix(".py")`. The old code stripped any trailing char in the set `{'.','p','y'}` not the suffix `.py`, corrupting paths like `app/copy.py` → `app/cop`.

### P3 — Nested functions invisible to parser
**Completed:** v2.8 (2026-04-08)
`code_parser.py`: replaced flat `for child in root.children` with recursive `_walk_nodes()` that descends into function bodies. Captures nested functions (Flask `create_app` factory pattern, closures, decorators).

### P3 — Multi-alias imports lose all but last alias
**Completed:** v2.8 (2026-04-08)
`code_parser.py`: fixed `_parse_imports` to accumulate per-name aliases into an `aliases: dict[str, str]` map. Added `aliases` key to import dicts. `alias` field retains backward compat (first alias if exactly one, else None).

### P3 — JS/TS brace-counting breaks on template literals
**Completed:** v2.8 (2026-04-08)
`code_parser.py` `_estimate_func_end`: rewritten with char-by-char scanner that tracks block comments, line comments, template literals, and quoted strings before counting braces. No longer mis-counts `{` inside `` `${expr}` ``.

### P3 — JS/TS parser feature parity with Python
**Completed:** v2.8 (2026-04-08)
`multi_lang_parser.py` full rewrite:
- Decorators: `@Controller`, `@Get`, `@Injectable` etc. extracted from tree-sitter `decorator` nodes
- Return types: TypeScript `: Promise<User>` annotations extracted
- Param types: full typed params `"id: number"` preserved (not stripped)
- Conditionals: `if`/`switch`/`ternary` extracted with `branch_count`
- Complexity: cyclomatic complexity computed
- Line boundaries: tree-sitter `end_point` used (not brace counting)
- Import symbols: `names: ["UserService", "AuthGuard"]` populated
- Import aliases: `aliases: {"UserService": "US"}` dict added
- Implements clause: TypeScript `implements Interface1, Interface2`
- Async flag: `is_async: True/False`
- JSDoc: checks prev sibling AND parent prev sibling (catches export-wrapped functions)
- Export tracking: `is_exported: True/False`
- React components: `is_react_component: True` for uppercase PascalCase functions with JSX
- Hooks: `is_hook: True` for `useXxx` functions
- Language labels: `"javascript"`, `"typescript"`, `"jsx"`, `"tsx"` per extension

### P2 — CALLS edges disabled for repos >2000 callables
**Completed:** v2.8 (2026-04-08)
`call_graph.py`: instead of skipping CALLS entirely for large repos, now uses focused resolution: top 30% of files by PageRank, per-function fanout capped at 50 unique targets. `_add_call_edges_focused()` + `_scan_calls_capped()` added.

### P4 — Leiden runs on undirected graph
**Completed:** v2.8 (2026-04-08)
`community.py`: `ig.Graph(directed=False)` → `directed=True`, `RBConfigurationVertexPartition` (undirected-only) → `CPMVertexPartition` (directed). Added seen-edge deduplication set.

### P4 — Community naming uses raw token frequency, not IDF
**Completed:** v2.8 (2026-04-08)
`community.py`: `_derive_community_name` now applies IDF weighting — `score = count_in_community / log(1 + count_in_all_communities)`. Generic path segments ("api", "backend", "frontend") that appear in many communities are penalised. Extended stopwords list.

### P3 — Multi-language repo support (JS/TS + Python)
**Completed:** v2.8 (2026-04-08)
`call_graph.py`:
- JS/TS relative imports (`"./foo"`, `"../bar"`) now resolved correctly in `_resolve_import_module` with extension probing (`.js`, `.ts`, `.jsx`, `.tsx`, `/index.js`, `/index.ts`)
- `_add_nodes` now handles `lineno` fallback for JS/TS nodes that emit `lineno` instead of `line_start`/`line_end`
- Mixed Python + JS/TS repos work end-to-end through graph build → flows → agent tools

### P3 — FailureRecord severity classification (basic)
**Completed:** v2.5 (estimated)
`failure_records.py`: `_severity_hint()` parses commit messages for P0/Sev0/critical → `"critical"`, P1/hotfix/incident → `"high"`, else `"unknown"`. Stored as `severity_hint` on FailureRecord nodes in Neo4j.
