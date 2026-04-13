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

### P1 — Learn-from-fix adaptation + FAIL_TO_PASS injection (v3.5)
**Completed:** v3.5 (2026-04-13)
Two improvements that together pushed pass rate from 60% → **80% (mission target hit)**:
- **FAIL_TO_PASS injection**: system prompt now includes the specific SWE-bench test IDs that must pass, plus guidance to run them LAST before submit_fix (fixes RICH-2187 style "agent's fix works but scorer reads unrelated last-test = FAIL").
- **Learn-from-fix**: `backend/agent/learn_from_fix.py` — after each run, a Haiku extractor writes a structured lesson (`**Pattern** / **Lesson** / **Tactic**`) to `{DATA_DIR}/{repo_name}/agent_lessons.md`. Next run on the same repo loads the 5 most recent lessons into the kickstart. Per-repo isolated, capped at 25 lessons/file with oldest-evicted, with rule-based fallback if the Haiku call fails. 24 unit tests.
Bug fix: initial impl read `state.run_outcome.tests_passed` (trace field, not state field). Fixed by `_derive_tests_passed(state)` helper that reads `test_result` string + verifier verdict with confidence threshold.

### P1 — Main-agent decisioning (P1-P4: thinking, context refresh, auto-replan, undo)
**Completed:** v3.4 (2026-04-13)
Four ports from Claude Code that improve how the main agent reasons and recovers:
- **P1 Extended thinking** on early turns (intake + plan production) — ChatAnthropic with `thinking={"type":"enabled","budget_tokens":2048}` until first edit, then switch to plain LLM. Defensive fallback for older anthropic SDKs that don't accept the kwarg.
- **P2 Dynamic context refresh** every 5 tool calls post-first-edit — short HumanMessage with current phase, files modified (git diff --stat), last test result, anti-pattern hints. Mirrors Claude Code's queryContext-per-turn pattern.
- **P3 Diminishing-returns stuck detection** — every 4 calls, compare last 3 deltas; if <500 tokens added AND 0 new edits/tests for 3 consecutive checks, inject REPLAN nudge with the current plan quoted, forcing the agent to either revise or commit.
- **P4 Per-edit rollback** via `undo_last_edit` tool — string_replace and create_file snapshot file content before writing; agent can surgically undo the most recent edit (restore prior content, or delete newly-created file). 17 unit tests.

### P1 — SDK dep-compat fixes for Flask/Werkzeug/Rich
**Completed:** v3.3 (2026-04-13)
`repo_manager.py`: dep-pinning matrix + test-dep installs for SWE-bench repos.
- Jinja2 Markup-gone detection → pin jinja2<3.1 + itsdangerous<2.1 + markupsafe<2.1
- Flask 2.0 → werkzeug>=2.0.0,<2.1 (as_tuple preserved)
- Flask 2.1/2.2 → werkzeug>=2.0.3,<2.3 (url_parse preserved)
- Werkzeug repo → install ephemeral_port_reserve + pytest-xprocess
- Rich repo → install attrs (for test_pretty.py)
Sentinel went from 1/5 → 3/5 pass; avg cost $0.79 → $0.38 (52% reduction); avg calls 32 → 14 (56% reduction).

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
