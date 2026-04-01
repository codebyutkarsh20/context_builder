# TODOS

## Active

### P1 — Multi-candidate patch sampling
**What:** Generate 3-5 candidate patches per bug, test all, pick the one that passes. Based on SWE-bench research showing 3-8% gain from ensembling.
**Why:** Current pipeline generates 1 patch. If it's wrong, the agent spirals trying to fix it. Multiple candidates + test filtering is the cheapest quality improvement.
**Context:** SWE-bench research (April 2026). Agentless uses 42 candidates.

### P1 — Expand eval dataset with multi-file bugs
**What:** Add 8-10 multi-file bugs from SWE-bench. Current 25 bugs are all single-file.
**Why:** Real production bugs are ~50% multi-file. We have get_callers/get_blast_radius tools but no eval data to test them.

### P1 — Real Jira integration
**What:** Connect to real Jira board, pull tickets, run agent, track PR review outcomes.
**Why:** The 80% approval target is unmeasurable without real human reviewers on real bugs.

### P2 — Continue extracting utilities from pipeline.py
**What:** pipeline.py is still 3300 lines. Utility functions (_redact_secrets, _fuzzy_match_replace, should_iterate, etc.) are still imported by tests. Continue extracting into focused modules (llm.py, graph_utils.py, linters.py pattern).
**Why:** Fixed pipeline is retired (run_ticket/build_agent_graph deprecated) but the file remains as a utility library. Smaller modules = faster imports, easier testing.

## Active (inherited)

### P2 — Validate ENFORCED_BY edge precision before production use
**What:** Run `BusinessLogicExtractor` on 20 manually annotated functions, measure precision/recall of ENFORCED_BY edges. Gate business context injection on >80% precision.
**Why:** The extractor links rules to functions by docstring keyword matching. If precision is low (<80%), repair prompts get noisy rule context that degrades fix quality (-3% per research findings).
**Context:** Open question #1 in design doc. Do this validation after Phase 1 ships, before Phase 3 is wired to the repair pipeline.
**Depends on:** Phase 1 shipped (ENFORCED_BY edges exist in Neo4j).

### P3 — FailureRecord severity classification
**What:** Add severity signal to FailureRecords — distinguish P0 incidents from cosmetic fixes. Options: parse `P0`/`sev1`/`critical` keywords from commit message, fetch PR labels via GitHub API, or infer from Jira priority field.
**Why:** All FailureRecords currently have implicit severity=unknown. Repair agent can't weight past incidents by urgency. A P0 production outage in the blast radius should trigger stricter constraints than a cosmetic fix.
**Context:** Open question #3 in design doc. `mine_failure_records()` in `backend/graph/business/failure_records.py`.
**Depends on:** Phase 2 shipped.

### P4 — Approach C: Forward-Looking Structured History (Lore protocol)
**What:** After 5-10 agent PRs are merged, implement agent-written git trailers on every PR: rules checked, blast radius, failure pattern ID. Git becomes the FailureRecords DB over time (arxiv 2603.15566).
**Why:** FailureRecords mined from git history are retrospective and limited by commit message quality. Agent-authored trailers create forward-looking, structured history with full agent context attached to each change.
**Context:** Explicitly deferred (Approach C) in design doc — cold start requires 5-10 real agent PRs to produce useful data. Completeness 7/10 vs Approach B's 9/10 because of cold start.
**Depends on:** Wedge demo shipped + 5-10 real agent PRs merged via Phase 3.

### P5 — Extract external side effects behind interface (dependency inversion)
**What:** `pr_creation_node()` has multiple external side effects (git push, gh pr create, feature flags, _enrich_from_fix) each guarded by `if not dry_run`. Extract behind a thin interface and no-op in test/eval mode.
**Why:** The boolean-flag approach works for 3-4 calls but creates maintenance burden as new side effects are added. Forgetting a guard is a silent bug.
**Context:** Identified by outside voice during eng review (2026-03-30). Current dry_run guards are: git push, feature flag creation, PR creation, enrichment. Each new external call needs a manual guard. Standard dependency inversion pattern.
**Depends on:** Production validation shipped (current sprint).

## Completed

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

