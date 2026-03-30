# TODOS

## Active

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

### P1 — Jira project prefix must be configurable
**Completed:** Already implemented (discovered 2026-03-30 during eng review)
`failure_records.py:37`: `JIRA_PROJECT_PREFIX` env var with default `PROJ`. Tests at `test_failure_records.py:54`.

### P0 — Bug: `repo_path` missing from `/api/repos` response
**Completed:** v0.1.1.0 (2026-03-29)
`api/repos.py`: initialize `repo_path: ""` in base entry and always assign from stats — field now present on all repos regardless of whether graph.json exists.

