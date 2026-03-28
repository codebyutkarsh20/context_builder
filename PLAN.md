<!-- /autoplan restore point: /Users/utkarshpatidar/.gstack/projects//main-autoplan-restore-20260329-012905.md -->
# Plan: Agent E2E Reliability — Phase 2 & Beyond

**Project:** AI Deploy Agent (context_builder)
**Branch:** main
**Date:** 2026-03-29
**Author:** Utkarsh Patidar

---

## Context

Previous plan (2026-03-27) targeted 3 failure modes blocking E2E success:
1. Patch application failure (LLM-generated `original_code` too short to fuzzy-match)
2. Test generation ignored by repair LLM
3. No multi-file coordination

**Today's status (2026-03-29):**
- E2E pipeline proven on taskflow-api (5 bugs, 20/20 tests pass, PR auto-created on GitHub)
- Infra fixes landed: git identity in sandbox, pip install before tests, rw volume mount, GH_TOKEN + gh CLI in container
- Patch application: working (fuzzy match succeeds on well-formed LLM patches)
- PR creation: fully autonomous end-to-end
- Single-file fixes: 97% confidence localization, 96% Opus review approval, 1 iteration

**What remains broken or unproven:**
1. Multi-file coordination (untested — taskflow-api bugs were all single-file)
2. Eval suite (no real-world bug dataset, no regression tracking)
3. Test generation enforcement (reviewer feedback loop not validated on real escalation)
4. BUG-6 (overdue tasks) was skipped by agent — only 4/5 bugs fixed in isolation runs
5. No GitHub Actions CI — test pass in sandbox != pass in CI

---

## Problem Statement

The agent can now fix simple single-file Python bugs autonomously and create PRs. But Milestone 1's target is **80% human-approved PRs** — not 80% E2E completion. We don't know if the patches are correct or if the PR would be approved without manual inspection.

Three gaps remain:

### Gap 1: Multi-File Coordination
Real bugs often span multiple files. No mechanism today:
- Caller files are identified via blast radius but not included in repair context
- Rename a function → callers break silently, tests pass in sandbox, fail in CI

### Gap 2: No Eval Suite = No Confidence
We tested on taskflow-api which we wrote ourselves. We know the bugs and answers. A real eval requires:
- ~20 real bugs from open-source Python repos (GitHub issues + fix PRs)
- Automated scoring: localization accuracy, patch quality, test pass, PR human approval
- Regression gate: pipeline changes must pass eval

### Gap 3: Jira Integration
Current intake is a mock (JSON files). Real milestone requires reading from Jira, not a form in the dashboard.

---

## Proposed Plan

### Phase 2A: Prerequisites (before any new capability)

**2A-0: BUG-6 root cause** (blocking)
Reproduce the silent-skip deterministically. Instrument an explicit escalation signal that fires every time the agent declines a bug. Add Jira comment write-back for skips.

**2A-1: Fix sandbox security** (blocking)
Replace `shell=True` → `shlex.split` in `sandbox.py:72`. Add symlink guard in `_find_file_in_repo`. Add Docker container isolation for eval execution (`--network none`, CPU/mem limits).

**2A-2: Fix same-file dedup after retry merge**
10-line fix in `pipeline.py:1817` — run `_deduplicate_patches` on `verified` list after retry merge.

**2A-3: Filter test files from caller lists**
Add `_noise_patterns` filter to `_find_callers_via_grep` and `_find_callers_from_graph` return values.

### Phase 2B: Multi-File Bug Fix (1 week CC) ← was 2A, moved up per gate

**2B-1: Blast radius in repair context**
After localization, pull all callers/importers of fault functions from Neo4j or grep. Include their full content in the repair prompt as "FILES THAT MAY NEED UPDATING."

**2B-2: Multi-file coordinator node**
Add a LangGraph node after repair that checks: for each caller file surfaced by blast radius, does `repair["patches"]` contain an entry? If not, re-run repair targeted at that caller.

**2B-3: Reviewer caller completeness check**
Extend reviewer prompt to validate caller files are patched, not only `fault_functions`.

**2B-4: Graph staleness check**
Before any repair run, check if graph was built within N commits. If stale, trigger rebuild.

**2B-5: Integration test**
Create a 2-file bug in taskflow-api: rename a service function + update the API router. Verify agent patches both files.

### Phase 2C: Eval Suite (1 week CC) ← moved after multi-file per gate override

**2C-1: Eval schema with multi-file support first**
Add `expected_patch_files` field and `multi_file_complete` metric to `eval_suite.py` before collecting dataset.

**2C-2: Dataset collection**
Collect 20 real Python bugs: 12 single-file + 8 multi-file. For each: issue text, repo URL + SHA, `expected_patch_files`, fix SHA. Use repos with passing test suites only. Run inside Docker.

**2C-3: Eval runner with containerization**
`python -m agent.eval.run --dataset eval/bugs.json` — clone inside Docker (`--network none`), score: localization hit, `multi_file_complete`, tests pass.

**2C-4: Regression gate**
Every pipeline.py change must pass eval. Track in `data/eval_history.json`. Alert on >5% drop.

### Phase 2D: Jira Integration (3 days CC) ← moved to last

**2D-1: Real Jira API connector**
Replace mock_jira.py with actual Jira REST API client. Read ticket by ID, parse description, attachments, comments.

**2C-2: Webhook intake**
FastAPI endpoint that receives Jira webhook on ticket creation/update. Trigger pipeline automatically.

**2C-3: Write-back**
After PR created, post comment on Jira ticket with PR URL + agent confidence score.

---

## Success Criteria

- [ ] Multi-file bug fix: 2-file bug in taskflow-api fixed autonomously
- [ ] Eval suite: 20 real bugs, score tracked baseline established
- [ ] E2E approval rate: ≥70% of eval PRs approved without changes (target: 80%)
- [ ] Jira round-trip: ticket in → PR created → comment written back to Jira

## Effort

| Phase | Human team | CC+gstack | Priority |
|-------|-----------|-----------|----------|
| 2A: Prerequisites (BUG-6 + security + dedup fixes) | 1 day | 2h | P0-BLOCKING |
| 2B: Multi-file coordination | 2 weeks | 3-4 hours | P0 |
| 2C: Eval suite (after multi-file) | 1 week | 2-3 hours | P0 |
| 2D: Jira | 3 days | 1-2 hours | P1 |

## Risks

1. **Graph coverage** — Multi-file blast radius requires indexed Neo4j graph. Repos without indexed graphs fall back to grep. Grep misses dynamic imports.
2. **Eval flakiness** — Open-source repos' tests may be flaky or slow. Needs timeout and retry.
3. **Jira auth** — Jira Cloud uses OAuth; self-hosted uses basic auth. Need to handle both.

## Not In Scope

- Non-Python languages (TypeScript, Go)
- Security vulnerability fixes
- Database migration changes
- Infrastructure/deployment changes
- New features (only bug fixes)

---

# CEO REVIEW (Phase 1 — /autoplan)

## 0A: Premise Challenge

| Premise | Status | Risk |
|---------|--------|------|
| E2E pipeline proven on single-file bugs | CONFIRMED (ran today) | None |
| Next bottleneck is multi-file + eval | CONFIRMED (user) | Medium — voices disagree on sequencing |
| 80% human-approved PRs = Milestone 1 success | CHALLENGED | Critical — unmeasurable on real bugs |
| Eval-first sequencing is correct | CHALLENGED | High — both voices say capability (multi-file) first |
| Jira integration is P1 (last) | CHALLENGED | Medium — gates real-world feedback |
| Neo4j graph is populated for target repos | ASSUMED, NOT CONFIRMED | High — graph build pipeline not complete |

## 0B: Existing Code Leverage

| Sub-problem | Existing code |
|-------------|---------------|
| Eval runner | `backend/agent/eval_suite.py` (13KB) — exists, needs dataset |
| Blast radius | `backend/agent/explore_tools.py` grep_repo + Neo4j queries |
| Multi-file patches | `backend/agent/pipeline.py` already applies N patches per job |
| Jira connector | `backend/agent/intake/mock_jira.py` — replace with REST client |
| Graph build | `backend/graph/builder.py` — exists, not wired to auto-run |

## 0C: Dream State Delta

```
NOW (2026-03-29):
  Single-file Python bugs → PR created autonomously
  Confidence: 97% localization, 96% Opus review
  Eval: taskflow-api only (we wrote the bugs ourselves)
  Intake: manual dashboard form

THIS PLAN:
  Multi-file Python bugs → PR created
  Eval: 20 real open-source bugs, scored
  Intake: Jira webhook

12-MONTH IDEAL (100 deploys/day):
  Any Python repo, any bug type → PR + canary deploy
  Real-time Jira round-trip
  Post-merge regression tracking
  Self-improving: failure patterns feed back into graph
```

**Delta gap:** This plan closes ~40% of the distance to the 12-month ideal.
Missing from plan: canary deploy, post-merge regression tracking, self-improvement loop.

## CODEX SAYS (CEO — strategy challenge)

Key findings:
- **Metric trap** (Critical): 80% approval rate optimizes for social engineering, not correctness. Need post-merge defect rate + rollback frequency.
- **Sequencing risk** (High): Eval-first benchmarks a toy agent. Capability-first, then measure.
- **Framework unknowns** (High): Novel stacks = dead stop without explicit cold-start playbook.
- **Competitive moat** (High): No differentiated answer vs. Devin/OpenHands/SWE-agent.
- **6-month regret**: "Green" eval scores on toy bugs while real customer performance is 40%.

## CLAUDE SUBAGENT (CEO — strategic independence)

Key findings (10 total):
- **Wrong unit of value** (Critical): Bugs fixed ≠ engineer hours saved. Reframe.
- **80% metric unmeasurable** (Critical): Only measurable at scale on real tickets with real reviewers.
- **BUG-6 silent skip** (High): Silent failure in autonomous agent = production safety issue. Must have root cause BEFORE Phase 2.
- **Neo4j graph not populated** (High): Multi-file blast radius requires indexed graph. Graph build pipeline not complete — blocking dependency not in plan.
- **Jira gates real feedback** (Medium): Move to P0 parallel with eval.
- **Opus self-review is circular** (High): Remove from success dashboard.

## CEO DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Premises valid?                   PARTIAL PARTIAL DISAGREE
  2. Right problem to solve?           PARTIAL YES    DISAGREE → TASTE
  3. Scope calibration correct?        NO      NO     CONFIRMED: too narrow
  4. Alternatives sufficiently explored? NO    NO     CONFIRMED: missing competitive audit
  5. Competitive/market risks covered? NO      NO     CONFIRMED: no analysis at all
  6. 6-month trajectory sound?         NO      NO     CONFIRMED: benchmark overfitting risk
═══════════════════════════════════════════════════════════════
```

**TASTE DECISIONS from CEO:**
- TD-1: Eval vs multi-file sequencing (user chose eval-first; both voices say capability-first)
- TD-2: Python-only depth vs polyglot breadth (unresolved product decision)

## Error & Rescue Registry

| Error scenario | Impact | Rescue |
|---------------|--------|--------|
| Agent silent-skips a bug (BUG-6 pattern) | Silent failure, bug stays open | Explicit escalation signal + Jira comment |
| Graph stale > N commits | Wrong blast radius → missed callers | Staleness check before every repair run |
| Eval dataset not representative | Metric looks good, prod fails | Add synthetic taskflow-api bugs + real Jira tickets |
| Opus approves bad patch (self-referential) | Bad code merged | Replace with test coverage delta as quality signal |
| Jira OAuth fails mid-run | Pipeline stuck | Graceful degradation to mock intake, alert |

## Failure Modes Registry

| Mode | Likelihood | Severity | Mitigation |
|------|-----------|---------|-----------|
| Benchmark overfitting | High | Critical | Real Jira project by month 2 |
| Multi-file missed caller | Medium | High | Graph staleness check |
| Silent skip (BUG-6) | Medium | High | Escalation signal, root cause first |
| Eval dataset not representative | High | High | Synthetic bugs + real tickets mix |
| Competitive commoditization | High | Medium | Identify moat now (telemetry? integrations?) |

## NOT In Scope (deferred to TODOS.md)

- Canary deploy integration
- Post-merge regression tracking
- Self-improvement feedback loop
- Non-Python languages
- Competitive audit vs Devin/OpenHands (deferred but should happen in week 1)
- GitHub Actions CI gate

## CEO Completion Summary

| Section | Status | Key finding |
|---------|--------|-------------|
| Premises | PARTIAL | 2 critical challenges (metric, sequencing) |
| Scope | NEEDS WORK | Graph build pipeline missing from plan |
| Alternatives | MISSING | No competitive analysis at all |
| Risks | PARTIAL | BUG-6 silent skip not root-caused |
| Sequencing | CHALLENGED | Both voices say multi-file before eval |
| Metric | WRONG | 80% approval rate is not the right KPI |

**PHASE 1 COMPLETE.**
Codex: 5 concerns. Claude subagent: 10 findings.
Consensus: 4/6 confirmed, 2 disagreements → surfaced at gate.
No UI scope → skipping Design phase. Passing to Eng Review.

---

## Decision Audit Trail

| # | Phase | Decision | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO | Keep plan scope (no expansion beyond Python) | P3 Pragmatic | Python-first gives faster iteration; polyglot adds complexity without validating core | Polyglot from day 1 |
| 2 | CEO | Add BUG-6 root cause as Phase 2 prerequisite | P1 Completeness | Silent failure in autonomous agent is safety-critical; cannot ship without fix | Defer to later |
| 3 | CEO | Add graph staleness check as Phase 2A prerequisite | P1 Completeness | Multi-file blast radius on stale graph is worse than no blast radius | Accept stale graph |
| 4 | CEO | Move competitive audit to week 1 (parallel, not blocking) | P6 Bias to action | Need to know moat before building 3 more months | Block on it |

---

# ENG REVIEW (Phase 3 — /autoplan)

## Architecture ASCII Diagram

```
Current pipeline.py (monolith):
┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐
│  intake  │──▶│ localize   │──▶│ read_source  │──▶│   repair   │
│  (Haiku) │   │ (Sonnet)   │   │ (fault files │   │  (Sonnet)  │
└──────────┘   └────────────┘   │ + callers)   │   └─────┬──────┘
                                │              │         │
                     MISSING ──▶│ No coordinator         │ flat patch list
                     multi-file │ node here    │         ▼
                     gate here  └──────────────┘   ┌────────────┐
                                                   │  reviewer  │
                                                   │  (Opus)    │
                                                   └─────┬──────┘
                                                         │
                                          MISSING: no ◀─┘
                                          caller completeness
                                          check in reviewer
                                                         ▼
                                                   ┌────────────┐
                                                   │  sandbox   │
                                                   │  test+PR   │
                                                   └────────────┘

After Phase 2A (proposed):
  + multi_file_coordinator node after repair
  + caller completeness check in reviewer prompt
  + dedup after retry merge path (pipeline.py:1817)
  + test-file filter in _find_callers_from_graph

Eval runner (new module needed — eval_suite.py is incomplete):
  eval_suite.py (read bugs.json) ──▶ repo_provisioner (clone + checkout SHA)
      ──▶ env_builder (venv + deps) ──▶ pipeline.run_ticket()
      ──▶ scorer (localization + patch + multi_file_complete + test_pass)
      ──▶ regression_tracker (compare vs baseline, alert on >5% drop)
  All inside: Docker container (--network none, CPU/mem limits)
```

## CODEX SAYS (Eng — architecture challenge)

- **Monolithic pipeline + no coordinator node** (High): Multi-file edits will race, re-localization doesn't run after each patch, caller completeness not checked
- **Sandbox integrity breaks on multi-file** (High): No atomic rollback if patch N+1 fails — worktree left dirty across retries
- **N+1 blast radius queries** (Medium): `_load_graph_data` called per repair, no caching, O(E) edge scan on every call
- **Eval runner incomplete** (High): `eval_suite.py` doesn't clone repos, checkout SHAs, or compare against ground-truth diff
- **Security: arbitrary repo test execution** (Critical): `setup_commands` with `shell=True`, no container isolation, symlink traversal via `rglob`

## CLAUDE SUBAGENT (Eng — independent review)

Key findings with specific file locations:
- **No multi-file coordinator node** (High): `pipeline.py:1958-1994` reviewer checks `fault_functions` only, not callers. `_build_source_section:1251-1293` truncates callers to 400 lines.
- **No `all_files_patched` eval metric** (High): `eval_suite.py:104-116` has no `expected_patch_files` field — multi-file fixes score as "fixed" even if callers aren't patched
- **Same-file conflict after retry not deduped** (High): `pipeline.py:1341-1378` retry merges new patches into `verified` but `_deduplicate_patches` at line 1805 only runs on `raw_patches` — not on the merged result
- **Test files appear as callers** (High): `pipeline.py:986-1019` — `_find_callers_via_grep` doesn't filter `_noise_patterns`, test files get patched via main patch path
- **`_load_graph_data` bypasses JSON cache** (Medium): `pipeline.py:919-935` — `explore_tools.py` has mtime-based cache but pipeline calls `_load_graph_data` directly
- **`shell=True` in setup_commands** (Critical): `sandbox.py:72`

## ENG DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Architecture sound?               NO      NO     CONFIRMED: missing coordinator node
  2. Test coverage sufficient?         NO      NO     CONFIRMED: no multi-file eval metric
  3. Performance risks addressed?      NO      NO     CONFIRMED: _load_graph_data not cached
  4. Security threats covered?         NO      NO     CONFIRMED: shell=True + no isolation
  5. Error paths handled?              NO      NO     CONFIRMED: dedup gap after retry merge
  6. Deployment risk manageable?       YES     YES    CONFIRMED: single container, reversible
═══════════════════════════════════════════════════════════════
5/6 confirmed — all critical/high findings have consensus.
```

## Critical Fixes Required Before Phase 2A Ships

| Priority | Fix | File | Effort |
|----------|-----|------|--------|
| P0-CRITICAL | Containerize eval test execution (Docker, `--network none`) | `sandbox.py`, `docker-compose.yml` | 4h |
| P0-HIGH | Dedup `verified` list after retry merge | `pipeline.py:1817` | 15min |
| P0-HIGH | Filter test/config files from caller lists | `pipeline.py:986-1019` | 30min |
| P0-HIGH | Add `expected_patch_files` + `multi_file_complete` metric to eval schema | `eval_suite.py:104-116` | 1h |
| P1-MEDIUM | Cache `_load_graph_data` with mtime invalidation | `pipeline.py:919-935` | 1h |
| P1-MEDIUM | Fix `shell=True` → `shlex.split` in setup_commands | `sandbox.py:72` | 15min |

## Test Plan

**Multi-file coordination tests (new):**
- `test_multi_file_patch_applies`: 2-file bug in taskflow-api (rename service fn + update router), verify both patched
- `test_caller_completeness_check`: reviewer rejects repair when caller file not patched
- `test_dedup_after_retry_merge`: same-file conflict after retry → best patch wins
- `test_test_file_not_in_callers`: test files filtered from blast radius results
- `test_graph_data_cached`: second `_load_graph_data` call uses cache, no disk read

**Eval runner tests (new):**
- `test_eval_runner_clones_repo`: provisions repo at SHA, verifies clean state
- `test_eval_multi_file_complete_metric`: `expected_patch_files` scoring works
- `test_eval_flaky_test_handling`: flaky test retried up to 3x, not scored as failure
- `test_eval_security_isolation`: subprocess in container, `--network none` verified

**Regression tests (must keep passing):**
- All 23 existing tests in `tests/`
- BUG-ALL-2 repro: 5 taskflow-api bugs, 20/20 tests, PR created

**Test plan artifact written to:** `~/.gstack/projects/context_builder/utkarshpatidar-main-eng-review-test-plan-20260329.md`

## NOT In Scope (Eng additions to CEO list)

- LangGraph node decomposition refactor (full monolith split) — too large, defer
- Prometheus/Grafana integration — deferred
- GitHub Actions CI gate for eval suite — defer to Phase 3

## Cross-Phase Themes

**Theme 1: Safety of autonomous action** — flagged in CEO (BUG-6 silent skip = production risk) AND Eng (shell=True + no container isolation = security risk). Both voices in both phases. High-confidence: any autonomous execution path must be hardened before expanding scope.

**Theme 2: Measurement before metrics** — flagged in CEO (80% approval rate unmeasurable) AND Eng (no `multi_file_complete` eval metric). The system needs better instrumentation of what "fixed" means before claiming progress.

**Theme 3: Sequencing of capability vs. measurement** — CEO both voices say multi-file before eval. Eng confirms: eval schema must be designed for multi-file before dataset is collected. These reinforce each other.

## Eng Completion Summary

| Section | Status | Key finding |
|---------|--------|-------------|
| Architecture | ISSUES | Missing multi-file coordinator node, caller completeness gap |
| Test coverage | MISSING | No multi-file eval metric, no `expected_patch_files` |
| Performance | PARTIAL | Graph data cache bypass — medium risk |
| Security | CRITICAL | `shell=True` + no container isolation for eval |
| Error paths | ISSUES | Dedup gap after retry merge, test files in callers |
| Deployment | CLEAN | Single container, all changes reversible |

**PHASE 3 COMPLETE.**
Codex: 5 concerns. Claude subagent: 5 findings, all specific with line numbers.
Consensus: 5/6 confirmed, 0 new taste decisions.

---

## Updated Decision Audit Trail

| # | Phase | Decision | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO | Keep plan scope (Python only) | P3 Pragmatic | Python-first gives faster iteration | Polyglot day 1 |
| 2 | CEO | BUG-6 root cause = Phase 2 prerequisite | P1 Completeness | Silent failure is safety-critical | Defer |
| 3 | CEO | Graph staleness check = Phase 2A prerequisite | P1 Completeness | Stale graph → wrong blast radius | Accept stale |
| 4 | CEO | Competitive audit week 1 (parallel) | P6 Bias to action | Need moat before building 3 months | Block on it |
| 5 | Eng | Containerize eval execution = P0 before Phase 2B | P1 Completeness | shell=True + no isolation = RCE risk | Accept risk |
| 6 | Eng | Fix dedup after retry merge before shipping | P5 Explicit | 10-line fix, prevents subtle multi-file corruption | Defer |
| 7 | Eng | Design eval schema (expected_patch_files) before collecting dataset | P1 Completeness | Schema change after collection invalidates all data | Collect first |
| 8 | Eng | TASTE: sequencing — multi-file before eval (both voices) | P1 Completeness | User overrode to multi-file first at gate | Eval first (original) |
| 9 | Gate | Containerize eval execution before dataset collection | P1 Completeness | User confirmed: security before scope | Accept risk |

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/autoplan` | Strategy + scope | 1 | issues_open | 2 critical (metric, sequencing), 4 high |
| Codex CEO Voice | `/autoplan` | Independent strategic challenge | 1 | issues_open | 5 concerns: metric trap, sequencing, moat, cold-start, 6mo regret |
| Claude Subagent CEO | `/autoplan` | Independent strategic review | 1 | issues_open | 10 findings, 2 critical |
| Eng Review | `/autoplan` | Architecture + tests + security | 1 | issues_open | 5 findings: 1 critical, 4 high |
| Codex Eng Voice | `/autoplan` | Architecture challenge | 1 | issues_open | 5 concerns: coordinator node, sandbox integrity, N+1, eval incomplete, RCE |
| Claude Subagent Eng | `/autoplan` | Independent eng review | 1 | issues_open | 5 findings with specific line numbers |
| Design Review | skipped | No UI scope (1 match, need 2+) | 0 | — | — |

**VERDICT:** APPROVED WITH OVERRIDES
- Sequencing changed: multi-file BEFORE eval (unanimous voice recommendation, user confirmed at gate)
- Security gate added: containerize eval execution before dataset collection (user confirmed)
- 2 critical issues flagged for immediate action: (1) BUG-6 silent skip must be root-caused before Phase 2B, (2) `shell=True` + no container isolation in sandbox.py must be fixed before eval
- Run `/ship` when prerequisites (Phase 2A) are complete
