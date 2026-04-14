# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [3.5.0] - 2026-04-13

### Added
- Concept-to-code mapping: `query_concept_to_code()` bridges business-language tickets to code via BusinessRule graph nodes.
- Hardened verifier prompt with anti-rationalization gate: APPROVE without adversarial-probe evidence auto-downgraded to REJECT.
- Autonomous Plan Mode: `produce_plan` tool gates `create_sandbox`; verifier cross-checks plan vs diff.
- Cache-friendly microcompact: in-place idempotent tool-result eviction preserving prompt-cache prefix bytes.
- Forked subagent infrastructure: `CacheSafeParams` + thread-local slot for cache-preserving subordinate LLM calls.
- Explore subagent: Haiku-backed read-only `delegate_explore` tool for parallel codebase search.
- WebFetch + WebSearch tools (disabled by default, `ENABLE_WEB_TOOLS=1`).
- Extended thinking on early turns (pre-first-edit); switches to fast LLM after. Configurable via `REACT_THINKING_BUDGET`.
- Dynamic context refresh every 5 tool calls (git diff stat, test status, anti-pattern hints).
- Diminishing-returns stuck detection with auto-replan nudge (not hard escalation).
- Per-edit rollback: `undo_last_edit` tool with file-content snapshots.
- Learn-from-fix: per-repo `agent_lessons.md` persists across runs; next run reads 5 most recent.
- FAIL_TO_PASS test injection into system prompt ("run these LAST before submit_fix").
- SWE-bench dep-compat for 10+ repo families (Django, Flask, Werkzeug, Rich, Matplotlib, Xarray, Requests, Sympy, Sklearn, Sphinx, Astropy, Pytest).
- Eval dataset expanded to 35 bugs (25 single-file + 10 multi-file from SWE-bench-Verified).
- 159 new unit tests (678 → 837 total).

### Changed
- Sentinel pass rate: 12% → 80% (mission target hit).
- Cost/bug: $0.79 → $0.37 (53% reduction).
- Tool calls/bug: 32 → 16 (50% reduction).
- Localization: 88% → 100%.
- CLAUDE.md rewritten following best practices (< 80 lines, progressive disclosure to agent_docs/).
- Stale root docs (PLAN.md, AGENTS.md, GEMINI.md, OUTPUT.md) removed or moved to agent_docs/.
- Anthropic SDK upgraded from 0.40.0 to 0.94.0 (enables extended thinking).

## [0.3.0.0] - 2026-04-01

### Added

- Unified eval package (`backend/agent/eval/`) with A/B pipeline comparison, 11 scoring metrics, regression gates, and GitHub PR review tracking.
- Context window management (`context_manager.py`) with 3-layer strategy: per-tool output caps, observation masking (15-turn window), and Haiku summarization safety net (120K trigger).
- `record_localization` tool for explicit fault location tracking in ReAct pipeline.
- Multi-file coordination tools: `get_callers` and `get_blast_radius` query knowledge graph for dependent files.
- CLI eval subgroup: `python cli.py eval run/curate/report/gate/track-prs`.
- 25-bug eval dataset expanded from SWE-bench Lite (9 repos: Django, Sympy, Requests, Pytest, Scikit-learn, Sphinx, Matplotlib, Astropy, xarray).
- 32 tests for the new eval package.
- Anthropic prompt caching on system prompt (~87% savings on static prefix across 30+ LLM calls).
- Tool call budget system with per-phase allocation (explore: 6-10, edit: 3-5, verify: 3-5).
- Anti-pattern detection: advisory warnings at 8+ greps, 10+ reads, 3+ tests, 4+ edits, 2+ reviews.
- Ground truth file matching in scoring (precision/recall/F1 against expected_patch_files).
- Context window usage logging every 10 tool calls.

### Fixed

- ReAct agent no longer wastes tool calls on absolute sandbox paths (auto-strips prefixes).
- Diff-scoped linting: only flags errors on lines the agent changed, not pre-existing repo issues.
- Pytest exit code 4 correctly classified as USAGE_ERROR (not "no tests collected").
- Pytest exit codes 2 (interrupted) and 3 (internal error) return "error:", not "failed:".
- `run_tests` exception no longer falls through to "passed" (was P0 bug).
- `run_tests(test_path)` now routes through `sandbox.run_tests` with full config support.
- `submit_fix` only marks terminal success when tool returns "OK:" (not on errors).
- `submit_fix` checks for agent-created commits vs base branch (not HEAD~1 false positive).
- `submit_fix` returns ERROR if no changes exist to commit.
- Guardrails accept "error" as valid test attempt (prompt/guardrail contract aligned).
- `full_pass` scoring requires `patch_hits_target` (not just `review_approved` — removes self-review bias).
- Metrics recording added to ReAct pipeline (was missing, dashboard would go stale).

### Changed

- ReAct is now the default pipeline everywhere (CLI, API, eval). Use `--no-react` for fixed pipeline.
- MAX_TOOL_CALLS reduced from 60 to 40 (no success ever hit 60).
- `grep_repo` default max_results reduced from 25 to 10.
- `read_file` default window: 80 lines. Prompt strongly prefers `read_function` over `read_file`.
- ReAct eval cost reduced from $2-4 to ~$0.11 per simple bug (prompt caching + tool efficiency).

## [0.2.0.0] - 2026-04-01

### Added

- ReAct agent pipeline as alternative to fixed 8-node LangGraph pipeline. Single agent loop where the LLM decides: explore, localize, edit, test, review, submit.
- 8 new sandbox-aware tools: string_replace, check_syntax, create_file, create_sandbox, run_tests, request_review, submit_fix, escalate.
- Safety guardrails: sandbox gate, submit gate, $5 cost cap, 60 tool call cap, 15-minute timeout.
- `python cli.py fix --react` command to run bugs through the ReAct pipeline.
- `--react` flag on eval runner for A/B comparison between pipelines.
- ReactAgentState TypedDict for the new pipeline's state management.

### Changed

- Eval runner now supports both fixed and ReAct pipelines via `--react` flag.

## [0.1.2.2] - 2026-03-31

### Changed

- Backend uvicorn now runs with `--reload` flag so code changes via Docker volume mount are picked up automatically without container restart.

### Added

- Debug logging for branch reuse check in `test_node` to diagnose retry iteration behavior.

## [0.1.2.1] - 2026-03-31

### Fixed

- Aggressive worktree cleanup on retry: scans ALL worktrees for the ticket (not just the computed path), prunes stale refs, and force-removes directories before recreating the branch. Fixes edge case where the prior iteration's worktree at a different path survived cleanup and caused a second branch to be created.

## [0.1.2.0] - 2026-03-31

### Fixed

- Agent pipeline now reuses a single branch per ticket across retry iterations instead of creating a new branch on each attempt. Previously, 3 review iterations produced 3 orphan branches (`fix/ticket-abc`, `fix/ticket-def`, `fix/ticket-ghi`). Now the same branch is reset and repatched on each retry, keeping the repo clean and the final PR on one branch.

### Tests

- Added `TestBranchReuseOnRetry` test class with 2 tests verifying branch reuse behavior and confirming no extra branches are created during retry iterations.

## [0.1.1.0] - 2026-03-29

### Security

- `sandbox.py`: Replaced `shell=True` with `shlex.split()` in `subprocess.run` for setup commands — eliminates shell injection risk when running user-supplied setup commands

### Fixed

- `pipeline.py`: Filter test files (`test_`, `conftest`, `/tests/`, `/test/`, `/__pycache__/`) from caller discovery results in both `_find_callers_from_graph` and `_find_callers_via_grep` — prevents noisy test-file callers from polluting the repair context
- `pipeline.py`: Deduplicate patches again after `_verify_and_fix_patches` — retry merge can reintroduce duplicate patches that were already removed pre-verify
- `pipeline.py` (`escalate_node`): Emit explicit `AGENT_DECLINED` ERROR-level log and dedicated `"escalation"` trace event with `ticket_id`, `iterations`, `reason` — fixes BUG-6 silent skip where escalation produced no external observable signal

### Tests

- Added 18 new tests in `backend/tests/test_phase2a_fixes.py` covering all four Phase 2A fixes: sandbox shlex tokenization, caller graph filter, caller grep filter, and escalate_node signal instrumentation

## [0.1.0.0] - 2026-03-01

### Added

- Initial AI Deploy Agent pipeline (LangGraph orchestration)
- Knowledge graph construction (Tree-sitter → Neo4j)
- Dashboard (React + FastAPI + force-directed graph)
- Sandbox test execution with worktree isolation
- PR creation pipeline
