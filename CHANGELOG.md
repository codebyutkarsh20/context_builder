# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0.0] - 2026-04-01

### Added

- Unified eval package (`backend/agent/eval/`) with A/B pipeline comparison, 11 scoring metrics, regression gates, and GitHub PR review tracking.
- Context window management (`context_manager.py`) with 3-layer strategy: per-tool output caps, observation masking, and Haiku summarization safety net.
- `record_localization` tool for explicit fault location tracking in ReAct pipeline.
- CLI eval subgroup: `python cli.py eval run/curate/report/gate/track-prs`.
- 32 tests for the new eval package.

### Fixed

- ReAct agent no longer wastes tool calls on absolute sandbox paths (auto-strips prefixes).
- Diff-scoped linting: only flags errors on lines the agent changed, not pre-existing repo issues.
- Pytest exit codes 4/5 (no tests collected) correctly return "skipped" instead of "failed".
- `submit_fix` guardrail bypass: blocked calls no longer set `state["submitted"] = True`.

### Changed

- ReAct eval cost reduced from $2-4 to ~$1.40 per bug (64% reduction) via prompt tuning + context management.

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
