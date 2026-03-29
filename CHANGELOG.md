# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
