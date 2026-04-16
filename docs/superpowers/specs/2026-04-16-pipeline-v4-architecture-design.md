# Pipeline v4 Architecture Design

**Date**: 2026-04-16
**Status**: Approved
**Author**: Utkarsh Patidar + Claude

## Summary

Redesign the agent pipeline from 5 sequential stages to 3 stages (parallel setup, free-form react loop, lightweight finalize). Move BRT and verification inside the loop as agent-controlled tools. Simplify the prompt from 270 lines of static rules to 80 lines of contracts + rich dynamic context. Drop Opus from all stages. Target: higher pass rate, lower cost, faster convergence.

## Motivation

### Problems identified (15 total, 4 clusters)

**Cluster A — Wasted pre-loop work**: BRT runs blind before agent reads code. Scout, intake, and agent query the graph 3 times independently. Baseline tests run after 8-12 tool calls are already burned. 11 tools are defined but never exported.

**Cluster B — LLM overconstraint**: Prompt mandates a phase sequence the LLM must follow. 22 tools at ~100 tokens each waste ~50K tokens/bug on definitions. Plan-gate forces 8-call minimum even for trivial bugs. 270-line prompt is 80% instructions, 20% data.

**Cluster C — Cost inversion**: Extended thinking runs during exploration (mechanical) not editing (where reasoning matters). Opus ($15/M) is used for in-loop review. Full Sonnet context is sent for every grep/read call. Context compaction Layer 3 breaks prompt cache.

**Cluster D — Feedback gaps**: Verifier judges without seeing agent's reasoning. Test failures are truncated to 4000 chars with no structured breakdown. Baseline regression filtering is invisible to the agent. Retry mechanism is disabled.

### Research context

| Agent | SWE-bench Verified | Architecture |
|-------|--------------------|-------------|
| Mini-SWE-agent | >74% | 100 lines, bash only, no tools |
| Augment | 65.4% | Claude + o1 ensemble, K=8 patches |
| OpenHands CodeAct 2.1 | 53% | Event-stream, sandboxed bash |
| Moatless | 39% at $0.14/bug | Finite state machine, semantic search |
| Agentless | 32% at $0.70/bug | No agent, 3-phase pipeline |
| Our agent (current) | ~21% at $0.88/bug (bugs that ran) | 5-stage pipeline, 22 tools |

Key insight: the top agents either give the LLM maximum freedom (mini-SWE-agent) or have the system handle orchestration while the LLM reasons freely (Augment, OpenHands). Our current architecture does neither — it constrains the LLM AND does pre-work poorly.

## Architecture

### Pipeline: 3 stages

```
setup_node (parallel, ~8s) → react_agent_node (~60s) → finalize_node (~5s)
```

### Stage 1: setup_node (parallel, no Opus)

Three independent threads merge results into a rich dynamic block:

**Thread 1 — Repo detection + sandbox + baseline tests** (no LLM):
- Detect language, write `.agent_config.json`
- Create git worktree sandbox (agent starts with ready workspace)
- Run tests on original broken code, capture pre-existing failures
- Output: `{sandbox_path, baseline_failures: set[str]}`

**Thread 2 — Scout localization** (Haiku + Sonnet, ~$0.04):
- Haiku extractor: entities from bug description
- Sonnet debugger: file-level suspects from entities + repo file listing
- Haiku skeleton narrowing: function-level suspects from file skeletons
- Path validation + fuzzy recovery on all suspects
- No Opus re-ranker (dropped — agent re-prioritizes itself)
- Output: `{suspects, entity_extraction, skeleton_data, blast_radius, business_rules, reasoning}`

**Thread 3 — Context assembly** (no LLM):
- Build repo tree (top 200 source files, multi-language)
- Load graph data (callers, hotspots, communities)
- Load per-repo lessons (`agent_lessons.md`)
- Load concept-to-code mappings
- Output: `{repo_tree, graph_context, lessons, concept_mappings}`

**Merge**: All three outputs assembled into the dynamic block. If scout finds nothing, dynamic block includes fallback: "No confident matches. Start from repo structure. Use delegate_explore for orientation."

### Stage 2: react_agent_node (Sonnet, free-form)

**Prompt architecture — lean static, rich dynamic:**

Static block (~80 lines, cached across all calls):
- Identity + role (2 lines)
- Soft workflow guidance: "Explore, understand, edit, test, verify, submit. Adapt freely." (3 lines)
- Hard contracts: sandbox ready, test before submit, verify_fix before submit (6 lines)
- Test result interpretation: passed/failed/skipped/error domain conventions (8 lines)
- Pre-existing failures note (3 lines)
- Path convention with examples (5 lines)
- BRT guidance: call write_brt after understanding the code, before editing (4 lines)
- Planning guidance: produce_plan is optional, use for complex bugs (3 lines)
- Cost guidance: delegate_explore for broad search, direct tools for precise lookups (4 lines)
- run_shell: non-interactive, for env diagnosis/repair (4 lines)
- verify_fix: forks conversation, call before submit, adapt on rejection (4 lines)
- Changelog anchor: "Known issues (add here when evals reveal consistent failures)" (3 lines)

Dynamic block (rich per-bug context, ~150-200 lines):
- Bug: title, priority, component, description
- Target tests: FAIL_TO_PASS tests that must pass, PASS_TO_PASS tests that must stay passing, specific run command
- Scout analysis: entity extraction, suspected files with reasoning + skeletons, blast radius, business rules. Or fallback if scout found nothing.
- Baseline test results: specific pre-existing failures listed by name
- Repo structure: top 200 source files
- Code map: function signatures + line numbers for top-5 suspected files
- Lessons from past fixes in this repo
- Concept-to-code mappings

Task message (minimal):
```
Fix this bug. The context above has everything you need to start.
Focus on the target tests — when they pass, you're done.
```

**Extended thinking**:

```
Thinking OFF: default state (exploration is mechanical)
Thinking ON triggers:
  - First string_replace call (entering edit phase)
  - Any test failure (reasoning about what went wrong)
  - verify_fix REJECTION (reasoning about feedback)
Thinking OFF triggers:
  - None — once ON, stays ON for the rest of the run
```

The "stays ON once triggered" simplification avoids bidirectional switching complexity. The agent explores cheaply, then thinks deeply once it starts editing. Test failures and rejections are already in the "thinking ON" window.

**Tools (17):**

| Category | Tool | Notes |
|----------|------|-------|
| Exploration (6) | `read_file` | Direct, precise |
| | `grep_repo` | Find where symbols live |
| | `get_file_structure` | Signatures + line numbers |
| | `list_files` | Directory listing |
| | `read_function` | Full function extraction |
| | `get_callers` | Who calls/imports this code |
| Cheap exploration (1) | `delegate_explore` | Haiku subagent for multi-step search |
| Editing (3) | `string_replace` | Auto-runs syntax check + ruff after each edit |
| | `create_file` | New files |
| | `undo_last_edit` | Surgical revert of last edit |
| Planning (1) | `produce_plan` | Optional, no gate |
| Testing (3) | `run_tests` | Structured output with new-vs-preexisting breakdown |
| | `run_shell` | Non-interactive shell for env diagnosis/repair |
| | `write_brt` | Agent-controlled BRT generation |
| Completion (3) | `verify_fix` | Forked subagent verification |
| | `submit_fix` | Commit + PR preparation |
| | `escalate` | Hand off to human |

**Guardrails (soft, not phase-mandated):**

Kept:
- Sandbox ready (setup created it — no gate needed, always true)
- Nudge: test before submit
- Nudge: verify_fix before submit
- Nudge: delegate_explore for multi-step search (cost guidance)
- Nudge: run_shell count >= 6 → "env likely unfixable, submit anyway"
- Localization: auto-inferred from all edited non-test files across session

Removed:
- Plan-gate (produce_plan before create_sandbox) — plan is optional now
- Sandbox-gate (create_sandbox before edit tools) — sandbox created by setup
- Read-before-edit gate (read_file before string_replace) — was WARNING-only, unnecessary
- Review-before-submit gate (request_review before submit_fix) — replaced by verify_fix nudge
- Grep count warning at 8 — let the LLM decide exploration strategy
- Run_tests retry warning at 3 — let the LLM decide

### Stage 3: finalize_node (no LLM)

- Create PR / commit (if submitted)
- Record lessons via learn_from_fix
- Cleanup sandbox
- Emit trace / metrics
- No retry logic (retries happen in-loop via verify_fix rejection)

## Tool Specifications

### verify_fix(explanation: str) -> str

Replaces both `request_review` (Opus, in-loop) and `verifier_node` (post-loop, disconnected).

**Mechanism**: Forked subagent using existing `CacheSafeParams`. Inherits the main agent's full message history as a read-only cached prefix. The forked Sonnet instance sees every exploration, rejected hypothesis, edit, and test result. Its reasoning stays in the fork and never enters the main agent's context. Returns ~200 tokens of structured verdict.

**Forked verifier task prompt**:
```
You are an independent reviewer. The conversation above shows
an agent fixing a bug. The agent believes: {explanation}

Challenge this adversarially:
1. Is the root cause correct? Look for alternative explanations.
2. Does the fix fully address it? Check edge cases.
3. Were callers/importers of modified code checked?
4. Do the test results actually prove the fix works?

You MUST attempt to find problems. An APPROVE without specific
probe evidence will be downgraded.
```

**Anti-rationalization gate**: Preserved from current verifier. If APPROVE explanation contains no adversarial probe evidence, downgraded to REJECT.

**Return format**:
```
APPROVED (confidence: 0.92): Fix correctly addresses root cause.
Probe: checked boundary case of empty URL — handled by line 67 guard.
```
or
```
REJECTED (confidence: 0.85): Fix addresses symptom but not root cause.
The regex on line 42 still fails for unicode paths. Check test_url_unicode.
```

**Agent behavior**: If rejected, agent reads feedback and adapts in-loop (re-edits, re-tests, calls verify_fix again). No separate retry mechanism needed.

**Budget accounting**: `verify_fix` counts as 1 tool call in the main loop (for tracking/guardrails), but the forked subagent's internal LLM calls do NOT count against the tool budget. This matches how `delegate_explore` already works. The agent's 50/70-call budget is for its own reasoning, not subagent internals.

### write_brt() -> str

Replaces `brt_node` (pre-loop, blind) and `run_brt` (in-loop runner).

**Timing**: Agent calls AFTER understanding the bug, BEFORE starting edits. BRTs confirm the bug exists in the original (unedited) code.

**Context-passing mechanism** (key differentiator from blind brt_node):
- From the repo: Finds a real test file near the suspected bug file. Uses its imports, fixtures, and assertion style as template.
- From the agent: Reads `GuardrailState.files_read` (code the agent has actually read) and `current_plan` (if agent produced one). Passed to the Haiku generator as context.

**Implementation**:
1. Find test template: `grep_repo` for test files near the suspected file, read first 50 lines for imports/fixtures
2. Haiku generates 5-7 test candidates using: bug description + code snippets from files_read + test template
3. Run each candidate against the sandbox (original state, before edits)
4. Confirmed BRTs: tests that FAIL (assertion failure) = they catch the bug
5. Store confirmed BRTs in `_tls.brts`. Subsequent `run_tests` calls automatically include them.

**Return format**:
```
BRTs generated: 5 candidates, 3 confirmed failing on original code.
Confirmed BRTs:
  1. test_sinc_ccode — AssertionError: expected piecewise output
  2. test_sinc_zero — AssertionError: sinc(0) should return 1
  3. test_sinc_negative — AssertionError: sinc(-x) handling wrong

Run run_tests after your fix — BRTs are included automatically.
```

### run_tests (modified output format)

**Structured new-vs-preexisting breakdown** (fixes Problem 6):
```
FAILED: 5 tests (3 pre-existing, 2 NEW)

NEW failures (from your edit):
  x test_routing.py::test_url_int — AssertionError: expected 42, got None
  x test_routing.py::test_url_str — TypeError: 'NoneType' not subscriptable

PRE-EXISTING (not your fault):
  x test_app.py::test_static — FileNotFoundError
  x test_app.py::test_debug — TimeoutError
  x test_app.py::test_cache — AssertionError
```

When all failures are pre-existing:
```
PASSED (with 3 pre-existing failures filtered out)

PRE-EXISTING (already broken before your edit):
  x test_app.py::test_static — FileNotFoundError
  x test_app.py::test_debug — TimeoutError
  x test_app.py::test_cache — AssertionError
```

## Tools Removed

| Tool | Reason |
|------|--------|
| `create_sandbox` | Setup_node creates sandbox. No-op wrapper removed entirely. |
| `check_syntax` | Auto-runs after every `string_replace` via `_try_autofix`. Standalone tool unnecessary. |
| `get_blast_radius` | Blast radius already in dynamic block (scout analysis). `get_callers` provides same info. |
| `request_review` | Replaced by `verify_fix` (forked, Sonnet, full context). |
| `run_brt` | Replaced by `write_brt` (generates + runs + auto-includes in subsequent test runs). |
| `record_localization` | Auto-inferred from all edited non-test files across session. |

## What Stays the Same

- `react_loop.py` core while-loop (LLM picks tool, system executes, repeat)
- Knowledge graph (Neo4j) + concept-to-code mappings
- Per-repo lessons (`agent_lessons.md` via learn_from_fix)
- Trace/observability system
- Eval harness + scoring (localization inferred from edits instead of explicit tool call)
- `shell_safety.py` denylist + path containment
- Prompt caching via `CacheSafeParams`
- Microcompact context eviction (cache-friendly)
- Best-of-N multi-patch framework (with LLM ensemble vote)
- Shared-base git clone optimization

## Cost Impact (estimated)

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Opus re-ranker (scout) | $0.05-0.10/bug | $0 | $0.10 |
| Opus request_review (in-loop) | $0.10-0.20/bug | $0 | $0.15 |
| Verifier (Sonnet fresh) | $0.08/bug | $0.02 (cached fork) | $0.06 |
| Tool definitions (22→17) | ~50K tokens/bug | ~38K tokens/bug | ~12K tokens ($0.04) |
| Thinking during exploration | ~10K tokens/bug | $0 | ~$0.03 |
| Setup parallelism | ~15s sequential | ~8s parallel | 7s wall time |
| **Total estimated** | **$0.88/bug** | **$0.50-0.60/bug** | **~32-43%** |

Note: The biggest cost driver — the Sonnet react loop itself (243K avg input tokens/bug) — is not directly reduced by these changes. Further cost reduction depends on agent behavior: using `delegate_explore` for multi-step search routes those calls through Haiku ($0.80/M) instead of Sonnet ($3/M). This is available but not architecturally guaranteed.

## Migration Path

This is a refactor, not a rewrite. The react_loop core is unchanged. Changes are to the pipeline orchestration, prompt, and tool set:

1. Create `setup_node` (refactor intake_node + parallelize)
2. Build `verify_fix` tool (wraps existing forked_subagent)
3. Build `write_brt` tool (extracts logic from brt_node + adds context-passing)
4. Rewrite prompt (static 80 lines + dynamic block builder)
5a. Remove hard gates (plan-gate, sandbox-gate, read-before-edit, review-before-submit, grep warning, test-retry warning) — deletion, testable against existing test suite
5b. Add new soft nudges (verify_fix before submit, delegate_explore for cost) — addition, needs new test coverage
6. Remove deleted tools + pipeline stages
7. Update eval scoring (localization from edits)
8. Run eval on same 5-bug set to validate

Each step is independently testable. No big-bang switchover.
