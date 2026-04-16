# AI Deploy Agent — Complete Project Reference (v4)

Everything about the project: what it is, how it works, every decision made, every component, every tool.

---

## 1. What This Project Is

**Autonomous bug-fix agent.** Give it a bug ticket → it reads the codebase → finds the bug → writes a fix → tests it → submits a PR. No human in the loop.

**Product name:** AI Deploy Agent (Primathon)
**Target:** Fix production bugs in any codebase (Python, JavaScript, TypeScript, Go, Rust)
**Benchmark:** SWE-bench Lite (300 bugs from 12 real open-source repos — Django, Flask, Sympy, etc.)

### Key Numbers

| Metric | Value |
|--------|-------|
| Pass rate (bugs that ran) | ~21% → targeting 40%+ with v4 |
| Localization accuracy | 100% (finds the right file) |
| Fix generation rate | 100% |
| Verifier approval rate | 100% at 0.93 avg confidence |
| Cost per bug | $0.50-0.60 |
| Time per bug | ~2.4 min |
| Tokens per bug | ~248K |
| Test suite | 1170 tests |
| Agent code | 15,671 lines across 28 files |
| Test code | 14,533 lines |

### Competitive Landscape

| Agent | SWE-bench Score | Architecture | Cost |
|-------|-----------------|-------------|------|
| Mini-SWE-agent | >74% | 100 lines, bash only | ~$0.30 |
| Augment | 65.4% | Claude + o1 ensemble | ~$1.50 |
| OpenHands | 53% | Event-stream, Docker | ~$1.50 |
| Moatless | 39% | State machine, semantic search | $0.14 |
| Agentless | 32% | No agent, 3-phase pipeline | $0.70 |
| **Our agent (v4)** | ~21% (pre-eval) | Smart system + free LLM | $0.50-0.60 |

---

## 2. Architecture

### The Core Idea (v4 design principle)

**"Smart system, free LLM."** The system handles orchestration, validation, and context assembly. The LLM reasons and acts freely with 18 tools, no mandated phases.

Why this beats alternatives:
- vs. Mini-SWE-agent (bash only): We add real value — knowledge graph, pre-localization, BRTs, verifier, lessons. Mini-SWE-agent has no moat.
- vs. Agentless (no agent): Our LLM can iterate, recover from failures, investigate env issues. Agentless is one-shot.
- vs. Old v3.5 (constrained agent): We removed phase mandates, plan gates, hardcoded sequences. The LLM decides its own workflow.

### Pipeline (3 stages)

```
setup_node (parallel, ~8s) → react_agent_node (~60s) → finalize_node (~5s)
```

#### Stage 1: setup_node

Three independent threads running in parallel:

| Thread | What it does | Uses LLM? | Output |
|--------|-------------|-----------|--------|
| Thread 1 | Repo detection + create sandbox + run baseline tests | No | sandbox_path, baseline_failures |
| Thread 2 | Scout localization (Haiku extractor + Sonnet debugger + skeleton narrowing) | Yes (Haiku + Sonnet) | suspects with reasoning, skeletons |
| Thread 3 | Context assembly (repo tree, graph data, lessons, concept-to-code) | No | rich context dict |

Results merge into a `_dynamic_context` dict → fed into the prompt.

**Why parallel?** These are independent. Sequential = 15s. Parallel = 8s. Saves 7s wall time per bug.

**Why no Opus?** The scout used to run an Opus re-ranker ($0.05-0.10/bug) to reorder 5 files. The agent re-prioritizes itself anyway. Cut.

#### Stage 2: react_agent_node

The core. A Sonnet 4.6 agent with 18 tools in a free-form while-loop.

**Prompt**: 60-line static block (contracts + conventions) + 150-200 line dynamic block (real data about THIS bug). Principle: don't tell the agent how to think — give it the information it needs to think well.

**Thinking**: OFF during exploration (mechanical), ON from first edit onward (reasoning matters for editing and recovery). Once ON, stays ON.

**Guardrails**: Soft nudges only. No phase mandates. Only 2 hard rules: test before submit, verify_fix before submit.

#### Stage 3: finalize_node

No LLM. Just: capture git diff → create PR → record lessons → cleanup sandbox → emit metrics.

**No retry logic.** Retries happen in-loop: verify_fix rejects → agent reads feedback → adapts → calls verify_fix again.

---

## 3. Every Architectural Decision and Why

### Decisions that shape the system

| Decision | What | Why | Alternative rejected |
|----------|------|-----|---------------------|
| **3-stage pipeline** | setup → react → finalize | Parallel setup saves 7s. Simple flow. | 5-stage sequential (v3.5): wasted pre-work, rigid |
| **Free-form agent** | No mandated phases | Mini-SWE-agent proves LLMs don't need a manual. 74% with 0 structure. | Mandatory explore→plan→edit→test→submit (v3.5): 10+ call minimum for trivial bugs |
| **Lean prompt** | 60 lines static + 200 lines data | Current prompt was 270 lines of rules, 80% instructions. Data > instructions. | Long instruction manual (v3.5): agent follows rules instead of reasoning |
| **Forked verifier** | verify_fix forks conversation | Full context (sees all reasoning), Sonnet not Opus, cached prefix = cheap | Disconnected post-loop verifier (v3.5): no context, Opus = expensive |
| **Agent-controlled BRT** | write_brt inside loop | Agent reads code first, uses real test patterns. BRT has context. | Blind pre-loop BRT (v3.5): generated tests for files it hadn't read |
| **Shell access** | run_shell tool | Agent can diagnose/repair env (pip install, etc). Universal escape hatch. | No shell (v3.5): agent couldn't fix "pytest exit 4" |
| **Baseline test snapshot** | Run tests BEFORE edits | Distinguishes pre-existing failures from regressions. Agent sees structured breakdown. | No baseline (v3.5): all test failures blamed on agent |
| **No plan gate** | produce_plan optional | Simple bugs don't need plans. Agent decides when to plan. | Mandatory plan before sandbox (v3.5): 4+ call overhead |
| **Thinking inversion** | OFF→ON at first edit | Exploration is mechanical (grep, read). Editing needs reasoning. | ON→OFF (v3.5): thinking during grep, no thinking during edits |
| **Haiku for cheap exploration** | delegate_explore | Routes broad search to $0.80/M instead of $3/M. Saves 60% on exploration. | All calls through Sonnet (v3.5): $0.88/bug avg |
| **Knowledge graph** | Neo4j with callers, business rules | Gives structural context (who calls what, blast radius) that grep can't. | No graph (most competitors): agent greps blindly |
| **Per-repo lessons** | agent_lessons.md persists | Agent learns from past failures. "Always check URL encoding in this repo." | No learning (all competitors): same mistakes repeated |
| **Shared-base git clone** | Clone repo once, local-copy per bug | 47 bugs from 12 repos = 12 clones instead of 47. Saves ~35 network calls. | Clone per bug (v3.5): 49% of 47-bug run failed at git clone |
| **Venv auto-injection** | PATH/VIRTUAL_ENV set for scorer's venv | Agent's `pip install` lands in the same venv the test scorer uses. | System pip (v3.5): agent installs deps, scorer doesn't see them |
| **Non-interactive shell** | stdin=DEVNULL, CI=true, EDITOR=true | No hangs on prompts, no interactive TUIs. Agent runs unattended. | Default shell (would hang on pip uninstall y/n, vim, etc.) |
| **Multi-language support** | Code map + check_syntax for JS/TS/Go/Rust | Not just Python. Detect language, route test runner, extract structure. | Python-only (v3.5): couldn't handle JS repos |
| **Multi-patch framework** | best_of_n + LLM ensemble vote | Generate K patches, pick best. +3-8% pass rate (research-proven). | Single-shot only (v3.5): no diversity in patches |
| **Anti-rationalization gate** | Verifier APPROVE without probe evidence → REJECT | Prevents rubber-stamp approvals. Forces adversarial thinking. | Blind trust in verifier (would approve plausible-sounding bad fixes) |

### Decisions we considered but rejected

| Rejected idea | Why rejected |
|---------------|-------------|
| Docker sandboxing | Adds 2-5s cold start, breaks venv access. Path validation sufficient for dev machine. Revisit for multi-tenant production. |
| Allowlist for shell commands | Agent needs arbitrary diagnostics. Denylist of 30+ dangerous patterns is more practical. |
| Separate retry mechanism | Retries happen naturally via verify_fix rejection. Agent adapts in-loop. |
| Architect/Editor model split | Aider's pattern: reasoning model + editing model. Good but adds latency + complexity. Sonnet handles both well enough. |
| AST patch normalization | Agentless normalizes patches for majority vote. Only matters with K>3 patches. |
| Full repo map in every prompt | Aider's approach: entire tree-sitter map. Too many tokens for SWE-bench repos (Django = 2000+ files). Top-200 listing is enough. |

---

## 4. All Tools (18)

### Exploration (7) — read-only

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 1 | `grep_repo` | `(pattern, file_glob="", max_results=20, context_lines=2)` | Regex search across source files with context |
| 2 | `read_file` | `(file_path, start_line=1, end_line=0)` | 100-line viewer with line numbers. Scroll with start_line. |
| 3 | `read_function` | `(file_path, function_name)` | Extract complete function source |
| 4 | `list_files` | `(directory="", extension="")` | Directory listing, filterable |
| 5 | `get_function_info` | `(function_id)` | Knowledge graph: callers, callees, business rules |
| 6 | `get_file_structure` | `(file_path)` | Class + function signatures, no bodies. Multi-language. |
| 7 | `get_blast_radius` | `(function_name)` | Who calls/imports this code. Impact assessment. |

### Cheap exploration (1) — Haiku subagent

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 8 | `delegate_explore` | `(question)` | Haiku does 5-8 grep/read cycles for ~10x cheaper than Sonnet |

### Editing (3) — requires sandbox

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 9 | `string_replace` | `(file_path, old_string, new_string)` | Exact string replacement. Auto syntax-check + ruff. |
| 10 | `create_file` | `(file_path, content)` | Create/overwrite file |
| 11 | `undo_last_edit` | `()` | Revert most recent edit surgically |

### Planning (1) — optional

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 12 | `produce_plan` | `(root_cause, target_files, approach, success_criteria, risk, rollback)` | Declare plan. Optional. Logged for verifier. |

### Testing (3)

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 13 | `run_tests` | `(test_path="")` | Tests + linters. Structured NEW vs PRE-EXISTING output. |
| 14 | `run_shell` | `(command, timeout=120, working_dir="")` | Non-interactive shell. For env diagnosis/repair. |
| 15 | `write_brt` | `()` | Generate repro tests from code agent has read |

### Completion (3)

| # | Tool | Signature | What it does |
|---|------|-----------|-------------|
| 16 | `verify_fix` | `(explanation)` | Fork conversation for independent Sonnet review |
| 17 | `submit_fix` | `(explanation)` | Commit + prepare PR |
| 18 | `escalate` | `(reason)` | Hand off to human |

---

## 5. System Prompt

### Static Block (~60 lines, cached)

```
You are an autonomous software engineer. You fix bugs in codebases.

## Workflow (adapt freely)
Explore the code to understand the bug. Edit the minimum needed. Test your fix.
Verify independently. Submit.
The order is flexible — use your judgment on when you know enough to act.

## Hard contracts
1. A sandbox MUST exist before any file edits (create_sandbox).
2. You MUST run the target tests at least once before calling submit_fix.
3. You MUST call verify_fix (independent fork) before submit_fix.
4. Do NOT modify files unrelated to the bug.
5. Keep changes minimal — fix the bug, nothing more.
6. ALWAYS use relative paths from the repo root for every tool argument.

## Test result interpretation
- "passed"  — tests ran and passed. Proceed.
- "failed"  — actual assertion failures. Investigate and re-fix.
- "skipped" — no tests collected (missing deps). Acceptable — proceed.
- "error"   — execution failed (import error). Acceptable — proceed.
Only "failed" blocks submission. Do NOT retry "skipped"/"error" > 3 times.

## Pre-existing failures
Do NOT fix pre-existing issues — only fix what YOUR edits introduced.

## Path convention
CORRECT: 'src/app/models.py'  |  WRONG: '/tmp/agent_sandbox_.../src/app/models.py'

## BRT guidance
Call write_brt AFTER understanding the bug, BEFORE editing.
After your fix, run_tests includes BRTs automatically.

## Planning guidance
produce_plan is optional. Recommended for multi-file fixes.

## Cost guidance
delegate_explore for broad search — 10x cheaper than your own tool calls.

## run_shell guidance
Non-interactive shell. stdin closed, CI=true.
Use for: pip install, pip list, python -c "import X", which pytest.

## verify_fix guidance
Forks your conversation. Independent review. Call before submit.
If rejected, read feedback and revise.

## Known issues (add when evals reveal consistent failures)
```

### Dynamic Block (per-bug, ~150-200 lines)

Assembled by setup_node from real data. Contains:
- Bug ticket (title, priority, component, description)
- Target tests (FAIL_TO_PASS that must pass, PASS_TO_PASS that must stay passing)
- Scout analysis (entity extraction, suspected files with reasoning + skeletons, blast radius, business rules — or fallback if scout found nothing)
- Baseline test results (specific pre-existing failures by name)
- Repo structure (top 200 source files)
- Code map (function signatures + line numbers for localized files)
- Lessons from past fixes in this repo
- Concept-to-code mappings from knowledge graph

### Task Message

```
Fix this bug. The context above has everything you need to start.
Focus on the target tests — when they pass, you're done.
```

---

## 6. Guardrails

### Kept (v4)
- Tool budget: 50 calls (single-file) / 70 calls (multi-file)
- Wall time: 1800s max
- Cost limit: $15 max
- Nudge: "call verify_fix before submit_fix"
- Nudge: "run_shell called 6+ times — env likely unfixable, submit anyway"

### Removed (from v3.5)
- Plan gate (plan before sandbox) — plan is optional now
- Sandbox gate (sandbox before edits) — setup creates it
- Read-before-edit warning — unnecessary
- Review-before-submit gate — verify_fix nudge replaces it
- Grep warning at 8 — LLM decides exploration
- run_tests warning at 3 — LLM decides testing

---

## 7. Observability

### Trace Events

Every bug run produces a JSON trace file with 17 event types:

| Event | Data |
|-------|------|
| `stage_start/end` | Stage name, timing |
| `llm_request` | Model, message count, context tokens, cost so far |
| `llm_response` | **input/output/cache tokens per call**, cost, **thinking_text**, tool calls |
| `tool_call` | Tool name, args, call number, phase, **agent's reasoning** |
| `tool_result` | Tool name, duration_ms, result preview |
| `state_transition` | Phase changes (explore → edit → test → review) |
| `context_compaction` | Tokens before/after eviction |
| `prompt_build` | **Full system prompt text**, static/dynamic sizes |
| `llm_mode_switch` | Thinking ON/OFF, trigger |
| `brt_confirmed` | Confirmed BRT descriptions |
| `brt_generated` | **Actual test code** for each confirmed BRT |
| `verifier_result` | Verdict, confidence, **full explanation**, fork status |
| `submission_diff` | **Complete git diff**, files changed, line count |
| `failure_diagnosis` | Failure mode, **replay steps**, edit churn, EPR |
| `run_outcome` | Submitted, cost, tool calls, tests passed, review verdict |
| `run_metrics` | Localization precision, verifier calibration |

### Summary-level metrics (top of trace JSON)
- `stage_timings` — how long each stage took
- `phase_breakdown` — time spent in explore/edit/test/review
- `wasted_calls` — repeated reads, grep streaks, test retries
- `context_timeline` — context window growth over time
- `token_usage_by_stage` — per-stage input/output/cache breakdown

### API Endpoints

| Endpoint | What |
|----------|------|
| `GET /api/traces` | List all eval runs |
| `GET /api/traces/{run_id}` | Bugs in a run with outcome/cost/tools |
| `GET /api/traces/{run_id}/{bug_id}` | Full trace JSON |

### Frontend

`TraceLogPanel.tsx` — live trace viewer with filters (LLM/tools/tests/stages/guardrails), expandable event rows, color-coded icons.

---

## 8. File Map

### Agent core (28 files, 15,671 lines)

| File | Lines | What it does |
|------|-------|-------------|
| `react_pipeline.py` | 2413 | Pipeline orchestration: setup_node, react_agent_node, finalize_node, verifier, BRT |
| `react_loop.py` | 958 | Core while-loop: LLM ↔ tools, thinking switch, context management |
| `react_tools.py` | 1680 | 10 react tools (edit, plan, test, completion) + verify_fix + write_brt |
| `react_prompt.py` | 524 | System prompt builder: static block + dynamic block |
| `react_guardrails.py` | 316 | Soft nudges, tool budget, cost/time limits |
| `explore_tools.py` | 1707 | 7 exploration tools (grep, read, list, structure, callers) |
| `explore_subagent.py` | 226 | delegate_explore — Haiku subagent for cheap search |
| `scout.py` | 963 | 3-agent fault localization: extractor → debugger → skeleton narrowing |
| `forked_subagent.py` | 226 | CacheSafeParams — fork conversation for verifier |
| `context_manager.py` | 434 | Microcompact eviction, summarization, context timeline |
| `sandbox.py` | ~500 | Git worktree creation, test runner, --noconftest fallback |
| `shell_tools.py` | ~200 | run_shell tool with non-interactive safety |
| `shell_safety.py` | ~150 | Denylist (30+ patterns), path containment, TUI blocks |
| `tool_metadata.py` | 171 | Per-tool metadata: read-only, concurrent-safe, output caps |
| `learn_from_fix.py` | ~200 | Record/load per-repo lessons |
| `repo_detection.py` | ~200 | Auto-detect language, test runner, package manager |
| `llm.py` | ~120 | LLM call wrappers: structured_call, simple_call, cost estimation |
| `linters.py` | ~200 | ESLint, ruff, Biome, pre-commit integration |
| `web_tools.py` | ~200 | web_fetch + web_search (disabled by default) |
| `graph_utils.py` | ~300 | Knowledge graph queries, concept-to-code, kickstart context |

### Eval harness

| File | What |
|------|------|
| `eval/runner.py` | Eval orchestration, per-bug case execution, report generation |
| `eval/scoring.py` | Localization, fix rate, patch correctness, test pass scoring |
| `eval/repo_manager.py` | Git clone/checkout, venv setup, shared-base clone, preflight |
| `eval/swebench_import.py` | Import SWE-bench Lite from HuggingFace → bugs.json |
| `eval/vaguify.py` | Haiku rewrites technical descriptions to vague product tickets |

### Frontend

| File | What |
|------|------|
| `frontend/src/pages/Agent.tsx` | Main agent UI page |
| `frontend/src/components/agent/TraceLogPanel.tsx` | Live trace viewer |
| `frontend/src/components/agent/LiveActivityFeed.tsx` | Activity feed |
| `frontend/src/pages/Knowledge.tsx` | Knowledge graph explorer |
| `frontend/src/pages/Overview.tsx` | Dashboard |

### Infrastructure

| File | What |
|------|------|
| `docker-compose.yml` | Neo4j + backend + frontend |
| `backend/main.py` | FastAPI app |
| `backend/cli.py` | CLI: eval run, build graph, fix ticket |
| `backend/api/eval.py` | Eval + trace browsing API endpoints |

---

## 9. Commands

```bash
# Run tests
python -m pytest backend/tests/ -q                    # 1170 tests

# Start services
docker-compose up                                      # Neo4j + backend + frontend

# Run eval
cd backend && python cli.py eval run                   # Full eval
cd backend && python cli.py eval run --sentinel        # Fast 5-bug check
cd backend && python cli.py eval run --bug BUG-ID      # Single bug
cd backend && python cli.py eval run --dataset FILE    # Custom dataset
cd backend && python cli.py eval run --nl              # Vague descriptions

# Build knowledge graph
cd backend && python cli.py build /path/to/repo

# Fix a single bug
cd backend && python cli.py fix TICKET-ID --repo /path/to/repo

# Access
http://localhost:5173          # Frontend UI
http://localhost:8001          # Backend API
http://localhost:7474          # Neo4j browser
http://localhost:8001/api/traces  # Trace browser API
```

## 10. Environment Variables

| Var | Default | What |
|-----|---------|------|
| `ANTHROPIC_API_KEY` | (required) | API key for Claude |
| `ANTHROPIC_MODEL` | claude-sonnet-4-6 | Main agent model |
| `REACT_THINKING_BUDGET` | 2048 | Thinking tokens per turn |
| `DISABLE_REACT_THINKING` | 0 | Skip extended thinking entirely |
| `REACT_REFRESH_INTERVAL` | 5 | Status refresh every N calls |
| `ENABLE_WEB_TOOLS` | 0 | Enable web_fetch + web_search |
| `DISABLE_LEARN_FROM_FIX` | 0 | Skip lesson recording |
| `REACT_MAX_RETRIES` | 0 | Pass@3 retry count (0=disabled) |
| `DATA_DIR` | /tmp/context_builder | Data storage root |
| `EVAL_REPOS_DIR` | eval/repos | Cloned repos cache |
| `EVAL_CLONE_TIMEOUT` | 900 | Git clone timeout (seconds) |

---

## 11. Eval Results History

### Cost breakdown across all runs

| Metric | Value |
|--------|-------|
| Total API spend | $72.37 |
| Total bugs attempted | 120 |
| Avg cost per bug | $0.60 |
| Cost per passed bug | $3.29 |
| Total tokens used | ~33M |

### 47-bug run failure analysis (adjusted)

| Failure mode | Count | % of bugs that ran |
|-------------|-------|-------------------|
| GIT_INFRA_FAIL | 23 | (infra, not agent) |
| TESTS_FAILED | 8 | 33% |
| LOCALIZATION_MISS | 6 | 25% |
| PASSED | 5 | 21% |
| NO_FIX | 4 | 17% |
| AGENT_TIMEOUT | 1 | 4% |

### v4 improvements (last eval run on same 5 bugs)

| Metric | v3.5 (first run) | v3.5 (after fixes) | v4 (latest) |
|--------|------------------|---------------------|-------------|
| Localization | 0% | 100% | 100% |
| Fix rate | 80% | 80% | 100% |
| Verifier approval | 80% | 80% | 100% |
| Confidence | 0.00 | 0.74 | 0.93 |
| Scout hallucinations | 20 | 0 | 0 |
| Test pass rate | 0% | 0% | pending v4 eval |
| Cost/bug | $0.85 | $1.00 | est $0.50-0.60 |

---

## 12. What's Next

### Immediate
1. Top up API credits → run v4 eval on 47 bugs
2. Target: 40% pass rate at <$0.60/bug

### Short-term
3. Best-of-N multi-patch sampling (framework built, needs eval)
4. Monorepo support (frontend + backend in same repo)
5. Historical trace browser in frontend UI

### Medium-term
6. Full SWE-bench Lite 300-bug run (~$150, ~10h)
7. Docker sandboxing for production multi-tenant
8. Multi-repo support (bug spans 2+ repos)
9. Architect/Editor model split (reasoning + editing models)
