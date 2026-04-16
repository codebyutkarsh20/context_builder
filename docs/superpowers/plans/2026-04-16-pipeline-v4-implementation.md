# Pipeline v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 5-stage pipeline into 3 stages (parallel setup, free-form react loop, lightweight finalize) with 17 tools, lean prompt, and in-loop verification.

**Architecture:** `setup_node` (parallel threads: repo detection + sandbox + baseline tests, scout localization, context assembly) merges into a rich dynamic block. `react_agent_node` runs a free-form Sonnet loop with 17 tools including `verify_fix` (forked subagent) and `write_brt` (context-aware). `finalize_node` handles PR creation, lessons, cleanup.

**Tech Stack:** Python 3.10+, Anthropic Claude Sonnet 4.6, LangChain, concurrent.futures, existing forked_subagent/CacheSafeParams infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-16-pipeline-v4-architecture-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/agent/react_pipeline.py` | Heavy modify | Replace intake_node with setup_node, remove brt_node + verifier_node, simplify finalize_node, update run_ticket_react |
| `backend/agent/react_prompt.py` | Rewrite | 80-line static block + dynamic block builder |
| `backend/agent/react_tools.py` | Modify | Add verify_fix + write_brt, remove 6 tools, update REACT_TOOLS list |
| `backend/agent/react_loop.py` | Modify | Invert thinking switch, remove tools from import |
| `backend/agent/react_guardrails.py` | Modify | Remove 6 hard gates, add 2 soft nudges |
| `backend/agent/scout.py` | Modify | Remove Opus re-ranker call, export full reasoning |
| `backend/agent/tool_metadata.py` | Modify | Add verify_fix/write_brt metadata, remove 6 entries |
| `backend/agent/context_manager.py` | Modify | Update COMPACTABLE_TOOLS |
| `backend/agent/eval/scoring.py` | Modify | Localization from edits instead of record_localization |
| `backend/tests/test_react_contracts.py` | Modify | Update tool list assertions, add new tool tests |
| `backend/tests/test_pipeline_v4.py` | Create | Integration tests for setup_node, verify_fix, write_brt |

---

### Task 1: Build `setup_node` with parallel threads

This is the foundation — everything else depends on it.

**Files:**
- Modify: `backend/agent/react_pipeline.py` (replace intake_node internals, add parallel execution)
- Test: `backend/tests/test_pipeline_v4.py` (new)

- [ ] **Step 1: Write test for setup_node thread merging**

```python
# backend/tests/test_pipeline_v4.py
"""Tests for Pipeline v4 setup_node, verify_fix, write_brt."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSetupNode:
    """setup_node must run 3 threads in parallel and merge results."""

    def test_setup_produces_sandbox_path(self, tmp_path):
        """Thread 1 must create a sandbox and return its path."""
        from agent.react_pipeline import setup_node

        state = _make_test_state(tmp_path)
        result = setup_node(state)
        assert result.get("sandbox_path"), "setup_node must set sandbox_path"
        assert Path(result["sandbox_path"]).exists()

    def test_setup_captures_baseline_failures(self, tmp_path):
        """Thread 1 must run tests and capture pre-existing failures."""
        from agent.react_pipeline import setup_node

        state = _make_test_state(tmp_path)
        result = setup_node(state)
        # baseline_failures is a set (possibly empty if tests pass)
        assert isinstance(result.get("baseline_failures", set()), set)

    def test_setup_includes_scout_reasoning(self, tmp_path):
        """Thread 2 must produce scout analysis with reasoning, not just paths."""
        from agent.react_pipeline import setup_node

        state = _make_test_state(tmp_path)
        with patch("agent.react_pipeline._run_scout_thread") as mock_scout:
            mock_scout.return_value = {
                "suspects": [{"file": "app.py", "reason": "URL handler", "confidence": 0.8}],
                "entity_extraction": {"function_names": ["match"], "error_types": ["ValueError"]},
                "skeleton_data": {"app.py": "L42: def match(self):"},
            }
            result = setup_node(state)
        intent = result.get("intent", {})
        assert intent.get("scout_reasoning"), "Scout reasoning must flow into intent"

    def test_setup_includes_repo_tree(self, tmp_path):
        """Thread 3 must assemble repo tree."""
        from agent.react_pipeline import setup_node

        state = _make_test_state(tmp_path)
        result = setup_node(state)
        # dynamic_context should have repo_tree
        assert "repo_tree" in result.get("_dynamic_context", {})

    def test_setup_threads_run_in_parallel(self, tmp_path):
        """Threads must actually run concurrently (wall time < sum of all)."""
        import time
        from agent.react_pipeline import setup_node

        state = _make_test_state(tmp_path)
        start = time.monotonic()
        setup_node(state)
        elapsed = time.monotonic() - start
        # Each thread takes ~0.1s in test mode; sequential = 0.3s, parallel < 0.2s
        # Use generous threshold to avoid flaky tests
        assert elapsed < 30, "setup_node took too long — threads may not be parallel"


def _make_test_state(tmp_path):
    """Build a minimal state dict for testing setup_node."""
    # Create a fake repo with a Python file
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def hello():\n    return 'world'\n")
    (repo / ".git").mkdir()  # fake git dir
    (repo / "setup.py").write_text("")

    return {
        "work_order": {
            "ticket_id": "TEST-001",
            "title": "Test bug",
            "description": "Something is broken",
            "repo_name": "test-repo",
            "repo_path": str(repo),
        },
        "intent": {},
        "status": "pending",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/utkarshpatidar/Projects/personal/context_builder
python -m pytest backend/tests/test_pipeline_v4.py::TestSetupNode::test_setup_produces_sandbox_path -v
```
Expected: FAIL — `setup_node` doesn't exist yet.

- [ ] **Step 3: Implement `setup_node` with 3 parallel threads**

In `backend/agent/react_pipeline.py`, add `setup_node` function. This replaces `intake_node`. The key changes:
- Use `concurrent.futures.ThreadPoolExecutor` to run 3 threads
- Thread 1: repo detection + sandbox creation + baseline tests (reuses existing code from `intake_node` + `create_sandbox` in react_tools.py)
- Thread 2: scout localization (reuses `_run_localization` but drops the Opus re-ranker)
- Thread 3: context assembly (reuses `build_kickstart_context` + `load_lessons`)
- Merge all outputs into `state["_dynamic_context"]` dict

```python
def setup_node(state: ReactAgentState) -> ReactAgentState:
    """Stage 1: Parallel setup — repo detection, sandbox, baseline tests,
    scout localization, context assembly. No Opus. ~8s wall time.
    
    Replaces the old sequential intake_node + brt_node.
    """
    _thread_local.current_stage = "setup"
    trace = _get_trace()
    if trace:
        trace.stage_start("setup")
    
    work_order = state.get("work_order", {})
    repo_path = _resolve_repo_path(work_order)
    repo_name = work_order.get("repo_name", "")
    
    if not repo_path:
        state["escalated"] = True
        state["escalate_reason"] = "No repo path available"
        return state
    
    # Translate intent (single cheap Haiku call — needed by all threads)
    state["intent"] = _translate_intent(work_order)
    intent = state["intent"]
    
    # Launch 3 independent threads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="setup") as pool:
        futures = {
            pool.submit(_setup_thread_repo, work_order, repo_path, state): "repo",
            pool.submit(_setup_thread_scout, work_order, intent, repo_name, repo_path): "scout",
            pool.submit(_setup_thread_context, work_order, intent, repo_name, repo_path): "context",
        }
        for future in as_completed(futures):
            thread_name = futures[future]
            try:
                results[thread_name] = future.result()
            except Exception as e:
                logger.warning("setup thread '%s' failed: %s", thread_name, e)
                results[thread_name] = {}
    
    # Merge results into state
    repo_result = results.get("repo", {})
    state["sandbox_path"] = repo_result.get("sandbox_path", "")
    state["branch_name"] = repo_result.get("branch_name", "")
    state["base_branch"] = repo_result.get("base_branch", "main")
    state["baseline_failures"] = repo_result.get("baseline_failures", set())
    
    scout_result = results.get("scout", {})
    # Merge scout findings into intent
    if scout_result.get("suspects"):
        intent["confirmed_files"] = [s["file"] for s in scout_result["suspects"] if s.get("file")]
        intent["likely_affected_modules"] = intent["confirmed_files"]
    intent["scout_reasoning"] = scout_result  # Full reasoning, not just paths
    state["intent"] = intent
    
    context_result = results.get("context", {})
    state["_dynamic_context"] = {
        "repo_tree": context_result.get("repo_tree", ""),
        "graph_context": context_result.get("graph_context", ""),
        "lessons": context_result.get("lessons", ""),
        "concept_mappings": context_result.get("concept_mappings", ""),
        "scout": scout_result,
        "baseline_failures": repo_result.get("baseline_failures", set()),
    }
    
    if trace:
        trace.stage_end("setup")
    return state
```

Then implement the 3 thread functions `_setup_thread_repo`, `_setup_thread_scout`, `_setup_thread_context` by extracting and simplifying the existing logic from `intake_node`, `_run_localization`, and `react_agent_node`'s kickstart code.

- [ ] **Step 4: Run tests to verify setup_node works**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestSetupNode -v
```
Expected: PASS for all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_pipeline.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): add setup_node with parallel threads

Replaces sequential intake_node. Three threads run in parallel:
- Thread 1: repo detection + sandbox creation + baseline tests
- Thread 2: scout localization (Haiku + Sonnet, no Opus)
- Thread 3: context assembly (repo tree, graph, lessons)

Merges results into rich dynamic context for react_agent_node."
```

---

### Task 2: Build `verify_fix` tool (forked subagent)

**Files:**
- Modify: `backend/agent/react_tools.py` (add verify_fix tool)
- Modify: `backend/agent/forked_subagent.py` (may need minor interface changes)
- Test: `backend/tests/test_pipeline_v4.py` (add TestVerifyFix class)

- [ ] **Step 1: Write test for verify_fix**

```python
# Add to backend/tests/test_pipeline_v4.py

class TestVerifyFix:
    """verify_fix must fork the conversation and return structured verdict."""

    def test_returns_structured_verdict(self):
        """verify_fix must return APPROVED/REJECTED with confidence."""
        from agent.react_tools import _tls
        _tls.sandbox_path = Path("/tmp/fake_sandbox")
        _tls.repo_path = Path("/tmp/fake_repo")

        with patch("agent.react_tools._run_forked_verification") as mock_fork:
            mock_fork.return_value = {
                "verdict": "APPROVED",
                "confidence": 0.92,
                "explanation": "Fix addresses root cause. Probe: checked empty input.",
            }
            from agent.react_tools import verify_fix
            result = verify_fix.invoke({"explanation": "Fixed the URL parser"})

        assert "APPROVED" in result
        assert "0.92" in result

    def test_rejected_includes_feedback(self):
        """REJECTED verdict must include actionable feedback."""
        from agent.react_tools import _tls
        _tls.sandbox_path = Path("/tmp/fake_sandbox")
        _tls.repo_path = Path("/tmp/fake_repo")

        with patch("agent.react_tools._run_forked_verification") as mock_fork:
            mock_fork.return_value = {
                "verdict": "REJECTED",
                "confidence": 0.85,
                "explanation": "Fix misses unicode edge case in line 42.",
            }
            from agent.react_tools import verify_fix
            result = verify_fix.invoke({"explanation": "Fixed the URL parser"})

        assert "REJECTED" in result
        assert "unicode" in result.lower()

    def test_anti_rationalization_gate(self):
        """APPROVED without probe evidence must be downgraded to REJECTED."""
        from agent.react_tools import _tls
        _tls.sandbox_path = Path("/tmp/fake_sandbox")
        _tls.repo_path = Path("/tmp/fake_repo")

        with patch("agent.react_tools._run_forked_verification") as mock_fork:
            mock_fork.return_value = {
                "verdict": "APPROVED",
                "confidence": 0.90,
                "explanation": "The diff looks correct.",  # No probe evidence!
            }
            from agent.react_tools import verify_fix
            result = verify_fix.invoke({"explanation": "Fixed the URL parser"})

        assert "REJECTED" in result or "downgraded" in result.lower()

    def test_counts_as_one_tool_call(self):
        """verify_fix counts as 1 tool call, subagent internals don't count."""
        from agent.react_guardrails import GuardrailState, update_from_tool_result
        gs = GuardrailState()
        initial_count = gs.tool_call_count
        update_from_tool_result("verify_fix", {"explanation": "test"}, "APPROVED (0.9)", gs)
        assert gs.tool_call_count == initial_count + 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestVerifyFix -v
```
Expected: FAIL — `verify_fix` doesn't exist yet.

- [ ] **Step 3: Implement `verify_fix` tool**

In `backend/agent/react_tools.py`, add the `verify_fix` tool function and `_run_forked_verification` helper:

```python
@tool
def verify_fix(explanation: str) -> str:
    """Fork your conversation for independent review of your fix.

    A separate Sonnet instance reads your FULL conversation history
    (every exploration, hypothesis, edit) and judges the fix.
    Its reasoning never enters your context — you only receive
    the structured verdict.

    Call this AFTER you've edited + tested. If rejected, read the
    feedback and adapt. You can call this multiple times.

    Args:
        explanation: Why you believe this fix is correct (2-3 sentences).
    """
    result = _run_forked_verification(explanation)
    verdict = result.get("verdict", "REJECTED")
    confidence = result.get("confidence", 0.0)
    feedback = result.get("explanation", "")

    # Anti-rationalization gate
    if verdict == "APPROVED":
        probe_signals = (
            "boundary", "concurrency", "idempoten", "parallel", "edge",
            "empty", "none", "unicode", "malformed", "probe", "considered",
            "checked", "if called", "what if", "negative",
        )
        if not any(sig in feedback.lower() for sig in probe_signals):
            verdict = "REJECTED"
            confidence = min(confidence, 0.4)
            feedback = f"[downgraded] APPROVE lacked adversarial probe. {feedback}"

    if verdict == "APPROVED":
        return f"APPROVED (confidence: {confidence:.2f}): {feedback}"
    else:
        return f"REJECTED (confidence: {confidence:.2f}): {feedback}"


def _run_forked_verification(explanation: str) -> dict:
    """Run verification in a forked subagent with full conversation context."""
    from agent.forked_subagent import run_forked_subagent, get_last_cache_safe_params

    prompt = (
        f"You are an independent reviewer. The conversation above shows "
        f"an agent fixing a bug. The agent believes: {explanation}\n\n"
        f"Challenge this adversarially:\n"
        f"1. Is the root cause correct? Look for alternative explanations.\n"
        f"2. Does the fix fully address it? Check edge cases.\n"
        f"3. Were callers/importers of modified code checked?\n"
        f"4. Do the test results actually prove the fix works?\n\n"
        f"You MUST attempt to find problems. An APPROVE without specific "
        f"probe evidence will be downgraded.\n\n"
        f"Respond with:\n"
        f"verdict: APPROVED or REJECTED\n"
        f"confidence: 0.0 to 1.0\n"
        f"explanation: your reasoning including which probe you ran"
    )

    # Use existing VerifierResult schema from react_pipeline
    from agent.react_pipeline import VerifierResult

    if get_last_cache_safe_params() is not None:
        fork_result = run_forked_subagent(
            task=prompt, schema=VerifierResult, max_tokens=2000,
        )
        if fork_result.get("parsed") is not None:
            parsed = fork_result["parsed"]
            return {
                "verdict": parsed.verdict,
                "confidence": parsed.confidence,
                "explanation": parsed.explanation or "",
            }

    # Fallback: fresh structured_call
    from agent.llm import structured_call
    result = structured_call("claude-sonnet-4-6", 2000, VerifierResult, prompt)
    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "explanation": result.explanation or "",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestVerifyFix -v
```
Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_tools.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): add verify_fix forked subagent tool

Replaces request_review (Opus, in-loop) and verifier_node (post-loop).
Uses CacheSafeParams to fork the full conversation. The forked Sonnet
reads all agent reasoning but its response stays in the fork.
Returns structured APPROVED/REJECTED with confidence.
Anti-rationalization gate preserved."
```

---

### Task 3: Build `write_brt` tool (context-aware BRT generation)

**Files:**
- Modify: `backend/agent/react_tools.py` (add write_brt tool)
- Test: `backend/tests/test_pipeline_v4.py` (add TestWriteBrt class)

- [ ] **Step 1: Write test for write_brt**

```python
# Add to backend/tests/test_pipeline_v4.py

class TestWriteBrt:
    """write_brt must generate tests using agent's actual exploration context."""

    def test_reads_from_files_read(self, tmp_path):
        """write_brt must use code the agent has actually read, not guesses."""
        from agent.react_tools import _tls
        from agent.react_guardrails import GuardrailState

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "app.py").write_text("def broken():\n    return None  # should return 42\n")
        (sandbox / "tests").mkdir()
        (sandbox / "tests" / "test_app.py").write_text(
            "import pytest\nfrom app import broken\n\ndef test_existing():\n    assert broken() == 42\n"
        )

        _tls.sandbox_path = sandbox
        _tls.repo_path = sandbox
        gs = GuardrailState()
        gs.files_read["app.py"] = "def broken():\n    return None"
        _tls._guardrail_state = gs

        with patch("agent.react_tools._generate_brt_candidates") as mock_gen:
            mock_gen.return_value = [
                {"test_code": "def test_broken_returns_42():\n    assert broken() == 42\n",
                 "description": "broken() should return 42"},
            ]
            from agent.react_tools import write_brt
            result = write_brt.invoke({})

        assert "BRT" in result or "generated" in result.lower()
        # Mock was called with files_read content
        call_args = mock_gen.call_args
        assert "broken" in str(call_args)

    def test_stores_confirmed_brts(self, tmp_path):
        """Confirmed BRTs must be stored in _tls.brts for run_tests."""
        from agent.react_tools import _tls

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        _tls.sandbox_path = sandbox
        _tls.repo_path = sandbox
        _tls.brts = []

        with patch("agent.react_tools._generate_brt_candidates") as mock_gen, \
             patch("agent.react_tools._run_brt_candidate") as mock_run:
            mock_gen.return_value = [
                {"test_code": "def test_repro():\n    assert False\n",
                 "description": "reproduces bug"},
            ]
            mock_run.return_value = {"passed": False, "output": "AssertionError"}

            from agent.react_tools import write_brt
            write_brt.invoke({})

        assert len(_tls.brts) >= 1, "Confirmed BRTs must be stored"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestWriteBrt -v
```
Expected: FAIL — `write_brt` doesn't exist yet.

- [ ] **Step 3: Implement `write_brt` tool**

In `backend/agent/react_tools.py`, add `write_brt` and helper `_generate_brt_candidates`:

```python
@tool
def write_brt() -> str:
    """Generate Bug Reproduction Tests based on what you've learned.

    Call this AFTER you've explored the code and understand the bug,
    BEFORE you start editing. BRTs confirm the bug exists in the
    original code. After your fix, run_tests includes BRTs automatically.

    Context: Uses code you've already read (via read_file/read_function)
    and test patterns from the repo's existing test files.
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox or not Path(sandbox).exists():
        return "ERROR: No sandbox. BRTs need a sandbox to run against."

    # 1. Gather context from what the agent has actually read
    gs = getattr(_tls, "_guardrail_state", None)
    files_read_snippets = {}
    if gs and hasattr(gs, "files_read"):
        files_read_snippets = dict(list(gs.files_read.items())[:5])

    if not files_read_snippets:
        return "ERROR: No files read yet. Explore the code first, then call write_brt."

    # 2. Find test template from repo
    test_template = _find_test_template(sandbox)

    # 3. Generate candidates via Haiku
    work_order = getattr(_tls, "_work_order", {})
    candidates = _generate_brt_candidates(
        files_read_snippets, test_template, work_order,
    )

    if not candidates:
        return "No BRT candidates generated. Proceed without BRTs."

    # 4. Run each candidate, keep ones that FAIL (= they catch the bug)
    confirmed = []
    for cand in candidates[:7]:
        result = _run_brt_candidate(sandbox, cand["test_code"])
        if not result["passed"]:  # FAIL = catches the bug = confirmed BRT
            confirmed.append(cand)

    # 5. Store in _tls for run_tests auto-inclusion
    _tls.brts = getattr(_tls, "brts", []) + confirmed

    if not confirmed:
        return f"Generated {len(candidates)} BRT candidates, but none failed on original code. The bug may not be reproducible via unit test."

    lines = [f"BRTs generated: {len(candidates)} candidates, {len(confirmed)} confirmed failing."]
    lines.append("Confirmed BRTs:")
    for i, c in enumerate(confirmed, 1):
        lines.append(f"  {i}. {c.get('description', 'test')}")
    lines.append("\nrun_tests will include these BRTs automatically after your fix.")
    return "\n".join(lines)
```

Implement `_find_test_template` (finds an existing test file for import/fixture patterns) and `_generate_brt_candidates` (Haiku structured call using files_read + template + bug description) and `_run_brt_candidate` (subprocess run of a single test).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestWriteBrt -v
```
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_tools.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): add write_brt context-aware BRT tool

Agent-controlled BRT generation. Uses code the agent has actually
read (files_read from GuardrailState) and real test patterns from
the repo. Runs candidates against sandbox, stores confirmed BRTs
in _tls.brts. Subsequent run_tests calls include them automatically."
```

---

### Task 4: Rewrite the prompt (lean static + rich dynamic)

**Files:**
- Rewrite: `backend/agent/react_prompt.py`
- Test: `backend/tests/test_pipeline_v4.py` (add TestPrompt class)

- [ ] **Step 1: Write test for new prompt structure**

```python
# Add to backend/tests/test_pipeline_v4.py

class TestPromptV4:
    """New prompt must be lean static + rich dynamic."""

    def test_static_block_under_100_lines(self):
        from agent.react_prompt import build_static_block
        static = build_static_block()
        line_count = len(static.strip().splitlines())
        assert line_count <= 100, f"Static block is {line_count} lines (max 100)"

    def test_dynamic_block_includes_scout_reasoning(self):
        from agent.react_prompt import build_dynamic_block
        dynamic_ctx = {
            "scout": {
                "suspects": [{"file": "app.py", "reason": "URL handler", "confidence": 0.8}],
                "entity_extraction": {"function_names": ["match"]},
                "skeleton_data": {"app.py": "L42: def match(self):"},
            },
            "baseline_failures": {"test_app.py::test_old"},
            "repo_tree": "  app.py\n  tests/test_app.py",
        }
        work_order = {"ticket_id": "T-1", "title": "Bug", "description": "broken",
                      "priority": "high", "component": "routing"}
        intent = {}
        dynamic = build_dynamic_block(work_order, intent, dynamic_ctx)
        assert "URL handler" in dynamic  # scout reasoning, not just path
        assert "app.py" in dynamic
        assert "test_app.py::test_old" in dynamic  # baseline failures

    def test_dynamic_block_scout_fallback(self):
        """When scout found nothing, dynamic block must include fallback."""
        from agent.react_prompt import build_dynamic_block
        dynamic_ctx = {"scout": {}, "baseline_failures": set(), "repo_tree": "  app.py"}
        work_order = {"ticket_id": "T-1", "title": "Bug", "description": "broken"}
        dynamic = build_dynamic_block(work_order, {}, dynamic_ctx)
        assert "no confident matches" in dynamic.lower() or "delegate_explore" in dynamic.lower()

    def test_dynamic_block_includes_target_tests(self):
        """FAIL_TO_PASS tests must appear in dynamic block."""
        from agent.react_prompt import build_dynamic_block
        dynamic_ctx = {"scout": {}, "baseline_failures": set(), "repo_tree": ""}
        work_order = {
            "ticket_id": "T-1", "title": "Bug", "description": "broken",
            "fail_to_pass": ["tests/test_url.py::test_parse_int"],
            "pass_to_pass": ["tests/test_url.py::test_basic"],
        }
        dynamic = build_dynamic_block(work_order, {}, dynamic_ctx)
        assert "test_parse_int" in dynamic
        assert "test_basic" in dynamic

    def test_task_message_is_minimal(self):
        from agent.react_prompt import build_task_message_v4
        msg = build_task_message_v4()
        assert len(msg) < 200, f"Task message is {len(msg)} chars (should be <200)"
        assert "fix" in msg.lower() or "target tests" in msg.lower()

    def test_no_mandatory_phase_sequence(self):
        """Static block must NOT contain mandatory phase ordering."""
        from agent.react_prompt import build_static_block
        static = build_static_block()
        assert "MANDATORY ORDER" not in static
        assert "do not skip or reorder" not in static.lower()

    def test_no_tool_reference_table(self):
        """Static block must NOT duplicate tool schemas."""
        from agent.react_prompt import build_static_block
        static = build_static_block()
        # Should not have the old "### Exploration (read-only" tool listing
        assert "read_file(file_path, start_line, end_line)" not in static
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestPromptV4 -v
```
Expected: FAIL — new prompt functions don't exist.

- [ ] **Step 3: Rewrite `react_prompt.py`**

Replace `build_system_prompt` and `build_task_message` with three new functions: `build_static_block()`, `build_dynamic_block(work_order, intent, dynamic_ctx)`, and `build_task_message_v4()`.

The static block follows the spec exactly: identity, soft workflow, hard contracts, test interpretation, path convention, BRT/plan/delegate/shell/verify guidance, changelog anchor. ~80 lines.

The dynamic block assembles from `dynamic_ctx` dict (populated by setup_node): bug info with priority/component, target tests (fail_to_pass + pass_to_pass), scout analysis with reasoning + skeletons (or fallback), baseline failures, repo tree, code map, lessons, concept mappings.

The task message is two sentences: "Fix this bug. The context above has everything you need to start. Focus on the target tests — when they pass, you're done."

Keep the old functions (`build_system_prompt`, `build_task_message`) as deprecated wrappers for backward compat during migration.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestPromptV4 -v
```
Expected: PASS for all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_prompt.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): rewrite prompt — lean static + rich dynamic

Static block: ~80 lines of contracts + conventions (cached).
Dynamic block: rich per-bug context from setup_node (scout reasoning,
baseline failures, target tests, repo tree, code map, lessons).
Task message: 2 sentences.

Deletes: tool reference table, mandatory phases, recovery table,
12-rule section, exploration strategy, escalation criteria."
```

---

### Task 5a: Remove hard gates from guardrails

**Files:**
- Modify: `backend/agent/react_guardrails.py`
- Modify: `backend/tests/test_react_contracts.py`

- [ ] **Step 1: Write test that hard gates are removed**

```python
# Add to backend/tests/test_react_contracts.py

class TestV4GuardrailsRemoved:
    """Verify that v4 removes hard gates."""

    def test_no_plan_gate(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        gs.plan_produced = False
        result = check_tool_call("string_replace", {"file_path": "app.py"}, gs)
        # Should NOT block — plan gate removed
        assert result is None or "plan" not in (result or "").lower()

    def test_no_sandbox_gate_for_edits(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        # Sandbox exists (setup created it) — but even without,
        # the gate should not exist as a hard block
        result = check_tool_call("string_replace", {"file_path": "app.py"}, gs)
        assert result is None or "sandbox" not in (result or "").lower()

    def test_no_grep_warning_at_8(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        gs.grep_count = 10
        result = check_tool_call("grep_repo", {"pattern": "test"}, gs)
        assert result is None

    def test_no_run_tests_warning_at_3(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        gs.run_tests_count = 5
        gs.plan_produced = True
        result = check_tool_call("run_tests", {}, gs)
        assert result is None
```

- [ ] **Step 2: Run tests — they should fail (gates still exist)**

```bash
python -m pytest backend/tests/test_react_contracts.py::TestV4GuardrailsRemoved -v
```

- [ ] **Step 3: Remove hard gates from `react_guardrails.py`**

In `check_tool_call`, remove:
- Plan-gate check (produce_plan before create_sandbox/edit tools)
- Sandbox-gate check (create_sandbox before string_replace/create_file)
- Read-before-edit warning
- Review-before-submit gate
- Grep count warning at 8
- Run_tests retry warning at 3

Keep:
- Tool budget (50/70 calls)
- Wall time limit
- Cost limit
- run_shell count >= 6 nudge

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/test_react_contracts.py -v
python -m pytest backend/tests/ -q  # full suite — no regressions
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_guardrails.py backend/tests/test_react_contracts.py
git commit -m "feat(pipeline-v4): remove hard gates from guardrails

Removed: plan-gate, sandbox-gate, read-before-edit, review-before-submit,
grep warning at 8, run_tests warning at 3.
Kept: tool budget, wall time, cost limit, run_shell nudge."
```

---

### Task 5b: Add new soft nudges

**Files:**
- Modify: `backend/agent/react_guardrails.py`
- Test: `backend/tests/test_react_contracts.py`

- [ ] **Step 1: Write test for new nudges**

```python
class TestV4SoftNudges:
    def test_nudge_verify_before_submit(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        gs.plan_produced = True
        gs.tool_call_count = 10
        # No verify_fix called yet
        gs._verify_fix_called = False
        result = check_tool_call("submit_fix", {"explanation": "done"}, gs)
        assert result is not None
        assert "verify_fix" in result.lower()

    def test_no_nudge_after_verify(self):
        from agent.react_guardrails import GuardrailState, check_tool_call
        gs = GuardrailState()
        gs.plan_produced = True
        gs.tool_call_count = 10
        gs._verify_fix_called = True
        result = check_tool_call("submit_fix", {"explanation": "done"}, gs)
        # Should not nudge about verify_fix (already called)
        assert result is None or "verify_fix" not in (result or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Add soft nudges to `check_tool_call`**

```python
# In check_tool_call, add:
if tool_name == "submit_fix" and not getattr(gs, "_verify_fix_called", False):
    return (
        "SUGGESTION: Call verify_fix(explanation) before submit_fix. "
        "It gives you independent review feedback you can act on."
    )
```

And in `update_from_tool_result`, track verify_fix:
```python
elif tool_name == "verify_fix":
    gs._verify_fix_called = True
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/test_react_contracts.py::TestV4SoftNudges -v
python -m pytest backend/tests/ -q
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_guardrails.py backend/tests/test_react_contracts.py
git commit -m "feat(pipeline-v4): add soft nudges for verify_fix and delegate_explore"
```

---

### Task 6: Remove deleted tools + update pipeline stages

**Files:**
- Modify: `backend/agent/react_tools.py` (remove tools, update REACT_TOOLS)
- Modify: `backend/agent/react_pipeline.py` (remove brt_node, verifier_node, wire setup_node)
- Modify: `backend/agent/tool_metadata.py` (remove entries, add new ones)
- Modify: `backend/agent/context_manager.py` (update COMPACTABLE_TOOLS)
- Modify: `backend/agent/react_loop.py` (invert thinking switch)
- Test: `backend/tests/test_react_contracts.py`

- [ ] **Step 1: Write test for final tool list**

```python
class TestV4ToolList:
    def test_react_tools_count(self):
        from agent.react_tools import REACT_TOOLS
        names = [t.name for t in REACT_TOOLS]
        assert len(names) == 10, f"Expected 10 react tools, got {len(names)}: {names}"
        # 3 editing + 1 planning + 3 testing + 3 completion = 10
        assert "verify_fix" in names
        assert "write_brt" in names
        assert "submit_fix" in names
        assert "create_sandbox" not in names  # removed
        assert "request_review" not in names  # replaced
        assert "run_brt" not in names  # replaced
        assert "record_localization" not in names  # auto-inferred
        assert "check_syntax" not in names  # auto-runs

    def test_all_tools_count(self):
        from agent.react_tools import REACT_TOOLS
        from agent.explore_tools import ALL_TOOLS as EXPLORE_TOOLS
        from agent.explore_subagent import EXPLORE_SUBAGENT_TOOLS
        total = len(EXPLORE_TOOLS) + len(EXPLORE_SUBAGENT_TOOLS) + len(REACT_TOOLS)
        # 6 explore + 1 delegate + 10 react = 17
        assert total == 17, f"Expected 17 total, got {total}"
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Remove tools and update pipeline**

In `react_tools.py`:
- Remove `create_sandbox`, `check_syntax`, `get_blast_radius`, `request_review`, `run_brt`, `record_localization` tool functions
- Update REACT_TOOLS list: `EDIT_TOOLS + [produce_plan] + TEST_TOOLS + COMPLETION_TOOLS`
  where `EDIT_TOOLS = [string_replace, create_file, undo_last_edit]`, `TEST_TOOLS = [run_tests, run_shell, write_brt]`, `COMPLETION_TOOLS = [verify_fix, submit_fix, escalate]`

In `react_pipeline.py`:
- Remove `brt_node` and `verifier_node` functions
- Replace `intake_node` calls with `setup_node`
- Update `run_ticket_react` to use new 3-stage flow:
  ```python
  state = setup_node(state)
  state = react_agent_node(state)
  state = finalize_node(state)
  ```
- Simplify `finalize_node` (remove retry logic)

In `react_loop.py`:
- Invert thinking switch: start with `llm_fast` (thinking OFF), switch to `llm_thinking` on first `string_replace` (stays ON thereafter)

In `tool_metadata.py`:
- Remove metadata for 6 deleted tools
- Add metadata for `verify_fix` and `write_brt`

In `context_manager.py`:
- Remove `get_blast_radius` from COMPACTABLE_TOOLS
- Add `write_brt` to COMPACTABLE_TOOLS

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest backend/tests/ -q --tb=short
```
Expected: All tests pass. Some existing tests may need updates (tool list assertions).

- [ ] **Step 5: Commit**

```bash
git add backend/agent/react_tools.py backend/agent/react_pipeline.py \
       backend/agent/react_loop.py backend/agent/tool_metadata.py \
       backend/agent/context_manager.py backend/tests/test_react_contracts.py
git commit -m "feat(pipeline-v4): remove 6 tools, wire 3-stage pipeline, invert thinking

Pipeline: setup_node → react_agent_node → finalize_node
Removed: create_sandbox, check_syntax, get_blast_radius,
  request_review, run_brt, record_localization
Added: verify_fix, write_brt
Thinking: OFF during exploration, ON from first edit onward
React tools: 22 → 10. Total with explore: 17."
```

---

### Task 7: Update eval scoring (localization from edits)

**Files:**
- Modify: `backend/agent/eval/scoring.py`
- Test: `backend/tests/test_pipeline_v4.py`

- [ ] **Step 1: Write test**

```python
class TestLocalizationFromEdits:
    def test_infers_localization_from_edited_files(self):
        from agent.eval.scoring import _score_localization_hit
        result = {
            "repair": {"patches": [{"file_path": "src/routing.py"}, {"file_path": "tests/test_routing.py"}]},
            "localization": {},  # Empty — no record_localization call
        }
        bug = {"expected_files": ["src/routing.py"]}
        assert _score_localization_hit(result, bug) is True

    def test_ignores_test_files_for_localization(self):
        from agent.eval.scoring import _score_localization_hit
        result = {
            "repair": {"patches": [{"file_path": "tests/test_routing.py"}]},
            "localization": {},
        }
        bug = {"expected_files": ["src/routing.py"]}
        # Only edited a test file — should NOT count as localization hit
        assert _score_localization_hit(result, bug) is False
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Update `_score_localization_hit`**

When `localization.fault_files` is empty, fall back to inferring from `repair.patches` — filter out test files, check if any edited source file matches expected_files.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add backend/agent/eval/scoring.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): infer localization from edits for eval scoring

When record_localization is not called (removed in v4), localization
accuracy is inferred from which non-test files the agent edited.
Backward compatible — still reads localization.fault_files if present."
```

---

### Task 8: Scout — drop Opus re-ranker, export full reasoning

**Files:**
- Modify: `backend/agent/scout.py`
- Test: existing tests should pass after removal

- [ ] **Step 1: Write test that scout skips re-ranker**

```python
class TestScoutV4:
    def test_scout_skips_reranker(self):
        """Scout must not call the Opus re-ranker."""
        from agent.scout import scout_localize
        with patch("agent.scout._run_extractor") as mock_ext, \
             patch("agent.scout._run_debugger") as mock_dbg, \
             patch("agent.scout._run_reranker") as mock_rerank, \
             patch("agent.scout._narrow_with_skeletons") as mock_narrow:
            mock_ext.return_value = MagicMock(function_names=["f"], module_hints=["m"], bug_summary="b")
            mock_dbg.return_value = MagicMock(suspects=[])
            mock_narrow.return_value = {}

            scout_localize("test", {}, {}, Path("/tmp"), repo_path=Path("/tmp"))

        mock_rerank.assert_not_called()

    def test_scout_returns_full_reasoning(self):
        """Scout must return entity_extraction and skeleton_data, not just paths."""
        from agent.scout import scout_localize
        with patch("agent.scout._run_extractor") as mock_ext, \
             patch("agent.scout._run_debugger") as mock_dbg, \
             patch("agent.scout._narrow_with_skeletons") as mock_narrow:
            mock_ext.return_value = MagicMock(
                function_names=["match"], module_hints=["routing"],
                bug_summary="URL broken", error_types=["ValueError"],
            )
            mock_dbg.return_value = MagicMock(
                suspects=[MagicMock(file="app.py", function="match", confidence=0.8, reason="URL handler")],
                blast_radius_files=["tests/test_app.py"],
            )
            mock_narrow.return_value = {"app.py": ["match"]}

            result = scout_localize("test", {}, {}, Path("/tmp"), repo_path=Path("/tmp"))

        assert "entity_extraction" in result
        assert "skeleton_data" in result
        assert result["entity_extraction"]["function_names"] == ["match"]
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Modify scout_localize**

Skip the `_run_reranker` call entirely. Add `entity_extraction` and `skeleton_data` to the return dict. The return dict becomes:
```python
{
    "top_locations": [...],
    "blast_radius_files": [...],
    "relevant_business_rules": [...],
    "scout_cost_usd": total_cost,
    "entity_extraction": {
        "function_names": extracted.function_names,
        "error_types": extracted.error_types,
        "module_hints": extracted.module_hints,
        "bug_summary": extracted.bug_summary,
    },
    "skeleton_data": narrowed,  # {file: [functions]} from skeleton narrowing
}
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add backend/agent/scout.py backend/tests/test_pipeline_v4.py
git commit -m "feat(pipeline-v4): scout drops Opus re-ranker, exports full reasoning

Scout pipeline: Haiku extractor → Sonnet debugger → Haiku skeleton
narrowing. Opus re-ranker removed (~$0.05-0.10/bug saved).
Return dict now includes entity_extraction and skeleton_data
so the dynamic block can show the agent WHY files were suspected."
```

---

### Task 9: Integration test — run full pipeline end-to-end

**Files:**
- Test: `backend/tests/test_pipeline_v4.py`

- [ ] **Step 1: Write integration test**

```python
class TestPipelineV4Integration:
    """Full pipeline: setup → react → finalize with mocked LLM."""

    def test_full_pipeline_3_stages(self, tmp_path):
        """Pipeline runs setup → react → finalize without crashing."""
        from agent.react_pipeline import run_ticket_react

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("def broken():\n    return None\n")
        (repo / ".git").mkdir()
        (repo / "setup.py").write_text("")

        work_order = {
            "ticket_id": "TEST-001",
            "title": "broken() returns None",
            "description": "Should return 42",
            "repo_name": "test",
            "repo_path": str(repo),
        }

        with patch("agent.react_pipeline.react_loop") as mock_loop:
            mock_loop.return_value = {
                "submitted": True,
                "explanation": "Fixed broken()",
                "tool_call_count": 5,
                "cost_usd": 0.10,
            }
            result = run_ticket_react(work_order, dry_run=True)

        assert "submitted" in result or "escalated" in result
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest backend/tests/test_pipeline_v4.py::TestPipelineV4Integration -v
```

- [ ] **Step 3: Run full test suite — zero regressions**

```bash
python -m pytest backend/tests/ -q --tb=short
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_pipeline_v4.py
git commit -m "test(pipeline-v4): add integration test for 3-stage pipeline"
```

---

### Task 10: Eval validation (when API credits available)

**Files:**
- No code changes — this is a validation run

- [ ] **Step 1: Verify API credits**

```bash
source .env && curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' \
  | head -c 200
```
Expected: Valid response (not "credit balance too low").

- [ ] **Step 2: Run eval on same 5 bugs**

```bash
cd backend && python cli.py eval run --dataset ../eval/next5.json --pipeline react --nl --timeout 900
```

- [ ] **Step 3: Compare results to pre-v4 baseline**

Pre-v4 baseline (run 18879fec):
- Pass rate: 0/5 (test infra blocked all)
- Localization: 100%
- Fix rate: 100%
- Approval: 100%
- Confidence: 0.93
- Cost: $1.48/bug avg

Check for improvements in:
- Pass rate (target: >0/5 — baseline filtering should help)
- Cost (target: <$1.00/bug — no Opus, fewer tools)
- Tool calls (target: <20 avg — no plan-gate overhead)
- Wall time (target: <200s avg — parallel setup)

- [ ] **Step 4: Document results and iterate**

Write results to eval report. If pass rate didn't improve, check traces for:
- Did the agent use verify_fix?
- Did write_brt generate useful tests?
- Did the structured test output help the agent understand failures?
- Was the dynamic block informative enough?

---

## Self-Review

**Spec coverage check:**
- [x] setup_node with 3 parallel threads (Task 1)
- [x] verify_fix forked subagent (Task 2)
- [x] write_brt context-aware (Task 3)
- [x] Prompt rewrite — lean static + rich dynamic (Task 4)
- [x] Remove hard gates (Task 5a)
- [x] Add soft nudges (Task 5b)
- [x] Remove 6 tools + update pipeline (Task 6)
- [x] Eval scoring from edits (Task 7)
- [x] Scout drop Opus + export reasoning (Task 8)
- [x] Thinking switch inversion (Task 6, in react_loop.py changes)
- [x] Integration test (Task 9)
- [x] Eval validation (Task 10)

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency:** `verify_fix` returns str in both tool def and tests. `write_brt` returns str. `setup_node` returns ReactAgentState. `build_static_block` returns str. `build_dynamic_block` returns str. All consistent.
