# Context Builder Agent: Operational Manifest

> **Last updated:** 2026-04-10
> **Agent architecture:** Single ReAct loop (react_pipeline.py → react_loop.py)
> **Model:** claude-sonnet-4-6 with prompt caching (~87% savings on prefix)
> **Tool budget:** 40 calls hard limit, ~30 recommended
> **Context window:** 160K tokens

## 1. What the Agent Does Well

- **Localization:** 96% accuracy on eval dataset. Scout pre-localization (Haiku + Sonnet + Opus reranker) narrows to top-5 files before the ReAct loop starts.
- **Single-file fixes:** When the bug is in one file and the agent reads the whole file, it produces correct fixes.
- **Code understanding:** When given enough code to read (not just grep snippets), the agent reasons correctly about logic errors.

## 2. What the Agent Struggles With

- **Grep spirals:** The #1 failure mode. Agent greps 15+ times for function names that don't exist in the code. The bug description uses different terminology than the code.
  - **Fix applied:** Prompt now says "read the whole file first, grep only to find which file."
  - **Fix applied:** read_file now returns entire files (30K cap), not 80-line windows.
  - **Remaining risk:** Agent may still grep excessively if the prompt isn't strong enough.

- **Multi-file refactors:** Performance bugs (like "fetch requisition once instead of 3 times") require understanding code flow across functions. Single ReAct loop struggles here.
  - **Not yet fixed:** Needs coordinator/worker architecture (planned for v0.4).

- **Test environment:** Target repos often don't have pytest installed in the sandbox. Tests fail with "No module named pytest" instead of running.
  - **Fix applied:** sandbox.py now detects infra failures and classifies them as "error:" (skippable), not "failed" (blocking).

- **Reviewer disagreements:** The Opus reviewer sometimes rejects correct fixes due to truncated diffs or stylistic concerns.
  - **Fix applied:** Diff truncation fixed (uses --unified=3, retries with --unified=1).
  - **Fix applied:** Submit gate no longer requires review approval (just warns).
  - **Fix applied:** PR created even on escalation if edits + review exist.

## 3. Current Tool Set (18 tools)

### Exploration (read-only, concurrent-safe)
| Tool | Output Cap | What It Does |
|------|-----------|--------------|
| `read_file` | 30K chars | Reads entire file by default. The primary exploration tool. |
| `read_function` | 15K chars | Extracts one complete function by name. |
| `grep_repo` | 8K chars | Regex search with 2 context lines. For finding WHICH file, not understanding code. |
| `get_file_structure` | 8K chars | Shows all classes/functions with line numbers. File map. |
| `get_function_info` | 3K chars | Function metadata from knowledge graph (callers, callees). |
| `list_files` | 3K chars | Directory listing. |
| `get_blast_radius` | 3K chars | Impact analysis — how many files depend on this file. |

### Editing (requires sandbox)
| Tool | What It Does |
|------|--------------|
| `string_replace` | Replace exact string. 2 strategies: exact match + whitespace-normalized. Auto-runs ruff --fix after. |
| `check_syntax` | Validates Python syntax via ast.parse(). |
| `create_file` | Creates new file in sandbox. |

### Sandbox & Testing
| Tool | What It Does |
|------|--------------|
| `create_sandbox` | Creates git worktree at /tmp/agent_sandbox_*. |
| `run_tests` | Auto-detects test framework. Classifies infra failures separately from assertion failures. |
| `run_brt` | Runs Bug Reproduction Tests if available. |

### Completion
| Tool | What It Does |
|------|--------------|
| `record_localization` | Records fault files/functions/hypothesis. |
| `request_review` | Independent Opus review of the diff. |
| `submit_fix` | Marks fix as done. Requires sandbox + tests attempted. |
| `escalate` | Hands off to human. |
| `get_callers` | Finds files that call/import a function. |

## 4. Guardrails (Soft, Not Blocking)

All guardrails are **warnings only** — the agent is never hard-blocked from exploring.

| Trigger | Warning |
|---------|---------|
| grep_repo >= 8 calls | "Consider read_function instead" |
| run_tests >= 3 calls | "If tests can't run, proceed to submit" |
| request_review >= 2 calls | "If reviewer keeps rejecting, submit directly" |
| Tests run without edits | "You haven't edited anything yet" |

Only **hard gates**:
- Sandbox required before editing
- Tests must be attempted before submit

## 5. Architecture

```
react_pipeline.py
  ├─ intake_node: Haiku parses ticket → IntentAnalysis
  ├─ scout: Haiku + Sonnet + Opus → top-5 file locations
  ├─ build_kickstart_context: graph + failure signals → orientation
  ├─ build_system_prompt: tools + context + rules
  └─ react_loop: Sonnet with 18 tools, 40-call budget
       ├─ Concurrent batching for read-only tools
       ├─ Three-layer context management (cap → mask → summarize)
       ├─ Multi-stage recovery (prompt-too-long, max-output-tokens)
       └─ finalize_node: create PR or escalate (PR on escalation if edits exist)
```

## 6. What's NOT Available

- **MCP graph tools:** Available to Claude Code users, NOT to the ReAct agent. The agent uses the knowledge graph via `graph_utils.py` during kickstart context assembly, but cannot call MCP tools during the loop.
- **Coordinator/worker pattern:** Single loop only. Multi-file bugs handled in one context.
- **Container isolation:** Sandbox is a git worktree, not a Docker container.
- **Production feedback:** No Sentry/PagerDuty integration. Failure records from git history only.
- **Embeddings/vector search:** Removed. Context comes from Neo4j graph + keyword search.

## 7. Cost Profile

| Phase | Model | Typical Cost |
|-------|-------|-------------|
| Intake | Haiku | $0.002 |
| Scout (3 calls) | Haiku + Sonnet + Opus | $0.05 |
| ReAct loop (15-25 calls) | Sonnet (cached) | $0.30-1.50 |
| Review | Opus | $0.10 |
| **Total per bug** | | **$0.50-1.70** |

Prompt caching saves ~87% on the system prompt across 30+ Sonnet calls.
