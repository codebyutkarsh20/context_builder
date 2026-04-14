# AI Deploy Agent — Primathon (v3.5)

Autonomous bug-fix agent: ticket → localize → plan → fix → test → submit PR.
**80% pass rate on sentinel, $0.37/bug avg, 16 tool calls avg.**

## Commands
```bash
python -m pytest backend/tests/ -q              # 837 tests, run from project root
cd backend && python cli.py eval run             # Full 35-bug eval
cd backend && python cli.py eval run --sentinel  # Fast 5-bug check
cd backend && python cli.py eval run --bug X     # Single bug
cd backend && python cli.py build /path/to/repo  # Build knowledge graph
cd backend && python cli.py fix TICKET --repo /  # Fix a bug
docker-compose up                                # Neo4j + backend + frontend
```

## Architecture (what you need to know)
- **react_loop.py** — the main agent while-loop. 24 tools. Extended thinking on early turns.
- **react_tools.py** — `produce_plan` gates `create_sandbox`. `undo_last_edit` reverts edits.
- **react_guardrails.py** — plan gate, sandbox gate, submit gate, stuck detection (auto-replan).
- **context_manager.py** — microcompact (cache-friendly eviction). Never compacts test/edit results.
- **explore_subagent.py** — Haiku delegation for "find me X" questions (`delegate_explore` tool).
- **learn_from_fix.py** — per-repo `agent_lessons.md` persists across runs. Disable: `DISABLE_LEARN_FROM_FIX=1`.
- **eval/repo_manager.py** — venv setup + dep-compat for SWE-bench repos (Django, Flask, Werkzeug, etc).
- **web_tools.py** — `web_fetch` + `web_search`. Disabled by default. Enable: `ENABLE_WEB_TOOLS=1`.
- **forked_subagent.py** — verifier reuses parent's prompt cache via `CacheSafeParams`.

## Critical rules
- MUST call `produce_plan()` before `create_sandbox()` — guardrail enforces this
- MUST run target repo's tests in sandbox before submitting
- NEVER modify files outside the sandbox worktree
- NEVER hardcode repo-specific knowledge — it goes in the knowledge graph or agent_lessons.md

## Env vars that change behavior
- `REACT_THINKING_BUDGET=2048` — thinking tokens on early turns (0 to disable)
- `DISABLE_REACT_THINKING=1` — skip extended thinking entirely
- `REACT_REFRESH_INTERVAL=5` — status refresh every N tool calls
- `ENABLE_WEB_TOOLS=1` — enable web_fetch + web_search
- `DISABLE_LEARN_FROM_FIX=1` — skip lesson recording/loading

## Skill routing
When the user's request matches a skill, invoke it FIRST:
- Bugs/errors → investigate
- Ship/deploy/PR → ship
- Code review → review
- Architecture → plan-eng-review
- Brainstorming → office-hours

## MCP: code-review-graph
**Use graph tools BEFORE Grep/Glob/Read.** The graph gives structural context (callers, tests, blast radius) that file scanning cannot.

| Tool | When |
|------|------|
| `detect_changes` | Code review |
| `get_impact_radius` | Blast radius |
| `query_graph` | Callers, callees, imports, tests |
| `semantic_search_nodes` | Find by name/keyword |
| `get_architecture_overview` | High-level structure |
