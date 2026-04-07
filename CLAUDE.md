# AI Deploy Agent — Primathon
## Mission
We are building an AI agent that autonomously fixes production bugs and makes
minor enhancements, targeting 100 reliable deploys/day within 3 months.
### Milestone 1 (Current Focus)
Agent reads a Jira bug → localizes the fault → generates a fix → reviews it → raises a PR.
**Success: 80% of PRs are approved by humans without changes.**
### Problem Statement 1: Context Creation
Pick any repo, create deep context on it. With that context alone (without code),
the LLM MUST be able to answer nuanced business logic questions.
This is solved by building a dual-layer knowledge graph:
- CODE LAYER: Files → Classes → Functions → Calls/Imports (via Tree-sitter + Neo4j)
- BUSINESS LAYER: BusinessRules, DomainConcepts, FailureRecords, ExternalDependencies
  linked to code entities via ENFORCED_BY, REPRESENTED_BY, RESULTED_IN_CHANGE edges
### Problem Statement 2: Autonomous Bug-Fixing Agent
ReAct pipeline: single agent loop with 19 tools decides explore → localize → edit → test → review → submit.
Evolved from fixed 8-node LangGraph to ReAct architecture (v0.2.0.0, April 2026).
## Tech Stack
- Python 3.11 / FastAPI (agent API)
- LangGraph (agent orchestration)
- Neo4j (code + business knowledge graph)
- ChromaDB (vector embeddings for semantic fallback)
- Claude API (LLM reasoning, repair, review)
- Tree-sitter (AST parsing for graph construction)
- GitHub Actions (CI/CD + PR creation)
- Jira API (bug ticket intake)
- Unleash (feature flags)
- Prometheus + Grafana (monitoring)
## Architecture
agent/
├── react_pipeline.py  # 3-node LangGraph: intake → react_agent → finalize (DEFAULT)
├── react_loop.py      # Core ReAct while-loop with 19 tools
├── react_tools.py     # Sandbox-aware edit/test/review/submit tools
├── react_prompt.py    # System prompt builder with context assembly
├── react_guardrails.py # Safety: sandbox gate, submit gate, anti-pattern detection
├── context_manager.py # Context window management (observation masking + Haiku summarization)
├── pipeline.py        # Legacy utility library (routing functions, helpers — runtime deprecated)
├── explore_tools.py   # Read-only codebase exploration tools (8 tools)
├── types.py           # AgentState, ReactAgentState, Pydantic models
├── trace.py           # RunTrace observability with SSE streaming
├── sandbox.py         # Git worktree sandbox + test execution
├── patch_utils.py     # Fuzzy patch matching + syntax checking
├── eval/              # Unified eval package (A/B comparison, 25-bug dataset)
├── graph/
│   ├── builder.py     # Tree-sitter → Neo4j pipeline
│   ├── business/      # Business knowledge extraction + ingestion
│   └── queries.py     # Cypher query templates
└── embeddings/        # ChromaDB pipeline
## Build & Test
```bash
pip install -r backend/requirements.txt
pytest tests/ -v                                           # All tests
cd backend && python cli.py build /path/to/repo            # Build knowledge graph
cd backend && python cli.py fix TICKET --repo /path        # Fix a bug
cd backend && python cli.py eval run                       # Run 25-bug eval suite
cd backend && python cli.py eval run --bug FLASK-2651      # Single bug eval
cd backend && python cli.py eval report                    # Show latest results
cd backend && python cli.py eval gate results/latest.json  # CI regression gate
```
## Key Design Rules
- MUST use the ReAct pipeline (react_pipeline.py) for all agent runs
- MUST query BOTH code graph AND business knowledge graph before generating fixes
- MUST include business context (rules, constraints, past incidents) in every repair prompt
- MUST run target repo's test suite in sandbox before creating PR
- MUST cap developer↔reviewer loop at 3 iterations, then escalate to human
- NEVER let agent modify files outside target repo
- NEVER hardcode repo-specific knowledge in agent code — it belongs in the knowledge graph
- NEVER remove validation checks from target code without explicit business rule confirmation

## Current State (April 2026)
- [x] Knowledge graph construction pipeline (Tree-sitter → Neo4j)
- [x] Business knowledge extraction from git history
- [x] Context assembly with graph + embeddings + failure signals
- [x] ReAct agent with 19 tools (explore, edit, test, review, submit)
- [x] Multi-file coordination tools (get_callers, get_blast_radius)
- [x] Prompt caching (Anthropic server-side, ~87% savings on prefix)
- [x] Context window management (observation masking + Haiku summarization)
- [x] Eval suite: 25 bugs from SWE-bench + open source, A/B comparison
- [x] PR creation pipeline (dry_run + real PR via gh CLI)
- [ ] Real Jira integration (currently mock intake)
- [ ] Multi-candidate patch sampling (generate 3-5, pick best)
- [ ] Production feedback loop (Sentry/PagerDuty → graph)
- [ ] Container sandbox isolation (Docker --network none)

### Eval Results (25-bug dataset, ReAct pipeline)
- Localization: 96% (24/25 find the right file)
- Submit rate: 40% (10/25 produce a fix)
- Pass rate: 40% (10/25 fix + correct file)
- Avg cost: $1.92/bug, avg 32 tool calls for successes

## Dashboard
A React + FastAPI dashboard visualizes the knowledge graph and context output.

### Stack
- Frontend: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui
- Graph viz: react-force-graph-2d (force-directed, node coloring by type)
- Charts: Recharts
- Backend: FastAPI serving graph data and context layers

### Panels
| Panel | Description |
|-------|-------------|
| Overview | Repo stats (files, functions, classes, LOC), tech stack badges, build progress |
| Knowledge Graph | Interactive force graph — File=blue, Class=green, Function=orange, BusinessRule=purple |
| Context Layers | 6-layer breakdown cards with token budget usage and completeness % |
| File Explorer | Tree view with PageRank hotspot score per file |
| Hotspots | Top-10 most-referenced symbols ranked by PageRank |
| Business Rules | Extracted rules table linked to code nodes |
| Context Viewer | Full context.md rendered as markdown |

### Running the Dashboard
```bash
docker-compose up           # Neo4j + backend + frontend
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
# Neo4j:    http://localhost:7474
```

## Reference Docs
@docs/architecture.md for detailed system architecture
@docs/business-rules.md for extracted business knowledge
@docs/failure-playbooks.md for past incident lessons
@docs/domain-glossary.md for domain terminology definitions

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
