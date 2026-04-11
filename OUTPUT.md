# Context Builder — Full Codebase Analysis

**Version**: 0.3.0.0 (April 2026)
**Analyzed**: 2026-04-09

---

## Project Overview

Context Builder is an AI-powered autonomous code repair agent that reads a bug ticket (Jira or mock), builds deep structural + business context about the target codebase using a dual-layer knowledge graph (code + business), then autonomously localizes the fault, generates a fix, runs tests in a sandboxed git worktree, gets an independent AI review, and raises a PR. The system targets 100 reliable deploys/day within 3 months, with the current milestone being 80% of PRs approved by humans without changes. Current eval results on a 25-bug dataset show 96% localization accuracy and 40% end-to-end pass rate at $1.92/bug average cost.

### Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.11 |
| Web Framework | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph (3-node pipeline), custom ReAct while-loop |
| LLM | Anthropic Claude (Sonnet 4.6 for agent, Haiku 4.5 for intake/summarization) |
| Graph Database | Neo4j 5.24 (code + business knowledge) |
| Vector Database | ChromaDB 0.5.15 (semantic search) |
| AST Parsing | Tree-sitter (Python, JS, TS, Go, Java) |
| Community Detection | Leiden algorithm (python-igraph + leidenalg), networkx fallback |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui |
| Graph Visualization | react-force-graph-2d |
| Charts | Recharts |
| CI/CD | GitHub Actions + `gh` CLI for PR creation |
| Ticket Intake | Jira API (currently mock) |
| Feature Flags | Unleash (planned), JSON-based (current) |
| Monitoring | Prometheus + Grafana (infrastructure ready) |
| Containerization | Docker Compose (Neo4j + backend + frontend) |

### Repo Structure

```
context_builder/
├── backend/                    # Python backend (FastAPI + agent + graph pipeline)
│   ├── agent/                  # AI agent: ReAct loop, tools, guardrails, sandbox
│   │   ├── eval/               # 25-bug evaluation suite with A/B comparison
│   │   └── (18 modules)        # react_pipeline, react_loop, react_tools, etc.
│   ├── analyzer/               # Code analysis: Tree-sitter parsing, call graphs, flows
│   ├── api/                    # FastAPI routers: repos, graph, agent, knowledge, search
│   ├── compiler/               # Context document assembly (context.md, summary.md)
│   ├── embeddings/             # ChromaDB vector embedding pipeline
│   ├── enricher/               # Business logic extraction, decision points, domain concepts
│   ├── graph/                  # Neo4j graph builder, queries, community detection, business rules
│   ├── rag/                    # Graph RAG: query analysis, retrieval, context assembly
│   ├── tests/                  # Backend test suite
│   ├── cli.py                  # Typer CLI: build, fix, eval commands
│   ├── main.py                 # FastAPI app entry point
│   └── requirements.txt        # Python dependencies
├── frontend/                   # React dashboard
│   ├── src/
│   │   ├── components/         # Layout, ErrorBoundary, agent subcomponents
│   │   ├── pages/              # Overview, Agent, Knowledge pages
│   │   └── lib/                # API client, utils, RepoContext
│   ├── package.json
│   └── Dockerfile
├── eval/                       # Eval dataset (bugs.json, swe_bench_20.json, run_experiment.py)
├── data/                       # Generated context files, graph caches, embeddings
├── docker-compose.yml          # Neo4j (7474/7687) + backend (8001) + frontend (5173)
├── .claude/                    # Claude Code config: launch.json, settings, skills
├── .mcp.json                   # MCP server config for code-review-graph
├── CLAUDE.md                   # Main architecture doc + design rules
├── PLAN.md                     # Roadmap with CEO/Eng/Design reviews
├── CHANGELOG.md                # v0.1.0.0 -> v0.3.0.0
├── TODOS.md                    # Active + completed tasks (P1-P5)
├── PRD.md                      # Product requirements
└── VERSION                     # 0.3.0.0
```

---

## How Things Are Built

### Core Abstractions and Design Patterns

1. **Thread-Local Context Isolation**: `explore_tools.py`, `react_tools.py`, and `react_pipeline.py` use `threading.local()` to store per-run state (repo path, sandbox path, trace object). This enables concurrent agent runs without interference. Set via `set_exploration_context()` (`explore_tools.py:~30`) and `set_react_context()` (`react_tools.py:34-48`).

2. **LangChain Tool Decoration**: All agent tools use `@tool` from `langchain_core.tools` for automatic schema generation, parameter validation, and LLM binding. Tools are split into read-only (`explore_tools.py`, 10 tools) and write (`react_tools.py`, 8 tools).

3. **Pydantic Structured Output**: LLM responses are constrained to Pydantic models (`types.py:17-66`) — `IntentAnalysis`, `LocalizationResult`, `Patch`, `RepairResult`, `ReviewResult`. This enforces schema on LLM outputs.

4. **TypedDict State Machine**: Pipeline state flows through `AgentState` (`types.py:106-127`) and `ReactAgentState` (`types.py:130-160`) TypedDicts, carrying work order, intent, localization, repair, review, sandbox info, and observability data.

5. **Three-Layer Token Budget**: Context window management (`context_manager.py`) uses: (a) per-tool character caps (`read_file: 8000`, `grep_repo: 4000`), (b) sliding-window observation masking (keep last 15 tool results), (c) Haiku LLM summarization at 120K token threshold for a 160K context window.

6. **Five-Strategy Fuzzy Patch Matching**: `patch_utils.py:70-318` tries strategies 0-5: line-number replace, exact substring, whitespace-normalized, stripped-whitespace, sliding-window (>=92% similarity), and anchor-based matching. This handles LLM-generated patches that don't exactly match source.

7. **Guardrail State Machine**: `react_guardrails.py` tracks mutable state (`GuardrailState`, line 31-61) — tool counts, cost, elapsed time, sandbox status, test results — and enforces constraints before each tool call (sandbox gate, submit gate, anti-pattern detection).

### Data Flow: Input to Output

```
Bug Ticket (Jira/mock)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ INTAKE NODE (react_pipeline.py)                          │
│  1. Parse ticket → IntentAnalysis (Haiku LLM)           │
│  2. Classify community (Leiden cluster mapping)          │
│  3. Pre-localize: LLM hints + ChromaDB + graph neighbors │
│     → top-5 candidate files                              │
│  4. Build kickstart context (graph_utils.py:433-589):   │
│     repo map + function locator + call subgraph +        │
│     flow context + vector search + past failures +       │
│     business rules + PageRank hotspots                   │
│  5. Build system prompt (react_prompt.py:46-278)         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ REACT AGENT NODE (react_loop.py:71-350+)                 │
│  ReAct while-loop with 19 tools:                         │
│  EXPLORE (3-8 calls) → grep, read, search, get_callers  │
│  EDIT (3-5 calls) → create_sandbox, string_replace       │
│  VERIFY (3-5 calls) → check_syntax, run_tests           │
│  FINISH (1-2 calls) → request_review, submit_fix         │
│                                                           │
│  Guards: MAX_TOOL_CALLS=40, MAX_WALL_TIME=900s,          │
│          MAX_COST_USD=$5.00, sandbox required for edits   │
│  Context: observation masking + Haiku summarization       │
│  Model: claude-sonnet-4-6 with prompt caching             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ FINALIZE NODE                                             │
│  1. Create PR via `gh pr create`                          │
│  2. Create feature flag for rollback                      │
│  3. Clean up sandbox worktree                             │
│  4. Write results + trace to disk                         │
└─────────────────────────────────────────────────────────┘
```

### Key Modules and What Each Owns

| Module | Owner | Key Files |
|--------|-------|-----------|
| **Agent Core** | ReAct pipeline orchestration | `react_pipeline.py` (3-node chain), `react_loop.py` (while-loop), `react_prompt.py` (system prompt builder) |
| **Agent Tools** | Tool definitions for LLM | `explore_tools.py` (10 read-only tools), `react_tools.py` (8 write tools: edit, sandbox, test, review, submit) |
| **Agent Safety** | Guardrails + cost control | `react_guardrails.py` (limits, gates, anti-patterns), `context_manager.py` (token budget) |
| **Sandbox** | Git worktree isolation + test runner | `sandbox.py` (create/cleanup worktree, auto-detect test framework, run tests) |
| **Patching** | Apply LLM patches to source | `patch_utils.py` (5-strategy fuzzy matching, syntax check, deduplication) |
| **Graph Construction** | Neo4j knowledge graph | `graph/builder.py` (Tree-sitter -> Neo4j), `graph/neo4j_client.py` (connection singleton) |
| **Graph Queries** | Cypher query templates | `graph/queries.py` (nodes, edges, hotspots, dead code, execution flows, change impact) |
| **Community Detection** | Codebase clustering | `graph/community.py` (Leiden algorithm, IDF-weighted naming, max 15 communities) |
| **Business Knowledge** | Rules, failures, decisions | `graph/business/failure_records.py` (git history mining), `graph/business/persist.py` (Neo4j persistence), `enricher/business_logic.py` (rule extraction) |
| **Code Analysis** | AST parsing + call graphs | `analyzer/code_parser.py` (Python), `analyzer/multi_lang_parser.py` (JS/TS/Go/Java), `analyzer/call_graph.py` (directed graph builder) |
| **Flow Analysis** | Execution paths + changes | `analyzer/flows.py` (flow detection, criticality scoring), `analyzer/changes.py` (git diff -> affected nodes) |
| **Enrichment** | Business intelligence layer | `enricher/decision_points.py` (conditional classification), `enricher/domain_concepts.py` (entity extraction), `enricher/summarizer.py` (LLM summaries) |
| **Embeddings** | Vector search | `embeddings/embedder.py` (ChromaDB pipeline, multiple model support) |
| **RAG** | Query-time retrieval | `rag/query_analyzer.py` (intent extraction), `rag/retriever.py` (vector + graph + keyword), `rag/context_assembler.py` (token-budgeted assembly) |
| **API** | FastAPI endpoints | `api/repos.py` (analysis jobs), `api/graph.py` (visualization), `api/agent.py` (agent jobs), `api/knowledge.py` (rules Q&A), `api/chat.py` (Graph RAG chat) |
| **Eval** | Performance measurement | `agent/eval/runner.py` (EvalRunner), `agent/eval/scoring.py` (11 metrics), `agent/eval/report.py` (markdown reports), `agent/eval/dataset.py` (25-bug schema) |
| **Frontend** | Dashboard UI | 3 pages (Overview, Agent, Knowledge), SSE trace streaming, force-graph visualization |

### How Modules Talk to Each Other

- **Direct Python imports**: All backend modules communicate via direct imports. `react_pipeline.py` imports from `react_loop`, `react_prompt`, `react_tools`, `graph_utils`, `types`, `trace`. `react_loop.py` imports `context_manager`, `react_guardrails`, `react_tools`.
- **Neo4j Cypher**: Graph queries go through `graph/neo4j_client.py` singleton → Neo4j bolt protocol (port 7687). Used by `graph/builder.py`, `graph/queries.py`, `graph_utils.py`, `enricher/summarizer.py`.
- **ChromaDB**: Vector operations go through `embeddings/embedder.py` `NodeEmbedder` class → ChromaDB local persistence.
- **Anthropic API**: LLM calls go through `langchain_anthropic.ChatAnthropic` (in `react_loop.py`) or raw `anthropic.Anthropic` client (in `enricher/summarizer.py`, `api/chat.py`).
- **HTTP REST**: Frontend (`api.ts`) → Vite dev proxy → FastAPI backend (port 8000). SSE streaming for agent traces via EventSource.
- **Git subprocess**: `sandbox.py` and `react_tools.py` use `subprocess.run(["git", ...])` for worktree creation, branch management, and PR creation.
- **Thread-local state**: Agent tools access repo path, sandbox path, and trace via `threading.local()` — no global mutable state.

---

## AI/Agent Architecture

### LLM APIs Used

| Model | Usage | Module |
|-------|-------|--------|
| `claude-sonnet-4-6` | Main ReAct agent loop (explore, edit, test, review) | `react_loop.py:~48` (`REACT_MODEL`) |
| `claude-haiku-4-5-20251001` | Intake node (ticket parsing), context summarization, community classification | `context_manager.py:34`, `react_pipeline.py:~42` |
| `claude-haiku-4-5-20251001` | File/function business summarization | `enricher/summarizer.py` (configurable via `SUMMARIZER_MODEL`) |

**Cost tracking** (`react_loop.py:51-68`): Per-call cost estimation accounting for prompt caching — cache write 1.25x, cache read 0.1x base price. Pricing: Sonnet $3/$15 per 1M tokens (input/output), Haiku $0.80/$4, Opus $15/$75.

### Prompt Strategy

The system uses a **multi-section system prompt** assembled at runtime (`react_prompt.py:46-278`):

1. **Role**: "You are a senior engineer fixing a bug in a production codebase"
2. **Workflow phases**: EXPLORE (3-8 calls) -> EDIT (3-5 calls) -> VERIFY (3-5 calls) -> FINISH (1-2 calls)
3. **Tool budget**: ~30 total tool calls, with per-phase guidance
4. **Recovery patterns table**: 11 rows mapping common failure states to corrective actions (e.g., "No sandbox exists" -> "Call create_sandbox NOW")
5. **Anti-patterns**: Explicit warnings against grep spam, read loops, test spiral, edit churn
6. **Full tool documentation**: Inline docs for all 19 tools
7. **Kickstart context**: Pre-assembled graph knowledge (repo map, call subgraph, flows, business rules, PageRank hotspots)
8. **Conventions section**: Project-specific linter/formatter/test rules from `project_conventions.json`
9. **Business rules section**: Extracted rules with ENFORCED_BY links to code

**Task message** (`react_prompt.py:281-327`): Adapts action verb per task type ("Fix this bug" / "Implement this feature" / "Make this change"), includes location hints from pre-localization, and provides an efficient 9-call reference sequence.

**Prompt caching**: System prompt uses `cache_control: {"type": "ephemeral"}` on the system message (`react_loop.py`). Since the system prompt is stable across 30+ LLM calls per bug fix, Anthropic server-side caching gives ~87% savings on the prefix.

### Tool/Function Calling Setup

**19 tools** split into two categories:

**Read-only exploration tools** (`explore_tools.py`):
| Tool | Purpose |
|------|---------|
| `grep_repo` | Ripgrep search with glob filtering (falls back to GNU grep) |
| `read_file` | Read file with optional line range |
| `read_function` | Extract single function via Tree-sitter or regex |
| `get_file_structure` | Classes/functions/imports without code body |
| `get_file_summary` | Semantic description of file purpose |
| `search_code` | ChromaDB semantic search |
| `list_files` | Directory listing with extension filter |
| `get_function_info` | Function metadata from graph |
| `get_callers` | Files that import/call a function |
| `get_blast_radius` | Risk assessment (LOW/MEDIUM/HIGH/CRITICAL) |

**Write/sandbox tools** (`react_tools.py`):
| Tool | Purpose |
|------|---------|
| `create_sandbox` | Create git worktree at `/tmp/agent_sandbox_{name}_{uuid}` |
| `string_replace` | Replace exact string in sandbox file (whitespace-normalized fallback) |
| `create_file` | Create new file in sandbox |
| `check_syntax` | Validate Python syntax via `ast.parse()` |
| `run_tests` | Auto-detect and run test suite (pytest/npm/make) |
| `run_brt` | Run Bug Reproduction Tests specifically |
| `request_review` | Get independent AI review (separate LLM call) |
| `submit_fix` | Terminal: mark as done |
| `escalate` | Terminal: hand off to human |

**Tool output capping** (`context_manager.py:63-68`): `read_file: 8000 chars`, `grep_repo: 4000`, `run_tests: 6000`, default: `4000`. This prevents a single tool result from consuming the context window.

### Context Assembly — How Repo Knowledge Reaches the LLM

Context is assembled in layers during the intake phase (`graph_utils.py:433-589`, `build_kickstart_context()`):

1. **Repo map** (lines 226-262): Directory tree structure showing file organization
2. **Function locator** (lines 265-311): Hint functions from pre-localization + their 1-hop graph neighbors
3. **Call subgraph** (lines 314-385): 2-hop call graph around the hint area showing CALLS/IMPORTS/INHERITS edges
4. **Flow context** (lines 388-430): Execution flows (API routes, CLI commands, event handlers) touching the hint area
5. **Vector search** (lines 463-480): ChromaDB semantic similarity results for the bug description
6. **Graph neighbors** (lines 482-514): Direct CALLS/IMPORTS relationships from Neo4j
7. **Past failures** (lines 516-537): FailureRecord nodes from git history (past incidents in the same area)
8. **Business rules** (lines 539-561): Extracted rules linked to the hint area via ENFORCED_BY edges
9. **PageRank hotspots** (lines 563-582): Most central functions in the codebase by PageRank score

**Pre-localization** (`react_pipeline.py:95-167`) narrows the search space before the agent starts:
- LLM hints from ticket analysis (score +3)
- ChromaDB semantic search on bug description (score +2)
- Graph neighbor expansion (score +1)
- Flow boost from high-criticality execution paths (score += criticality)
- Returns top-5 non-test files

### Evaluation Approach

**25-bug dataset** (`eval/bugs.json`): 20 from SWE-bench Lite + 5 custom open-source bugs (Flask, Click, Rich, Werkzeug).

**EvalRunner** (`agent/eval/runner.py`): Process-level isolation per bug case. Clones repo at specific SHA, builds knowledge graph, runs ReAct pipeline, captures trace. Multiprocessing with timeout kill.

**11 scoring metrics** (`agent/eval/scoring.py`):
- `localization_hit`: Found the right file
- `fix_generated`: Patches produced
- `review_approved`: AI review approved
- `patch_hits_target`: File-level correctness
- `multi_file_complete`: All expected files patched
- `test_pass`: Repo tests pass after fix
- `patch_correctness`: File-level overlap with ground truth (0.0-1.0)
- Ground truth precision/recall/F1 at file level
- Cost + duration tracking

**A/B comparison** (`agent/eval/ab_eval.py`): Runs same bugs through two pipeline configurations, compares pass rate, cost, tool calls, per-bug winners.

**Baseline** (`agent/eval/baseline.py`): Single-shot Claude API call with no tools/retries. Measures infrastructure value.

**Current results** (25 bugs):
- Localization: 96% (24/25)
- Submit rate: 40% (10/25)
- Pass rate: 40% (10/25)
- Avg cost: $1.92/bug
- Avg tool calls: 32 for successes

---

## Knowledge/Context Strategy

### How Codebase Context Is Built

**Step 1 — Structure Analysis** (`analyzer/structure.py`):
Walks the repo (max depth 4, skipping `.git`, `node_modules`, `__pycache__`, etc.), detects tech stack via marker files (`requirements.txt` -> Python, `package.json` -> JS/Node), identifies entry points (`main.py`, `app.py`, `index.js`), counts files and LOC.

**Step 2 — AST Parsing** (`analyzer/code_parser.py`, `analyzer/multi_lang_parser.py`):
Tree-sitter parses Python (primary), JS, TS, Go, Java. Extracts per-file: classes (name, methods, bases, decorators, docstring), functions (name, params, return type, decorators, docstring, is_test flag), imports (module, names, alias). Falls back to stdlib `ast` for Python if Tree-sitter fails.

**Step 3 — Call Graph Construction** (`analyzer/call_graph.py`):
Builds a directed graph with node IDs like `path::ClassName::method_name`. Edge types: CONTAINS (structural), IMPORTS (file-to-file via import resolution), CALLS (function references detected by scanning function bodies for known callable names), INHERITS (class inheritance). Applies PageRank scoring for hotspot detection.

**Step 4 — Enrichment Layer**:
- **Business rules** (`enricher/business_logic.py`): Mines docstrings for "must", "should", "validates" keywords; extracts TODO/FIXME comments; captures named constants matching `MAX_*`, `LIMIT_*`, `THRESHOLD_*` patterns; detects route decorators.
- **Decision points** (`enricher/decision_points.py`): Classifies conditionals as role_check, status_check, feature_flag, error_guard, threshold, or logic_branch.
- **Domain concepts** (`enricher/domain_concepts.py`): Extracts concepts from CamelCase class names; maps type suffixes (Service->process, Repository->entity, Model->entity, Event->event).
- **Failure records** (`graph/business/failure_records.py`): Mines git history for "fix" commits (regex: "fixes #123", "incident", "regression"), parses issue refs, classifies severity (P0/critical/hotfix), matches diff hunks to function boundaries via Tree-sitter.
- **LLM summaries** (`enricher/summarizer.py`): Claude Haiku generates 3-5 sentence business purpose per file and 1 sentence per function. Persisted to Neo4j.

**Step 5 — Neo4j Ingestion** (`graph/builder.py`):
Atomic repo snapshot: upserts Repo, File, Class, Function nodes with properties. Creates CONTAINS, IMPORTS, CALLS, INHERITS, TESTED_BY edges. Applies PageRank scores. Supports incremental mode (deletes stale nodes in batches of 100).

**Step 6 — Community Detection** (`graph/community.py`):
Leiden algorithm (or greedy_modularity fallback) clusters nodes into communities. Edge weights: CALLS=1.0, INHERITS=0.8, IMPORTS=0.7. IDF-weighted token scoring names communities from file paths. Caps at 15 communities, merges <3-node clusters into "misc".

**Step 7 — Vector Embeddings** (`embeddings/embedder.py`):
ChromaDB indexes enriched node content (docstrings, params, imports, classes, functions, PageRank). Supports multiple models: all-MiniLM-L6-v2 (default), CodeBERT, UniXcoder, CodeSage.

**Step 8 — Context Compilation** (`compiler/context_doc.py`):
Generates `context.md` (full document) and `summary.md` (executive summary) from Neo4j graph. Includes: tech stack badges, ASCII file tree, file/class/function documentation, business rules, decision points, execution flows, dead code warnings, test coverage analysis.

### What Gets Indexed and What Doesn't

**Indexed**:
- All source files in supported languages (Python, JS, TS, Go, Java)
- Classes, functions/methods with full metadata (params, return types, decorators, docstrings)
- Import relationships and call relationships
- Business rules (from docstrings, constants, TODOs, route decorators)
- Decision points (conditionals classified by type)
- Domain concepts (from class names and module paths)
- Failure records (from git history "fix" commits)
- LLM-generated business summaries per file and function

**Not indexed**:
- Binary files (images, compiled assets, archives — skip list in `explore_tools.py:76-81`)
- `node_modules`, `.git`, `__pycache__`, `venv`, `dist`, `build` directories
- Files beyond depth 4 in structure analysis
- Non-supported languages (Ruby, C/C++, Rust, etc. — no Tree-sitter parser loaded)
- Git history beyond "fix" commits (general refactors, features not mined)

### How Context Is Retrieved at Query Time

**For agent runs** (`graph_utils.py:build_kickstart_context`):
1. Pre-localization narrows to top-5 files via LLM hints + ChromaDB + graph neighbors
2. 2-hop call subgraph around hint files extracted from Neo4j
3. Execution flows touching hint area fetched
4. Business rules linked via ENFORCED_BY edges
5. Past failure records from same code area
6. PageRank hotspots for orientation
7. All assembled into system prompt sections

**For Q&A chat** (`rag/retriever.py`):
1. Query intent extraction via regex (`rag/query_analyzer.py`) — entity types, mentioned names, scope, relationship focus
2. Vector search via ChromaDB for semantic similarity
3. Keyword search via regex patterns
4. Graph expansion: BFS up to 2 hops from seed nodes
5. Context assembly (`rag/context_assembler.py`): Priority ordering (primary full detail -> expanded name+summary -> edges compact -> business rules). Token-budgeted using tiktoken or 3.5 chars/token heuristic.

### Token Budget Management

Three layers in `context_manager.py`:

1. **Layer 1 — Per-tool output caps** (line 63-68): `read_file: 8000`, `grep_repo: 4000`, `run_tests: 6000`, default: `4000` characters.
2. **Layer 2 — Sliding window masking** (line 77-118): Tool results older than 15 iterations (`OBSERVATION_WINDOW = 15`) are replaced with `"[masked — older observation]"`. Keeps recent results in full.
3. **Layer 3 — Haiku summarization** (line 121-169): At `SUMMARIZATION_TRIGGER = 120_000` tokens (in a 160K context), early messages are compressed by Haiku into a summary. Uses `SUMMARIZATION_MODEL = "claude-haiku-4-5-20251001"`.

Token estimation: `chars * 0.25` ratio (`count_tokens_approx`, line 71-74).

---

## Development Workflow

### How to Build and Run Locally

```bash
# Backend
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:5173

# Docker (all services)
docker-compose up   # Neo4j :7474/:7687, Backend :8001, Frontend :5173

# CLI commands
cd backend && python cli.py build /path/to/repo            # Build knowledge graph
cd backend && python cli.py fix TICKET --repo /path        # Fix a bug
cd backend && python cli.py eval run                       # Run 25-bug eval suite
cd backend && python cli.py eval run --bug FLASK-2651      # Single bug eval
cd backend && python cli.py eval report                    # Show latest results
cd backend && python cli.py eval gate results/latest.json  # CI regression gate
```

**Environment variables** (`.env.example`):
- `ANTHROPIC_API_KEY` — Required for all LLM calls
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Graph database (default: `bolt://localhost:7687`)
- `DATA_DIR` — Output directory for caches/context (default: `/tmp/context_builder`)
- `EMBEDDING_MODEL` — Vector model (default: `all-MiniLM-L6-v2`)
- `CORS_ORIGINS` — API CORS (default: `localhost:5173,localhost:3000`)

### How Tests Are Structured and Run

```bash
pytest tests/ -v                    # All tests
pytest tests/test_community_detection.py -v   # Specific module
```

Test files in `backend/tests/`:
- `test_community_detection.py` — Tests Leiden community detection with mock graph data
- `test_changes.py` — Tests git diff parsing and change-to-node mapping
- `test_flows.py` — Tests execution flow detection and criticality scoring
- Plus additional test files for other modules

**Eval suite** (`backend/agent/eval/`): Not unit tests but integration eval — runs real bugs through the full pipeline with process-level timeout isolation.

### CI/CD Pipeline

No explicit CI config file (`.github/workflows/`) was found in the repo. The eval suite (`cli.py eval gate`) serves as a regression gate:
- `eval run` executes all 25 bugs through the pipeline
- `eval gate results/latest.json` checks for regressions against previous results
- Reports are generated in markdown format for review

PR creation is done via `gh pr create` in the finalize node of the pipeline.

### Claude Code Specific Setup

**`.claude/` directory**:
- `launch.json` — Run configs for frontend (`npm run dev`) and backend (`uvicorn`)
- `settings.local.json` — MCP server enablement, permission grants for browsing `/data` and temp files
- `skills/` — 4 custom skill definitions:
  - `debug-issue.md` — Debug workflow emphasizing graph tools first
  - `explore-codebase.md` — Exploration workflow with token efficiency targets
  - `refactor-safely.md` — Safe refactoring with blast radius analysis
  - `review-changes.md` — Code review using `detect_changes` + `get_review_context`

**`.mcp.json`**: Configures `code-review-graph` MCP server pointing to the project's knowledge graph. Tools include `detect_changes`, `get_impact_radius`, `get_affected_flows`, `query_graph`, `semantic_search_nodes`, `get_architecture_overview`, `refactor_tool`, and more.

**CLAUDE.md design rules** (enforced in all Claude Code sessions):
- MUST use ReAct pipeline for all agent runs
- MUST query BOTH code graph AND business knowledge graph
- MUST use MCP graph tools BEFORE Grep/Glob/Read
- MUST run tests in sandbox before PR
- Cap dev-reviewer loop at 3 iterations
- NEVER modify files outside target repo

---

## Frontend Architecture

### Pages and Components

**Overview page** (`frontend/src/pages/Overview.tsx`, 467 lines):
- Analyze form: input repo path + name, submit triggers background analysis job
- Job progress: polls `getJobStatus()` every 1.5s, shows progress bar
- Stats cards: Files, Functions, Classes, LOC from `getGraphStats()`
- Tech stack pills from `stats.tech_stack`
- Node distribution pie chart (Recharts), language bar chart, top-5 PageRank hotspots

**Agent page** (`frontend/src/pages/Agent.tsx`, 506 lines):
- Custom ticket form: repo selector, ticket ID, title, description, affected file
- Sample tickets list from backend
- Job polling: `getAgentJobStatus()` every 2s, 30-minute timeout
- SSE trace subscription: EventSource to `/api/agent/trace/{jobId}`
- `LiveActivityFeed` (`components/agent/LiveActivityFeed.tsx`, 399 lines): Human-readable activity stream with cost tracker, tool call counter, phase labels
- `TraceLogPanel` (`components/agent/TraceLogPanel.tsx`, 232 lines): Raw trace events with filtering (phases/LLM/tools/tests/guards), stage timing bars
- `EndToEndSummary` (`components/agent/EndToEndSummary.tsx`, 161 lines): Final results — localization, patches (with `CodeDiff` toggle), test output, review checks, PR link
- Past runs sidebar with clickable history

**Knowledge page** (`frontend/src/pages/Knowledge.tsx`, 590 lines):
- Two tabs: Questions (decision points for human review) and Rules (business rules)
- `QuestionCard`: Expandable with condition type badge, suggested answers, rule type/severity selectors
- `RuleCard`: Severity badge, rule type icon, description, file/function reference
- Add Rule modal with node search picker (debounced 280ms search via `searchGraphNodes()`)

**Shared components**:
- `Layout.tsx` (171 lines): Sidebar with repo selector dropdown, 3-tab navigation (Overview/Agent/Knowledge)
- `ErrorBoundary.tsx` (46 lines): Class component error catcher with retry button
- `RepoContext.tsx` (77 lines): React Context for active repo state, persisted to localStorage

**API client** (`lib/api.ts`, 328 lines): Generic `request<T>()` fetch wrapper with 30s timeout. All requests proxied via Vite (`/api` -> `http://localhost:8000`). SSE via `subscribeToTrace()` using EventSource.

---

## Strengths & Gaps

### What's Well-Designed or Clever

1. **Pre-localization pipeline** (`react_pipeline.py:95-167`): Multi-signal scoring (LLM hints +3, ChromaDB +2, graph neighbors +1, flow criticality boost) narrows from all files to top-5 before the agent starts. This reportedly saves ~40% of exploration cost.

2. **Five-strategy fuzzy patch matching** (`patch_utils.py`): The graduated fallback from exact match through whitespace normalization to sliding-window similarity (>=92%) with function-name guards is sophisticated. Most LLM-generated patches have whitespace/indentation drift, and this handles it without brittle string matching.

3. **Three-layer context window management** (`context_manager.py`): The cap -> mask -> summarize approach is directly inspired by SWE-Agent research (arXiv 2508.21433). The observation masking is particularly clever — keeping the last 15 tool results in full while replacing older ones preserves the agent's recent working memory.

4. **Prompt caching economics**: By making the system prompt static across 30+ LLM calls per bug fix and using Anthropic's server-side caching (`cache_control: ephemeral`), the system gets ~87% savings on the largest part of each request. At Sonnet pricing, this drops per-bug cost significantly.

5. **Guardrail anti-pattern detection** (`react_guardrails.py:83-157`): Rather than hard-blocking, the system warns at thresholds (8+ greps, 10+ reads, 3+ test runs, 4+ edits) — a pragmatic approach that nudges the agent without preventing legitimate deep investigation.

6. **Dual-layer knowledge graph**: Separating CODE LAYER (Files/Classes/Functions with CALLS/IMPORTS) from BUSINESS LAYER (BusinessRules/DomainConcepts/FailureRecords linked via ENFORCED_BY/REPRESENTED_BY/RESULTED_IN_CHANGE) is architecturally clean. The business layer gives the agent context that pure code analysis misses.

7. **Failure record mining** (`graph/business/failure_records.py`): Automatically mining git history for past production incidents and linking them to the functions they affected is genuinely useful — it gives the agent "institutional memory" about fragile code areas.

8. **Community detection with IDF naming** (`graph/community.py`): Using Leiden clustering with IDF-weighted token scoring from file paths to auto-name communities is clever. The names (e.g., "auth-middleware", "payment-processing") give meaningful labels to code clusters without human curation.

9. **Eval infrastructure maturity**: Process-level isolation per bug case, 11 scoring metrics, A/B pipeline comparison, baseline comparison, regression detection with history tracking — this is a serious evaluation framework, not a toy.

10. **Thread-local context isolation**: Using `threading.local()` for per-run state enables concurrent agent runs without global mutable state. This is critical for scaling to multiple simultaneous bug fixes.

### What's Missing, Incomplete, or Fragile

1. **`shell=True` security risk** (`sandbox.py`): Test commands are executed with `shell=True` in subprocess calls. This is a known P0-CRITICAL issue (documented in PLAN.md). A malicious test command in `.agent_config.json` could execute arbitrary code. Fix: `shlex.split()`.

2. **No container isolation**: The sandbox is a git worktree with no network or filesystem isolation. The agent runs tests with full host access. Docker `--network none` is on the roadmap but not implemented. This means a malicious test could exfiltrate data or modify the host.

3. **No CI/CD pipeline file**: No `.github/workflows/` configuration was found. The eval gate (`cli.py eval gate`) exists but isn't wired into automated CI. Tests and eval runs must be triggered manually.

4. **Mock Jira integration**: `jira_intake.py` is a mock. Real Jira API integration is listed as Phase 2D. The agent can only process manually-crafted tickets or the sample dataset.

5. **Legacy pipeline not removed**: `pipeline.py` (the original 8-node LangGraph pipeline) is still in the codebase, listed as "runtime deprecated" in CLAUDE.md. It shares some utilities with the ReAct pipeline but creates confusion about which is active. The TODOS.md mentions extracting utilities as P2.

6. **40% pass rate**: While 96% localization is strong, only 40% of bugs result in correct fixes. 60% of bugs are localized but not fixed — indicating the repair/patch generation step is the bottleneck, not exploration.

7. **Single-candidate patching**: The system generates one patch per bug. Multi-candidate sampling (generate 3-5, pick best) is on the roadmap (TODOS.md P1) but not implemented. This is likely the most impactful improvement for pass rate.

8. **No production feedback loop**: Sentry/PagerDuty -> graph integration is listed as TODO. The system can't learn from production incidents in real-time; failure records are only mined from git history.

9. **Missing multi-file coordination**: The PLAN.md Eng Review identifies that the repair node doesn't re-localize callers after editing. If a function signature changes, callers aren't automatically updated. A multi-file coordinator node is needed.

10. **Eval dataset size**: 25 bugs is small for statistical confidence. The dataset heavily skews toward Python (Django, Flask, Sympy, Pytest) with no Java, Go, or TypeScript bugs despite the parser supporting those languages.

11. **Frontend graph visualization unused**: `react-force-graph-2d` is listed as a dependency in `package.json`, and the CLAUDE.md mentions an interactive force graph, but the current frontend pages (Overview, Agent, Knowledge) don't render an actual force-directed graph. The graph API endpoints exist (`api/graph.py`) but aren't consumed by the frontend in the current code.

12. **No authentication**: The FastAPI backend has no auth middleware. Anyone with network access to port 8000 can trigger analysis jobs, run the agent, or query the graph. Fine for local dev, problematic for any shared deployment.

13. **Stale routers in `main.py`**: Several API routers are imported but commented out in `main.py:~85-95` (context, search, chat, metrics, flags, tools). These modules exist but aren't mounted, making their functionality inaccessible.

14. **Test coverage appears thin**: Only 3 test files were found in `backend/tests/` (`test_community_detection.py`, `test_changes.py`, `test_flows.py`). Core modules like `react_loop.py`, `patch_utils.py`, `graph/builder.py`, and `context_manager.py` don't have corresponding unit tests. The eval suite tests end-to-end behavior but doesn't isolate component correctness.

### Where the Approach Differs from Conventional Patterns

1. **ReAct over fixed DAG**: The v0.2.0 switch from an 8-node LangGraph DAG to a single ReAct while-loop with 19 tools is unusual. Most agent frameworks use fixed graphs for predictability. The ReAct approach gives the agent more autonomy but relies heavily on the guardrails (`react_guardrails.py`) and prompt engineering (`react_prompt.py`) to stay on track.

2. **Knowledge graph for code understanding**: Most AI coding tools use RAG over raw files or embeddings alone. This project's dual-layer graph (code + business) with community detection, PageRank hotspots, and execution flow tracing provides structural understanding that flat retrieval can't match.

3. **Business knowledge as first-class data**: Extracting and persisting business rules, decision points, domain concepts, and failure records as graph nodes — not just code entities — is distinctive. The ENFORCED_BY edges linking rules to code functions enable the reviewer to check business compliance, not just code correctness.

4. **Eval-driven development**: Having a 25-bug eval suite with 11 metrics, A/B comparison, and regression gating from early in the project lifecycle is more rigorous than most AI agent projects. The SWE-bench methodology (clone at specific SHA, run real tests) provides grounded evaluation.

5. **Pre-localization before agent starts**: Rather than letting the agent explore freely, the intake node narrows to top-5 files using multi-signal scoring. This is a bet that cheaper pre-processing (Haiku + ChromaDB + graph traversal) can significantly reduce expensive Sonnet exploration.

6. **Observation masking from SWE-Agent research**: Directly adopting the sliding-window masking technique from academic research (arXiv 2508.21433) for production use is unusual. Most teams implement simpler truncation strategies.

---

## Appendix: File Reference

### Backend Agent Module (18 files)

| File | Lines | Purpose |
|------|-------|---------|
| `agent/__init__.py` | 1 | Package marker |
| `agent/agent_config.py` | ~234 | Per-repo config from `.agent_config.json` |
| `agent/context_manager.py` | ~169 | Three-layer token budget (cap + mask + summarize) |
| `agent/eval_suite.py` | ~361 | Legacy eval harness (replaced by `eval/` package) |
| `agent/explore_tools.py` | ~300+ | 10 read-only exploration tools |
| `agent/feature_flags.py` | ~121 | JSON-based feature flags per PR |
| `agent/graph_utils.py` | ~589 | Graph queries for context assembly + kickstart |
| `agent/jira_intake.py` | ~50 | Mock Jira ticket intake |
| `agent/patch_utils.py` | ~438 | 5-strategy fuzzy patch matching |
| `agent/pipeline.py` | ~1800+ | Legacy 8-node LangGraph pipeline (deprecated) |
| `agent/react_guardrails.py` | ~213 | Safety: limits, gates, anti-pattern detection |
| `agent/react_loop.py` | ~350+ | Core ReAct while-loop |
| `agent/react_pipeline.py` | ~200+ | Modern 3-node pipeline (intake -> react -> finalize) |
| `agent/react_prompt.py` | ~361 | System prompt builder (task-type-aware) |
| `agent/react_tools.py` | ~400+ | 8 write tools (edit, sandbox, test, review, submit) |
| `agent/sandbox.py` | ~299 | Git worktree sandbox + test auto-detection |
| `agent/trace.py` | ~150+ | Observability: event capture + SSE streaming |
| `agent/types.py` | ~160 | Pydantic models + TypedDict state definitions |

### Backend Eval Module (11 files)

| File | Purpose |
|------|---------|
| `agent/eval/__init__.py` | Package exports |
| `agent/eval/ab_eval.py` | A/B pipeline comparison |
| `agent/eval/baseline.py` | Single-shot baseline (no tools) |
| `agent/eval/dataset.py` | 25-bug schema + SWE-bench curation |
| `agent/eval/graph_builder.py` | Graph construction for eval repos |
| `agent/eval/pr_tracker.py` | PR creation tracking |
| `agent/eval/regression.py` | Regression detection vs previous runs |
| `agent/eval/repo_manager.py` | Thread-safe repo clone + venv setup |
| `agent/eval/report.py` | Markdown report generation |
| `agent/eval/runner.py` | EvalRunner with process-level isolation |
| `agent/eval/scoring.py` | 11 metrics (localization, patch correctness, cost, etc.) |

### Backend Graph Module (5+ files)

| File | Purpose |
|------|---------|
| `graph/neo4j_client.py` | Neo4j driver singleton + query executor |
| `graph/builder.py` | Tree-sitter -> Neo4j ingestion (atomic snapshot) |
| `graph/queries.py` | Cypher query builders for API endpoints |
| `graph/community.py` | Leiden community detection + naming |
| `graph/business/persist.py` | Business rule Neo4j persistence |
| `graph/business/failure_records.py` | Git history mining for past incidents |

### Backend Analyzer Module (6 files)

| File | Purpose |
|------|---------|
| `analyzer/structure.py` | Repo structure, tech stack, entry point detection |
| `analyzer/code_parser.py` | Tree-sitter AST parsing (Python primary) |
| `analyzer/multi_lang_parser.py` | JS/TS/Go/Java Tree-sitter parsing |
| `analyzer/call_graph.py` | Directed call/import graph construction |
| `analyzer/flows.py` | Execution flow detection + criticality scoring |
| `analyzer/changes.py` | Git diff -> affected node mapping |

### Backend Other Modules

| File | Purpose |
|------|---------|
| `enricher/business_logic.py` | Mine code for business rules |
| `enricher/decision_points.py` | Classify conditionals |
| `enricher/domain_concepts.py` | Extract domain entities |
| `enricher/summarizer.py` | LLM business summaries |
| `embeddings/embedder.py` | ChromaDB vector pipeline |
| `compiler/context_doc.py` | context.md + summary.md generation |
| `rag/query_analyzer.py` | Regex-based query intent extraction |
| `rag/retriever.py` | Multi-strategy retrieval (vector + graph + keyword) |
| `rag/context_assembler.py` | Token-budgeted context assembly |

### Frontend (18 files)

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.tsx` | 17 | React entry + BrowserRouter |
| `src/App.tsx` | 25 | Route definitions (Overview, Agent, Knowledge) |
| `src/index.css` | 157 | Global styles, dark mode, prose styling |
| `src/lib/api.ts` | 328 | REST API client + SSE subscription |
| `src/lib/utils.ts` | 25 | cn(), formatNumber(), truncate() |
| `src/lib/RepoContext.tsx` | 77 | Global repo state (React Context) |
| `src/components/Layout.tsx` | 171 | Sidebar + nav + repo selector |
| `src/components/ErrorBoundary.tsx` | 46 | Error catcher |
| `src/pages/Overview.tsx` | 467 | Analysis dashboard + stats + charts |
| `src/pages/Agent.tsx` | 506 | Agent interface + trace streaming |
| `src/pages/Knowledge.tsx` | 590 | Business rules + decision points |
| `src/components/agent/TicketCard.tsx` | 54 | Ticket display card |
| `src/components/agent/StatusBadge.tsx` | 38 | Job status indicator |
| `src/components/agent/ReviewChecks.tsx` | 24 | Review check results |
| `src/components/agent/CodeDiff.tsx` | 86 | Code diff viewer (3 modes) |
| `src/components/agent/PatchCard.tsx` | 38 | Patch display with expandable diff |
| `src/components/agent/TraceLogPanel.tsx` | 232 | Raw trace events with filtering |
| `src/components/agent/EndToEndSummary.tsx` | 161 | Final results display |
| `src/components/agent/LiveActivityFeed.tsx` | 399 | Human-readable activity stream |

---

## Learnings from Reference Project (Ticket-to-PR Pipeline)

*Source: `~/Downloads/output.md` — a parallel implementation of the same problem (ticket -> knowledge graph -> agent -> PR) by another engineer. Their system uses Memgraph + Qdrant + Celery + Docker sandbox with a "minions blueprint" DAG engine that replaced their original linear pipeline.*

### 1. Signal-Based Model Routing (High Impact, Low Effort)

**What they do**: Their agent (`layer45-agent/agent.py:_pick_model()`) dynamically escalates models based on signals: Haiku for exploration, Sonnet for writing, Opus when stuck (3+ failed searches or 2+ failed fixes).

**What we do**: We use Sonnet 4.6 for the entire ReAct loop (`react_loop.py:~48`, `REACT_MODEL`), regardless of phase.

**Learning**: Exploration (grep, read_file, get_callers) doesn't need Sonnet reasoning. Haiku can handle tool routing at ~4x lower cost. Our `react_guardrails.py` already tracks phase transitions — adding model switching at phase boundaries is straightforward. **Estimated savings**: 40-60% on exploration phase tokens (which are ~50% of total), so ~20-30% overall cost reduction per bug. That would bring avg cost from $1.92 to ~$1.35.

### 2. Tool Subsets Per Phase (High Impact, Medium Effort)

**What they do**: They split tools into phase-specific subsets — EXPLORE_TOOLS (23 read-only), WRITE_TOOLS (15 including write_file), FIX_TOOLS (14 targeted). Each agentic node only sees its relevant tools.

**What we do**: We give all 19 tools to the agent at once. The system prompt guides phase behavior, but the agent can technically call `string_replace` during exploration or `grep_repo` during the review phase.

**Learning**: Too many tools causes "tool paralysis" — the LLM wastes reasoning on which tool to pick. We already split tools into `explore_tools.py` (read-only) and `react_tools.py` (write). Wiring this so the LLM only sees explore tools during EXPLORE phase and edit tools during EDIT phase would reduce prompt size and improve tool selection accuracy. The `react_guardrails.py` sandbox gate partially enforces this (blocks write tools without sandbox), but showing fewer tools is better than blocking after selection.

### 3. Deterministic Autofix Catalog Before Agent (Medium Impact, Low Effort)

**What they do**: `minions/autofix/catalog.py` applies deterministic regex fixes for trivial issues (trailing whitespace, unused imports via F401, blank line violations) BEFORE the agentic fix_lint node runs. Zero LLM tokens spent on mechanical fixes.

**What we do**: We have `_diff_scoped_lint()` in `react_tools.py:259-324` that runs ruff on changed files, but the agent must interpret and fix lint errors via LLM.

**Learning**: Many lint errors have deterministic fixes (remove unused import, add blank line, fix trailing whitespace). Running `ruff --fix` or a small regex catalog after the agent's edit but before `run_tests` would reduce unnecessary LLM iterations. Our `check_syntax` tool already validates Python — extending this to auto-fix trivial lint violations saves 1-2 tool calls per bug.

### 4. Separate Exploration and Writing as Distinct Agent Steps (High Impact, High Effort)

**What they do**: Their minions blueprint splits exploration (Haiku, read-only tools, fresh conversation) and writing (Sonnet, write tools, receives explorer's file cache in prompt) into completely separate nodes. The explorer outputs a PLAN; the writer receives the plan + cached files.

**What we do**: Our single ReAct loop handles both exploration and writing in one continuous conversation. The agent transitions from EXPLORE to EDIT naturally.

**Learning**: Splitting has a key advantage: the **writer starts with a clean context** containing only the plan and relevant files, not 15-20 exploration tool results. Our `context_manager.py` masks old observations, but the writer still carries the full message history. Their approach means the write phase gets maximum context for code generation. **Trade-off**: Our continuous loop lets the agent revisit exploration during editing (read a caller it missed). A strict split prevents this. The middle ground: keep our ReAct loop but implement an explicit "summarize and reset" between phases — compress exploration results into a structured plan, then continue with a lighter context for editing.

### 5. Git Co-Change Coupling (COUPLED_WITH Edges) (Medium Impact, Medium Effort)

**What they do**: `layer2-indexer/src/vcs/coupling.py` scans git log, computes pairwise Jaccard scores on commit sets for files that frequently change together, and creates COUPLED_WITH edges in the graph.

**What we do**: We mine git history for failure records (`graph/business/failure_records.py`) but don't track co-change coupling.

**Learning**: Co-change coupling captures implicit dependencies that call graphs miss. If `auth.py` and `middleware.py` always change together but have no import relationship, the call graph won't link them — but coupling will. Adding COUPLED_WITH edges to our Neo4j graph and exposing them in `get_blast_radius` would improve multi-file fix completeness. Our `analyzer/call_graph.py` and `graph/builder.py` already handle edge creation — adding a coupling pass during graph build is a contained addition.

### 6. LLM-Generated Descriptions as Embedding Input (Medium Impact, Low Effort)

**What they do**: `layer2-indexer/src/semantic/descriptions.py:enrich_file()` uses Claude Haiku to generate 1-2 sentence descriptions per symbol, then embeds those descriptions (not raw code) in Qdrant.

**What we do**: `enricher/summarizer.py` generates LLM summaries and stores them in Neo4j, but `embeddings/embedder.py` builds embeddings from concatenated metadata (docstrings, params, imports) — not the LLM-generated summaries.

**Learning**: Embedding LLM-generated descriptions makes semantic search dramatically better for natural-language queries. "The function that validates user input" matches a description "validates user-provided form data against schema constraints" far better than it matches raw code `def _check_params(data, schema)`. Our `summarizer.py` already generates these descriptions — feeding them into `embedder.py:build_enriched_nodes()` is a one-line change in the enriched node content builder.

### 7. Hard Gate Nodes Instead of Soft Guardrails (Medium Impact, High Effort)

**What they do**: Their blueprint DAG uses GATE nodes (`lint_gate`, `test_gate`, `review_gate`) that are hard routing decisions. If tests fail, the DAG routes to `fix_tests`, not the same agent. Hard cap: `ci_round >= max_ci_rounds` -> escalate. The agent cannot choose to skip tests or ignore lint.

**What we do**: Our `react_guardrails.py` warns at thresholds (8+ greps, 3+ test runs) but the agent can still call tools past the warning. The only hard gates are `submit_fix` requiring tests attempted + review approved, and the global MAX_TOOL_CALLS=40.

**Learning**: Their hard gates are more reliable because they don't depend on LLM cooperation. Our anti-pattern detection is good for nudging, but determined agents can ignore warnings. The key insight: **lint/test/review retry should be a loop with a hard cap, not an LLM decision**. If tests fail, automatically route back to edit; after 2 rounds, escalate. This matches our MAX_TEST_FAILURES=3 in guardrails but enforces it structurally rather than via prompt instruction.

### 8. Cross-Node File Cache (Medium Impact, Low Effort)

**What they do**: `PipelineContext.file_cache` carries file contents between exploration and writing nodes. The writer's prompt includes pre-cached files, avoiding re-reads.

**What we do**: When our agent reads a file during EXPLORE, it's in the message history. After observation masking (`context_manager.py`), the content may be replaced with `"[masked — older observation]"`, forcing the agent to re-read during EDIT.

**Learning**: File content that was relevant during exploration is almost certainly relevant during editing. Instead of masking and re-reading, we should cache the top N most-read files and inject them into the context when entering EDIT phase. This avoids 2-4 redundant `read_file` calls per bug. Implementation: extend `GuardrailState` to track files read + their content, then inject a "cached files" section before the EDIT phase.

### 9. Scoped Pre-Commit (Low Impact, Low Effort)

**What they do**: `layer7-pr-publisher/git_ops.py:_run_pre_commit_fixes()` runs prettier/eslint only on changed files and sets `HUSKY=0` to prevent a 1-line fix from reformatting 20 unrelated files.

**What we do**: Our `_diff_scoped_lint()` in `react_tools.py:259-324` already runs ruff only on changed files. But our PR creation step doesn't scope pre-commit hooks.

**Learning**: Minor but worth noting — if target repos have pre-commit hooks, our PR creation should scope them. Already partially implemented via diff-scoped lint.

### 10. PR Retry with Vision Context (Low Impact, Medium Effort)

**What they do**: `mcp-server/server.py:retry_pipeline_pr()` extracts screenshots from PR review comments, base64-encodes them, and passes to the agent as Claude vision blocks.

**What we do**: No PR retry mechanism. Our pipeline is one-shot: fix -> PR. If the PR is rejected, the human reviews manually.

**Learning**: Adding a retry loop that reads PR review comments and re-runs the agent would close the feedback loop. The vision context for screenshot-based bug reports is a nice touch for frontend bugs. Not high priority for our current 40% pass rate (improving first-pass quality matters more than retry), but worth planning for.

### Decisions Made

**Embeddings removed**: ChromaDB and all vector search functionality has been stripped from the codebase. The Neo4j knowledge graph + keyword search in the RAG layer provide sufficient retrieval capability. The `enriched_nodes.json` cache is retained (built by `embeddings/embedder.py:build_enriched_nodes()`) for function lookups and context assembly, but no vectors are computed or stored.

**Sonnet throughout (no model switching)**: We keep `claude-sonnet-4-6` for the entire ReAct loop rather than switching to Haiku for exploration. Reason: Anthropic's server-side prompt caching gives ~87% savings when the same model is used across all 30+ LLM calls per bug. Switching models between phases would break the cache and lose more money than the per-token Haiku discount saves. The reference project's signal-based routing is clever but incompatible with our prompt caching strategy.

### Summary: Priority-Ordered Adoption Recommendations

| # | Learning | Impact | Effort | Recommendation |
|---|----------|--------|--------|----------------|
| 1 | ~~Signal-based model routing~~ | ~~High~~ | ~~Low~~ | **Rejected.** Breaks prompt caching; net cost increase. |
| 2 | Tool subsets per phase | High | Medium | **Do first.** Only show explore tools during EXPLORE, edit tools during EDIT. |
| 6 | ~~LLM descriptions as embeddings~~ | ~~Medium~~ | ~~Low~~ | **N/A.** Embeddings removed entirely. |
| 3 | Deterministic autofix catalog | Medium | Low | **Quick win.** Run `ruff --fix` before agent sees lint errors. |
| 8 | Cross-node file cache | Medium | Low | **Quick win.** Cache files read during EXPLORE, inject into EDIT phase context. |
| 5 | Git co-change coupling | Medium | Medium | **Next sprint.** Add COUPLED_WITH edges for better blast radius. |
| 7 | Hard gate nodes | Medium | High | **Plan for v0.4.** Structural lint/test/review loops with hard caps. |
| 4 | Separate explore/write agents | High | High | **Plan for v0.4.** Requires ReAct loop restructuring. |
| 10 | PR retry with vision | Low | Medium | **Backlog.** First-pass quality matters more now. |
| 9 | Scoped pre-commit | Low | Low | **Backlog.** Already partially implemented. |

---

## Learnings from Production AI Coding Tool (src/ Analysis)

*Source: `~/Downloads/src/` — a production-grade AI coding CLI with tools, context management, coordinator/worker orchestration, cost tracking, task management, and permission system. ~200+ TypeScript files.*

### Architecture Overview

The system has these key layers:
- **QueryEngine** (`QueryEngine.ts`, 1295 lines) — wraps the agent loop lifecycle
- **query()** (`query.ts`, 1729 lines) — the actual ReAct-style while-loop with streaming, recovery, and compression
- **Tool system** (`Tool.ts`, 793 lines) — generic `Tool<Input, Output>` interface with Zod schemas, permissions, progress callbacks, and result persistence
- **Tool orchestration** (`services/tools/toolOrchestration.ts`) — concurrent batching for read-only tools, serial execution for writes
- **Context management** (`context.ts`) — layered caching: system context (frozen at session start), user context (semi-cached), conversation (dynamic)
- **Coordinator** (`coordinator/coordinatorMode.ts`, 369 lines) — multi-agent orchestration with research/synthesis/implementation/verification phases
- **Cost tracking** (`cost-tracker.ts`, 323 lines) — per-model breakdown with session persistence and recursive sub-tool cost accounting
- **Task system** (`Task.ts`, `tasks/`) — polymorphic task types with disk-backed output, abort controllers, and lifecycle management
- **Skill system** (`skills/`) — skills are prompts + tool calls (not code), routed by the model

### 11. Three-Level Tool Permission Gates (High Impact, Medium Effort)

**What they do**: Every tool call passes through three gates in sequence:
1. `validateInput(input, context)` — structural validation (file exists? path safe? old_string present in file?)
2. `checkPermissions(input, context)` — permission decision (`allow` / `ask` / `deny`) with reason tracking
3. `call(input, context)` — actual execution, only reached if both gates pass

Permission results carry a discriminated union `decisionReason` explaining *why* (rule match, mode, classifier, hook, sandbox override). Denied calls are tracked in `permissionDenials[]` and reported in the final result.

**What we do**: Our `react_guardrails.py:check_tool_call()` (line 83-157) does a single pre-execution check combining validation and permission. No structured reason tracking. No post-denial reporting.

**Learning**: Separating validation from permission makes each concern cleaner. Validation is "is this input well-formed?" (file exists, no path traversal). Permission is "is this operation allowed?" (sandbox gate, write to protected file). Our guardrail already checks sandbox state — splitting it into `validate_tool_input()` and `check_tool_permission()` with structured denial reasons would improve debuggability and enable permission analytics (which tools get denied most, why).

**Where to apply**: `react_guardrails.py:check_tool_call()` — split into two functions. Add a `PermissionDenial` dataclass to `types.py` and accumulate in `GuardrailState`.

### 12. Concurrent Tool Batching (High Impact, Medium Effort)

**What they do** (`services/tools/toolOrchestration.ts`): When the LLM returns multiple tool calls in one response, they partition them into batches:
- **Concurrent batch**: Multiple consecutive read-only tools (`isConcurrencySafe=true`) run in parallel (up to 10)
- **Serial batch**: A single write tool runs alone

```
[grep, read_file, read_file, edit_file, grep, read_file]
→ Batch 1: [grep, read_file, read_file] — concurrent
→ Batch 2: [edit_file] — serial
→ Batch 3: [grep, read_file] — concurrent
```

Each tool declares `isConcurrencySafe(input) -> bool` and `isReadOnly(input) -> bool`. Default is `false` (fail-closed).

**What we do**: Our `react_loop.py` executes tool calls sequentially, one at a time. Even when the LLM requests multiple `grep_repo` calls in one response, they run in series.

**Learning**: Our explore tools (`grep_repo`, `read_file`, `read_function`, `list_files`, `get_function_info`, etc.) are all read-only and safe to parallelize. Running 3-4 grep calls concurrently instead of sequentially would cut exploration phase wall time by 60-70%. Implementation: add `is_concurrent_safe` flag to each `@tool` definition in `explore_tools.py`, then batch tool calls in `react_loop.py` using `asyncio.gather()` for concurrent batches.

**Where to apply**: `react_loop.py` tool execution section. Add `is_concurrent_safe = True` to all explore tools in `explore_tools.py`. Write tools (`string_replace`, `create_file`, `create_sandbox`) get `False`.

### 13. Multi-Stage Context Compression Pipeline (High Impact, Medium Effort)

**What they do** (`query.ts` lines 1062-1250): Three recovery levels for context overflow, tried in sequence:
1. **Context-collapse drain** — cheap, structural collapse of old tool results (like our observation masking)
2. **Reactive compact** — full LLM summarization of middle turns (keep first + last N messages)
3. **Surface error and exit** — only if both fail

For `max_output_tokens` exceeded: escalate from 8K → 64K default, then retry up to 3 times with a recovery message ("Output token limit hit. Resume directly — no apology").

**What we do**: Our `context_manager.py` has two layers: observation masking at 15 messages (`mask_old_observations`), and Haiku summarization at 120K tokens (`maybe_summarize`). But we don't have graceful recovery for prompt-too-long errors — if the API rejects, the loop fails.

**Learning**: Adding a reactive compact step (triggered by 413 prompt-too-long) that aggressively summarizes and retries would prevent hard failures in long exploration sessions. Also, the max-output-tokens recovery pattern (escalate then retry with "resume directly") is directly applicable — our agent sometimes gets cut off mid-response.

**Where to apply**: `react_loop.py` — wrap the LLM call in a try/except for `OverflowError`/`BadRequestError`. On prompt-too-long, call `maybe_summarize()` forcefully (regardless of token threshold), then retry once. On max-output-tokens, append "Your output was truncated. Continue from where you stopped." and retry up to 2 times.

### 14. File State Cache with Read-Before-Edit Enforcement (Medium Impact, Low Effort)

**What they do** (`Tool.ts` `ToolUseContext.readFileState: FileStateCache`): An LRU cache tracks every file the agent has read. The `FileEditTool.validateInput()` checks that `readFileState` has the file before allowing edits — **you can't edit a file you haven't read**. The cache also stores the content hash, so if the file changed between read and edit, the edit is rejected.

**What we do**: Our `string_replace` tool in `react_tools.py` doesn't enforce read-before-edit. The agent can attempt to edit a file it hasn't read, leading to failed replacements (old_string doesn't match because the agent is guessing).

**Learning**: Adding a `files_read: dict[str, str]` (path -> content hash) to our thread-local context and checking it in `string_replace` before attempting the edit would catch "blind edit" attempts early. Return a helpful error: "You haven't read this file yet. Call read_file first." This prevents wasted tool calls and gives the agent actionable feedback.

**Where to apply**: `react_tools.py:set_react_context()` — add `files_read = {}`. In `read_file` tool, populate on successful read. In `string_replace` tool, check before executing.

### 15. Coordinator/Worker Pattern for Multi-File Fixes (High Impact, High Effort)

**What they do** (`coordinator/coordinatorMode.ts`): A coordinator agent orchestrates workers through 4 phases:
1. **Research** — spawn parallel read-only workers to explore
2. **Synthesis** — coordinator reads findings, crafts detailed specs (file paths, line numbers, exact changes)
3. **Implementation** — spawn workers per-file sequentially (avoid conflicts)
4. **Verification** — spawn independent test workers

Critical rule: **"Never lazy-delegate."** The coordinator must synthesize — "based on your findings, fix the bug" is forbidden. It must include file paths, line numbers, what specifically to change.

Decision logic for continue vs. spawn fresh: high context overlap → continue existing worker. Low overlap → spawn fresh. Correcting failure → continue. Unrelated task → spawn.

**What we do**: Single ReAct loop handles everything. PLAN.md identifies "missing multi-file coordinator node" as a P0-HIGH issue.

**Learning**: For multi-file bugs (8 of our 25-bug eval set), the single ReAct loop struggles because it exhausts context on exploration before having room for multi-file edits. The coordinator pattern solves this by giving the writer a clean context with only the plan + cached files. **This is the #1 architecture change for improving pass rate beyond 40%.**

**Where to apply**: New module `agent/coordinator.py`. For bugs where `len(intent.likely_affected_modules) > 1`, use coordinator mode: spawn explore worker → synthesize plan → spawn edit worker per file → verify. Falls back to single ReAct loop for single-file bugs.

### 16. Tool Definition with Safe Defaults via Builder Pattern (Medium Impact, Low Effort)

**What they do** (`Tool.ts` lines 783-792): A `buildTool(def)` function merges tool definitions with safe defaults:
```
isConcurrencySafe: () => false   // Fail-closed: assume sequential
isReadOnly: () => false          // Fail-closed: assume writes
isDestructive: () => false
isEnabled: () => true
```

Each tool also declares:
- `searchHint` — 3-10 word phrase for keyword matching (separate from the tool name)
- `getActivityDescription(input)` — human-readable spinner text ("Reading src/foo.ts")
- `maxResultSizeChars` — threshold for result persistence (20K-100K chars, varies by tool)
- `isSearchOrReadCommand(input)` — for UI collapsing of read-only operations

**What we do**: Our tools are decorated with `@tool` from LangChain, which provides basic schema but no metadata about concurrency safety, read-only status, output caps, or activity descriptions. Our `TOOL_OUTPUT_CAPS` in `context_manager.py` is a separate dict that must be kept in sync with tool names manually.

**Learning**: Attaching metadata directly to each tool (rather than in separate dicts) prevents drift. A Python `@agent_tool` decorator wrapping LangChain's `@tool` could add `is_read_only`, `is_concurrent_safe`, `max_output_chars`, and `activity_description` fields. The output cap currently in `context_manager.py:TOOL_OUTPUT_CAPS` would move to each tool definition.

**Where to apply**: Create a thin wrapper in `explore_tools.py` that adds metadata attributes to each tool function. Then `context_manager.py:cap_tool_output()` reads from `tool.max_output_chars` instead of the separate dict.

### 17. Cost Tracking with Per-Model Breakdown and Cache Accounting (Medium Impact, Low Effort)

**What they do** (`cost-tracker.ts`): Track per-API-call cost with:
- Per-model accumulation (input tokens, output tokens, cache read, cache creation, cost USD)
- Session-aware persistence (restore on resume, don't mix sessions)
- Recursive sub-tool cost (if a tool call triggers another LLM call, add its cost recursively)
- Display: `Total cost: $X.XX | API duration: HH:MM:SS | Code changes: N lines added, M removed`

**What we do**: Our `react_loop.py:_estimate_cost()` (lines 51-68) estimates cost per call with cache accounting. `GuardrailState.cost_usd` accumulates. But we don't persist across sessions, don't track per-model breakdown, and don't account for sub-calls (reviewer, intake Haiku calls).

**Learning**: Our cost tracking misses the Haiku intake call, Haiku summarization calls, and reviewer calls — only tracking the main Sonnet loop. Adding a `CostTracker` class that accumulates across all LLM calls (not just the ReAct loop) would give accurate per-bug costs. Session persistence would enable cost trending over eval runs.

**Where to apply**: New class in `agent/cost_tracker.py`. Called from `react_loop.py`, `react_pipeline.py:intake_node()`, `context_manager.py:maybe_summarize()`, and `react_tools.py:request_review()`.

### 18. Disk-Backed Task Output with Streaming Offset (Low Impact, Medium Effort)

**What they do** (`Task.ts`): Each task writes output to a disk file (`outputFile`) with a streaming `outputOffset`. Consumers read from `offset` to get new output without re-reading the whole file. Tasks are polymorphic (discriminated union by `type` field) with type-specific kill implementations.

**What we do**: Our `trace.py:RunTrace` stores events in memory and supports SSE streaming via subscriber queues. No disk persistence during the run (only `save_report(path)` at the end).

**Learning**: For long-running agent tasks (our 15-minute wall time limit), writing trace events to disk incrementally means we don't lose data if the process crashes. Also enables resuming from a checkpoint. Low priority since our current runs complete within limits, but useful for the production feedback loop (Sentry/PagerDuty → graph) planned in TODOS.md.

### 19. Deferred Tool Loading to Reduce Prompt Size (Medium Impact, Low Effort)

**What they do** (`Tool.ts`): Tools can be marked `shouldDefer: true`, meaning they're excluded from the initial prompt and only loaded when the model calls `ToolSearch`. This reduces the system prompt size when tool count is high (60+ tools). Some tools are `alwaysLoad: true` (never deferred, shown on turn 1).

**What we do**: All 18 tools are included in every system prompt. The system prompt in `react_prompt.py` includes full documentation for all tools regardless of whether they'll be used.

**Learning**: With 18 tools, our prompt overhead is manageable (~2K tokens). But if we add more tools (graph query tools, git tools, etc.), deferred loading would help. More immediately: we could **skip tool documentation for tools the agent won't need** based on task type. Bug fixes rarely need `create_file`; refactors rarely need `get_blast_radius`. Trimming 3-4 unused tools saves ~500 tokens per call × 30 calls = 15K tokens per bug.

**Where to apply**: `react_prompt.py:build_system_prompt()` — accept a `task_type` parameter and include tool docs conditionally. Bug fixes skip `create_file`. Refactors skip `get_failure_history`.

### Updated Priority Table (All Sources Combined)

| # | Learning | Source | Impact | Effort | Recommendation |
|---|----------|--------|--------|--------|----------------|
| 12 | Concurrent tool batching | src/ | **High** | Medium | **Do first.** Parallelize read-only tools. 60-70% exploration speedup. |
| 2 | Tool subsets per phase | ref | **High** | Medium | **Do second.** Show only relevant tools per phase. |
| 13 | Multi-stage compression + recovery | src/ | **High** | Medium | **Do third.** Prevent hard failures on context overflow. |
| 14 | Read-before-edit enforcement | src/ | Medium | Low | **Quick win.** Prevent blind edits, save wasted tool calls. |
| 3 | Deterministic autofix (ruff --fix) | ref | Medium | Low | **Quick win.** Zero LLM tokens on trivial lint. |
| 8 | Cross-node file cache | ref | Medium | Low | **Quick win.** Avoid re-reading files after masking. |
| 16 | Tool metadata via builder pattern | src/ | Medium | Low | **Quick win.** Attach caps/flags directly to tools. |
| 17 | Full cost tracking with sub-calls | src/ | Medium | Low | **Quick win.** Accurate per-bug cost accounting. |
| 19 | Deferred/conditional tool docs | src/ | Medium | Low | **Quick win.** Trim unused tool docs by task type. |
| 11 | Three-level permission gates | src/ | Medium | Medium | **Next sprint.** Split validation from permission. |
| 5 | Git co-change coupling | ref | Medium | Medium | **Next sprint.** COUPLED_WITH edges. |
| 15 | Coordinator/worker for multi-file | src/ | **High** | High | **Plan for v0.4.** Key to breaking 40% pass rate ceiling. |
| 7 | Hard gate nodes | ref | Medium | High | **Plan for v0.4.** Structural retry loops. |
| 4 | Separate explore/write agents | ref | High | High | **Plan for v0.4.** Subsumes coordinator pattern. |
| 18 | Disk-backed task output | src/ | Low | Medium | **Backlog.** Useful for crash recovery. |
| 10 | PR retry with vision | ref | Low | Medium | **Backlog.** First-pass quality first. |
