# Context Builder — Complete Project Documentation

> **One line:** Take any code repository, break it into a knowledge graph, and generate a context document so rich that an LLM can answer complex questions about the code and business logic — without reading a single source file.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [The Analysis Pipeline](#3-the-analysis-pipeline)
4. [The 8-Layer Context Document](#4-the-8-layer-context-document)
5. [Knowledge Graph Schema](#5-knowledge-graph-schema)
6. [API Reference](#6-api-reference)
7. [CLI Reference](#7-cli-reference)
8. [Frontend Dashboard](#8-frontend-dashboard)
9. [Running the Project](#9-running-the-project)
10. [File Map](#10-file-map)

---

## 1. Executive Summary

Context Builder solves a fundamental problem in AI-assisted software engineering: **how do you give an LLM enough understanding of a codebase to answer nuanced questions without feeding it every source file?**

The answer is a **dual-layer knowledge graph**:

- **Code Layer** — Files, Classes, Functions, and the relationships between them (imports, calls, inheritance)
- **Business Layer** — Business rules, API endpoints, domain constants, and AI-generated purpose summaries

The pipeline works like this:

```
Any Git repository
       |
       v
  [ Tree-sitter AST parsing ]
  [ Call graph + PageRank    ]
  [ Business rule extraction ]
  [ LLM summarization        ]
       |
       v
  Neo4j Knowledge Graph
       |
       v
  context.md (8 layers)
       |
       v
  LLM reads context.md --> answers questions about the repo
```

The generated `context.md` is a structured markdown document (~15-25KB) containing everything the LLM needs: file purposes, class hierarchies, function signatures with docstrings, API endpoints, business constants, call relationships, and PageRank-ranked hotspots.

This is **Milestone 1** of the AI Deploy Agent project — once the context is good enough, subsequent milestones will build on it to create agents that autonomously fix bugs and raise PRs.

---

## 2. Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Entry                          │
│                                                                 │
│   cli.py build /path/to/repo        POST /api/analyze           │
│   cli.py query repo "question?"     GET  /api/context/full      │
└───────────┬─────────────────────────────────┬───────────────────┘
            │                                 │
            v                                 v
┌───────────────────────────────────────────────────────────────┐
│                     Analysis Pipeline                         │
│                                                               │
│  StructureAnalyzer ──> CodeParser ──> CallGraphBuilder        │
│         │                  │                │                  │
│         │                  │                v                  │
│         │                  │          GitAnalyzer              │
│         │                  │                                   │
│         v                  v                v                  │
│  ┌─────────────────────────────────────────────────┐          │
│  │           In-Memory Parsed Data                  │          │
│  │  structure: tech stack, entry points, stats      │          │
│  │  parsed:    classes, functions, imports, docs     │          │
│  │  graph:     nodes, edges, PageRank scores        │          │
│  │  git:       change hotspots                      │          │
│  └──────────────────┬──────────────────────────────┘          │
└─────────────────────┼─────────────────────────────────────────┘
                      │
        ┌─────────────┼──────────────┐
        v             v              v
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  GraphBuilder│ │ Business │ │  Summarizer  │
│  (Neo4j)     │ │ Logic    │ │  (Claude AI) │
│              │ │ Extractor│ │              │
│ Repo─>File   │ │          │ │ Per-file     │
│ File─>Class  │ │ Endpoints│ │ business     │
│ Class─>Func  │ │ Constants│ │ purpose      │
│ IMPORTS      │ │ Rules    │ │ summaries    │
│ CALLS        │ │ TODOs    │ │              │
│ INHERITS     │ │          │ │              │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │
       v              v              v
┌─────────────────────────────────────────────┐
│              Neo4j Knowledge Graph           │
│                                             │
│  Nodes: Repo, File, Class, Function,        │
│         BusinessRule, DomainConcept          │
│  Edges: CONTAINS, IMPORTS, CALLS,           │
│         INHERITS, FOUND_IN, ENFORCED_BY     │
└──────────────────┬──────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────┐
│           Context Compiler                   │
│                                             │
│  Queries Neo4j ──> Renders 8-layer          │
│  context.md + summary.md                    │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          v                 v
   context.md         React Dashboard
   (for LLM)         (for humans)
```

### The Dual-Layer Concept

| Layer | Node Types | Edge Types | Purpose |
|-------|-----------|------------|---------|
| **Code** | File, Class, Function | CONTAINS, IMPORTS, CALLS, INHERITS | Structural understanding — what code exists and how it connects |
| **Business** | BusinessRule, DomainConcept | FOUND_IN, ENFORCED_BY | Semantic understanding — why the code exists and what constraints govern it |

The power comes from linking these layers together. A `BusinessRule` node like `"RATE_LIMIT = 50"` is connected via `FOUND_IN` to the File it lives in, and that File is connected via `CONTAINS` to the Function that enforces it. An LLM can traverse this to answer: *"What's the rate limit and where is it enforced?"*

---

## 3. The Analysis Pipeline

When you run `cli.py build /path/to/repo`, eight steps execute in sequence:

### Step 1: Structure Analyzer

**File:** `backend/analyzer/structure.py`

Scans the repository to understand what it is:

- **Tech stack detection** — Looks for config files (`requirements.txt` → Python, `package.json` → Node.js, `Dockerfile` → Docker) and scans their contents for framework names (`fastapi`, `react`, `django`, etc.)
- **Entry points** — Finds files like `main.py`, `app.py`, `index.js`, `server.ts` in the first 3 directory levels
- **File statistics** — Counts total files, Python/JS/TS files, and lines of code
- **README** — Reads the first 2000 characters of README.md

**Output:** `{tech_stack, entry_points, file_stats, readme_content, tree}`

### Step 2: Code Parser (Tree-sitter)

**File:** `backend/analyzer/code_parser.py`

Parses every Python file using [tree-sitter](https://tree-sitter.github.io/), a fast incremental parser. Falls back to Python's stdlib `ast` module for edge cases.

For each `.py` file, it extracts:

| Extracted | Example |
|-----------|---------|
| Module docstring | `"""This module handles authentication."""` |
| Classes | Name, base classes, decorators, docstring |
| Methods (per class) | Name, parameters, return type, decorators, docstring |
| Top-level functions | Name, parameters, return type, decorators, docstring |
| Imports | `from auth.utils import verify_token` → `{module: "auth.utils", names: ["verify_token"], is_from: True}` |
| Line counts | Total lines of code per file |

**Skips:** Files >512KB, and directories like `__pycache__`, `.venv`, `node_modules`, `migrations`.

**Output:** List of `ParsedFile` dicts, one per Python file.

### Step 3: Call Graph Builder

**File:** `backend/analyzer/call_graph.py`

Builds a directed graph (using NetworkX) from the parsed data:

**Node ID convention:**
```
File:     "path/to/file.py"
Class:    "path/to/file.py::ClassName"
Function: "path/to/file.py::func_name"
Method:   "path/to/file.py::ClassName::method_name"
```

**Phases:**
1. **Add nodes** — One node per file, class, method, and function
2. **CONTAINS edges** — File→Class, File→Function, Class→Method (structural hierarchy)
3. **IMPORTS edges** — Resolves `import` statements to file nodes (handles relative + absolute imports)
4. **INHERITS edges** — Maps `class Child(Parent)` to an edge from Child to Parent
5. **CALLS edges** — Scans function bodies for identifier references to known callables (best-effort, skipped if >2000 callables for performance)
6. **PageRank** — Computes importance scores (alpha=0.85) so the most-referenced symbols rank highest

**Output:** `{nodes: [...], edges: [...], hotspots: [top 20 by PageRank]}`

### Step 4: Git Analyzer

**File:** `backend/analyzer/git_analyzer.py`

Analyzes the last 50 commits to find **change hotspots** — files that change most frequently, which often indicates critical or problematic code.

- Runs `git log --name-only` with a 30-second timeout
- Counts how many commits touched each file
- Returns the top 10 most-changed files with their last change date

### Step 5: Business Logic Extractor

**File:** `backend/enricher/business_logic.py`

Mines source files for business knowledge using four scanners:

| Scanner | Pattern | Example Match |
|---------|---------|---------------|
| **Docstring keywords** | `must`, `should`, `validates`, `required`, `constraint`, `policy` | `"""Users must verify email before login."""` |
| **TODO/FIXME** | `# TODO: ...` or `# FIXME: ...` | `# TODO: Add rate limiting to this endpoint` |
| **Constants** | `[A-Z]+_(LIMIT\|MAX\|MIN\|TIMEOUT\|RATE\|...)` | `MAX_RETRY_COUNT = 3` |
| **API Endpoints** | `@app.get("/path")`, `@router.post("/path")` | `@router.post("/api/users")` → `API Endpoint: POST /api/users → handler: create_user()` |

Each extracted rule becomes a `BusinessRule` node in Neo4j, linked to its source File via `FOUND_IN` and to the enforcing Function via `ENFORCED_BY`.

### Step 6: LLM Summarizer

**File:** `backend/enricher/summarizer.py`

Calls Claude Haiku to generate a 3-5 sentence business-purpose summary for each file.

**What it sends to Claude:**
```
Module path: auth/middleware.py
Module docstring: "Authentication middleware for FastAPI"
Classes defined: AuthMiddleware, TokenValidator
Functions defined: verify_jwt, refresh_token
```

**What Claude returns:**
> This module provides the authentication layer for the API, ensuring all incoming requests carry valid JWT tokens. It serves as a gateway that intercepts requests before they reach route handlers, rejecting unauthorized access. The token validator supports both access and refresh token flows, with configurable expiration policies.

These summaries are stored on the `File` node's `summary` property in Neo4j, and appear in Layer 6 of the context document.

### Step 7: Graph Builder (Neo4j Ingestion)

**File:** `backend/graph/builder.py`

Takes all the in-memory data and writes it to Neo4j:

1. **Upsert Repo node** — With tech_stack, entry_points, file_count, readme
2. **Upsert File nodes** — With path, language, LOC, docstring, linked to Repo via CONTAINS
3. **Upsert Class nodes** — With name, bases, docstring, linked to File via CONTAINS
4. **Upsert Function nodes** — With name, params, return_type, docstring, decorators
5. **Create edges** — IMPORTS and CALLS from the call graph data
6. **Apply PageRank** — Write pre-computed scores to node properties

All operations use `MERGE` (upsert) so re-runs are idempotent.

### Step 8: Context Compiler

**File:** `backend/compiler/context_doc.py`

Queries Neo4j and assembles two output files:

- **`context.md`** — The full 8-layer context document (~15-25KB)
- **`summary.md`** — A compact ~3K-token summary with top hotspots and business rules

Written to `/tmp/context_builder/{repo_name}/`.

---

## 4. The 8-Layer Context Document

The context document is the **primary artifact** of the system. It's what gets fed to an LLM so it can answer questions about the codebase.

### Layer 1: Repository Structure

**Contains:** Tech stack badges, file count, LOC, entry points, directory tree, README excerpt.

**Answers:** *"What kind of project is this? What technologies does it use? Where does execution start?"*

```markdown
## 1. Repository Structure
### Tech Stack
![Python] ![FastAPI] ![Docker] ![React]

**22 files** | **4,476 lines of code**

### Entry Points
- `main.py`

### README
(first 2000 chars)
```

### Layer 2: File Index

**Contains:** Every file with its purpose (from module docstring), imports, and exports.

**Answers:** *"What file handles payments? Where is the database connection configured?"*

```markdown
| File | Purpose | Imports | Exports |
|------|---------|---------|---------|
| `auth/middleware.py` | JWT authentication middleware | jwt, fastapi | AuthMiddleware, verify_token |
| `billing/processor.py` | Stripe payment processing | stripe, models | process_payment, refund |
```

### Layer 3: Symbol Map

**Contains:** Every class (with base classes and methods) and every function (with params, return types, and docstrings).

**Answers:** *"What parameters does `create_user()` accept? What methods does `PaymentProcessor` have?"*

```markdown
### Classes
**PaymentProcessor** — Handles Stripe payment lifecycle
  Methods: `__init__, charge, refund, get_receipt, validate_card`

### Functions
- `create_user(name: str, email: str)` → `User` — Create and persist a new user account
```

### Layer 4: Call Graph Hotspots

**Contains:** Top-20 symbols ranked by PageRank importance score.

**Answers:** *"What's the most critical code? Which functions are called the most?"*

```markdown
| Rank | Symbol | Type | PageRank |
|------|--------|------|----------|
| 1 | `run` | function | 0.0668 |
| 2 | `query` | function | 0.0512 |
| 3 | `connect` | function | 0.0333 |
```

### Layer 5: Data Models

**Contains:** Full class details with all methods and their signatures.

**Answers:** *"What fields does the User model have? What methods are available on the API client?"*

### Layer 6: Business Logic Summaries

**Contains:** AI-generated plain-English description of each file's business purpose.

**Answers:** *"What does the billing module do? Why does the auth middleware exist?"*

### Layer 7: Business Rules & Constraints

**Contains:** Extracted API endpoints, business constants, docstring rules, and TODOs — grouped by type.

**Answers:** *"What are all the API endpoints? What's the rate limit? Are there any known issues?"*

```markdown
### API Endpoints
- API Endpoint: POST /analyze → handler: analyze_repo()
- API Endpoint: GET /graph → handler: get_graph()
- API Endpoint: GET /health → handler: health()

### Business Constants & Limits
- RECENT_COMMITS_LIMIT = 50
- GIT_TIMEOUT = 30
```

### Layer 8: Call Flow & Module Dependencies

**Contains:** Import graph (which files import which) and call relationships (which functions call which).

**Answers:** *"How does data flow from the API endpoint to the database? What depends on the auth module?"*

```markdown
### Module Import Graph
| Importer | Imports |
|----------|---------|
| `main.py` | `api/repos.py` |
| `main.py` | `api/graph.py` |

### Key Call Relationships
| Caller | Callee |
|--------|--------|
| `cli.py::build` | `CodeParser::parse_all` |
| `cli.py::build` | `CallGraphBuilder::build` |
```

### Token Budget Strategy

The context document is designed to fit within LLM context windows:

- **Full context.md:** ~15-25KB (~4,000-7,000 tokens) — fits easily in any modern LLM
- **summary.md:** ~3KB (~800 tokens) — for quick queries or smaller context windows
- Layers are ordered by importance: structure → symbols → hotspots → business rules
- If token-constrained, the first 4 layers alone provide enough for most questions

---

## 5. Knowledge Graph Schema

### Node Types

| Node | Key Properties | Description |
|------|---------------|-------------|
| **Repo** | `name`, `path`, `tech_stack[]`, `entry_points[]`, `file_count`, `readme` | The analyzed repository |
| **File** | `id`, `path`, `language`, `loc`, `docstring`, `summary`, `pagerank` | A source file |
| **Class** | `id`, `name`, `file`, `bases[]`, `docstring`, `pagerank` | A class definition |
| **Function** | `id`, `name`, `file`, `params[]`, `return_type`, `docstring`, `decorators[]`, `pagerank` | A function or method |
| **BusinessRule** | `id`, `content`, `source_file`, `source_line`, `rule_type` | An extracted business rule |

### Relationship Types

| Relationship | From → To | Description |
|-------------|-----------|-------------|
| **CONTAINS** | Repo→File, File→Class, File→Function, Class→Function | Structural hierarchy |
| **IMPORTS** | File→File | Module import dependency |
| **CALLS** | Function→Function | Function call relationship |
| **INHERITS** | Class→Class | Class inheritance |
| **FOUND_IN** | BusinessRule→File | Where a rule was extracted from |
| **ENFORCED_BY** | BusinessRule→Function | Which function enforces the rule |

### Uniqueness Constraints

```cypher
CREATE CONSTRAINT file_id_unique     FOR (f:File)     REQUIRE f.id IS UNIQUE
CREATE CONSTRAINT class_id_unique    FOR (c:Class)    REQUIRE c.id IS UNIQUE
CREATE CONSTRAINT function_id_unique FOR (fn:Function) REQUIRE fn.id IS UNIQUE
CREATE CONSTRAINT repo_name_unique   FOR (r:Repo)     REQUIRE r.name IS UNIQUE
```

### Full-Text Search Index

```cypher
CREATE FULLTEXT INDEX nodeSearch
  FOR (n:File|Class|Function|BusinessRule|DomainConcept)
  ON EACH [n.name, n.summary, n.content]
```

### Useful Cypher Queries

**Find all files in a repo:**
```cypher
MATCH (r:Repo {name: "my-repo"})-[:CONTAINS*1..]->(f:File)
RETURN f.path, f.loc, f.summary
ORDER BY f.path
```

**Get the top hotspots by PageRank:**
```cypher
MATCH (r:Repo {name: "my-repo"})-[:CONTAINS*1..]->(n)
WHERE n.pagerank IS NOT NULL
RETURN n.name, labels(n), n.pagerank
ORDER BY n.pagerank DESC
LIMIT 10
```

**Trace what a function calls:**
```cypher
MATCH (fn:Function {name: "build"})-[:CALLS]->(target)
RETURN fn.name, target.name, labels(target)
```

**Find business rules in a file:**
```cypher
MATCH (br:BusinessRule)-[:FOUND_IN]->(f:File {path: "api/repos.py"})
RETURN br.content, br.rule_type, br.source_line
```

**Get a file's full neighborhood:**
```cypher
MATCH (f:File {path: "cli.py"})-[r]-(neighbor)
RETURN f.path, type(r), neighbor.name, labels(neighbor)
```

---

## 6. API Reference

Base URL: `http://localhost:8000/api`

### Repository Management

#### `POST /analyze`

Start analyzing a repository (runs as a background job).

**Request body:**
```json
{
  "repo_path": "/path/to/repo",
  "repo_name": "my-project",
  "include_git_history": true,
  "generate_llm_summaries": false
}
```

**Response** (202 Accepted):
```json
{
  "job_id": "abc-123",
  "status": "pending",
  "progress": 0,
  "stage": "queued"
}
```

#### `GET /status/{job_id}`

Check analysis job progress.

**Response:**
```json
{
  "job_id": "abc-123",
  "status": "running",
  "progress": 65,
  "stage": "Writing to Neo4j...",
  "repo_name": "my-project"
}
```

Status values: `pending`, `running`, `done`, `failed`

#### `GET /repos`

List all analyzed repositories.

**Response:**
```json
[
  {
    "name": "my-project",
    "path": "/path/to/repo",
    "tech_stack": ["Python", "FastAPI"],
    "entry_points": ["main.py"],
    "file_count": 22
  }
]
```

### Graph Endpoints

#### `GET /graph?repo=...&layer=code&limit=2000`

Fetch graph nodes and edges for visualization.

**Query parameters:**
- `repo` (required) — Repository name
- `layer` — `code` (File/Class/Function), `business` (BusinessRule/DomainConcept), or omit for all
- `node_type` — Exact label filter: `File`, `Class`, `Function`, `BusinessRule`
- `limit` — Max nodes (default 2000, max 10000)

**Response:**
```json
{
  "nodes": [
    {"id": "cli.py", "name": "cli.py", "labels": ["File"], "pagerank": 0.045}
  ],
  "edges": [
    {"source_id": "cli.py", "target_id": "api/repos.py", "type": "IMPORTS"}
  ]
}
```

#### `GET /graph/hotspots?repo=...&top_n=20`

Get the most important nodes by PageRank.

#### `GET /graph/node/{node_id}?repo=...`

Get full detail for a single node plus its 1-hop neighbors (incoming + outgoing).

#### `GET /graph/stats?repo=...`

Get aggregate statistics: file count, class count, function count, edge counts, tech stack.

### Context Endpoints

#### `GET /context/layers?repo=...`

Get the 6-layer breakdown with token estimates and completeness percentages.

**Response:**
```json
{
  "repo": "my-project",
  "layers": [
    {
      "layer": 1,
      "name": "Repository Structure",
      "node_count": 22,
      "token_estimate": 800,
      "completeness": 100
    }
  ],
  "total_tokens": 6500,
  "token_budget": 200000
}
```

#### `GET /context/summary?repo=...`

Returns the compact summary.md content.

#### `GET /context/full?repo=...`

Returns the full context.md content.

### Search

#### `GET /search?q=...&repo=...&limit=20`

Full-text search across all node names, summaries, and content.

**Response:**
```json
[
  {
    "id": "auth/middleware.py::AuthMiddleware",
    "name": "AuthMiddleware",
    "labels": ["Class"],
    "path": "auth/middleware.py",
    "score": 4.82
  }
]
```

### Health Check

#### `GET /health`

```json
{"status": "ok", "neo4j": "connected"}
```

---

## 7. CLI Reference

### `build` — Analyze a Repository

```bash
python backend/cli.py build /path/to/repo [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--name, -n` | Override the repository name (default: folder name) |
| `--summaries, -s` | Generate LLM business summaries using Claude (requires `ANTHROPIC_API_KEY`) |
| `--no-neo4j` | Skip Neo4j, only generate context.md files from in-memory data |

**Examples:**
```bash
# Basic analysis (no Neo4j, no LLM)
python backend/cli.py build ~/projects/my-api --no-neo4j

# Full analysis with Neo4j and LLM summaries
python backend/cli.py build ~/projects/my-api --name my-api --summaries

# Analyze this project itself
python backend/cli.py build backend --name context-builder --no-neo4j
```

**Output:** Creates `/tmp/context_builder/{name}/context.md` and `summary.md`

### `query` — Ask Questions About a Repo

```bash
python backend/cli.py query <repo_name> "<question>"
```

Loads the generated `context.md` and sends it to Claude Sonnet along with your question.

**Examples:**
```bash
python backend/cli.py query my-api "How does user authentication work?"
python backend/cli.py query my-api "What happens when a payment fails?"
python backend/cli.py query my-api "What are all the API endpoints?"
```

Requires `ANTHROPIC_API_KEY` environment variable.

### `list` — Show Analyzed Repos

```bash
python backend/cli.py list
```

Shows all repos with their context.md/summary.md status and file sizes.

---

## 8. Frontend Dashboard

The React dashboard at `http://localhost:5173` provides visual exploration of the knowledge graph.

### Overview Page (`/`)

- **Analyze form** — Enter a repo path and name to start analysis
- **Stats cards** — Files, functions, classes, lines of code
- **Tech stack badges** — Detected technologies
- **Charts** — Node distribution, files by language
- **Hotspots table** — Top symbols by PageRank

### Knowledge Graph Page (`/graph`)

Interactive force-directed graph powered by `react-force-graph-2d`.

**Node colors:**
| Type | Color |
|------|-------|
| File | Blue (`#3b82f6`) |
| Class | Green (`#22c55e`) |
| Function | Orange (`#f97316`) |
| BusinessRule | Purple (`#a855f7`) |
| DomainConcept | Pink (`#ec4899`) |

**Edge colors:**
| Type | Color |
|------|-------|
| CONTAINS | Gray |
| CALLS | Red |
| IMPORTS | Yellow |
| INHERITS | Cyan |

**Features:**
- Click a node to see its detail panel (properties, incoming/outgoing neighbors)
- Filter by layer: All / Code / Business
- Zoom, pan, reset controls
- Hover to highlight connected nodes

### Context Layers Page (`/context`)

- Six cards showing each context layer with token estimate and completeness
- Token budget progress bar (200K token target)
- Full context.md rendered as formatted markdown

### Search Page (`/search`)

- Full-text search input
- Results table with name, type, file path, and relevance score
- Click a result to view details

---

## 9. Running the Project

### Option A: Docker Compose (recommended for full features)

```bash
# 1. Copy environment config
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY if you want LLM summaries

# 2. Start all services
docker-compose up

# Services:
#   Neo4j Browser:  http://localhost:7474  (user: neo4j, pass: contextbuilder)
#   Backend API:    http://localhost:8000
#   Frontend:       http://localhost:5173
```

### Option B: Local Development (without Docker)

```bash
# 1. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Start backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start frontend (new terminal)
cd frontend && npm run dev
```

### Option C: CLI Only (no servers needed)

```bash
source .venv/bin/activate

# Analyze any repo without Neo4j
python backend/cli.py build /path/to/any/repo --name my-repo --no-neo4j

# Read the output
cat /tmp/context_builder/my-repo/context.md

# Ask questions (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python backend/cli.py query my-repo "How does the payment system work?"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `contextbuilder` | Neo4j password |
| `ANTHROPIC_API_KEY` | (none) | Required for LLM summaries and CLI query |

---

## 10. File Map

```
context_builder/
├── CLAUDE.md                          # Project mission, design rules, architecture guide
├── docker-compose.yml                 # Neo4j + Backend + Frontend orchestration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
│
├── backend/
│   ├── main.py                        # FastAPI app entry point, CORS, lifespan
│   ├── cli.py                         # Typer CLI: build, query, list commands
│   │
│   ├── analyzer/
│   │   ├── structure.py               # Tech stack detection, entry points, file stats
│   │   ├── code_parser.py             # Tree-sitter Python AST parsing
│   │   ├── call_graph.py              # NetworkX graph building + PageRank
│   │   └── git_analyzer.py            # Git history mining for change hotspots
│   │
│   ├── enricher/
│   │   ├── business_logic.py          # Rule extraction: endpoints, constants, docstrings, TODOs
│   │   └── summarizer.py              # Claude Haiku LLM summaries per file
│   │
│   ├── graph/
│   │   ├── neo4j_client.py            # Singleton Neo4j driver, connection, schema
│   │   ├── builder.py                 # Ingest parsed data into Neo4j nodes + edges
│   │   └── queries.py                 # Cypher query builders for all API operations
│   │
│   ├── compiler/
│   │   └── context_doc.py             # Assemble 8-layer context.md + summary.md
│   │
│   └── api/
│       ├── repos.py                   # POST /analyze, GET /status, GET /repos
│       ├── graph.py                   # GET /graph, /hotspots, /node, /stats
│       ├── context.py                 # GET /context/layers, /summary, /full
│       └── search.py                  # GET /search (full-text via Neo4j index)
│
├── frontend/
│   ├── package.json                   # React 18, Vite, Tailwind, recharts, force-graph
│   ├── vite.config.ts                 # Dev proxy /api → localhost:8000
│   └── src/
│       ├── App.tsx                    # React Router: /, /graph, /context, /search
│       ├── pages/
│       │   ├── Overview.tsx           # Repo stats, analyze form, hotspots
│       │   ├── Graph.tsx              # Force-directed graph + node detail panel
│       │   ├── Context.tsx            # 6-layer cards + context.md viewer
│       │   └── Search.tsx             # Full-text search with results table
│       ├── components/
│       │   ├── Layout.tsx             # Sidebar navigation + repo selector
│       │   ├── KnowledgeGraph.tsx     # ForceGraph2D with node coloring + controls
│       │   ├── Hotspots.tsx           # Top-N PageRank table
│       │   ├── ContextLayers.tsx      # Layer breakdown cards
│       │   ├── ContextViewer.tsx      # Markdown renderer for context.md
│       │   └── FileExplorer.tsx       # Tree view with PageRank scores
│       └── lib/
│           ├── api.ts                 # API client: types + fetch functions
│           ├── RepoContext.tsx         # Global repo selection state (React Context)
│           └── utils.ts               # formatNumber, truncate, cn()
│
└── docs/
    └── PROJECT.md                     # This file
```
