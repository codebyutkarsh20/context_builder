# AI Deploy Agent

Autonomous bug-fix agent: give it a ticket → it reads your codebase → finds the bug → writes a fix → tests it → submits a PR.

**44% pass rate on SWE-bench Lite, $0.50–$0.60/bug average.** [v4 architecture](docs/v4-complete-agent-reference.md)

---

## Quickstart (Docker — 5 minutes)

```bash
# 1. Clone
git clone <this-repo>
cd context_builder

# 2. Configure
cp .env.example .env
# Edit .env — at minimum, set ANTHROPIC_API_KEY=sk-ant-...

# 3. Start
docker-compose up

# 4. Open
#    Frontend:  http://localhost:5173
#    API:       http://localhost:8001
#    Neo4j UI:  http://localhost:7474  (user: neo4j, pass: contextbuilder)
```

That's it. Services auto-start, volumes persist across restarts.

## What you need

- **Docker** (Docker Desktop on macOS/Windows, or docker + docker-compose on Linux)
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com)
- Optional: **GitHub token** (`GH_TOKEN` in `.env`) if you want the agent to open PRs

## Fix your first bug

### Via the UI

1. Open http://localhost:5173
2. Go to **Overview** → paste the absolute path to your repo → **Analyze**
3. Wait ~30s for the knowledge graph to build
4. Go to **Agent** → describe the bug (title + description) → **Run**
5. Watch the live trace. The agent will explore, edit, test, and submit.

### Via the CLI

```bash
cd backend
python cli.py fix MY-BUG-001 \
  --title "Dates parse wrong in timezone-aware mode" \
  --desc "When TIMEZONE is set, ISO strings lose their offset..." \
  --repo /path/to/your/repo
```

The agent will:
1. Create an isolated git worktree (doesn't touch your working tree)
2. Explore the codebase using grep, read_file, and graph queries
3. Write a plan, make edits, run tests, verify with an AI reviewer
4. Submit a PR (or print the branch + diff if you pass `--dry-run`)

## Architecture (3-stage pipeline)

```
setup_node  →  react_agent_node  →  finalize_node
   8s             ~60s                5s
   
• 3 parallel threads:                 • Free-form Sonnet 4.6 loop
  - repo detection + sandbox          • 20 tools (grep, read, edit,
  - scout localization (Haiku+Sonnet)   run_shell, write_brt, verify_fix)
  - context assembly (graph, lessons) • Forked-subagent verifier
```

Full reference: [docs/v4-complete-agent-reference.md](docs/v4-complete-agent-reference.md)

## Features

| | |
|---|---|
| **Multi-language** | Python, JavaScript, TypeScript, Go, Rust |
| **Knowledge graph** | Neo4j-backed — callers, blast radius, business rules |
| **Per-repo learning** | `agent_lessons.md` persists across runs |
| **Cross-repo patterns** | Global lessons transfer between codebases |
| **Multi-repo tickets** | `switch_repo` tool for bugs spanning 2+ repos |
| **Forked verifier** | Independent AI review with full conversation context |
| **Multi-patch sampling** | `--best-of-n 3` generates K candidates, picks best |
| **Safety** | Denylist blocks destructive shell commands, path containment |

## Running without Docker

If you prefer native:

```bash
# 1. Start Neo4j (still needed for the knowledge graph)
docker run -d \
  --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/contextbuilder \
  neo4j:5.23-community

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set env
export $(cat ../.env | xargs)
export DATA_DIR=$HOME/.context_builder  # or wherever you want persistence

# Start API
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Config reference

See `.env.example` for all options. Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Main agent model |
| `NEO4J_PASSWORD` | `contextbuilder` | Neo4j auth (matches docker-compose) |
| `DATA_DIR` | `/data` (docker) | Where lessons + graphs are stored |
| `USER_REPOS_HOST` | `$HOME` | Host dir mounted into backend container |
| `GH_TOKEN` | (optional) | For PR creation |

## Running the evaluation

The project ships with SWE-bench Lite datasets for benchmarking:

```bash
cd backend
python cli.py eval run --dataset ../eval/swebench_50_vague.json --nl
```

Results are written to `backend/eval/results/`. See current scores in the [agent reference](docs/v4-complete-agent-reference.md).

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** — Edit `.env` and set your key.

**"Cannot connect to Neo4j"** — Run `docker-compose ps`; make sure `context_builder-neo4j-1` is `healthy`.

**"No .git directory"** — The repo you point at must be a real git repo (the agent uses worktrees).

**"Port already in use"** — Something else is on 5173/8001/7474. Either stop it, or edit `docker-compose.yml` to use different host ports.

**Agent fails on Sympy/older repos** — Some SWE-bench repos need Python 3.9 (we handle this via `_find_compat_python`). If you add new repos, check their Python version requirement.

## Contributing

- Tests: `python -m pytest backend/tests/ -q` (1170 tests)
- Lint: `ruff check backend/`
- Docs: Main reference at `docs/v4-complete-agent-reference.md`

## License

See `LICENSE`.
