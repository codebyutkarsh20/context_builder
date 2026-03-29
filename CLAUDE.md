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
### Problem Statement 2: Developer + Reviewer Agent Pipeline
Build a LangGraph pipeline: Intake → Developer → Reviewer → PR Agent.
Agentless-first (hierarchical localization), evolve to agentic (graph-guided navigation).
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
├── intake/         # Jira → WorkOrder parsing
├── context/        # Knowledge graph queries + context assembly
├── localization/   # Hierarchical + graph-guided fault localization
├── repair/         # Multi-patch generation + test validation
├── review/         # AI code review with business rule checking
├── pr/             # GitHub PR creation + test evidence
├── graph/
│   ├── builder/    # Tree-sitter → Neo4j pipeline
│   ├── business/   # Business knowledge extraction + ingestion
│   └── queries/    # Cypher query templates
├── embeddings/     # ChromaDB pipeline
└── eval/           # Evaluation suite
## Build & Test
```bash
pip install -r backend/requirements.txt
pytest tests/ -v                                    # All tests
pytest tests/test_localization.py -v                # Single module
python -m agent.run --ticket=PROJ-1234 --dry-run    # Test on one ticket
python -m agent.graph.build --repo=/path/to/repo    # Build knowledge graph
python -m agent.eval.run --dataset=eval/bugs.json   # Run eval suite
```
## Key Design Rules
- MUST use LangGraph for all agent orchestration
- MUST query BOTH code graph AND business knowledge graph before generating fixes
- MUST include business context (rules, constraints, past incidents) in every repair prompt
- MUST run target repo's test suite in sandbox before creating PR
- MUST cap developer↔reviewer loop at 3 iterations, then escalate to human
- NEVER let agent modify files outside target repo
- NEVER hardcode repo-specific knowledge in agent code — it belongs in the knowledge graph
- NEVER remove validation checks from target code without explicit business rule confirmation

## Current State & Next Steps
[UPDATE THIS WEEKLY]
- [ ] Knowledge graph construction pipeline (Tree-sitter → Neo4j)
- [ ] Business knowledge extraction from git history + Jira
- [ ] Context assembly with dual-layer graph queries
- [ ] Localization agent with graph-guided traversal
- [ ] Repair agent with multi-patch sampling
- [ ] Review agent with business rule verification
- [ ] PR creation pipeline
- [ ] Eval suite against real bug dataset

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
