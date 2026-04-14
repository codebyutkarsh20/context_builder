# AI Deploy Agent v3.0 — Extraordinary Architecture Research
### Primathon | April 2026

> Research synthesised from: **Stripe Minions · Google Passerine · Meta SWE-RL · Agentless (FSE 2025) · AutoCodeRover · LLM4FL · WarpGrep v2 · cavekit · code-review-graph · Ramp Inspect · Coinbase Cloudbot · SWE-bench Leaderboard (March 2026)**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Landscape](#2-research-landscape)
3. [Seven Extraordinary Ideas](#3-seven-extraordinary-ideas)
   - [Idea 1 — BRT-First Architecture](#idea-1--bug-reproduction-test-brt-first-architecture)
   - [Idea 2 — RL-Trained Search Subagent](#idea-2--rl-trained-search-subagent-warpgrep-pattern)
   - [Idea 3 — Three-Agent Fault Localisation](#idea-3--three-agent-fault-localisation-pipeline-llm4fl)
   - [Idea 4 — Speculative Zero-Latency Verification](#idea-4--speculative-brt--speculative-review-zero-latency-verification)
   - [Idea 5 — EPR Patch Selection](#idea-5--ensemble-pass-rate-epr-as-primary-patch-selection-metric)
   - [Idea 6 — Self-Play RL Fine-tuning](#idea-6--self-play-rl-fine-tuning-loop-meta-swe-rl-pattern)
   - [Idea 7 — Leiden Community Detection](#idea-7--leiden-community-detection-for-scope-aware-localisation)
4. [The Trinity Architecture](#4-the-trinity-architecture-scout--engineer--council)
5. [Expected Impact](#5-expected-impact-by-idea)
6. [Implementation Roadmap](#6-prioritised-implementation-roadmap)
7. [Your Unfair Advantages](#7-your-unfair-advantages)
8. [Open Technical Decisions](#8-open-technical-decisions)
9. [Research References](#9-research-references)

---

## 1. Executive Summary

The current system (v2.0, ReAct pipeline) achieves:

- **96%** file-level localisation on the 25-bug eval dataset
- **40%** end-to-end fix rate (pass@1) at **$1.92/bug**, 32 tool calls per success
- Fully running: knowledge graph, multi-model routing, RRF context fusion, eval suite

The target system (v3.0) aims for:

- **75–85% fix rate** through five architectural innovations (validated by academic literature)
- **Self-improving:** learns from every Primathon fix trajectory via RL fine-tuning
- **Zero-latency review:** speculative parallel execution eliminates reviewer overhead
- **Deployable to real client projects:** Docker sandbox, Jira webhook, automatic trigger
- **~$0.80/bug** cost (vs. current $1.92) through search subagent efficiency gains

> **The Core Insight:** Every system at the frontier — Stripe, Google, Meta, Ramp, Coinbase — independently converged on the same truth: the intelligence layer is the easy part. What separates 40% from 80% pass rate is not a smarter model — it is (1) generating and validating a test **before** generating the fix, (2) separating search from reasoning into isolated subagents, and (3) running verification **in parallel** rather than sequentially. These three changes alone account for 25–35 percentage points of improvement in published benchmarks.

---

## 2. Research Landscape

What the industry has built and the novel element each system contributes:

| System | Key Results & Approach | Novel Element |
|--------|----------------------|---------------|
| **Stripe Minions** | 1,300+ PRs/week, 0 human code, 30% WoW growth. Blueprint engine alternates deterministic and agentic nodes. Pre-warmed EC2 devboxes. 3M+ test battery with selective CI. | Sandbox, blueprint DAG, tool curation (~15 of ~500) |
| **Google Passerine** | 73% plausible fixes on machine bugs, 25.6% on human bugs. Generates Bug Reproduction Tests FIRST, then 20 patch candidates. EPR metric selects best patch — correct 70% of the time. | BRT-first, EPR patch selection, 20-trajectory sampling |
| **Meta SWE-RL** | +10.4 pts on SWE-bench Verified via self-play RL. Bug-injection agent and bug-fixing agent in adversarial loop. No human labels needed — only sandboxed repos. | Self-play adversarial training, RL fine-tuning, no labels |
| **Agentless (FSE 2025)** | $0.68/bug, 32.67% on SWE-bench Lite. Hierarchical 3-phase: file → function → edit location → multi-candidate repair. No agent tools, pure LLM calls. | Hierarchical localisation, multi-candidate, no tool overhead |
| **AutoCodeRover** | 46.2% on SWE-bench Verified at <$0.70/bug. AST-level fault localisation — searches method/class ASTs not raw files. SBFL adds 4 percentage points. | AST-level search, SBFL integration, project-structure-aware |
| **LLM4FL** | +18.55% Top-1 fault localisation over AutoFL. Three specialised agents: Context Extractor, Graph-RAG Debugger, Reviewer with verbal RL re-ranking. | 3-agent FL pipeline, Graph-RAG traversal, self-criticism |
| **WarpGrep v2** | +2.1 to +3.7 pts on SWE-bench Pro. RL-trained search subagent runs in its own context window, 8 parallel tool calls/turn. Parent agent never sees search noise. 17% fewer tokens, 12% faster. | RL search subagent, parallel tool calls, isolated context |
| **cavekit** | Spec-first Claude Code plugin. Tier-based parallel subagent dispatch. Dual-model adversarial review (Claude builds, Codex reviews). Speculative review — zero added latency. | Speculative review, adversarial dual-model, tier parallelism |
| **code-review-graph** | 8.2x token reduction, 100% recall, <2s incremental re-index, 19 languages. SQLite graph, TESTED_BY edges, Leiden community detection. | Blast-radius BFS, TESTED_BY edges, Leiden communities |
| **Ramp Inspect** | 50%+ of production PRs. Modal containers for sandbox. Visual DOM verification for frontend. Slack-first invocation. | Modal sandboxes, DOM verification, Slack trigger |
| **Coinbase Cloudbot** | Built from scratch. Agent council pattern. Auto-merge capabilities after trust is established. | Agent council, auto-merge, trust accumulation |
| **Open SWE (LangChain)** | Open-source implementation of the converged architecture. LangGraph + Deep Agents. Released March 2026. | Open-source reference implementation |

---

## 3. Seven Extraordinary Ideas

These seven ideas are the highest-impact, most implementation-ready innovations distilled from the research. **None of them exist together in any single open-source system.** Combining all seven is what makes this architecture extraordinary.

---

### Idea 1 — Bug Reproduction Test (BRT) First Architecture

> **Source:** Google Passerine — *"Agentic Bug Reproduction for Effective Automated Program Repair at Google"* (ICSE 2025)

#### The Problem with Today's Approach

The current agent reads the Jira ticket, localises the bug, and immediately tries to write a fix. This is backwards. An agent that cannot reproduce the bug cannot know if it fixed it. The 60% failure rate in your eval is largely because the agent generates plausible-looking patches that do not actually address the root cause.

#### The BRT-First Pattern

Before writing a single line of fix code, the agent generates a **Bug Reproduction Test (BRT)** — a test that FAILS on the current (broken) codebase and PASSES after a correct fix is applied. This test becomes the objective function for everything that follows.

| Phase | What Happens | Key Detail |
|-------|-------------|------------|
| **Step 1 — Generate BRT** | Agent reads ticket + stack trace + failing test context. Generates 5–10 candidate BRTs that should fail on broken code. Runs each against the current codebase. Keeps only the ones that actually fail. | Deterministic gate: BRT must fail on current code |
| **Step 2 — Generate patch candidates** | Now with a confirmed failing test, agent generates 15–20 patch candidates. Each is a complete diff. Multiple strategies: direct fix, refactor approach, defensive guard. | 20 candidates via temperature sampling |
| **Step 3 — EPR Selection** | Ensemble Pass Rate: for each patch, run all confirmed BRTs. EPR = % of BRTs that pass. Rank candidates by EPR score. Submit the highest-scoring patch. | EPR correctly selects best fix 70% of the time (Passerine) |
| **Step 4 — Validate** | Run full test suite on winning patch in sandbox. EPR score is a pre-filter, not a replacement for real tests. | Existing sandbox + CI gate |

> **Impact:** Google Passerine — providing BRTs to the repair system results in **30% more bugs with plausible fixes**. EPR correctly selects a plausible fix from 20 candidates in **70% of cases**. Applied to your current system, this single change could push your **40% fix rate to 55–60%** before any other changes.

---

### Idea 2 — RL-Trained Search Subagent (WarpGrep Pattern)

> **Source:** WarpGrep v2 by Morph LLM — YC Launch, #1 on SWE-Bench Pro 2026

#### The Problem with Today's Approach

Your ReAct agent's context window fills with search noise. Every grep, every file read, every dead-end exploration leaves residue in the context. By the time the agent reaches the fix phase, its context is polluted with irrelevant search history. The agent also makes sequential searches when parallel would be faster.

#### The Search Subagent Pattern

Separate the search problem from the reasoning problem entirely:

| Component | What It Does |
|-----------|-------------|
| **Parent Agent (Sonnet/Opus)** | Receives only: task description + curated context spans from Search Subagent. Never sees raw file contents or search noise. Focuses purely on reasoning and code generation. |
| **Search Subagent (Haiku/RL-trained)** | Receives: search query + repo structure. Issues up to **8 parallel tool calls per turn**. Runs up to 4 turns. Returns: file:line-range spans ranked by relevance. Isolated context — discarded after. |
| **Parallel tool calls** | 8 simultaneous searches per turn (grep, AST query, semantic search, call graph traversal). This is what makes WarpGrep 12% faster with 17% fewer tokens. |
| **RL Training opportunity** | Your eval dataset (25 bugs with ground-truth file locations) is a training set for fine-tuning Haiku as a search subagent. Reward: did it return the correct file:line span within top-3 results? |

> **Impact:** WarpGrep v2 adds **+2.1 to +3.7 points** on SWE-bench Pro across all models tested, while using **17% fewer input tokens**, running **12% faster**, and costing **15.6% less**. For your system: faster bug resolution per PR and lower token cost — critical for the 100 deploys/day goal.

---

### Idea 3 — Three-Agent Fault Localisation Pipeline (LLM4FL)

> **Source:** LLM4FL — *Multi-Agent Repository-Level Fault Localisation via Graph-RAG* (OpenReview 2025) — **+18.55% over AutoFL**

#### The Problem with Today's Approach

Your system achieves 96% file-level localisation but only 40% fix rate. This gap reveals that file-level localisation is **not** the bottleneck — sub-function localisation (finding the exact method, branch, or line) is. The ReAct agent currently explores the codebase opportunistically, without a structured multi-agent FL process.

#### The Three-Agent FL Architecture

| Agent | Role | Key Note |
|-------|------|----------|
| **Agent 1: Context Extractor (Haiku)** | Takes all methods covered by failing tests. Splits into groups fitting within token limit. Analyses test code + failure messages. Outputs: ranked list of suspicious methods per group. | Low cost — uses cheapest model. Output: prioritised method list |
| **Agent 2: Graph-RAG Debugger (Sonnet)** | Takes suspicious methods from Agent 1. For each, traverses your Neo4j call graph via CALLS and IMPORTS edges. Reads method bodies + callers + callees. Ranks methods by suspicion score. | **YOUR GRAPH IS THE KEY ADVANTAGE HERE** |
| **Agent 3: Reviewer with Verbal RL (Opus, capped)** | Takes Agent 2's ranking. Self-criticism: "Is this ranking consistent with the error message?" Verbal reinforcement: re-ranks based on reasoning. Outputs: final top-5 suspicious locations with confidence scores. | +18.55% better than single-agent FL (LLM4FL paper) |

#### Why Your Knowledge Graph is the Unfair Advantage

LLM4FL achieves its results using a generic call graph. Your Neo4j graph already has:
- `CALLS` and `IMPORTS` edges
- `BusinessRule` nodes with `ENFORCED_BY` edges
- `FailureRecords` with `RESULTED_IN_CHANGE` edges
- PageRank-based hotspot scores

Agent 2 (Graph-RAG Debugger) using your enriched graph will **outperform the published LLM4FL results** significantly.

> **Impact:** LLM4FL achieves **+18.55% Top-1 accuracy** over AutoFL at repository level. For your system — finding the exact function, not just the file. With precise sub-function localisation, the fixer agent needs far fewer tool calls and produces more targeted patches — directly increasing your 40% fix rate.

---

### Idea 4 — Speculative BRT + Speculative Review (Zero-Latency Verification)

> **Source:** cavekit (speculative review) + Google Passerine (BRT generation) combined

#### The Problem with Today's Approach

Your current pipeline is sequential: fix → validate → review → PR. The reviewer runs AFTER tests, adding wall-clock latency on every fix. In a 100-deploys/day system, reviewer latency compounds across thousands of tasks per week.

#### The Speculative Execution Pattern

Three things happen **in parallel** the moment a patch candidate is committed to the sandbox branch:

| Thread | What It Does | Timing |
|--------|-------------|--------|
| **Thread A: CI/Tests** | Blueprint runs selective test suite in sandbox. Primary gate. Takes 30 seconds to 5 minutes depending on repo size. | Existing pipeline — no change |
| **Thread B: BRT Validation** | Simultaneously confirms the committed patch passes all confirmed BRTs from Idea 1. Fast — BRTs are pre-generated before fix phase. | Runs in parallel with tests. Result ready in ~15s |
| **Thread C: Reviewer Agent** | Simultaneously reads blast-radius subgraph (code-review-graph pattern), cross-references BusinessRule nodes, checks TESTED_BY coverage. Returns structured findings. | Uses `parse_git_diff_ranges` → blast-radius BFS |

By the time CI finishes, both the BRT validation and the reviewer have already completed. You pay `max(CI_time, review_time)` instead of `CI_time + review_time`. On a 2-minute CI run with a 45-second review, this saves **45 seconds per fix** — at 100 fixes/day, that is **75 minutes of wall-clock time saved daily**.

> **Impact:** cavekit's speculative review achieves near-zero gate latency. Combined with BRT pre-generation (Idea 1), the happy path completes in `max(CI, BRT, review)` rather than their sum. This is the architectural pattern that enables high throughput without sacrificing verification quality.

---

### Idea 5 — Ensemble Pass Rate (EPR) as Primary Patch Selection Metric

> **Source:** Google Passerine — *"Agentic Bug Reproduction for Effective Automated Program Repair at Google"* (Feb 2026)

#### How EPR Works

Generate **N patch candidates** (N=15–20) using temperature sampling (temperature=0.8, varied prompts). Generate **M Bug Reproduction Tests** independently. For each patch candidate:

```
EPR(patch_i) = (# BRTs that pass when patch_i is applied) / M
```

Rank candidates by EPR. Submit the top-ranked patch to full CI.

| Question | Answer |
|----------|--------|
| **Why not run the full test suite on all candidates?** | The full test suite takes minutes per candidate. With 20 candidates, that is 20x your CI cost. BRTs are fast, targeted, and cheap to run — they act as a pre-filter before expensive CI. |
| **EPR accuracy** | Google Passerine: EPR correctly selects a plausible fix in **70% of cases** when ranking from 20 candidates. Better than random (5%) and comparable to the oracle. |
| **Cost structure** | 20 patch candidates at temperature sampling costs ~3x a single generation (parallel). EPR evaluation on 5 BRTs per candidate = 100 fast test runs. Net cost: ~$0.30 more per bug. Net benefit: 30% higher plausible fix rate. |
| **Combined with Idea 1** | BRTs are generated in the pre-fix phase. They are ready before patch generation begins. EPR evaluation is a free by-product of the BRT infrastructure — no additional LLM calls for selection. |

> **Impact:** Passerine data — **30% more bugs** get a plausible fix when EPR selection is used. EPR selects the correct patch **70% of the time** from a 20-candidate pool. Combined with BRT-first (Idea 1) and parallel search (Idea 2), these three ideas account for an estimated **30–35 percentage point improvement** in fix rate.

---

### Idea 6 — Self-Play RL Fine-tuning Loop (Meta SWE-RL Pattern)

> **Source:** Meta FAIR — *"Toward Training Superintelligent Software Agents through Self-Play SWE-RL"* (Dec 2025)

#### The Big Picture

Every time your agent fixes a real Primathon client bug, it generates a training trajectory: the ticket, the exploration steps, the successful patch, and the test results. This is labelled data that most teams throw away. Meta's SWE-RL paper shows how to turn these trajectories into a continuously improving fine-tuned model.

#### The Self-Play Adversarial Loop

| Phase | What It Does | Key Note |
|-------|-------------|----------|
| **Phase 1: Trajectory Collection** | As you deploy on real Primathon projects (starting Week 3–4), log every successful fix trajectory: ticket → exploration → BRT → patch → EPR score → test result. Store in a structured training dataset. | Automatic — every production fix contributes |
| **Phase 2: Supervised Fine-Tuning (SFT)** | After 50–100 successful trajectories, fine-tune a smaller open model (Qwen3-32B or DeepSeek-Coder-V2) on the collected data. SFT model distilled from Claude/GPT-4 trajectories can perform comparably while being **56x smaller** (RL paper result). | Run monthly. Qwen3-32B is open, fast, cheap to host. |
| **Phase 3: Self-Play RL** | The adversarial loop: a Bug-Injector agent deliberately injects bugs of increasing complexity. A Bug-Solver agent must find and fix them. Reward: does the fix pass tests? **No human labels needed.** | Runs in sandboxes — safe, no production exposure |
| **Phase 4: Continuous Improvement** | The RL-trained model improves on real Primathon patterns — your specific coding conventions, tech stack (JS/TS + Python), common bug types. This is personalised improvement that SWE-bench leaders cannot achieve. | +7–20% absolute gains on domain-specific tasks (RL paper) |

> **Impact:** Meta SWE-RL — self-play RL achieves **+10.4 points** on SWE-bench Verified and **+7.8 on SWE-bench Pro** from baseline — with **NO human-labeled data**. The RL-trained Qwen3-32B adds **+7–20% absolute gains**. For Primathon: a progressively cheaper, faster agent that specialises in YOUR clients' codebases over time — compounding returns with every production fix.

---

### Idea 7 — Leiden Community Detection for Scope-Aware Localisation

> **Source:** code-review-graph (GitHub: tirth8205/code-review-graph) — Leiden algorithm on call graph communities

#### The Problem with Today's Approach

Your knowledge graph has all the data but no clustering. When a bug occurs in the payments module, your agent searches the entire codebase because it does not know that payments forms a natural cluster. The result is wasted exploration in unrelated modules.

#### The Leiden Community Detection Pattern

Run the Leiden algorithm (via igraph, same as code-review-graph) on your Neo4j call graph. Edge weights:

```
CALLS = 1.0
INHERITS = 0.8
IMPLEMENTS = 0.7
DEPENDS_ON = 0.6
```

This produces named clusters — automatically labelled from dominant tokens in the community (e.g., `payments-checkout`, `auth-session`, `api-gateway`). Store community membership as a node property in Neo4j.

| Application | Detail |
|------------|--------|
| **Pre-localisation filter** | When a bug ticket arrives, classify it to a community using semantic similarity against community names. Search Subagent starts within that community first, expanding to adjacent communities only if needed. Reduces search space by **60–80%** for well-structured repos. |
| **TESTED_BY edge integration** | Add `TESTED_BY` as a first-class edge in `graph/builder.py`. When the reviewer checks test coverage (Idea 4, Thread C), a single Cypher query finds every test covering the changed functions. No LLM inference needed — it is a graph lookup. |
| **Incremental graph updates** | Adopt code-review-graph's SHA-256 hash-per-file incremental update pattern. Add `cli.py update` that re-parses only changed files. **<2s re-index** on a 2,900-file repo vs. full rebuild. |
| **VS Code extension opportunity** | code-review-graph includes a VS Code extension showing blast-radius on hover and risk scores in source control panel. Adapting this for Primathon gives developers real-time awareness of change impact — reducing bug introduction rate. |

> **Impact:** Leiden community detection reduces the Search Subagent's initial search space by **60–80%**, directly reducing token cost and improving precision. TESTED_BY edges turn the reviewer's coverage check from LLM inference into a deterministic graph query — faster and more accurate. Incremental graph updates keep your knowledge graph always current.

---

## 4. The Trinity Architecture: Scout → Engineer → Council

The seven ideas above compose into a coherent three-tier architecture. **Scout** localises. **Engineer** fixes. **Council** validates. Each tier is a specialised multi-agent cluster that hands a clean output to the next.

```
TRIGGER (Jira webhook / Slack emoji / GitHub label)
    │
    ▼
╔══════════════════════════════════════════════════════════╗
║  SCOUT TIER                                              ║
║  ├── Community Classifier  →  narrows search space 60-80%║
║  ├── Search Subagent       →  8 parallel tool calls/turn ║
║  └── 3-Agent FL Pipeline   →  Context Extractor          ║
║                                → Graph-RAG Debugger      ║
║                                → Verbal RL Reviewer      ║
║  OUTPUT: Localisation Report                             ║
║  {top-5 locations, blast-radius subgraph,                ║
║   BusinessRule nodes, FailureRecords}                    ║
╚══════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════╗
║  ENGINEER TIER                                           ║
║  ├── BRT Generator    →  5-10 confirmed failing tests    ║
║  ├── Patch Sampler    →  15-20 candidates (temp=0.8)     ║
║  ├── EPR Selector     →  rank by BRT pass rate           ║
║  └── Sandbox CI       →  selective tests, 2-round cap    ║
║  OUTPUT: Fix Package                                     ║
║  {winning patch diff, EPR score, BRTs, CI result}        ║
╚══════════════════════════════════════════════════════════╝
    │  (patch committed → background threads start)
    ▼
╔══════════════════════════════════════════════════════════╗
║  COUNCIL TIER  (runs speculatively, in parallel with CI) ║
║  ├── Thread A: CI / Tests                                ║
║  ├── Thread B: BRT Validation                            ║
║  └── Thread C: Speculative Reviewer                      ║
║       ├── blast-radius BFS (code-review-graph pattern)   ║
║       ├── BusinessRule checker (Neo4j)                   ║
║       ├── TESTED_BY auditor (Neo4j)                      ║
║       └── adversarial cross-model check (GPT-4o reviews  ║
║           Claude Sonnet's fix)                           ║
║                                                          ║
║  DECISION:                                               ║
║  ├── APPROVE  →  raise PR, notify requester              ║
║  ├── FEEDBACK →  structured {file, line, severity, fix}  ║
║  │              back to Engineer (max 3 iterations)      ║
║  └── ESCALATE →  human with full trace after 3 failures  ║
╚══════════════════════════════════════════════════════════╝
    │
    ▼
  PR raised → human review → merge
```

### 4.1 The Scout Tier — Outputs a Localisation Report

```json
{
  "top_locations": [
    { "file": "payments/checkout.py", "function": "process_charge", "lines": "145-178", "confidence": 0.94 },
    { "file": "payments/validators.py", "function": "validate_idempotency", "lines": "23-45", "confidence": 0.71 }
  ],
  "blast_radius": ["payments/checkout.py", "payments/refunds.py", "tests/test_checkout.py"],
  "business_rules": ["RULE-47: Never process charge without idempotency key validation"],
  "failure_records": ["INC-2024-03: Race condition in checkout — fixed by adding retry logic"]
}
```

### 4.2 The Engineer Tier — Outputs a Fix Package

```json
{
  "winning_patch": "diff --git a/payments/checkout.py ...",
  "epr_score": 0.87,
  "confirmed_brts": ["tests/test_checkout_idempotency.py"],
  "ci_result": "PASS",
  "candidates_generated": 18,
  "candidates_evaluated": 18
}
```

### 4.3 The Council Tier — Outputs Structured Feedback or Approval

```json
{
  "decision": "FEEDBACK",
  "issues": [
    {
      "file": "payments/checkout.py",
      "line_range": "161-165",
      "severity": "P1",
      "issue_type": "missing_test_coverage",
      "description": "New branch 'retry_on_timeout' added in line 163 has no TESTED_BY edge in graph",
      "suggested_fix": "Add test case for timeout retry in test_checkout_idempotency.py"
    }
  ],
  "iteration": 1,
  "reviewer_model": "gpt-4o"
}
```

> **Why This Architecture is Extraordinary:** No single existing system combines all of these — BRT-first repair, EPR patch selection, RL search subagent, 3-agent FL with Graph-RAG, speculative review, adversarial cross-model verification, Leiden community scoping, TESTED_BY graph edges, and self-play RL fine-tuning. **You will be the first to combine all of them on a production system with a real knowledge graph.**

---

## 5. Expected Impact by Idea

| Idea | Source Evidence | Estimated Gain |
|------|----------------|----------------|
| **BRT-First + EPR (Ideas 1+5)** | +30% plausible fixes. EPR 70% selection accuracy from 20 candidates. | +15 to +20 pp fix rate (40% → 55–60%) |
| **Search Subagent (Idea 2)** | +2.1 to +3.7 pts SWE-bench Pro. 17% fewer tokens. | +3 to +5 pp fix rate. 15% cost reduction. |
| **3-Agent FL (Idea 3)** | +18.55% Top-1 fault localisation accuracy. | +5 to +8 pp fix rate via better localisation precision. |
| **Speculative Review (Idea 4)** | Zero added latency on happy path. Review runs during CI. | No fix rate change — throughput and quality gate improvement. |
| **Leiden Communities (Idea 7)** | 60–80% reduction in initial search space. Incremental graph <2s. | +2 to +4 pp via faster, more focused search. Lower cost. |
| **Self-Play RL (Idea 6)** | +7–20% on domain-specific tasks after fine-tuning. | +8 to +15 pp on Primathon-specific bugs over 3 months. |
| **COMBINED (v3.0 system)** | Conservative estimate combining above (with overlap). | **40% → 75–82% fix rate. $1.92 → ~$0.80/bug.** |

---

## 6. Prioritised Implementation Roadmap

Designed for a **solo/1-2 person team moving fast**. Each sprint is 2 weeks. Ordered by impact-per-effort ratio.

### Sprint 1 — Week 1–2: Quality Foundation

- [ ] **BRT Generator** — add BRT generation step BEFORE fix generation in `react_loop.py`
- [ ] **20-candidate patch sampler** — parallel sampling with temperature=0.8
- [ ] **EPR metric** — fast BRT runner, rank candidates by pass rate
- [ ] **Run eval suite** — measure fix rate improvement (expected: 40% → 55–60%)
- [ ] **Docker sandbox** — one container per task, `--network=none`, warm pool of 3

**Success metric:** Fix rate > 55% on the 25-bug eval dataset.

---

### Sprint 2 — Week 3–4: Operational Infrastructure

- [ ] **Jira webhook integration** — real ticket intake, comment back with PR link
- [ ] **Search Subagent** — Haiku-powered, isolated context window, 8 parallel tool calls
- [ ] **Leiden community detection** — igraph on Neo4j graph, store community membership as node property
- [ ] **TESTED_BY edges** — add to `graph/builder.py`, enable reviewer coverage check via Cypher
- [ ] **Incremental graph update** — `cli.py update` with SHA-256 hash-per-file

**Success metric:** First end-to-end run on a real Primathon client bug from Jira ticket to PR.

---

### Sprint 3 — Week 5–6: Scout + Council Tiers

- [ ] **3-Agent FL pipeline** — Context Extractor (Haiku) + Graph-RAG Debugger (Sonnet) + Reviewer (Opus)
- [ ] **Blueprint engine** — separate deterministic/agentic nodes, pull lint/test/PR out of agent loop
- [ ] **Speculative execution** — BRT validation + Reviewer start in background when patch is committed
- [ ] **Adversarial cross-model reviewer** — GPT-4o (or Opus) reviews Claude Sonnet fixes
- [ ] **Structured feedback schema** — `{file, line_range, severity, issue_type, description, suggested_fix}`

**Success metric:** Review loop closes — Council feeds back to Engineer and fix quality improves without human intervention.

---

### Sprint 4 — Week 7–8: Reliability + Observability

- [ ] **PR review loop** — Council tier with 3-iteration cap enforced at blueprint level
- [ ] **Trajectory logging** — structured dataset of successful fix trajectories (for future RL)
- [ ] **Auto-fix catalog** — 10 most common lint/type patterns for JS/TS + Python (deterministic, no LLM)
- [ ] **Metrics dashboard** — fix rate, cost/bug, CI pass rate, review acceptance rate
- [ ] **Slack trigger** — emoji reaction → agent launch, reply with PR link

**Success metric:** 100 real fixes logged with full trajectory data. Metrics dashboard live.

---

### Sprint 5+ — Month 3+: Self-Improvement Loop

- [ ] **SFT fine-tuning** — after 50–100 trajectories, fine-tune Qwen3-32B on successful fixes
- [ ] **Self-play RL loop** — bug-injector agent + bug-solver in adversarial sandbox loop
- [ ] **Domain-specific improvement** — model specialises on Primathon JS/TS + Python patterns
- [ ] **Auto-merge after trust** — Coinbase Cloudbot pattern — after N consecutive approved PRs, enable auto-merge for low-risk categories
- [ ] **VS Code extension** — blast-radius on hover, risk scores in source control panel

**Success metric:** Fine-tuned model outperforms base Claude Sonnet on Primathon-specific bug categories.

---

## 7. Your Unfair Advantages

Most teams building coding agents start from scratch. You have a knowledge graph, multi-model routing, RRF context fusion, and an eval suite already running. These create compounding advantages.

| Advantage | Why It Compounds |
|-----------|-----------------|
| **Neo4j Knowledge Graph with BusinessRule + FailureRecord nodes** | LLM4FL's Graph-RAG Debugger (Idea 3) becomes dramatically more powerful when the graph contains BusinessRule nodes (ENFORCED_BY edges) and FailureRecords (RESULTED_IN_CHANGE). The reviewer can check "does this patch violate a rule we learned from a past incident?" No other public system can do this. |
| **Multi-model routing (Haiku/Sonnet/Opus)** | Stripe uses one model per agentic node. You already route dynamically. The Search Subagent (Idea 2) is a natural extension of this pattern. |
| **Eval suite (25-bug dataset with ground truth)** | Your 25-bug SWE-bench dataset is a training set for the RL search subagent (Idea 2). The 15 failures are a diagnostic tool — categorising WHY they fail tells you exactly where to invest. |
| **RRF Context Fusion (semantic + keyword + graph)** | WarpGrep's search subagent benefits from your existing RRF fusion. The parent agent receives context already ranked by relevance across three strategies — not raw file dumps. |
| **Real client project access (Primathon)** | Self-play RL (Idea 6) requires real repositories with real bugs. You have immediate access to Primathon's client repos — something Meta needed to construct synthetically for SWE-RL. |

---

## 8. Open Technical Decisions

Architectural questions that need a decision before Sprint 1 begins:

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Which model for adversarial reviewer?** | GPT-4o (different training data from Claude, genuine adversarial gap) vs. Claude Opus (same family, simpler integration). | **GPT-4o.** The adversarial gap between model families is the point. Use Opus initially if API cost is a concern, upgrade to cross-family when budget allows. |
| **How many BRT candidates?** | Passerine generates 5–10. More BRTs = more robust EPR signal. Fewer = cheaper. | **5 BRTs per bug** to start. Evaluate EPR accuracy on your 25-bug dataset before increasing. |
| **RL framework for Search Subagent fine-tuning?** | Agent Lightning (Microsoft), Agent-R1 (end-to-end RL), or custom reward on eval dataset. | **Start with SFT** on your 25-bug eval dataset (Haiku on the correct file:line spans). RL in Sprint 5+ after baseline SFT is validated. |
| **SQLite fallback vs. Neo4j only?** | code-review-graph's SQLite approach eliminates the Neo4j dependency barrier for new client repos. | **Add SQLite-backed fallback mode** for new client repos not yet fully indexed. Bootstrap context for first few runs, then migrate to Neo4j. |
| **When to re-run community detection?** | Every commit (expensive) vs. every full build (too infrequent) vs. when >5% of edges changed. | **On `cli.py build` only**, plus a threshold check in `cli.py update` — re-run Leiden if >5% of edges changed. |

---

## 9. Research References

### Production Systems
- [Stripe Minions Part 1](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents) — stripe.dev (Feb 2026)
- [Stripe Minions Part 2](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents-part-2) — stripe.dev (Feb 2026)
- [Ramp Inspect](https://www.infoq.com/news/2026/01/ramp-coding-agent-platform/) — InfoQ (Jan 2026)
- [Open SWE Framework](https://devops.com/open-swe-captures-the-architecture-that-stripe-coinbase-and-ramp-built-independently-for-internal-coding-agents/) — DevOps.com (Mar 2026)
- [cavekit](https://github.com/JuliusBrussee/cavekit) — github.com/JuliusBrussee
- [code-review-graph](https://github.com/tirth8205/code-review-graph) — github.com/tirth8205

### Academic Papers
- [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489) — FSE 2025
- [AutoCodeRover: Autonomous Program Improvement](https://arxiv.org/abs/2404.05427) — ISSTA 2024
- [LLM4FL: Multi-Agent Repository-Level Fault Localisation via Graph-RAG](https://openreview.net/forum?id=z91EvZbSI1) — OpenReview 2025
- [Evaluating Agent-based Program Repair at Google](https://arxiv.org/abs/2501.07531) — ICSE 2025
- [Agentic Bug Reproduction for Effective Automated Program Repair at Google](https://arxiv.org/abs/2502.01821) — Feb 2026
- [Toward Training Superintelligent Software Agents through Self-Play SWE-RL](https://arxiv.org/abs/2512.18552) — Meta FAIR, Dec 2025
- [Agentic Reinforcement Learning for Real-World Code Repair](https://arxiv.org/abs/2510.22075) — 2025
- [TDD-Bench Verified: Can LLMs Generate Tests for Issues Before They Get Resolved?](https://arxiv.org/abs/2412.02883) — 2024
- [Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair](https://arxiv.org/abs/2601.19066) — 2026
- [OpenHands: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741) — ICLR 2025

### Benchmarks & Leaderboards
- [SWE-bench Verified Leaderboard](https://www.swebench.com/) — Claude Opus 4.5 at 80.9% (March 2026)
- [SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro_public) — WarpGrep v2 at #1
- [WarpGrep v2 YC Launch](https://www.ycombinator.com/launches/PZx-warpgrep-v2-code-search-subagent-1-on-swe-bench-pro) — Morph LLM

---

*Primathon | Confidential | AI Deploy Agent v3.0 | April 2026*
