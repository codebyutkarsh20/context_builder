# TODOS

Roadmap of known follow-ups. Completed items live in [`CHANGELOG.md`](CHANGELOG.md). For larger discussions, open a GitHub issue.

## Active

### P2 — Continue extracting utilities from `pipeline.py`
**What:** `pipeline.py` is still 3300 lines. Utility functions (`_redact_secrets`, `_fuzzy_match_replace`, `should_iterate`, etc.) are still imported by tests. Extract into focused modules: `llm_utils.py`, `secrets.py`, `file_utils.py`, `analysis_utils.py`, `patch_utils_extended.py`.
**Why:** The fixed pipeline is retired but the file remains as a utility library. A 6-module extraction plan is sketched in prior eng review notes.

### P2 — Validate `ENFORCED_BY` edge precision before production use
**What:** Run `BusinessLogicExtractor` on 20 manually annotated functions, measure precision/recall of `ENFORCED_BY` edges. Gate business-context injection on >80% precision.
**Why:** The extractor links rules to functions by docstring keyword matching. Low precision means repair prompts get noisy rule context that degrades fix quality.

### P3 — `FailureRecord` severity classification (enhanced)
**What:** Basic severity classification exists (`_severity_hint` parses P0/Sev0/critical/hotfix keywords). Enhance with PR labels via GitHub API, or infer from issue-tracker priority for richer signal.
**Why:** Current classification is keyword-only from commit messages. Labels and tracker priority give cleaner signal for distinguishing P0 incidents from cosmetic fixes.
**Where:** `mine_failure_records()` in `backend/graph/business/failure_records.py`.

### P4 — Forward-Looking Structured History (Lore protocol)
**What:** After 5–10 agent PRs are merged, implement agent-written git trailers on every PR: rules checked, blast radius, failure pattern ID. Git becomes the `FailureRecords` DB over time (arxiv 2603.15566).
**Why:** Records mined from git history are retrospective and limited by commit-message quality. Agent-authored trailers create forward-looking, structured history with full agent context attached.

### P5 — Extract external side effects behind an interface (dependency inversion)
**What:** `pr_creation_node()` has multiple external side effects (git push, `gh pr create`, feature flags, `_enrich_from_fix`) each guarded by `if not dry_run`. Extract behind a thin interface and no-op in test/eval mode.
**Why:** The boolean-flag approach works for 3–4 calls but creates maintenance burden as new side effects are added. Forgetting a guard is a silent bug.

## How to pick something up

1. Comment on (or open) the corresponding GitHub issue so we don't duplicate work.
2. Read the linked source file and the relevant `CHANGELOG.md` entries that touched it.
3. Open a PR per [`CONTRIBUTING.md`](CONTRIBUTING.md).
