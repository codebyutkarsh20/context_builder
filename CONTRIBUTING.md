# Contributing

Thanks for your interest. This project is in early-stage open source — issues and PRs are welcome.

## Quick local setup

```bash
git clone <this-repo>
cd context_builder
cp .env.example .env  # set ANTHROPIC_API_KEY at minimum
docker-compose up      # Neo4j + backend + frontend
```

For native (no Docker) setup, see the README's *Running without Docker* section.

## Running the test suite

```bash
# from project root
python -m pytest backend/tests/ -q
```

Sub-targets:

```bash
python -m pytest backend/tests/test_react_loop.py -q   # one file
python -m pytest backend/tests/ -k "not e2e" -q        # skip e2e
```

The end-to-end suite (`test_hardening_e2e.py`) is **not run by default** — it requires a live backend, frontend, and a real repo to point at. See the docstring at the top of that file for env-var configuration.

## Lint / format

```bash
ruff check backend/
```

We don't enforce a formatter yet; please match the surrounding style.

## Branch and PR workflow

- Branch off `main`. Use a short descriptive name: `fix/sandbox-leak`, `feat/multi-repo-tickets`.
- Keep PRs focused — one concern per PR.
- Add tests for any behavioral change. The repo currently sits at ~1180 tests; we'd like to keep the ratio.
- Update `CHANGELOG.md` under `[Unreleased]` with a one-line summary.
- Make sure `python -m pytest backend/tests/ -q` passes locally before pushing.

## What kind of changes are most welcome

- **Bug fixes** — especially around sandbox isolation, dependency-compat for new SWE-bench repo families, or shell-safety edge cases.
- **New language frontends** — adding tree-sitter parsers for languages we don't yet cover.
- **Eval improvements** — new datasets, more rigorous scoring, additional benchmarks.
- **Docs** — the architecture is genuinely complex; clearer explanations help everyone.

## What to avoid

- Hardcoded paths, personal API keys, or references to private repos in code or tests.
- Vendored binaries / large datasets — link to a download script instead.
- Sweeping refactors with no behavioral motivation.

## Reporting bugs

Please include: command you ran, the bug ticket / input, the exit log or trace ID, and the version of `claude-sonnet-4-x` you're on (printed at startup).

## Questions

Open a GitHub Discussion or Issue. For security reports, see `SECURITY.md`.
