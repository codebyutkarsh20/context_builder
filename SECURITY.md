# Security Policy

## Reporting a vulnerability

If you find a security issue, **please do not open a public GitHub issue**. Email the maintainer privately:

- **Contact:** open a GitHub security advisory on this repository (Settings → Security → Advisories), or email the address listed in the repo's GitHub profile.
- **Response target:** acknowledgement within 72 hours, fix or mitigation within 14 days for high-severity issues.

Please include: a clear description, reproduction steps, affected version/commit, and the impact you observed.

## Threat model

This project runs an autonomous LLM agent that **edits files and executes shell commands** against repositories you point it at. Anyone running it should understand the trust boundary.

### What the agent can do by default

- **Read any file** under `USER_REPOS_HOST` (defaults to `$HOME` when running via Docker). The bind-mount gives the backend container read access to that entire directory tree.
- **Execute shell commands** in an isolated git worktree under the target repo. A denylist (`backend/agent/shell_safety.py`) blocks destructive operations such as `rm -rf`, `git push --force`, `git reset --hard` on dirty repos, and unbounded recursive deletes. The denylist is best-effort, **not** a sandbox — do not run the agent against repos you cannot afford to lose without backups.
- **Make outbound HTTP calls** to the Anthropic API (always) and to the Web (only when `ENABLE_WEB_TOOLS=1`).
- **Push branches and open PRs** when `GH_TOKEN` is set. The token is used as-is for `git push` and `gh pr create`.

### What the agent cannot do

- Modify files outside the per-job git worktree (path containment is enforced in `backend/agent/path_safety.py`).
- Execute commands on the host outside the container when running via Docker.
- Read or transmit your `.env` — secret-redaction patterns scrub `sk-ant-*`, `ghp_*`, `gho_*` and similar from prompts and logs (`backend/agent/llm.py`, `backend/agent/explore_tools.py`).

### Recommendations

- **Set `API_TOKEN` before exposing the backend on any non-loopback interface.** Without it, anyone who can reach port 8001 can submit jobs that run shell commands and edit files. Generate one with `openssl rand -hex 32` and set the same value as `VITE_API_TOKEN` for the frontend. `/health` stays open for orchestrators; everything under `/api/*` is gated.
- **Never run against a repo you have uncommitted work in** unless you've reviewed `react_guardrails.py`. The agent defends against `git reset --hard` on dirty trees but other footguns exist.
- **Tighten `USER_REPOS_HOST`** in `.env` to a specific code directory rather than `$HOME` if you store credentials, SSH keys, or personal documents there.
- **Change `NEO4J_PASSWORD`** before exposing port `7687`/`7474` on any network. The default in `docker-compose.yml` is `contextbuilder` and is intended for local-only use.
- **Use a scoped GitHub token** for `GH_TOKEN` — `repo` scope on the specific target repos, not a personal `repo`-everywhere PAT.
- **Don't forward your real Anthropic key to evaluators or shared CI.** API calls cost money and the agent makes many of them per bug.

### API authentication

The HTTP API ships with optional bearer-token auth (`backend/api/auth.py`). Behavior:

- `API_TOKEN` unset/empty → all endpoints open (back-compat for existing local installs).
- `API_TOKEN` set → every `/api/*` request must present `Authorization: Bearer <token>`, or `?token=<token>` for SSE clients (`EventSource` cannot set headers). `/health` and CORS preflights stay exempt.
- Comparison uses `hmac.compare_digest` to avoid timing leaks.

This is a single shared token, not per-user identity. For multi-tenant deployments, replace `BearerTokenMiddleware` with a real auth layer (OAuth, mTLS, etc.).

### Known dual-use behaviors

- `run_shell` can execute arbitrary commands inside the worktree. The denylist is documented in `backend/agent/shell_safety.py`. Treat any LLM output as untrusted before relaxing it.
- The verifier subagent reuses the parent's prompt cache. If you change verifier prompts, make sure cache keys are still independent of secret material.
- `ENABLE_WEB_TOOLS=1` lets the agent fetch arbitrary URLs. Disabled by default for a reason.

## Supported versions

The `main` branch is the only actively maintained line. Security fixes will not be backported to tagged releases.
