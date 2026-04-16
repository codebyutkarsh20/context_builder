"""
shell_tools.py — General shell access for the agent.

Why this exists
---------------
Industry-standard coding agents (Claude Code, OpenHands, SWE-Agent, Aider,
Devin) all give the model a general shell. Without it, the agent cannot
investigate or repair its environment when tests fail with infra issues
(`pytest exit 4: ImportError`, `ModuleNotFoundError`, missing deps).

Our specialized tools (run_tests, read_file, etc.) cover the happy path,
but `run_shell` is the universal escape hatch for the long tail of "I
need to figure out why my env is broken" cases.

Architecture
------------
Single-file tool with three layers:
1. `shell_safety.check_command_safety` — denylist for catastrophic patterns
2. `shell_safety.validate_working_dir` — path containment via safe_resolve
3. Subprocess execution — same pattern as `sandbox.run_tests` but with
   `shell=True` for pipes/&&/$VARS, head+tail truncation, exit-code-aware
   formatting.

Adapted from Claude Code's BashTool architecture
(/Downloads/src/tools/BashTool/BashTool.tsx + bashSecurity.ts), but
simplified for an unattended (no human-in-the-loop) eval agent.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent.shell_safety import check_command_safety, validate_working_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output formatting constants
# ---------------------------------------------------------------------------
MAX_OUTPUT_CHARS = 4000   # Reduced from 8K — discourages using shell for reading code
HEAD_CHARS = 2800         # Show first N chars
TAIL_CHARS = 1000         # Show last N chars
TIMEOUT_DEFAULT = 120     # Default subprocess timeout (pip install pandas ≈ 90s)
TIMEOUT_MAX = 300         # Hard ceiling (5 min)
TIMEOUT_MIN = 5           # Minimum (prevent foot-guns)


# ---------------------------------------------------------------------------
# Non-interactive environment overrides
# ---------------------------------------------------------------------------
# These force common tools into non-interactive mode so they NEVER prompt
# for user input. Critical for "no human in the loop" — without these, a
# command like `pip uninstall foo` or `git commit` would hang on stdin
# until the timeout kills it (wasting up to 5 min of budget).
NON_INTERACTIVE_ENV: dict[str, str] = {
    # Pip — never prompt (e.g. for index credentials), no version check spam
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_YES": "1",  # auto-confirm uninstall (defense in depth — safety also blocks)
    # Git — never prompt for credentials/passphrases
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/echo",  # any prompt → empty answer → fail fast
    # Apt — defense in depth (we don't have sudo so this rarely matters)
    "DEBIAN_FRONTEND": "noninteractive",
    # CI markers — most tools detect these and disable progress bars / prompts
    "CI": "true",
    "CONTINUOUS_INTEGRATION": "true",
    # Python — unbuffered output (we see stdout immediately on timeout)
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",  # no .pyc clutter in sandbox
    # Discourage interactive UIs / colorized output that breaks parsing
    "TERM": "dumb",
    "NO_COLOR": "1",
    # Pager — stop tools from piping into less/more (would hang)
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "MANPAGER": "cat",
    # Editor — if anything tries to open one, fall back to true (no-op)
    "EDITOR": "true",
    "VISUAL": "true",
}


def _find_venv_bin_dir(sandbox: Path | None) -> Path | None:
    """Return the venv `bin/` dir the eval scorer uses for this sandbox.

    Reuses `_find_brt_python` (already battle-tested for SWE-bench's per-bug
    venvs at `eval/repos/{repo}_{hash}_venv/`). The returned bin dir gets
    prepended to PATH so the agent's `pip install foo` lands in the venv
    that the scorer's `pytest` will see — without this, pip installs go to
    system python and the scorer's pytest still hits ImportError.
    """
    if not sandbox:
        return None
    try:
        # Late import: react_tools and shell_tools have a small dependency
        # cycle (react_tools → shell_tools for the tool, shell_tools →
        # react_tools for _tls + _find_brt_python).
        from agent.react_tools import _find_brt_python
        py_path = _find_brt_python(sandbox)
        bin_dir = Path(py_path).parent
        # Sanity check: must be a real bin/ dir with a python in it
        if bin_dir.name in ("bin", "Scripts") and (bin_dir / "python").exists():
            return bin_dir
    except Exception as e:
        logger.debug("venv bin detection failed: %s", e)
    return None


def _build_subprocess_env(sandbox: Path | None = None) -> dict[str, str]:
    """Merge OS env with non-interactive + venv-aware overrides.

    Layered:
      1. Inherit OS env (PATH, HOME, USER, etc.) so basic tools work
      2. Apply NON_INTERACTIVE_ENV (PIP_NO_INPUT, GIT_TERMINAL_PROMPT=0, ...)
      3. If a venv exists for this sandbox, prepend its bin/ to PATH and set
         VIRTUAL_ENV so `pip`/`python`/`pytest` resolve to the venv versions
    """
    env = os.environ.copy()
    env.update(NON_INTERACTIVE_ENV)

    # Venv injection: makes `pip install` land in the SAME venv the eval
    # scorer uses, so the agent's env repairs are visible to the scorer.
    venv_bin = _find_venv_bin_dir(sandbox)
    if venv_bin is not None:
        venv_root = venv_bin.parent
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
        env["VIRTUAL_ENV"] = str(venv_root)
        # Drop PYTHONHOME if set — it overrides venv discovery
        env.pop("PYTHONHOME", None)
        logger.debug("run_shell venv injected: VIRTUAL_ENV=%s", venv_root)
    return env


def _truncate_output(text: str, label: str) -> str:
    """Truncate `text` to MAX_OUTPUT_CHARS using head+tail strategy.

    Test output usually has the summary at the end (assertion errors, pass
    counts), so we keep both head AND tail, dropping the middle.
    """
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    dropped = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{head}\n\n... [{dropped:,} chars truncated from middle of {label}] ...\n\n{tail}"


def _format_shell_result(
    command: str,
    exit_code: int,
    duration_s: float,
    stdout: str,
    stderr: str,
    cwd: Path,
    warning: str = "",
) -> str:
    """Format subprocess output into a single string for the LLM.

    Structured so the agent can quickly see exit_code at the top, then
    skim STDOUT/STDERR. Cap each stream individually + total — protects
    context window from runaway commands.
    """
    stdout_t = _truncate_output(stdout.rstrip(), "stdout")
    stderr_t = _truncate_output(stderr.rstrip(), "stderr")

    parts = [
        f"[exit_code={exit_code}, duration={duration_s:.1f}s, cwd={cwd.name}]",
    ]
    if warning:
        parts.append(f"⚠️  {warning}")

    if stdout_t:
        parts.append(f"STDOUT:\n{stdout_t}")
    else:
        parts.append("STDOUT: (empty)")

    if stderr_t:
        parts.append(f"STDERR:\n{stderr_t}")

    full = "\n".join(parts)
    # Final guard: total cap (in case both streams are huge)
    if len(full) > MAX_OUTPUT_CHARS * 2:
        full = full[:MAX_OUTPUT_CHARS * 2] + f"\n\n... [{len(full) - MAX_OUTPUT_CHARS * 2:,} chars truncated overall]"
    return full


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

@tool
def run_shell(command: str, timeout: int = 120, working_dir: str = "") -> str:
    """Execute a shell command in the sandbox (or repo if no sandbox yet).

    USE THIS WHEN: you need to investigate or repair the test environment.
    The most common pattern is: `run_tests` returns "exit code 4 — bad
    invocation or missing deps", and you need to figure out *why*.

    Common diagnostic patterns:
      - `pip list | grep <pkg>` — check if a dependency is installed
      - `pip install <pkg>` — install a missing dependency
      - `python -c "import <module>"` — test if an import works
      - `which pytest` — verify the test runner is on PATH
      - `python --version` — check Python version
      - `ls tests/` — inspect the test directory structure
      - `cat conftest.py` — read a config file
      - `find . -name conftest.py -maxdepth 3` — locate config files

    DO NOT use this for:
      - editing code → use `string_replace` or `create_file`
      - reading code → use `read_file` or `read_function`
      - searching code → use `grep_repo`
      - running test suite → use `run_tests`

    BLOCKED commands: `rm -rf /`, `sudo`, `dd if=...`, fork bombs, raw
    device writes, `curl ... | sh`, system shutdown/reboot.

    Args:
        command: Shell command string (supports pipes, &&, $VARS).
        timeout: Seconds before the command is killed. 5-300, default 120
                 (sized for `pip install pandas`-class waits).
        working_dir: Optional cwd relative to the sandbox/repo root. Empty
                     means use the sandbox root (or repo root if no sandbox).

    Returns:
        A string with this structure:
            [exit_code=N, duration=X.Xs, cwd=...]
            ⚠️  warning if applicable
            STDOUT:
            <captured stdout>
            STDERR:
            <captured stderr>

        Output is capped at ~8KB total (head + tail truncation).
    """
    # Late import to avoid circular dependency: react_tools imports shell_tools
    from agent.react_tools import _tls

    # --- Safety: validate command ----------------------------------------
    allowed, reason = check_command_safety(command)
    if not allowed:
        return f"REJECTED: {reason}"
    warning = reason  # may be empty or a soft-warn note

    # --- Safety: clamp timeout -------------------------------------------
    timeout = max(TIMEOUT_MIN, min(TIMEOUT_MAX, int(timeout)))

    # --- Safety: resolve + contain working_dir ---------------------------
    sandbox_root = getattr(_tls, "sandbox_path", None)
    repo_root = getattr(_tls, "repo_path", None)
    cwd, err = validate_working_dir(working_dir, sandbox_root, repo_root)
    if cwd is None:
        return f"REJECTED: {err}"

    # --- Execute ----------------------------------------------------------
    # Defense-in-depth for "no human intervention":
    #   stdin=DEVNULL  → any tool reading from stdin gets EOF immediately,
    #                    can't hang waiting for "y/n" or password input
    #   env injection  → forces pip/git/apt into non-interactive mode
    #   start_new_session → puts subprocess in its own process group so
    #                       os.killpg() can cleanly kill it AND its children
    #                       on timeout (otherwise children leak as zombies)
    logger.info("run_shell (cwd=%s, timeout=%ds): %s", cwd.name, timeout, command[:200])
    started = time.monotonic()
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
            env=_build_subprocess_env(sandbox_root),
            start_new_session=True,  # → its own process group for clean kill
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.monotonic() - started
            return _format_shell_result(
                command, proc.returncode, duration,
                stdout or "", stderr or "", cwd, warning,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            # Kill the entire process group (subprocess + any children).
            # SIGTERM first (graceful), then SIGKILL after 2s if still alive.
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=2)
            except (ProcessLookupError, PermissionError) as e:
                logger.debug("killpg failed (already dead?): %s", e)
            # Drain whatever output was buffered before kill
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:
                stdout, stderr = "", ""
            return _format_shell_result(
                command, -1, duration, stdout or "",
                (stderr or "") + f"\n[TIMEOUT after {timeout}s — process group killed]",
                cwd, f"command timed out after {timeout}s (no human in the loop, no input given)",
            )
    except FileNotFoundError as e:
        return f"REJECTED: shell not available ({e})"
    except OSError as e:
        return f"ERROR: OSError ({e})"
    except Exception as e:
        logger.exception("run_shell unexpected error")
        # Best-effort cleanup
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return f"ERROR: unexpected ({type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# Tool collection (consumed by react_tools.py REACT_TOOLS)
# ---------------------------------------------------------------------------

SHELL_TOOLS = [run_shell]
