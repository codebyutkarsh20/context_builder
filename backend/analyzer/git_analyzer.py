"""
GitAnalyzer: Analyzes git history to surface change frequency, recent commits, and hotspot files.
Uses only Python stdlib (subprocess, pathlib).
"""

import logging
import os
import re
import subprocess
import shlex
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# Maximum number of recent commits to retrieve
RECENT_COMMITS_LIMIT = 50

# Number of top hotspot files to return
HOTSPOT_LIMIT = 10

# Timeout (seconds) for each git subprocess call
GIT_TIMEOUT = 30


def _run_git(args: list[str], cwd: Path) -> tuple[str, str, int]:
    """
    Run a git command and return (stdout, stderr, returncode).
    Returns ("", error_message, 1) on timeout or other OS-level errors.
    """
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"git command timed out after {GIT_TIMEOUT}s: {shlex.join(cmd)}", 1
    except FileNotFoundError:
        return "", "git executable not found", 1
    except OSError as exc:
        return "", str(exc), 1


def _is_git_repo(repo_path: Path) -> bool:
    """Return True if the path is inside a git repository."""
    _, _, rc = _run_git(["rev-parse", "--is-inside-work-tree"], repo_path)
    return rc == 0


def _parse_log_name_only(output: str) -> list[dict]:
    """
    Parse the output of:
        git log --pretty=format:"COMMIT|<hash>|<date>|<subject>" --name-only

    Returns a list of commit dicts:
        {hash, message, date, files_changed: [str, ...]}
    """
    commits: list[dict] = []
    current: dict | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("COMMIT|"):
            # Save previous commit if any
            if current is not None:
                commits.append(current)
            parts = line.split("|", 3)
            current = {
                "hash": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
                "message": parts[3] if len(parts) > 3 else "",
                "files_changed": [],
            }
        elif current is not None and line:
            # Non-empty lines after the header are file paths
            current["files_changed"].append(line)

    if current is not None:
        commits.append(current)

    return commits


class GitAnalyzer:
    """
    Analyzes the git history of a repository to surface change frequency,
    recent commits, and hotspot files.

    Parameters
    ----------
    repo_path : Path
        Path to the root of the repository (or any directory within it).
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> dict:
        """
        Perform a git history analysis.

        Returns
        -------
        dict
            A dictionary containing:
            - change_frequency: {file_path: commit_count} for every file
              that appears in the commit history
            - recent_changes: list of the last RECENT_COMMITS_LIMIT commits,
              each as {hash, message, date, files_changed}
            - hotspot_files: top HOTSPOT_LIMIT files by commit count,
              as a sorted list of {path, commits}

        If the directory is not a git repository, or has no history, or git
        is unavailable, all values are returned as empty collections.
        """
        empty = {
            "change_frequency": {},
            "recent_changes": [],
            "hotspot_files": [],
        }

        if not self.repo_path.exists():
            return empty

        if not _is_git_repo(self.repo_path):
            return empty

        recent_changes = self._get_recent_changes()
        change_frequency = self._build_change_frequency(recent_changes)
        hotspot_files = self._compute_hotspots(change_frequency)

        return {
            "change_frequency": change_frequency,
            "recent_changes": recent_changes,
            "hotspot_files": hotspot_files,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_recent_changes(self) -> list[dict]:
        """
        Retrieve the last RECENT_COMMITS_LIMIT commits with their changed files.

        Uses --name-only to list every file touched by each commit.
        """
        # The custom pretty format uses a sentinel prefix so we can reliably
        # detect commit header lines even when commit messages contain paths.
        fmt = "COMMIT|%H|%aI|%s"
        stdout, _, rc = _run_git(
            [
                "log",
                f"--pretty=format:{fmt}",
                "--name-only",
                f"-{RECENT_COMMITS_LIMIT}",
            ],
            self.repo_path,
        )
        if rc != 0 or not stdout.strip():
            return []

        return _parse_log_name_only(stdout)

    def _build_change_frequency(self, recent_changes: list[dict]) -> dict[str, int]:
        """
        Build a {file_path: commit_count} mapping from the full commit history.

        We query the full history separately (not limited to RECENT_COMMITS_LIMIT)
        so that change_frequency reflects all-time activity.
        """
        # Ask git for a flat list of all filenames from all commits
        stdout, _, rc = _run_git(
            [
                "log",
                "--pretty=format:",  # no commit line, just file names
                "--name-only",
            ],
            self.repo_path,
        )
        if rc != 0 or not stdout.strip():
            # Fall back to what we already have from recent_changes
            freq: dict[str, int] = defaultdict(int)
            for commit in recent_changes:
                for filepath in commit["files_changed"]:
                    if filepath:
                        freq[filepath] += 1
            return dict(freq)

        freq = defaultdict(int)
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if line:
                freq[line] += 1

        return dict(freq)

    def _compute_hotspots(self, change_frequency: dict[str, int]) -> list[dict]:
        """
        Return the top HOTSPOT_LIMIT files sorted by commit count descending.

        Each entry: {path: str, commits: int}
        """
        sorted_files = sorted(
            change_frequency.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"path": path, "commits": count}
            for path, count in sorted_files[:HOTSPOT_LIMIT]
        ]

    # ------------------------------------------------------------------
    # Decision context mining
    # ------------------------------------------------------------------

    _DECISION_KEYWORDS = re.compile(
        r"\b(because|per |required by|SLA|contract|compliance|security|"
        r"performance|workaround|hotfix|revert|breaking change|migration|"
        r"rollback|incident|outage|critical fix|legal|regulation|GDPR|"
        r"mandate|policy|agreement)\b",
        re.IGNORECASE,
    )

    _TYPE_MAP = {
        "sla": "sla", "contract": "sla", "agreement": "sla",
        "compliance": "requirement", "legal": "requirement",
        "regulation": "requirement", "gdpr": "requirement",
        "mandate": "requirement", "policy": "requirement",
        "required by": "requirement",
        "security": "security",
        "performance": "performance",
        "workaround": "workaround",
        "hotfix": "hotfix", "critical fix": "hotfix",
        "incident": "hotfix", "outage": "hotfix",
        "revert": "workaround", "rollback": "workaround",
        "breaking change": "requirement", "migration": "requirement",
    }

    def extract_decision_context(self, llm_enhance: bool = False) -> list[dict]:
        """
        Mine git history for commits that encode business decisions.

        Pass 1: Scans commit messages for decision-related keywords and classifies them.
        Pass 2 (if llm_enhance=True): Sends matched commits to Claude for deeper analysis.

        Returns list of:
            {commit_hash, date, message, decision_type, affected_files, llm_analysis?}
        """
        if not self.repo_path.exists() or not _is_git_repo(self.repo_path):
            return []

        # Get commits with full body (subject + body)
        fmt = "COMMIT|%H|%aI|%s %b"
        stdout, _, rc = _run_git(
            [
                "log",
                f"--pretty=format:{fmt}",
                "--name-only",
                f"-{RECENT_COMMITS_LIMIT * 2}",  # scan more commits for decisions
            ],
            self.repo_path,
        )
        if rc != 0 or not stdout.strip():
            return []

        commits = _parse_log_name_only(stdout)
        decisions: list[dict] = []

        for commit in commits:
            msg = commit.get("message", "")
            match = self._DECISION_KEYWORDS.search(msg)
            if not match:
                continue

            keyword = match.group(1).lower()
            decision_type = self._TYPE_MAP.get(keyword, "general")

            decisions.append({
                "commit_hash": commit.get("hash", "")[:8],
                "date": commit.get("date", ""),
                "message": msg[:300],  # cap message length
                "decision_type": decision_type,
                "affected_files": commit.get("files_changed", []),
            })

        # Pass 2: LLM-enhanced analysis
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if llm_enhance and api_key and decisions:
            self._llm_enhance_decisions(decisions, api_key)

        return decisions

    def _llm_enhance_decisions(self, decisions: list[dict], api_key: str) -> None:
        """Send decision commits to Claude for deeper business analysis."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except (ImportError, Exception) as e:
            logger.debug("Skipping LLM decision analysis: %s", e)
            return

        BATCH_SIZE = 5
        for i in range(0, min(len(decisions), 20), BATCH_SIZE):
            batch = decisions[i:i + BATCH_SIZE]
            prompt_lines = []
            for d in batch:
                # Get commit diff summary
                diff_out, _, _ = _run_git(
                    ["show", "--stat", "--format=", d["commit_hash"]],
                    self.repo_path,
                )
                diff_summary = (diff_out or "")[:500]
                prompt_lines.append(
                    f"- Commit {d['commit_hash']} ({d['date']}):\n"
                    f"  Message: {d['message'][:200]}\n"
                    f"  Changed files: {diff_summary}\n"
                )

            prompt = (
                "For each commit below, analyze: Does this encode a business decision? "
                "What constraint, SLA, policy, or architectural choice was made and WHY?\n"
                "Respond with one line per commit: <hash>: <one paragraph analysis>\n\n"
                + "\n".join(prompt_lines)
            )

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else ""
                for line in text.strip().splitlines():
                    if ":" not in line:
                        continue
                    hash_part, _, analysis = line.partition(":")
                    for d in batch:
                        if d["commit_hash"] in hash_part.strip():
                            d["llm_analysis"] = analysis.strip()[:500]
                            break
            except Exception as e:
                logger.warning("LLM git decision analysis failed: %s", e)
