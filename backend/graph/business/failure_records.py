"""Mine FailureRecords from git history and persist to Neo4j.

A FailureRecord represents a past production bug, incident, or regression
captured from commit messages. It is linked to the Function (or File as
fallback) that was changed to fix the issue, via RESULTED_IN_CHANGE edges.

Feature flag: set ENABLE_FAILURE_RECORDS=true to enable mining.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

from analyzer.git_analyzer import _is_git_repo, _parse_log_name_only, _run_git

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Fix-commit classifiers (case-insensitive)
_FIX_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bfixes?\s+#\d+\b", re.IGNORECASE),
    re.compile(r"\bcloses?\s+#\d+\b", re.IGNORECASE),
    re.compile(r"\b(incident|hotfix|hot.fix|regression|bugfix|bug.fix)\b", re.IGNORECASE),
]

# Issue reference extractors
_ISSUE_REF_PATTERN = re.compile(r"#(\d+)")
_JIRA_PATTERN = re.compile(
    r"\b(" + os.environ.get("JIRA_PROJECT_PREFIX", "PROJ") + r"-\d+)\b"
)

# Severity hints from commit message keywords
_SEVERITY_PATTERNS = {
    "critical": re.compile(r"\b(p0|sev.?0|sev.?1|critical|production.down|outage)\b", re.IGNORECASE),
    "high": re.compile(r"\b(p1|sev.?2|urgent|hotfix|incident)\b", re.IGNORECASE),
}

# Maximum number of files changed before treating a commit as squash noise
_SQUASH_FILE_THRESHOLD = 20

# Maximum fix commits to process per run (performance cap)
_DEFAULT_MAX_COMMITS = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    return os.environ.get("ENABLE_FAILURE_RECORDS", "false").lower() in ("1", "true", "yes")


def _classify_fix_commit(message: str) -> tuple[bool, str | None, float]:
    """Return (is_fix, issue_ref, confidence) for a commit message."""
    for pat in _FIX_PATTERNS:
        if pat.search(message):
            issue_ref = None
            m = _ISSUE_REF_PATTERN.search(message)
            if m:
                issue_ref = f"#{m.group(1)}"
            jira_m = _JIRA_PATTERN.search(message)
            if jira_m:
                issue_ref = jira_m.group(1)
            return True, issue_ref, 1.0

    # JIRA ref with fix keyword — high confidence (e.g. "fixes ACME-789: ...")
    jira_m = _JIRA_PATTERN.search(message)
    if jira_m:
        _fix_keyword_pat = re.compile(r"\b(fixes?|closes?|resolves?)\b", re.IGNORECASE)
        if _fix_keyword_pat.search(message):
            return True, jira_m.group(1), 1.0

    # Keyword-only match (no issue ref) — lower confidence
    keyword_pat = re.compile(r"\b(bug|fix|patch|correct|revert)\b", re.IGNORECASE)
    if keyword_pat.search(message):
        return True, None, 0.5

    return False, None, 0.0


def _severity_hint(message: str) -> str:
    for severity, pat in _SEVERITY_PATTERNS.items():
        if pat.search(message):
            return severity
    return "unknown"


def _failure_record_id(commit_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"failure:{commit_hash}"))


# ---------------------------------------------------------------------------
# Tree-sitter function boundary matching (Python only, fallback for others)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=500)
def _parse_python_functions(content: str) -> list[tuple[str, int, int]]:
    """Return list of (name, start_line, end_line) for all top-level functions.

    Uses tree-sitter-python. Returns [] on parse failure so callers fall back
    to File-level linking.
    """
    try:
        from tree_sitter import Parser, Language
        import tree_sitter_python as tspython

        lang = Language(tspython.language())
        parser = Parser(lang)
        source_bytes = content.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)

        results: list[tuple[str, int, int]] = []
        for node in tree.root_node.children:
            # Handle decorated functions
            actual = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in ("function_definition", "async_function_definition"):
                        actual = child
                        break

            if actual.type in ("function_definition", "async_function_definition"):
                name_node = actual.child_by_field_name("name")
                if name_node:
                    name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                    start = actual.start_point[0] + 1  # 1-indexed
                    end = actual.end_point[0] + 1
                    results.append((name, start, end))

        return results
    except Exception as exc:
        logger.debug("Tree-sitter parse failed: %s", exc)
        return []


def _match_hunk_to_function(
    functions: list[tuple[str, int, int]],
    hunk_start: int,
    hunk_end: int,
) -> str | None:
    """Return the function name whose body contains the hunk, or None."""
    for name, fn_start, fn_end in functions:
        if fn_start <= hunk_start and hunk_end <= fn_end:
            return name
    return None


def _parse_diff_hunks(diff_output: str) -> list[tuple[int, int]]:
    """Extract (start_line, end_line) pairs from unified diff hunk headers.

    Parses lines like: @@ -old +new,count @@ ...
    Returns the new-file line ranges.
    """
    hunks: list[tuple[int, int]] = []
    hunk_pat = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff_output.splitlines():
        m = hunk_pat.match(line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            hunks.append((start, start + max(count - 1, 0)))
    return hunks


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def mine_failure_records(
    repo_path: Path,
    lookback_days: int = 90,
    max_commits: int = _DEFAULT_MAX_COMMITS,
) -> list[dict]:
    """Mine fix commits from git history and return FailureRecord dicts.

    Each dict has:
        id, commit_hash, message, date, issue_ref, severity_hint,
        confidence, files_changed, function_hits (list of {file, function})

    Does NOT write to Neo4j — callers decide whether to persist.
    Returns [] if ENABLE_FAILURE_RECORDS is not set or repo is not a git repo.
    """
    if not _is_enabled():
        logger.debug("FailureRecords mining disabled (ENABLE_FAILURE_RECORDS not set)")
        return []

    if not _is_git_repo(repo_path):
        logger.warning("mine_failure_records: %s is not a git repo", repo_path)
        return []

    # 1. Fetch all commits in window
    log_out, _, rc = _run_git(
        [
            "log",
            f"--since={lookback_days}.days.ago",
            "--pretty=format:COMMIT|%H|%as|%s",
            "--name-only",
        ],
        repo_path,
    )
    if rc != 0 or not log_out.strip():
        return []

    # 2. Parse into commit records
    all_commits = _parse_log_name_only(log_out)

    # 3. Classify fix commits, cap at max_commits
    fix_commits = []
    for commit in all_commits:
        is_fix, issue_ref, confidence = _classify_fix_commit(commit.get("message", ""))
        if is_fix:
            commit["issue_ref"] = issue_ref
            commit["confidence"] = confidence
            fix_commits.append(commit)

    fix_commits = fix_commits[:max_commits]
    logger.info("mine_failure_records: found %d fix commit(s) in last %d days", len(fix_commits), lookback_days)

    records: list[dict] = []

    for i, commit in enumerate(fix_commits):
        if i % 50 == 0 and i > 0:
            logger.info("FailureRecords: processing commit %d/%d", i, len(fix_commits))

        commit_hash = commit.get("hash", "")
        files_changed: list[str] = commit.get("files_changed", [])

        # Skip squash commits (noise)
        if len(files_changed) > _SQUASH_FILE_THRESHOLD:
            logger.debug("Skipping squash commit %s (%d files)", commit_hash[:8], len(files_changed))
            continue

        severity = _severity_hint(commit.get("message", ""))
        issue_ref = commit.get("issue_ref")
        confidence = commit.get("confidence", 1.0)

        function_hits: list[dict] = []

        for file_path in files_changed:
            if not file_path.endswith(".py"):
                # Non-Python: link to File only (no function boundary matching)
                function_hits.append({"file": file_path, "function": None})
                continue

            # Get file content at this commit
            content_out, _, content_rc = _run_git(
                ["show", f"{commit_hash}:{file_path}"],
                repo_path,
            )
            if content_rc != 0 or not content_out:
                # Deleted or inaccessible file — link to File
                function_hits.append({"file": file_path, "function": None})
                continue

            # Check for binary content
            if "\x00" in content_out[:1024]:
                continue

            # Get diff hunks for this file
            diff_out, _, _ = _run_git(
                ["diff", f"{commit_hash}^", commit_hash, "--", file_path],
                repo_path,
            )
            hunks = _parse_diff_hunks(diff_out) if diff_out else []

            if not hunks:
                function_hits.append({"file": file_path, "function": None})
                continue

            # Parse function boundaries (cached by content)
            functions = _parse_python_functions(content_out)

            matched_functions: set[str] = set()
            for hunk_start, hunk_end in hunks:
                fn_name = _match_hunk_to_function(functions, hunk_start, hunk_end)
                if fn_name:
                    matched_functions.add(fn_name)

            if matched_functions:
                for fn_name in matched_functions:
                    function_hits.append({"file": file_path, "function": fn_name})
            else:
                function_hits.append({"file": file_path, "function": None})

        if not function_hits and not files_changed:
            continue

        label = " (unverified)" if confidence < 1.0 else ""
        records.append(
            {
                "id": _failure_record_id(commit_hash),
                "commit_hash": commit_hash,
                "message": commit.get("message", "") + label,
                "date": commit.get("date", ""),
                "issue_ref": issue_ref,
                "severity_hint": severity,
                "confidence": confidence,
                "files_changed": files_changed,
                "function_hits": function_hits,
            }
        )

    logger.info("mine_failure_records: produced %d FailureRecord(s)", len(records))
    return records


# ---------------------------------------------------------------------------
# Neo4j persistence
# ---------------------------------------------------------------------------


def persist_failure_records(records: list[dict], repo_name: str) -> int:
    """Write FailureRecord nodes + RESULTED_IN_CHANGE edges to Neo4j.

    Idempotent via MERGE on FailureRecord.id.
    Returns number of records written (existing records are skipped).
    """
    from graph.neo4j_client import neo4j_client as _neo4j_client

    if not _neo4j_client.is_connected() or not records:
        return 0

    written = 0
    for record in records:
        try:
            _neo4j_client.run(
                "MERGE (fr:FailureRecord {id: $id}) "
                "ON CREATE SET "
                "  fr.commit_hash = $commit_hash, "
                "  fr.message = $message, "
                "  fr.date = $date, "
                "  fr.issue_ref = $issue_ref, "
                "  fr.severity_hint = $severity_hint, "
                "  fr.confidence = $confidence, "
                "  fr.repo = $repo",
                {
                    "id": record["id"],
                    "commit_hash": record["commit_hash"],
                    "message": record["message"],
                    "date": record["date"],
                    "issue_ref": record.get("issue_ref"),
                    "severity_hint": record.get("severity_hint", "unknown"),
                    "confidence": record.get("confidence", 1.0),
                    "repo": repo_name,
                },
            )

            for hit in record.get("function_hits", []):
                file_path = hit.get("file", "")
                fn_name = hit.get("function")

                if fn_name:
                    # Link to Function node (best case)
                    _neo4j_client.run(
                        "MATCH (fr:FailureRecord {id: $frid}) "
                        "MATCH (fn:Function {name: $name}) "
                        "WHERE fn.path ENDS WITH $file "
                        "MERGE (fr)-[:RESULTED_IN_CHANGE]->(fn)",
                        {"frid": record["id"], "name": fn_name, "file": file_path},
                    )
                else:
                    # Fallback: link to File node
                    _neo4j_client.run(
                        "MATCH (fr:FailureRecord {id: $frid}) "
                        "MATCH (f:File) WHERE f.path ENDS WITH $file "
                        "MERGE (fr)-[:RESULTED_IN_CHANGE]->(f)",
                        {"frid": record["id"], "file": file_path},
                    )

            written += 1
        except Exception as exc:
            logger.warning(
                "Failed to persist FailureRecord %s: %s", record.get("commit_hash", "?")[:8], exc
            )

    logger.info("persist_failure_records: wrote %d/%d record(s) to Neo4j", written, len(records))
    return written
