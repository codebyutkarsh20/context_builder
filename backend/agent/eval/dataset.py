"""
dataset.py — Bug schema, dataset loading, validation, and SWE-bench curation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "ticket_id", "title", "description", "repo_url", "repo_sha",
    "expected_files", "expected_root_cause", "difficulty",
    "source", "priority",
}

OPTIONAL_FIELDS_CORE = {"fix_sha"}  # present in internal bugs, absent in SWE-bench

OPTIONAL_FIELDS = {
    "expected_patch_files", "category", "language", "repo_name",
    "swe_bench_id", "setup_commands", "test_command", "tags", "comments",
    "estimated_cost_usd", "local_repo_path", "fail_to_pass", "pass_to_pass",
    "nl_description",  # Business-language variant for natural-language eval
    # SWE-bench Lite specific fields
    "gold_patch", "test_patch", "version", "environment_setup_commit",
    "hints_text",
}

VALID_DIFFICULTIES = {"single-file", "multi-file"}
VALID_CATEGORIES = {
    "logic-error", "type-error", "missing-check", "regression",
    "api-misuse", "config", "data-handling", "bug-fix", "unknown",
}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}


class EvalBug(TypedDict, total=False):
    """Schema for a single evaluation bug entry."""
    # Required
    ticket_id: str
    title: str
    description: str
    repo_url: str
    repo_sha: str
    fix_sha: str
    expected_files: list[str]
    expected_root_cause: str
    difficulty: str
    source: str
    priority: str

    # Optional (new fields)
    expected_patch_files: list[str]
    category: str
    language: str
    repo_name: str
    swe_bench_id: str | None
    setup_commands: list[str]
    test_command: str | None
    tags: list[str]
    comments: list[str]
    estimated_cost_usd: float | None
    local_repo_path: str | None  # Absolute path to local repo (skips GitHub clone)
    nl_description: str | None   # Business-language description (no code terms) for NL eval


# ---------------------------------------------------------------------------
# Loading + Validation
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path | str) -> list[EvalBug]:
    """Load and validate an eval bug dataset from a JSON file.

    Parameters
    ----------
    path : Path or str
        Path to bugs.json file.

    Returns
    -------
    list[EvalBug]
        Validated list of bug entries.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any bug entry fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(raw).__name__}")

    bugs: list[EvalBug] = []
    errors: list[str] = []

    for i, entry in enumerate(raw):
        entry_errors = _validate_bug(entry, i)
        if entry_errors:
            errors.extend(entry_errors)
        else:
            # Fill defaults for optional fields
            bug = _normalize_bug(entry)
            bugs.append(bug)

    if errors:
        error_summary = "\n".join(errors[:20])
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n{error_summary}"
        )

    logger.info("Loaded %d eval bugs from %s", len(bugs), path)
    return bugs


def _validate_bug(entry: dict, index: int) -> list[str]:
    """Validate a single bug entry. Returns list of error strings."""
    errors: list[str] = []
    prefix = f"Bug [{index}]"

    if not isinstance(entry, dict):
        return [f"{prefix}: expected dict, got {type(entry).__name__}"]

    # Check required fields
    missing = REQUIRED_FIELDS - set(entry.keys())
    if missing:
        errors.append(f"{prefix} ({entry.get('ticket_id', '?')}): missing fields: {missing}")

    tid = entry.get("ticket_id", "")
    if tid:
        prefix = f"Bug [{index}] ({tid})"

    # Type checks
    if "expected_files" in entry and not isinstance(entry["expected_files"], list):
        errors.append(f"{prefix}: expected_files must be a list")

    if "difficulty" in entry and entry["difficulty"] not in VALID_DIFFICULTIES:
        errors.append(f"{prefix}: difficulty must be one of {VALID_DIFFICULTIES}, got '{entry['difficulty']}'")

    if "category" in entry and entry["category"] not in VALID_CATEGORIES:
        errors.append(f"{prefix}: category must be one of {VALID_CATEGORIES}, got '{entry['category']}'")

    if "priority" in entry and entry["priority"] not in VALID_PRIORITIES:
        errors.append(f"{prefix}: priority must be one of {VALID_PRIORITIES}, got '{entry['priority']}'")

    # repo_url format (skip validation for local repos that supply local_repo_path)
    repo_url = entry.get("repo_url", "")
    if repo_url and not entry.get("local_repo_path") and not repo_url.startswith("https://github.com/"):
        errors.append(f"{prefix}: repo_url must be a GitHub URL, got '{repo_url}'")

    # SHA format
    for sha_field in ("repo_sha", "fix_sha"):
        sha = entry.get(sha_field, "")
        if sha and not re.match(r"^[0-9a-f]{7,40}$", sha):
            errors.append(f"{prefix}: {sha_field} must be a hex SHA, got '{sha}'")

    return errors


def _normalize_bug(entry: dict) -> EvalBug:
    """Fill defaults for optional fields."""
    bug = dict(entry)

    # Derive repo_name from repo_url if not set
    if "repo_name" not in bug or not bug["repo_name"]:
        url = bug.get("repo_url", "")
        bug["repo_name"] = url.rstrip("/").split("/")[-1] if url else ""

    # Default optional fields
    bug.setdefault("expected_patch_files", bug.get("expected_files", []))
    bug.setdefault("category", "unknown")
    bug.setdefault("language", "python")
    bug.setdefault("swe_bench_id", None)
    bug.setdefault("setup_commands", [])
    bug.setdefault("test_command", None)
    bug.setdefault("tags", [])
    bug.setdefault("comments", [])
    bug.setdefault("estimated_cost_usd", None)

    return bug  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SWE-bench Curation
# ---------------------------------------------------------------------------

def curate_from_swe_bench(
    dataset_path: str | Path,
    output_path: str | Path,
    max_bugs: int = 25,
    allowed_repos: list[str] | None = None,
    difficulty_mix: dict[str, int] | None = None,
) -> list[EvalBug]:
    """Cherry-pick bugs from a SWE-bench-lite dataset and convert to our schema.

    Parameters
    ----------
    dataset_path : str or Path
        Path to downloaded SWE-bench-lite JSON (array of SWE-bench instances).
    output_path : str or Path
        Where to write the curated bugs.json.
    max_bugs : int
        Maximum number of bugs to include.
    allowed_repos : list[str] or None
        If set, only include bugs from these repos (e.g. ["pallets/flask"]).
    difficulty_mix : dict or None
        Target distribution, e.g. {"single-file": 16, "multi-file": 9}.

    Returns
    -------
    list[EvalBug]
        Curated and converted bugs.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"SWE-bench dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        swe_data = json.load(f)

    if difficulty_mix is None:
        difficulty_mix = {"single-file": 16, "multi-file": 9}

    # Convert SWE-bench entries
    candidates: list[EvalBug] = []
    for entry in swe_data:
        bug = _swe_bench_to_eval_bug(entry)
        if bug is None:
            continue
        if allowed_repos and _extract_repo_slug(entry) not in allowed_repos:
            continue
        candidates.append(bug)

    logger.info("Converted %d / %d SWE-bench entries", len(candidates), len(swe_data))

    # Select based on difficulty mix
    selected: list[EvalBug] = []
    for difficulty, count in difficulty_mix.items():
        pool = [b for b in candidates if b["difficulty"] == difficulty and b not in selected]
        selected.extend(pool[:count])

    # Fill remaining slots if mix didn't reach max_bugs
    remaining = max_bugs - len(selected)
    if remaining > 0:
        extras = [b for b in candidates if b not in selected]
        selected.extend(extras[:remaining])

    selected = selected[:max_bugs]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)

    logger.info("Curated %d bugs → %s", len(selected), output_path)
    return selected


def _swe_bench_to_eval_bug(entry: dict) -> EvalBug | None:
    """Convert a single SWE-bench instance to EvalBug format."""
    try:
        instance_id = entry["instance_id"]
        repo_slug = _extract_repo_slug(entry)
        patch = entry.get("patch", "")
        problem = entry.get("problem_statement", "")

        if not patch or not problem:
            return None

        # Extract file paths from unified diff
        patched_files = _extract_files_from_patch(patch)
        if not patched_files:
            return None

        difficulty = "single-file" if len(patched_files) == 1 else "multi-file"

        # Derive category heuristically from problem statement
        category = _guess_category(problem)

        # Build test_command from FAIL_TO_PASS test IDs (ground-truth verification)
        fail_to_pass = entry.get("FAIL_TO_PASS", "[]")
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except Exception:
                fail_to_pass = []
        test_command = None
        if fail_to_pass:
            # Cap at 5 tests to keep command short; -x stops on first failure
            test_ids = " ".join(fail_to_pass[:5])
            test_command = f"pytest {test_ids} -x --no-header -rN -q"

        return {
            "ticket_id": instance_id,
            "title": _extract_title(problem),
            "description": problem,
            "repo_url": f"https://github.com/{repo_slug}",
            "repo_sha": entry.get("base_commit", ""),
            "fix_sha": "",  # SWE-bench fix is in the patch field, not a SHA
            "expected_files": patched_files,
            "expected_patch_files": patched_files,
            "expected_root_cause": _extract_keywords(problem),
            "difficulty": difficulty,
            "source": "swe-bench-lite",
            "priority": "medium",
            "category": category,
            "language": "python",
            "repo_name": repo_slug.split("/")[-1],
            "swe_bench_id": instance_id,
            "setup_commands": ["pip install -e ."],
            "test_command": test_command,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": entry.get("PASS_TO_PASS", []),
            "tags": [],
            "comments": [],
            "estimated_cost_usd": None,
        }
    except (KeyError, TypeError) as e:
        logger.debug("Skipping SWE-bench entry: %s", e)
        return None


def _extract_repo_slug(entry: dict) -> str:
    """Extract 'owner/repo' from SWE-bench instance_id like 'django__django-15695'."""
    instance_id = entry.get("instance_id", "")
    repo = entry.get("repo", "")
    if repo:
        return repo
    # Fallback: parse from instance_id (format: owner__repo-number)
    parts = instance_id.split("-", 1)
    if parts:
        slug = parts[0].replace("__", "/")
        return slug
    return ""


def _extract_files_from_patch(patch: str) -> list[str]:
    """Parse file paths from a unified diff."""
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            # diff --git a/path/to/file b/path/to/file
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                if path not in files:
                    files.append(path)
    return files


def _extract_title(problem: str) -> str:
    """Extract first line of problem statement as title."""
    first_line = problem.strip().split("\n")[0]
    return first_line[:120]


def _extract_keywords(problem: str) -> str:
    """Extract likely root-cause keywords from problem statement."""
    # Take first 2 sentences, extract non-trivial words
    sentences = problem.strip().split(".")[:2]
    text = ". ".join(sentences).lower()
    # Remove common stop words
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "with", "and", "or", "but", "not", "this", "that", "it",
            "be", "has", "have", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "i", "we", "you", "they", "my", "your",
            "when", "if", "then", "from", "by", "as", "so"}
    words = re.findall(r"[a-z_][a-z0-9_]+", text)
    keywords = [w for w in words if w not in stop and len(w) > 2]
    return " ".join(keywords[:10])


def _guess_category(problem: str) -> str:
    """Heuristically guess bug category from problem statement."""
    text = problem.lower()
    if any(w in text for w in ("typeerror", "attributeerror", "type error", "wrong type")):
        return "type-error"
    if any(w in text for w in ("regression", "used to work", "broke", "no longer")):
        return "regression"
    if any(w in text for w in ("missing check", "validation", "not validated", "missing validation")):
        return "missing-check"
    if any(w in text for w in ("config", "setting", "environment", "configuration")):
        return "config"
    if any(w in text for w in ("api", "endpoint", "request", "response", "http")):
        return "api-misuse"
    if any(w in text for w in ("data", "parse", "serialize", "format", "encoding")):
        return "data-handling"
    return "logic-error"
