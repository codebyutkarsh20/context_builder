"""
swebench_import.py — Import SWE-bench Lite dataset into our eval format.

Downloads from HuggingFace, converts to our bugs.json schema, and maps
repo names to clone URLs + repo_name identifiers.

Usage:
    python -m agent.eval.swebench_import [--output eval/swebench_lite.json] [--limit 50]
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping from SWE-bench repo slug → clone URL
_REPO_URLS = {
    "django/django": "https://github.com/django/django",
    "sympy/sympy": "https://github.com/sympy/sympy",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx",
    "astropy/astropy": "https://github.com/astropy/astropy",
    "psf/requests": "https://github.com/psf/requests",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint",
    "pydata/xarray": "https://github.com/pydata/xarray",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn",
    "pallets/flask": "https://github.com/pallets/flask",
}

# Difficulty heuristic based on number of files in the gold patch
def _classify_difficulty(patch: str) -> str:
    """Classify bug difficulty from the gold patch."""
    files = set(re.findall(r"^diff --git a/(\S+)", patch, re.MULTILINE))
    if len(files) <= 1:
        return "single-file"
    elif len(files) <= 3:
        return "multi-file"
    return "complex"


def _extract_patch_files(patch: str) -> list[str]:
    """Extract modified file paths from a unified diff."""
    return sorted(set(re.findall(r"^diff --git a/(\S+)", patch, re.MULTILINE)))


def _parse_test_ids(raw: str) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS which is JSON-encoded list as string."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _make_ticket_id(instance_id: str) -> str:
    """Convert SWE-bench instance_id to our ticket_id format.

    'django__django-16315' → 'DJANGO-16315'
    'scikit-learn__scikit-learn-25570' → 'SKLEARN-25570'
    """
    # instance_id format: <owner>__<repo>-<number>
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2:
        prefix = parts[0].split("__")[-1].upper().replace("-", "")
        # Shorten common ones
        short = {
            "SCIKITLEARN": "SKLEARN",
            "MATPLOTLIB": "MPL",
        }
        prefix = short.get(prefix, prefix)
        return f"{prefix}-{parts[1]}"
    return instance_id.upper().replace("__", "-")


def _make_repo_name(instance_id: str) -> str:
    """Convert instance_id to our repo cache name.

    'django__django-16315' → 'django' (just the project name)
    """
    # Take the repo part: django__django-16315 → django
    return instance_id.split("__")[0].split("/")[-1].lower()


def convert_dataset(limit: int = 0, difficulty_filter: str = "") -> list[dict]:
    """Download SWE-bench Lite and convert to our bugs.json format.

    Args:
        limit: Max bugs to include (0 = all 300)
        difficulty_filter: "single-file", "multi-file", "complex", or "" for all

    Returns:
        List of bug dicts in our format.
    """
    from datasets import load_dataset

    logger.info("Loading SWE-bench Lite from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    bugs = []
    for item in ds:
        patch = item.get("patch", "")
        difficulty = _classify_difficulty(patch)

        if difficulty_filter and difficulty != difficulty_filter:
            continue

        fail_to_pass = _parse_test_ids(item.get("FAIL_TO_PASS", ""))
        pass_to_pass = _parse_test_ids(item.get("PASS_TO_PASS", ""))
        patch_files = _extract_patch_files(patch)
        repo_slug = item["repo"]

        bug = {
            "ticket_id": _make_ticket_id(item["instance_id"]),
            "swe_bench_id": item["instance_id"],
            "repo_name": _make_repo_name(item["instance_id"]),
            "title": item["problem_statement"].split("\n")[0][:120],
            "description": item["problem_statement"],
            "repo_url": _REPO_URLS.get(repo_slug, f"https://github.com/{repo_slug}"),
            "repo_sha": item["base_commit"],
            "fix_sha": "",  # Not available in SWE-bench Lite directly
            "expected_files": patch_files,
            "expected_patch_files": patch_files,
            "expected_root_cause": "",
            "difficulty": difficulty,
            "source": "swe-bench-lite",
            "priority": "high" if difficulty == "complex" else "medium",
            "category": "bug-fix",
            "language": "python",
            "tags": [repo_slug.split("/")[1], difficulty],
            # SWE-bench specific fields
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "gold_patch": patch,
            "test_patch": item.get("test_patch", ""),
            "version": item.get("version", ""),
            "environment_setup_commit": item.get("environment_setup_commit", ""),
            "hints_text": item.get("hints_text", ""),
        }
        bugs.append(bug)

        if limit and len(bugs) >= limit:
            break

    logger.info(
        "Converted %d bugs (%d single-file, %d multi-file, %d complex)",
        len(bugs),
        sum(1 for b in bugs if b["difficulty"] == "single-file"),
        sum(1 for b in bugs if b["difficulty"] == "multi-file"),
        sum(1 for b in bugs if b["difficulty"] == "complex"),
    )
    return bugs


def print_stats(bugs: list[dict]) -> None:
    """Print dataset statistics."""
    from collections import Counter
    repos = Counter(b["repo_url"].split("/")[-1] for b in bugs)
    diffs = Counter(b["difficulty"] for b in bugs)

    print(f"\nTotal: {len(bugs)} bugs")
    print(f"\nBy difficulty:")
    for d, c in diffs.most_common():
        print(f"  {d}: {c}")
    print(f"\nBy repo:")
    for r, c in repos.most_common():
        print(f"  {r}: {c}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Import SWE-bench Lite")
    parser.add_argument("--output", default="eval/swebench_lite.json")
    parser.add_argument("--limit", type=int, default=0, help="Max bugs (0=all)")
    parser.add_argument("--difficulty", default="", choices=["", "single-file", "multi-file", "complex"])
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    bugs = convert_dataset(limit=args.limit, difficulty_filter=args.difficulty)

    if args.stats_only:
        print_stats(bugs)
        sys.exit(0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bugs, indent=2))
    print(f"Wrote {len(bugs)} bugs to {out}")
    print_stats(bugs)
