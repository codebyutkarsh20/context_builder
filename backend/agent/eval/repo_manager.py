"""
repo_manager.py — Git clone, SHA checkout, caching, and cleanup for eval repos.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.environ.get("EVAL_REPOS_DIR", "eval/repos"))
CLONE_TIMEOUT = 300  # 5 minutes
CHECKOUT_TIMEOUT = 60


class RepoManager:
    """Thread-safe, cache-aware repository manager for eval.

    Clones repos using partial clone (--filter=blob:none) for speed,
    caches by repo_name + SHA prefix, and supports ground truth diff extraction.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create a per-repo lock."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def ensure_repo(self, bug: dict) -> Path:
        """Clone repo at buggy SHA if not cached. Returns repo path.

        Thread-safe: concurrent calls for the same repo will block on the lock.
        Cache-aware: if the repo already exists at the correct SHA, returns immediately.

        Parameters
        ----------
        bug : dict
            EvalBug with repo_url, repo_sha, ticket_id, and optionally repo_name.

        Returns
        -------
        Path
            Absolute path to the checked-out repo at the buggy SHA.

        Raises
        ------
        RuntimeError
            If clone or checkout fails.
        """
        repo_url = bug["repo_url"]
        repo_sha = bug["repo_sha"]
        ticket_id = bug["ticket_id"]
        repo_name = bug.get("repo_name", "") or ticket_id.lower()

        cache_key = f"{repo_name}_{repo_sha[:8]}"
        repo_dir = self.cache_dir / cache_key
        lock = self._get_lock(cache_key)

        with lock:
            if repo_dir.exists() and self._verify_sha(repo_dir, repo_sha):
                logger.info("Cache hit: %s at %s", repo_name, repo_sha[:8])
                self._reset_repo(repo_dir)
                return repo_dir.resolve()

            # Remove stale cache entry
            if repo_dir.exists():
                logger.info("Removing stale cache: %s", repo_dir)
                shutil.rmtree(repo_dir, ignore_errors=True)

            # Clone
            self._clone(repo_url, repo_sha, repo_dir)
            return repo_dir.resolve()

    def _clone(self, repo_url: str, sha: str, target_dir: Path) -> None:
        """Clone repo with partial clone and checkout specific SHA."""
        logger.info("Cloning %s at %s → %s", repo_url, sha[:8], target_dir)

        # Partial clone (blobless) — downloads tree objects but fetches blobs on demand.
        # This is faster than full clone for large repos.
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", repo_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT,
            )
        except subprocess.CalledProcessError:
            # Fallback: full clone if partial clone is unsupported
            logger.warning("Partial clone failed, falling back to full clone")
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            subprocess.run(
                ["git", "clone", "--quiet", repo_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT,
            )

        # Checkout buggy SHA
        try:
            subprocess.run(
                ["git", "checkout", "--quiet", sha],
                cwd=target_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=CHECKOUT_TIMEOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to checkout SHA {sha}: {e.stderr}"
            ) from e

        logger.info("Cloned %s at %s", repo_url, sha[:8])

    def _verify_sha(self, repo_dir: Path, expected_sha: str) -> bool:
        """Check if repo HEAD matches the expected SHA prefix."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            current = result.stdout.strip()
            return current.startswith(expected_sha[:8])
        except (subprocess.SubprocessError, OSError):
            return False

    def _reset_repo(self, repo_dir: Path) -> None:
        """Reset repo to clean state (discard any sandbox leftovers)."""
        try:
            subprocess.run(
                ["git", "checkout", "--quiet", "."],
                cwd=repo_dir,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "clean", "-fd", "--quiet"],
                cwd=repo_dir,
                capture_output=True,
                timeout=30,
            )
        except subprocess.SubprocessError:
            logger.warning("Failed to reset repo at %s", repo_dir)

    def get_ground_truth_diff(self, bug: dict) -> str:
        """Get the ground-truth patch by diffing repo_sha..fix_sha.

        Parameters
        ----------
        bug : dict
            EvalBug with repo_sha and fix_sha.

        Returns
        -------
        str
            Unified diff of the ground-truth fix, or empty string on failure.
        """
        repo_dir = self.ensure_repo(bug)
        fix_sha = bug.get("fix_sha", "")
        repo_sha = bug["repo_sha"]

        if not fix_sha or fix_sha == repo_sha:
            return ""

        try:
            # Fetch the fix commit if not present
            subprocess.run(
                ["git", "fetch", "--quiet", "origin", fix_sha],
                cwd=repo_dir,
                capture_output=True,
                timeout=60,
            )
            result = subprocess.run(
                ["git", "diff", f"{repo_sha}..{fix_sha}"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except subprocess.SubprocessError as e:
            logger.warning("Failed to get ground truth diff: %s", e)

        return ""

    def cleanup_stale(self, max_age_hours: int = 48) -> int:
        """Remove cached repos not accessed within max_age_hours.

        Returns
        -------
        int
            Number of repos cleaned up.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        cleaned = 0

        if not self.cache_dir.exists():
            return 0

        for entry in self.cache_dir.iterdir():
            if not entry.is_dir():
                continue
            # Use directory mtime as "last accessed"
            try:
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    logger.info("Cleaning stale repo cache: %s", entry.name)
                    shutil.rmtree(entry, ignore_errors=True)
                    cleaned += 1
            except OSError:
                continue

        if cleaned:
            logger.info("Cleaned %d stale repo cache entries", cleaned)
        return cleaned

    def list_cached(self) -> list[dict]:
        """List all cached repos with metadata."""
        entries = []
        if not self.cache_dir.exists():
            return entries

        for entry in sorted(self.cache_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                stat = entry.stat()
                size_mb = sum(
                    f.stat().st_size for f in entry.rglob("*") if f.is_file()
                ) / (1024 * 1024)
                entries.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size_mb": round(size_mb, 1),
                    "last_modified": stat.st_mtime,
                })
            except OSError:
                continue

        return entries
