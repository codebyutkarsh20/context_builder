"""
repo_manager.py — Git clone, SHA checkout, caching, and cleanup for eval repos.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
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

        Supports local repos via ``local_repo_path`` field in the bug dict.
        When set, skips cloning and returns the local path directly (after
        checking out the requested SHA). Useful for eval against the
        context_builder repo itself.

        Parameters
        ----------
        bug : dict
            EvalBug with repo_url, repo_sha, ticket_id, and optionally:
            - repo_name: short name for cache key
            - local_repo_path: absolute path to an already-cloned local repo

        Returns
        -------
        Path
            Absolute path to the checked-out repo at the buggy SHA.

        Raises
        ------
        RuntimeError
            If clone or checkout fails.
        """
        # Local path shortcut — clone from local dir into cache to avoid touching
        # the working directory. git clone file:///path is fast (hardlinks on same FS).
        local_path = bug.get("local_repo_path")
        if local_path:
            local_dir = Path(local_path).resolve()
            if not local_dir.exists():
                raise RuntimeError(f"local_repo_path does not exist: {local_path}")
            repo_sha = bug["repo_sha"]
            ticket_id = bug["ticket_id"]
            repo_name = bug.get("repo_name", "") or ticket_id.lower()
            cache_key = f"{repo_name}_local_{repo_sha[:8]}"
            repo_dir = self.cache_dir / cache_key
            lock = self._get_lock(cache_key)

            with lock:
                if repo_dir.exists() and self._verify_sha(repo_dir, repo_sha):
                    logger.info("Local cache hit: %s at %s", repo_name, repo_sha[:8])
                    self._reset_repo(repo_dir)
                    return repo_dir.resolve()

                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)

                logger.info("Cloning local repo %s → %s at %s", local_dir, repo_dir, repo_sha[:8])
                # Clone from local path (fast — uses hardlinks on same filesystem)
                subprocess.run(
                    ["git", "clone", "--quiet", str(local_dir), str(repo_dir)],
                    check=True, capture_output=True, text=True, timeout=CLONE_TIMEOUT,
                )
                subprocess.run(
                    ["git", "checkout", "--quiet", repo_sha],
                    cwd=repo_dir, check=True, capture_output=True,
                    text=True, timeout=CHECKOUT_TIMEOUT,
                )
                return repo_dir.resolve()

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

    @staticmethod
    def _find_compat_python() -> str:
        """Return path to Python 3.10.x if available via pyenv, else sys.executable.

        Most SWE-bench instances target Python 3.8-3.10. Running them on Python 3.12+
        causes failures (greenlet build errors, werkzeug API removals, py==1.10.0
        incompatibility with Python 3.13). Using 3.10 avoids these issues.
        """
        pyenv_root = Path.home() / ".pyenv" / "versions"
        if pyenv_root.exists():
            for candidate in sorted(pyenv_root.iterdir(), reverse=True):
                if candidate.name.startswith("3.10."):
                    python = candidate / "bin" / "python"
                    if python.exists():
                        logger.info("Using pyenv Python %s for SWE-bench venv", candidate.name)
                        return str(python)
        logger.warning("Python 3.10.x not found in pyenv — falling back to %s", sys.executable)
        return sys.executable

    def setup_venv(self, repo_dir: Path, bug: dict) -> Path | None:
        """Create an isolated virtualenv for a cloned repo and install test dependencies.

        Creates the venv at ``{repo_dir.parent}/{repo_dir.name}_venv/``, tries common
        test-extra install specs (``.[dev]``, ``.[testing]``, ``.[tests]``, ``.[test]``)
        before falling back to plain ``pip install -e .``.  Always ensures pytest is
        installed.  Writes ``{repo_dir}/.agent_config.json`` so the sandbox picks up the
        venv's pytest instead of ``sys.executable``.

        Cache-aware: skips reinstallation if ``{venv_dir}/.install_ok`` exists and the
        venv python is still present.

        Parameters
        ----------
        repo_dir : Path
            Path to the cloned repo (must contain setup.py / pyproject.toml).
        bug : dict
            EvalBug — used to pull ``setup_commands`` and ``test_command`` overrides.

        Returns
        -------
        Path or None
            Path to the venv directory, or None if setup failed non-fatally.
        """
        # Always use absolute paths — subprocess cwd changes break relative paths
        repo_dir = repo_dir.resolve()
        venv_dir = repo_dir.parent / f"{repo_dir.name}_venv"
        venv_python = venv_dir / "bin" / "python"
        venv_pytest = venv_dir / "bin" / "pytest"
        stamp = venv_dir / ".install_ok"

        # Cache hit — skip if venv exists, stamp is present, and imports work
        if stamp.exists() and venv_python.exists():
            # Verify the package still imports (catches stale werkzeug etc.)
            pkg_name = repo_dir.name.split("_")[0]
            import_ok = subprocess.run(
                [str(venv_python), "-c", f"import {pkg_name}"],
                capture_output=True, text=True, timeout=15,
            ).returncode == 0
            if import_ok:
                # For Django repos, also ensure pytest-django is installed in the cached venv
                # (old cached venvs may not have it)
                if self._is_django_repo(repo_dir):
                    has_pytest_django = subprocess.run(
                        [str(venv_python), "-c", "import pytest_django"],
                        capture_output=True, timeout=10,
                    ).returncode == 0
                    if not has_pytest_django:
                        logger.info("Cached Django venv missing pytest-django — installing")
                        pip = str(venv_dir / "bin" / "pip")
                        subprocess.run(
                            [pip, "install", "--quiet", "pytest-django"],
                            capture_output=True, timeout=60,
                        )
                        self._write_django_pytest_ini(repo_dir)
                logger.info("Venv cache hit: %s", venv_dir)
                self._write_agent_config(repo_dir, venv_pytest, bug)
                return venv_dir
            # Import broken — remove stamp and rebuild
            logger.warning("Venv cache stale (import %s failed) — rebuilding", pkg_name)
            stamp.unlink(missing_ok=True)

        logger.info("Creating venv for %s at %s", repo_dir.name, venv_dir)

        # Prefer Python 3.10 for SWE-bench repos — most SWE-bench instances target
        # Python 3.8-3.10 and have deps that are incompatible with Python 3.12+
        # (e.g., old greenlet, py==1.10.0, werkzeug api removals).
        python_bin = self._find_compat_python()

        # Create venv
        try:
            subprocess.run(
                [python_bin, "-m", "venv", str(venv_dir)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to create venv for %s: %s", repo_dir.name, e.stderr[:200])
            return None

        pip = str(venv_dir / "bin" / "pip")

        # Upgrade pip silently
        subprocess.run(
            [pip, "install", "--quiet", "--upgrade", "pip"],
            capture_output=True, timeout=60,
        )

        # Step 1: Install the package in editable mode first
        # Try .in (unpinned) files before .txt (pinned) to avoid old-compiler failures
        # on new Python (e.g., old greenlet pinned in tests.txt won't build on 3.13)
        install_specs = [".[dev]", ".[testing]", ".[tests]", ".[test]", "."]
        for spec in install_specs:
            result = subprocess.run(
                [pip, "install", "--quiet", "-e", spec],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info("Installed %s with spec '%s'", repo_dir.name, spec)
                break
            logger.debug("Install spec '%s' failed: %s", spec, result.stderr[:80])

        # Step 2: Werkzeug compat patch.
        # Flask 2.0-2.2 (and similar 2021-era packages) require werkzeug 2.0.x:
        # - werkzeug 2.1+ removed as_tuple from EnvironBuilder (breaks flask testing)
        # - werkzeug 3.0+ removed url_quote (breaks flask import)
        # pip with no upper bound installs the latest (3.x), so we downgrade after install.
        pkg_name = repo_dir.name.split("_")[0]  # e.g., "flask" from "flask_d8c37f43"
        import_check = subprocess.run(
            [str(venv_dir / "bin" / "python"), "-c", f"import {pkg_name}"],
            capture_output=True, text=True, timeout=15,
        )
        if import_check.returncode != 0 and "werkzeug" in import_check.stderr:
            # Old package uses werkzeug API removed in 3.0+ — pin to 2.x
            logger.info("werkzeug incompatibility in %s — pinning to <3.0", pkg_name)
            subprocess.run(
                [pip, "install", "--quiet", "werkzeug>=2.2.2,<3.0"],
                capture_output=True, timeout=60,
            )

        # Step 3: Run any custom setup_commands from the bug spec
        for cmd in bug.get("setup_commands", []):
            if not cmd.strip():
                continue
            logger.info("Running setup_command: %s", cmd)
            try:
                subprocess.run(
                    cmd, shell=True, cwd=str(repo_dir),
                    env={**os.environ, "PATH": f"{venv_dir / 'bin'}:{os.environ.get('PATH', '')}"},
                    capture_output=True, text=True, timeout=300,
                )
            except Exception as e:
                logger.warning("setup_command failed (continuing): %s", e)

        # Always ensure pytest is available
        subprocess.run(
            [pip, "install", "--quiet", "pytest", "pytest-timeout"],
            capture_output=True, timeout=120,
        )

        # Django-specific: install pytest-django and write pytest.ini with settings.
        # Django tests require DJANGO_SETTINGS_MODULE; plain pytest without pytest-django
        # will fail at setUpClass because django.setup() is never called.
        if self._is_django_repo(repo_dir):
            logger.info("Detected Django repo — installing pytest-django + writing pytest.ini")
            subprocess.run(
                [pip, "install", "--quiet", "pytest-django"],
                capture_output=True, timeout=60,
            )
            self._write_django_pytest_ini(repo_dir)

        # Write stamp
        stamp.write_text("ok")

        self._write_agent_config(repo_dir, venv_pytest, bug)
        logger.info("Venv ready: %s", venv_dir)
        return venv_dir

    @staticmethod
    def _is_django_repo(repo_dir: Path) -> bool:
        """Return True if this looks like the Django source repo."""
        return (
            (repo_dir / "django" / "__init__.py").exists()
            and (repo_dir / "tests").is_dir()
        )

    @staticmethod
    def _extract_django_test_apps(fail_to_pass: list[str]) -> list[str]:
        """Extract Django internal test app names from FAIL_TO_PASS test IDs.

        SWE-bench test IDs come in two formats:
          unittest: "test_name (app.module.ClassName.test_name)"
          pytest:   "app/module.py::Class::test_name"

        Returns a deduplicated list of top-level module names (e.g. "auth_tests",
        "queries") that need to be in INSTALLED_APPS for models to resolve.

        Skips anything that starts with "django." (those are built-in apps already
        in the standard ALWAYS_INSTALLED_APPS list).
        """
        apps: list[str] = []
        for test_id in fail_to_pass:
            top = ""
            if "(" in test_id and ")" in test_id:
                # e.g. "test_x (auth_tests.test_basic.TestGetUser)"
                inner = test_id[test_id.index("(") + 1 : test_id.index(")")]
                top = inner.split(".")[0]
            elif "::" in test_id:
                # e.g. "auth_tests/test_basic.py::Class::test_method"
                top = test_id.split("::")[0].split("/")[0].removesuffix(".py")
            if top and not top.startswith("django"):
                apps.append(top)
        seen: set[str] = set()
        return [a for a in apps if not (a in seen or seen.add(a))]  # type: ignore[func-returns-value]

    @staticmethod
    def _django_test_app_has_models(app_dir: Path) -> bool:
        """Return True if a Django internal test app directory defines models.

        Apps that only contain SimpleTestCase/functional tests don't need to be
        in INSTALLED_APPS; only apps that define ORM models do.
        """
        if not app_dir.is_dir():
            return False
        return (app_dir / "models.py").exists() or (app_dir / "models").is_dir()

    @staticmethod
    def _find_site_packages(venv_dir: Path) -> Path | None:
        """Return the site-packages directory inside a venv, or None if not found."""
        lib_dir = venv_dir / "lib"
        if not lib_dir.is_dir():
            return None
        for entry in lib_dir.iterdir():
            if entry.name.startswith("python") and (entry / "site-packages").is_dir():
                return entry / "site-packages"
        return None

    @staticmethod
    def _write_full_django_settings(
        site_packages: Path, module_name: str, extra_apps: list[str] | None = None
    ) -> None:
        """Write a complete Django settings module to site-packages.

        Used when the repo's test_sqlite.py lacks INSTALLED_APPS (common in
        SWE-bench Django instances).  Writing to site-packages means the module
        is importable by the venv's pytest regardless of the worktree path.

        Parameters
        ----------
        extra_apps : list of str, optional
            Django internal test app names (e.g. ["auth_tests", "queries"]) to
            append to INSTALLED_APPS so their models resolve correctly.
        """
        extra_apps_list = extra_apps or []
        extra_app_lines = "".join(f"    '{app}',\n" for app in extra_apps_list)
        # Disable migrations for core Django apps AND internal test apps.
        # Mirrors what Django's runtests.py does (see setup_collect_tests).
        # Without this, auth_tests.UserProxy (which inherits auth.User) raises
        # InvalidBasesError because the migration state can't resolve the base class
        # before auth migrations have been applied.
        # Setting these to None makes Django create tables directly via CREATE TABLE
        # instead of applying migrations, sidestepping the state resolution issue.
        all_no_migrate = ["auth", "contenttypes", "sessions"] + extra_apps_list
        migration_module_lines = "".join(
            f"    '{app}': None,\n" for app in all_no_migrate
        )
        migration_modules_block = (
            "MIGRATION_MODULES = {\n"
            f"{migration_module_lines}"
            "}\n"
        )
        content = (
            "# Auto-generated by SWE-bench eval infrastructure\n"
            "# Provides a complete Django settings module for repos whose\n"
            "# test_sqlite.py is minimal and lacks INSTALLED_APPS.\n"
            "DATABASES = {\n"
            "    'default': {\n"
            "        'ENGINE': 'django.db.backends.sqlite3',\n"
            "        'NAME': ':memory:',\n"
            "    },\n"
            "    'other': {\n"
            "        'ENGINE': 'django.db.backends.sqlite3',\n"
            "        'NAME': ':memory:',\n"
            "    },\n"
            "}\n"
            "USE_TZ = False\n"
            "DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'\n"
            "SECRET_KEY = 'swe-bench-test-secret-key'\n"
            "PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']\n"
            "INSTALLED_APPS = [\n"
            "    'django.contrib.auth',\n"
            "    'django.contrib.contenttypes',\n"
            "    'django.contrib.sessions',\n"
            "    'django.contrib.messages',\n"
            "    'django.contrib.staticfiles',\n"
            "    'django.contrib.admin',\n"
            "    'django.contrib.sites',\n"
            f"{extra_app_lines}"
            "]\n"
            f"{migration_modules_block}"
        )
        target = site_packages / f"{module_name}.py"
        target.write_text(content)
        logger.debug("Wrote full Django settings to %s", target)

    @staticmethod
    def _write_django_pytest_ini(repo_dir: Path) -> None:
        """Write pytest.ini that configures pytest-django with SQLite settings.

        Django's `tests/test_sqlite.py` (or `tests/test_settings.py`) is the
        standard settings module used by SWE-bench.  pytest-django reads
        `DJANGO_SETTINGS_MODULE` from pytest.ini so no shell env var is needed.
        """
        # Prefer test_sqlite.py (Django 3.x+); fallback to test_settings.py
        tests_dir = repo_dir / "tests"
        if (tests_dir / "test_sqlite.py").exists():
            django_settings = "tests.test_sqlite"
        elif (tests_dir / "test_settings.py").exists():
            django_settings = "tests.test_settings"
        else:
            # Older Django — generate a minimal settings file
            django_settings = "tests.test_sqlite"
            minimal = (
                "DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', "
                "'NAME': ':memory:'}}\n"
                "USE_TZ = True\n"
                "DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'\n"
                "SECRET_KEY = 'swe-bench-test-secret-key'\n"
                "INSTALLED_APPS = [\n"
                "    'django.contrib.auth',\n"
                "    'django.contrib.contenttypes',\n"
                "    'django.contrib.sessions',\n"
                "    'django.contrib.messages',\n"
                "    'django.contrib.sites',\n"
                "    'django.contrib.admin',\n"
                "]\n"
            )
            (tests_dir / "test_sqlite.py").write_text(minimal)
            logger.info("Wrote minimal tests/test_sqlite.py for Django pytest-django")

        pytest_ini_path = repo_dir / "pytest.ini"
        if pytest_ini_path.exists():
            # Don't overwrite existing pytest.ini — just ensure DJANGO_SETTINGS_MODULE
            existing = pytest_ini_path.read_text()
            if "DJANGO_SETTINGS_MODULE" not in existing:
                pytest_ini_path.write_text(
                    existing.rstrip() + f"\nDJANGO_SETTINGS_MODULE = {django_settings}\n"
                )
        else:
            pytest_ini_path.write_text(
                "[pytest]\n"
                f"DJANGO_SETTINGS_MODULE = {django_settings}\n"
                "addopts = -x --timeout=60 -q\n"
            )
        logger.info("Wrote pytest.ini: DJANGO_SETTINGS_MODULE = %s", django_settings)

    def _write_agent_config(self, repo_dir: Path, venv_pytest: Path, bug: dict) -> None:
        """Write .agent_config.json so the sandbox uses the venv pytest."""
        pytest_path = str(venv_pytest)

        # Build test_command: prefer bug's test_command, else use venv pytest with default args
        test_cmd_override = bug.get("test_command")
        if test_cmd_override:
            # Replace bare 'pytest' with full path if the command starts with it
            if test_cmd_override.startswith("pytest "):
                test_command = pytest_path + test_cmd_override[len("pytest"):]
            else:
                test_command = test_cmd_override
        else:
            test_command = f"{pytest_path} -x --timeout=60 -q"

        config: dict = {
            "test_command": test_command,
            "pytest_path": pytest_path,
        }

        # Django repos need DJANGO_SETTINGS_MODULE in the test environment.
        # The sandbox is a git worktree and pytest.ini from the original repo dir
        # isn't automatically present there, so we inject the env var via config.
        if self._is_django_repo(repo_dir):
            tests_dir = repo_dir / "tests"
            if (tests_dir / "test_sqlite.py").exists():
                django_settings = "tests.test_sqlite"
                settings_text = (tests_dir / "test_sqlite.py").read_text()
            elif (tests_dir / "test_settings.py").exists():
                django_settings = "tests.test_settings"
                settings_text = (tests_dir / "test_settings.py").read_text()
            else:
                django_settings = "tests.test_sqlite"
                settings_text = ""

            # If the detected settings file is missing INSTALLED_APPS (as many
            # minimal SWE-bench test_sqlite.py files are), write a complete settings
            # module to the venv's site-packages so auth/contenttypes tests pass.
            # site-packages is always on sys.path when using the venv's pytest binary,
            # so the module will be importable from the sandbox worktree too.
            #
            # Also add `tests/` to PYTHONPATH so Django's internal test apps
            # (auth_tests, queries, etc.) can be imported as top-level modules.
            # The sandbox prepends the worktree root to PYTHONPATH; this value
            # appended after it so Django source from the worktree takes priority.
            extra_pythonpath: str | None = None
            if "INSTALLED_APPS" not in settings_text:
                # Only include test apps that define Django models.
                # SimpleTestCase-only apps (utils_tests, template_tests, etc.)
                # don't need INSTALLED_APPS registration, and including them can
                # cause pytest duplicate-module-path errors when PYTHONPATH is set.
                raw_apps = self._extract_django_test_apps(bug.get("fail_to_pass", []))
                test_apps = [
                    app for app in raw_apps
                    if self._django_test_app_has_models(repo_dir / "tests" / app)
                ]
                venv_dir = venv_pytest.parent.parent
                site_packages = self._find_site_packages(venv_dir)
                if site_packages:
                    settings_module = "swe_bench_django_settings"
                    self._write_full_django_settings(site_packages, settings_module, test_apps)
                    django_settings = settings_module
                    logger.info(
                        "Django settings missing INSTALLED_APPS — wrote full settings "
                        "(model_apps=%s, skipped=%s) to %s",
                        test_apps,
                        [a for a in raw_apps if a not in test_apps],
                        site_packages / f"{settings_module}.py",
                    )
                # Add tests/ to PYTHONPATH only for model-bearing test apps.
                # Use *relative* path "tests" so Python resolves it against the
                # subprocess cwd (the sandbox worktree root) — this avoids the
                # pytest duplicate-module error that occurs with absolute paths
                # pointing at the original repo's tests/ dir.
                if test_apps:
                    extra_pythonpath = "tests"

            django_env: dict[str, str] = {"DJANGO_SETTINGS_MODULE": django_settings}
            if extra_pythonpath:
                django_env["PYTHONPATH"] = extra_pythonpath
            config["env"] = django_env
            logger.debug("Django repo: injecting DJANGO_SETTINGS_MODULE = %s", django_settings)

        config_path = repo_dir / ".agent_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        logger.debug("Wrote .agent_config.json: test_command=%s", test_command)

    def build_graph(self, repo_path: Path, bug: dict) -> None:
        """Build knowledge graph for a repo during the setup phase.

        Cache-aware: delegates to build_eval_graph which skips if SHA matches.
        Failures are logged and swallowed so eval continues without graph.

        Parameters
        ----------
        repo_path : Path
            Path to the cloned repo at the correct SHA.
        bug : dict
            EvalBug — used to derive repo_name.
        """
        repo_name = bug.get("repo_name") or bug["ticket_id"].lower().replace("-", "_")
        try:
            from agent.eval.graph_builder import build_eval_graph, DATA_DIR
            logger.info("Building graph for %s...", repo_name)
            build_eval_graph(repo_name, repo_path, data_dir=DATA_DIR)
            logger.info("Graph ready for %s", repo_name)
        except Exception as e:
            logger.warning(
                "Graph build failed for %s (%s) — continuing without graph",
                repo_name, e,
            )

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
