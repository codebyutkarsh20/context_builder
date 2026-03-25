"""
StructureAnalyzer: Analyzes repository structure, tech stack, entry points, and file statistics.
Uses only Python stdlib (pathlib, os).
"""

import os
from pathlib import Path


SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".eggs",
}

SKIP_DIR_SUFFIXES = {".egg-info"}

MAX_DEPTH = 4

ENTRY_POINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "run.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "index.js",
    "index.ts",
    "app.js",
    "app.ts",
    "server.js",
    "server.ts",
    "main.go",
    "main.rs",
    "main.rb",
    "main.java",
    "Program.cs",
    "__main__.py",
}

# Tech stack detection: maps filename -> list of tech names to add
TECH_DETECTION_FILES = {
    "requirements.txt": ["Python"],
    "setup.py": ["Python"],
    "setup.cfg": ["Python"],
    "pyproject.toml": ["Python"],
    "Pipfile": ["Python"],
    "poetry.lock": ["Python"],
    "package.json": [],  # handled separately for JS/TS/React/etc.
    "package-lock.json": [],
    "yarn.lock": ["Node.js"],
    "pnpm-lock.yaml": ["Node.js"],
    "Dockerfile": ["Docker"],
    "docker-compose.yml": ["Docker"],
    "docker-compose.yaml": ["Docker"],
    "go.mod": ["Go"],
    "go.sum": ["Go"],
    "Cargo.toml": ["Rust"],
    "Cargo.lock": ["Rust"],
    "Gemfile": ["Ruby"],
    "Gemfile.lock": ["Ruby"],
    "pom.xml": ["Java", "Maven"],
    "build.gradle": ["Java", "Gradle"],
    "build.gradle.kts": ["Kotlin", "Gradle"],
    "*.csproj": ["C#", ".NET"],
    "*.fsproj": ["F#", ".NET"],
    "*.sln": [".NET"],
    "composer.json": ["PHP"],
    "mix.exs": ["Elixir"],
    "pubspec.yaml": ["Dart", "Flutter"],
    "CMakeLists.txt": ["C/C++", "CMake"],
    "Makefile": [],  # too generic to infer a stack
    "terraform.tf": ["Terraform"],
    "main.tf": ["Terraform"],
    "*.tf": ["Terraform"],
    "ansible.cfg": ["Ansible"],
    "Chart.yaml": ["Helm", "Kubernetes"],
    "kubernetes.yml": ["Kubernetes"],
    "kubernetes.yaml": ["Kubernetes"],
    "k8s.yml": ["Kubernetes"],
    "k8s.yaml": ["Kubernetes"],
}

# Keywords to scan inside package.json for framework detection
PACKAGE_JSON_DEPS_FRAMEWORKS = {
    "react": "React",
    "react-dom": "React",
    "next": "Next.js",
    "nuxt": "Nuxt.js",
    "vue": "Vue",
    "@angular/core": "Angular",
    "svelte": "Svelte",
    "express": "Express",
    "fastify": "Fastify",
    "koa": "Koa",
    "hapi": "Hapi",
    "nestjs": "NestJS",
    "@nestjs/core": "NestJS",
    "gatsby": "Gatsby",
    "remix": "Remix",
    "@remix-run/react": "Remix",
    "vite": "Vite",
    "webpack": "Webpack",
    "typescript": "TypeScript",
    "eslint": "ESLint",
    "jest": "Jest",
    "vitest": "Vitest",
    "tailwindcss": "Tailwind CSS",
    "prisma": "Prisma",
    "sequelize": "Sequelize",
    "mongoose": "Mongoose",
    "graphql": "GraphQL",
    "apollo-server": "Apollo",
    "@apollo/server": "Apollo",
    "socket.io": "Socket.IO",
    "axios": "Axios",
    "redux": "Redux",
    "@reduxjs/toolkit": "Redux",
    "mobx": "MobX",
    "zustand": "Zustand",
    "electron": "Electron",
    "react-native": "React Native",
}

# Keywords inside requirements.txt / pyproject.toml for Python framework detection
PYTHON_FRAMEWORK_KEYWORDS = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "tornado": "Tornado",
    "starlette": "Starlette",
    "aiohttp": "aiohttp",
    "sanic": "Sanic",
    "falcon": "Falcon",
    "bottle": "Bottle",
    "pyramid": "Pyramid",
    "sqlalchemy": "SQLAlchemy",
    "alembic": "Alembic",
    "celery": "Celery",
    "pydantic": "Pydantic",
    "uvicorn": "Uvicorn",
    "gunicorn": "Gunicorn",
    "pytest": "pytest",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "scikit-learn": "scikit-learn",
    "tensorflow": "TensorFlow",
    "torch": "PyTorch",
    "keras": "Keras",
    "transformers": "HuggingFace Transformers",
    "langchain": "LangChain",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "boto3": "AWS SDK",
    "google-cloud": "Google Cloud",
    "azure": "Azure SDK",
    "redis": "Redis",
    "pymongo": "MongoDB",
    "motor": "MongoDB (async)",
    "psycopg2": "PostgreSQL",
    "asyncpg": "PostgreSQL (async)",
    "aiomysql": "MySQL",
    "pymysql": "MySQL",
    "graphene": "GraphQL",
    "strawberry": "GraphQL",
}


def _should_skip_dir(name: str) -> bool:
    if name in SKIP_DIRS:
        return True
    for suffix in SKIP_DIR_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _build_tree(path: Path, depth: int = 0) -> dict:
    """Recursively build directory tree up to MAX_DEPTH."""
    node = {
        "name": path.name,
        "type": "dir" if path.is_dir() else "file",
        "path": str(path),
        "children": [],
    }

    if path.is_dir() and depth < MAX_DEPTH:
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return node

        for entry in entries:
            if entry.is_dir() and _should_skip_dir(entry.name):
                continue
            child = _build_tree(entry, depth + 1)
            node["children"].append(child)

    return node


def _read_file_safe(path: Path, max_bytes: int = 65536) -> str:
    """Read a file safely, returning empty string on error."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except (OSError, PermissionError):
        return ""


def _detect_tech_stack(repo_path: Path) -> list:
    """Detect technologies used in the repository."""
    techs: set = set()

    # Walk only the top level and one level deep for config files
    # to avoid scanning the whole tree again
    search_paths = [repo_path]
    try:
        for entry in repo_path.iterdir():
            if entry.is_dir() and not _should_skip_dir(entry.name):
                search_paths.append(entry)
    except PermissionError:
        pass

    found_files: dict[str, Path] = {}
    for search_dir in search_paths:
        try:
            for entry in search_dir.iterdir():
                if entry.is_file():
                    found_files[entry.name] = entry
        except PermissionError:
            pass

    # Check exact filename matches
    for filename, file_techs in TECH_DETECTION_FILES.items():
        if "*" in filename:
            # Glob pattern match
            suffix = filename.lstrip("*")
            for fname in found_files:
                if fname.endswith(suffix):
                    techs.update(file_techs)
                    break
        elif filename in found_files:
            techs.update(file_techs)

    # Detect Node.js presence via package.json
    if "package.json" in found_files:
        content = _read_file_safe(found_files["package.json"])
        if content:
            techs.add("Node.js")
            content_lower = content.lower()
            # Check for TypeScript files in the repo root as well
            for dep_key, tech_name in PACKAGE_JSON_DEPS_FRAMEWORKS.items():
                if f'"{dep_key}"' in content_lower or f"'{dep_key}'" in content_lower:
                    techs.add(tech_name)

    # Detect Python frameworks from requirements.txt
    for req_file in ("requirements.txt", "requirements-dev.txt", "requirements/base.txt"):
        req_path = repo_path / req_file
        if req_path.exists():
            content = _read_file_safe(req_path).lower()
            for kw, tech in PYTHON_FRAMEWORK_KEYWORDS.items():
                if kw in content:
                    techs.add(tech)

    # Detect Python frameworks from pyproject.toml
    pyproject_path = repo_path / "pyproject.toml"
    if pyproject_path.exists():
        content = _read_file_safe(pyproject_path).lower()
        for kw, tech in PYTHON_FRAMEWORK_KEYWORDS.items():
            if kw in content:
                techs.add(tech)

    # Detect Python frameworks from Pipfile
    pipfile_path = repo_path / "Pipfile"
    if pipfile_path.exists():
        content = _read_file_safe(pipfile_path).lower()
        for kw, tech in PYTHON_FRAMEWORK_KEYWORDS.items():
            if kw in content:
                techs.add(tech)

    # Check for TypeScript source files
    ts_extensions = {".ts", ".tsx"}
    has_ts = any(
        entry.suffix in ts_extensions
        for entry in repo_path.rglob("*")
        if entry.is_file() and not any(_should_skip_dir(p) for p in entry.parts)
    ) if False else False  # skip rglob here; will use file_stats from collect_file_stats

    # Simpler TS detection: check common TS config files
    ts_configs = ["tsconfig.json", "tsconfig.base.json", "tsconfig.build.json"]
    for ts_cfg in ts_configs:
        if (repo_path / ts_cfg).exists():
            techs.add("TypeScript")
            break

    # Kubernetes / Helm detection from yaml files
    k8s_keywords = ["apiVersion:", "kind: Deployment", "kind: Service", "kind: Pod"]
    for yaml_file in list(repo_path.glob("*.yml")) + list(repo_path.glob("*.yaml")):
        content = _read_file_safe(yaml_file, max_bytes=4096)
        if any(kw in content for kw in k8s_keywords):
            techs.add("Kubernetes")
            break

    return sorted(techs)


def _collect_file_stats(repo_path: Path) -> dict:
    """Collect file statistics across the repository."""
    total_files = 0
    python_files = 0
    js_files = 0
    ts_files = 0
    total_lines = 0

    for root, dirs, files in os.walk(repo_path):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

        for filename in files:
            total_files += 1
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()

            is_text = False
            if ext == ".py":
                python_files += 1
                is_text = True
            elif ext in (".js", ".jsx", ".mjs", ".cjs"):
                js_files += 1
                is_text = True
            elif ext in (".ts", ".tsx"):
                ts_files += 1
                is_text = True
            elif ext in (
                ".go", ".rs", ".java", ".rb", ".php", ".cs", ".cpp", ".c", ".h",
                ".html", ".css", ".scss", ".sass", ".less", ".json", ".yaml",
                ".yml", ".toml", ".ini", ".cfg", ".md", ".txt", ".sh", ".bash",
                ".zsh", ".fish", ".sql", ".graphql", ".proto",
            ):
                is_text = True

            if is_text:
                try:
                    with open(filepath, "rb") as f:
                        content = f.read(131072)  # read up to 128KB for line counting
                    total_lines += content.count(b"\n")
                except (OSError, PermissionError):
                    pass

    return {
        "total_files": total_files,
        "python_files": python_files,
        "js_files": js_files,
        "ts_files": ts_files,
        "total_lines": total_lines,
    }


def _find_entry_points(repo_path: Path) -> list:
    """Find likely entry point files in the repository."""
    entry_points = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
        # Limit depth to 3 for entry point scanning
        rel = Path(root).relative_to(repo_path)
        if len(rel.parts) > 3:
            dirs[:] = []
            continue

        for filename in files:
            if filename in ENTRY_POINT_NAMES:
                filepath = Path(root) / filename
                entry_points.append(str(filepath.relative_to(repo_path)))

    return sorted(entry_points)


class StructureAnalyzer:
    """
    Analyzes the structure of a code repository.

    Parameters
    ----------
    repo_path : Path
        Path to the root of the repository to analyze.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def analyze(self) -> dict:
        """
        Perform a full structural analysis of the repository.

        Returns
        -------
        dict
            A dictionary containing:
            - repo_path: absolute path string
            - name: repository folder name
            - tree: recursive directory/file tree (4 levels deep)
            - tech_stack: list of detected technologies
            - entry_points: list of likely entry point file paths
            - file_stats: counts of files and estimated line count
            - readme_content: first 2000 chars of README.md if present
        """
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")

        tree = _build_tree(self.repo_path)
        tech_stack = _detect_tech_stack(self.repo_path)
        entry_points = _find_entry_points(self.repo_path)
        file_stats = _collect_file_stats(self.repo_path)

        readme_content = ""
        for readme_name in ("README.md", "README.rst", "README.txt", "README"):
            readme_path = self.repo_path / readme_name
            if readme_path.exists():
                content = _read_file_safe(readme_path, max_bytes=2000)
                readme_content = content[:2000]
                break

        return {
            "repo_path": str(self.repo_path),
            "name": self.repo_path.name,
            "tree": tree,
            "tech_stack": tech_stack,
            "entry_points": entry_points,
            "file_stats": file_stats,
            "readme_content": readme_content,
        }
