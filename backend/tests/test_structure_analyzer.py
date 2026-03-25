"""
Unit tests for analyzer/structure.py — StructureAnalyzer and helper functions.

Covers:
  - _should_skip_dir
  - _build_tree (depth, pruning)
  - _read_file_safe
  - _detect_tech_stack (Python, Node.js, Docker, Go, React, frameworks)
  - _collect_file_stats (Python/JS/TS counts, line counts)
  - _find_entry_points
  - StructureAnalyzer.analyze (full integration)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.structure import (
    MAX_DEPTH,
    SKIP_DIRS,
    StructureAnalyzer,
    _build_tree,
    _collect_file_stats,
    _detect_tech_stack,
    _find_entry_points,
    _read_file_safe,
    _should_skip_dir,
)


# ---------------------------------------------------------------------------
# _should_skip_dir
# ---------------------------------------------------------------------------

class TestShouldSkipDir:

    def test_skips_git(self):
        assert _should_skip_dir(".git") is True

    def test_skips_pycache(self):
        assert _should_skip_dir("__pycache__") is True

    def test_skips_node_modules(self):
        assert _should_skip_dir("node_modules") is True

    def test_skips_venv(self):
        assert _should_skip_dir(".venv") is True
        assert _should_skip_dir("venv") is True

    def test_skips_dist_and_build(self):
        assert _should_skip_dir("dist") is True
        assert _should_skip_dir("build") is True

    def test_skips_egg_info_suffix(self):
        assert _should_skip_dir("mypackage.egg-info") is True

    def test_allows_src(self):
        assert _should_skip_dir("src") is False

    def test_allows_tests(self):
        assert _should_skip_dir("tests") is False

    def test_allows_lib(self):
        assert _should_skip_dir("lib") is False


# ---------------------------------------------------------------------------
# _build_tree
# ---------------------------------------------------------------------------

class TestBuildTree:

    def test_file_node_structure(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        node = _build_tree(f)
        assert node["name"] == "hello.py"
        assert node["type"] == "file"
        assert node["children"] == []

    def test_dir_node_has_children(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        node = _build_tree(tmp_path)
        assert node["type"] == "dir"
        child_names = {c["name"] for c in node["children"]}
        assert "a.py" in child_names
        assert "b.py" in child_names

    def test_skips_git_dir(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
        node = _build_tree(tmp_path)
        child_names = {c["name"] for c in node["children"]}
        assert ".git" not in child_names

    def test_respects_max_depth(self, tmp_path):
        # Create MAX_DEPTH + 2 levels of nesting
        current = tmp_path
        for i in range(MAX_DEPTH + 2):
            current = current / f"level{i}"
            current.mkdir()
            (current / "file.py").write_text(f"# level {i}")

        node = _build_tree(tmp_path)

        # Traverse the tree to find max depth
        def max_depth(n, depth=0):
            if not n["children"]:
                return depth
            return max(max_depth(c, depth + 1) for c in n["children"])

        actual_depth = max_depth(node)
        assert actual_depth <= MAX_DEPTH + 1  # +1 for the file leaf

    def test_path_stored_on_node(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("")
        node = _build_tree(f)
        assert "path" in node


# ---------------------------------------------------------------------------
# _read_file_safe
# ---------------------------------------------------------------------------

class TestReadFileSafe:

    def test_reads_normal_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert _read_file_safe(f) == "hello world"

    def test_returns_empty_on_missing_file(self, tmp_path):
        result = _read_file_safe(tmp_path / "nonexistent.txt")
        assert result == ""

    def test_respects_max_bytes(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("A" * 200_000)
        result = _read_file_safe(f, max_bytes=100)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# _detect_tech_stack
# ---------------------------------------------------------------------------

class TestDetectTechStack:

    def test_detects_python_from_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Python" in techs

    def test_detects_fastapi(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi==0.115.0\nuvicorn\n")
        techs = _detect_tech_stack(tmp_path)
        assert "FastAPI" in techs

    def test_detects_django(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django>=4.0\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Django" in techs

    def test_detects_flask(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Flask" in techs

    def test_detects_docker_from_dockerfile(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Docker" in techs

    def test_detects_docker_from_compose(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("version: '3'\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Docker" in techs

    def test_detects_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/app\n\ngo 1.21\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Go" in techs

    def test_detects_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "app"\n')
        techs = _detect_tech_stack(tmp_path)
        assert "Rust" in techs

    def test_detects_node_from_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app", "dependencies": {}}')
        techs = _detect_tech_stack(tmp_path)
        assert "Node.js" in techs

    def test_detects_react_from_package_json(self, tmp_path):
        pkg = {"dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        techs = _detect_tech_stack(tmp_path)
        assert "React" in techs

    def test_detects_typescript_from_tsconfig(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
        techs = _detect_tech_stack(tmp_path)
        assert "TypeScript" in techs

    def test_detects_sqlalchemy(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("sqlalchemy\n")
        techs = _detect_tech_stack(tmp_path)
        assert "SQLAlchemy" in techs

    def test_detects_redis(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("redis\n")
        techs = _detect_tech_stack(tmp_path)
        assert "Redis" in techs

    def test_empty_repo_no_stack(self, tmp_path):
        # No config files — should return empty list
        techs = _detect_tech_stack(tmp_path)
        assert isinstance(techs, list)

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\nredis\n")
        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        techs = _detect_tech_stack(tmp_path)
        assert techs == sorted(techs)

    def test_detects_from_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\npython = "^3.11"\nfastapi = "*"\n'
        )
        techs = _detect_tech_stack(tmp_path)
        assert "FastAPI" in techs

    def test_detects_kubernetes_from_yaml(self, tmp_path):
        (tmp_path / "deploy.yaml").write_text(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: app\n"
        )
        techs = _detect_tech_stack(tmp_path)
        assert "Kubernetes" in techs


# ---------------------------------------------------------------------------
# _collect_file_stats
# ---------------------------------------------------------------------------

class TestCollectFileStats:

    def test_counts_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        stats = _collect_file_stats(tmp_path)
        assert stats["python_files"] == 2

    def test_counts_js_files(self, tmp_path):
        (tmp_path / "app.js").write_text("console.log(1);\n")
        (tmp_path / "util.mjs").write_text("export default {};\n")
        stats = _collect_file_stats(tmp_path)
        assert stats["js_files"] == 2

    def test_counts_ts_files(self, tmp_path):
        (tmp_path / "app.ts").write_text("const x: number = 1;\n")
        (tmp_path / "comp.tsx").write_text("export default () => null;\n")
        stats = _collect_file_stats(tmp_path)
        assert stats["ts_files"] == 2

    def test_counts_total_files(self, tmp_path):
        for name in ["a.py", "b.js", "c.go", "d.png"]:
            (tmp_path / name).write_text("x")
        stats = _collect_file_stats(tmp_path)
        assert stats["total_files"] == 4

    def test_counts_lines(self, tmp_path):
        (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
        stats = _collect_file_stats(tmp_path)
        assert stats["total_lines"] >= 3

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("x = 1\n")
        (tmp_path / "app.js").write_text("y = 2\n")
        stats = _collect_file_stats(tmp_path)
        # Should only count app.js, not node_modules/lib.js
        assert stats["js_files"] == 1

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "app.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "app.py").write_text("x = 1\n")
        stats = _collect_file_stats(tmp_path)
        assert stats["python_files"] == 1


# ---------------------------------------------------------------------------
# _find_entry_points
# ---------------------------------------------------------------------------

class TestFindEntryPoints:

    def test_finds_main_py(self, tmp_path):
        (tmp_path / "main.py").write_text("if __name__ == '__main__': pass")
        eps = _find_entry_points(tmp_path)
        assert "main.py" in eps

    def test_finds_app_py(self, tmp_path):
        (tmp_path / "app.py").write_text("app = FastAPI()")
        eps = _find_entry_points(tmp_path)
        assert "app.py" in eps

    def test_finds_manage_py(self, tmp_path):
        (tmp_path / "manage.py").write_text("# Django manage")
        eps = _find_entry_points(tmp_path)
        assert "manage.py" in eps

    def test_finds_nested_entry_point(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("x = 1")
        eps = _find_entry_points(tmp_path)
        assert any("main.py" in ep for ep in eps)

    def test_finds_index_js(self, tmp_path):
        (tmp_path / "index.js").write_text("module.exports = {};")
        eps = _find_entry_points(tmp_path)
        assert "index.js" in eps

    def test_no_entry_points(self, tmp_path):
        (tmp_path / "utils.py").write_text("def helper(): pass")
        eps = _find_entry_points(tmp_path)
        assert "utils.py" not in eps

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        (tmp_path / "app.py").write_text("")
        eps = _find_entry_points(tmp_path)
        assert eps == sorted(eps)


# ---------------------------------------------------------------------------
# StructureAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestStructureAnalyzer:

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            StructureAnalyzer(Path("/no/such/path")).analyze()

    def test_result_has_all_keys(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        result = StructureAnalyzer(tmp_path).analyze()
        for key in ("repo_path", "name", "tree", "tech_stack", "entry_points",
                    "file_stats", "readme_content"):
            assert key in result, f"Missing key: {key}"

    def test_name_matches_directory(self, tmp_path):
        result = StructureAnalyzer(tmp_path).analyze()
        assert result["name"] == tmp_path.name

    def test_readme_content_loaded(self, tmp_path):
        (tmp_path / "README.md").write_text("# My Project\nThis is cool.\n")
        result = StructureAnalyzer(tmp_path).analyze()
        assert "My Project" in result["readme_content"]

    def test_readme_content_empty_when_no_readme(self, tmp_path):
        result = StructureAnalyzer(tmp_path).analyze()
        assert result["readme_content"] == ""

    def test_file_stats_populated(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        result = StructureAnalyzer(tmp_path).analyze()
        assert result["file_stats"]["python_files"] >= 1

    def test_tech_stack_list(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        result = StructureAnalyzer(tmp_path).analyze()
        assert isinstance(result["tech_stack"], list)
        assert "Flask" in result["tech_stack"]

    def test_entry_points_list(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        result = StructureAnalyzer(tmp_path).analyze()
        assert "main.py" in result["entry_points"]

    def test_tree_root_is_dir(self, tmp_path):
        result = StructureAnalyzer(tmp_path).analyze()
        assert result["tree"]["type"] == "dir"

    def test_readme_truncated_at_2000_chars(self, tmp_path):
        (tmp_path / "README.md").write_text("X" * 10_000)
        result = StructureAnalyzer(tmp_path).analyze()
        assert len(result["readme_content"]) <= 2000
