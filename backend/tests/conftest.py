"""
conftest.py — Shared fixtures for the test suite.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so tests can find ANTHROPIC_API_KEY etc.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv()


# ── Temp repo fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary Git repo with some Python files."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    # Initialize git
    import subprocess
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)

    # Create source files
    (repo / "app.py").write_text(
        'def normalize_email(email):\n'
        '    return email.lower().strip()\n'
        '\n'
        'def get_user(user_id):\n'
        '    user = db.find(user_id)\n'
        '    user["email"] = normalize_email(user["email"])\n'
        '    return user\n'
    )
    (repo / "utils.py").write_text(
        'import os\n'
        '\n'
        'API_KEY = "sk-1234567890abcdefghij"\n'
        'DB_PASSWORD = "SuperSecret123456789"\n'
        '\n'
        'def calculate_total(items):\n'
        '    return sum(item["price"] for item in items)\n'
    )
    (repo / "binary_file.pyc").write_bytes(b'\x00\x01\x02\x03')
    (repo / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n')
    (repo / "data.sqlite").write_bytes(b'SQLite format 3\x00')

    # Add a nested file
    (repo / "src").mkdir()
    (repo / "src" / "models.py").write_text(
        'class User:\n'
        '    def __init__(self, name, email=None):\n'
        '        self.name = name\n'
        '        self.email = email\n'
    )

    # Initial commit
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

    return repo


@pytest.fixture
def tmp_repo_dirty(tmp_repo):
    """A repo with uncommitted changes."""
    (tmp_repo / "dirty.txt").write_text("uncommitted file")
    return tmp_repo


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary DATA_DIR with a mock graph.json."""
    data = tmp_path / "data"
    data.mkdir()
    repo_dir = data / "test-repo"
    repo_dir.mkdir()

    graph = {
        "nodes": [{"id": "f:app.py", "name": "app.py", "type": "File"}],
        "edges": [],
        "stats": {
            "repo": "test-repo",
            "repo_path": "",
            "files": 2,
            "classes": 0,
            "functions": 3,
            "lines_of_code": 15,
            "tech_stack": ["python"],
        },
    }
    (repo_dir / "graph.json").write_text(json.dumps(graph))
    (repo_dir / "summary.md").write_text("# Test Repo\nA test repository.")
    return data


@pytest.fixture
def mock_work_order(tmp_repo, tmp_data_dir):
    """A work order dict pointing at the tmp_repo."""
    # Store repo_path in graph.json
    graph_path = tmp_data_dir / "test-repo" / "graph.json"
    data = json.loads(graph_path.read_text())
    data["stats"]["repo_path"] = str(tmp_repo)
    graph_path.write_text(json.dumps(data))

    return {
        "ticket_id": "TEST-001",
        "title": "Bug in normalize_email",
        "description": "NoneType error when email is null",
        "repo_name": "test-repo",
        "repo_path": str(tmp_repo),
        "priority": "high",
        "comments": [],
    }


@pytest.fixture
def pipeline_module(tmp_data_dir):
    """Import pipeline module with DATA_DIR pointed at tmp_data_dir."""
    with patch.dict(os.environ, {"DATA_DIR": str(tmp_data_dir)}):
        # Force reimport isn't practical due to module-level graph compilation,
        # so we just import and patch DATA_DIR at usage points
        import agent.pipeline as pipeline
        original_data_dir = pipeline.DATA_DIR
        pipeline.DATA_DIR = tmp_data_dir
        yield pipeline
        pipeline.DATA_DIR = original_data_dir
