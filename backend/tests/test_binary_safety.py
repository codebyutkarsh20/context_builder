"""
Tests for binary file safety — Phase 1.5

Ensures binary files are skipped during source reading
and UnicodeDecodeError is handled gracefully.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import _BINARY_EXTENSIONS, read_source_node, _redact_secrets
from agent.types import PipelineStatus


class TestBinaryExtensions:
    """The blocklist correctly identifies binary files."""

    # ── Compiled / native ────────────────────────────────────────────
    def test_pyc_blocked(self):
        assert '.pyc' in _BINARY_EXTENSIONS

    def test_pyo_blocked(self):
        assert '.pyo' in _BINARY_EXTENSIONS

    def test_so_blocked(self):
        assert '.so' in _BINARY_EXTENSIONS

    def test_dll_blocked(self):
        assert '.dll' in _BINARY_EXTENSIONS

    def test_exe_blocked(self):
        assert '.exe' in _BINARY_EXTENSIONS

    def test_dylib_blocked(self):
        assert '.dylib' in _BINARY_EXTENSIONS

    # ── Images ───────────────────────────────────────────────────────
    def test_png_blocked(self):
        assert '.png' in _BINARY_EXTENSIONS

    def test_jpg_blocked(self):
        assert '.jpg' in _BINARY_EXTENSIONS

    def test_jpeg_blocked(self):
        assert '.jpeg' in _BINARY_EXTENSIONS

    def test_gif_blocked(self):
        assert '.gif' in _BINARY_EXTENSIONS

    def test_svg_blocked(self):
        assert '.svg' in _BINARY_EXTENSIONS

    def test_webp_blocked(self):
        assert '.webp' in _BINARY_EXTENSIONS

    # ── Fonts ────────────────────────────────────────────────────────
    def test_woff_blocked(self):
        assert '.woff' in _BINARY_EXTENSIONS

    def test_woff2_blocked(self):
        assert '.woff2' in _BINARY_EXTENSIONS

    def test_ttf_blocked(self):
        assert '.ttf' in _BINARY_EXTENSIONS

    # ── Archives ─────────────────────────────────────────────────────
    def test_zip_blocked(self):
        assert '.zip' in _BINARY_EXTENSIONS

    def test_tar_blocked(self):
        assert '.tar' in _BINARY_EXTENSIONS

    def test_gz_blocked(self):
        assert '.gz' in _BINARY_EXTENSIONS

    # ── Documents ────────────────────────────────────────────────────
    def test_pdf_blocked(self):
        assert '.pdf' in _BINARY_EXTENSIONS

    def test_docx_blocked(self):
        assert '.docx' in _BINARY_EXTENSIONS

    # ── Media ────────────────────────────────────────────────────────
    def test_mp3_blocked(self):
        assert '.mp3' in _BINARY_EXTENSIONS

    def test_mp4_blocked(self):
        assert '.mp4' in _BINARY_EXTENSIONS

    # ── Databases ────────────────────────────────────────────────────
    def test_sqlite_blocked(self):
        assert '.sqlite' in _BINARY_EXTENSIONS

    def test_db_blocked(self):
        assert '.db' in _BINARY_EXTENSIONS

    # ── Source files ALLOWED ─────────────────────────────────────────
    def test_py_allowed(self):
        assert '.py' not in _BINARY_EXTENSIONS

    def test_js_allowed(self):
        assert '.js' not in _BINARY_EXTENSIONS

    def test_ts_allowed(self):
        assert '.ts' not in _BINARY_EXTENSIONS

    def test_java_allowed(self):
        assert '.java' not in _BINARY_EXTENSIONS

    def test_go_allowed(self):
        assert '.go' not in _BINARY_EXTENSIONS

    def test_rs_allowed(self):
        assert '.rs' not in _BINARY_EXTENSIONS

    def test_rb_allowed(self):
        assert '.rb' not in _BINARY_EXTENSIONS

    def test_html_allowed(self):
        assert '.html' not in _BINARY_EXTENSIONS

    def test_css_allowed(self):
        assert '.css' not in _BINARY_EXTENSIONS

    def test_json_allowed(self):
        assert '.json' not in _BINARY_EXTENSIONS

    def test_yaml_allowed(self):
        assert '.yaml' not in _BINARY_EXTENSIONS

    def test_md_allowed(self):
        assert '.md' not in _BINARY_EXTENSIONS


class TestReadSourceNodeBinarySafety:
    """read_source_node skips binary files and handles UnicodeDecodeError."""

    def test_skips_binary_by_extension(self, tmp_repo):
        """Binary files listed in localization are skipped."""
        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": ["binary_file.pyc", "image.png", "app.py"]},
            "status": "",
        }
        result = read_source_node(state)
        source = result["source_code"]

        assert "binary_file.pyc" not in source
        assert "image.png" not in source
        assert "app.py" in source

    def test_handles_unicode_decode_error(self, tmp_repo):
        """Files that fail UTF-8 decode are skipped gracefully."""
        # Write a file with invalid UTF-8
        bad_file = tmp_repo / "broken.py"
        bad_file.write_bytes(b'\x80\x81\x82\x83' * 100)

        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": ["broken.py", "app.py"]},
            "status": "",
        }
        result = read_source_node(state)
        source = result["source_code"]

        assert "broken.py" not in source  # skipped due to decode error
        assert "app.py" in source  # still read successfully

    def test_caps_at_5_files(self, tmp_repo):
        """Only first 5 files are read."""
        for i in range(10):
            (tmp_repo / f"file{i}.py").write_text(f"x = {i}")

        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": [f"file{i}.py" for i in range(10)]},
            "status": "",
        }
        result = read_source_node(state)
        assert len(result["source_code"]) <= 5

    def test_truncates_long_files(self, tmp_repo):
        """Files longer than the max line limit are truncated."""
        # read_source_node uses max_lines=3000, so file must exceed that
        long_content = "\n".join(f"line_{i} = {i}" for i in range(4000))
        (tmp_repo / "long.py").write_text(long_content)

        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": ["long.py"]},
            "status": "",
        }
        result = read_source_node(state)
        content = result["source_code"]["long.py"]
        lines = content.strip().split("\n")
        # Should be truncated below the original 4000 lines
        assert len(lines) < 4000
        assert "truncated" in content or "omitted" in content

    def test_redacts_secrets_in_source(self, tmp_repo):
        """Secrets in source code are redacted."""
        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": ["utils.py"]},
            "status": "",
        }
        result = read_source_node(state)
        content = result["source_code"]["utils.py"]

        assert "sk-1234567890abcdefghij" not in content
        assert "SuperSecret123456789" not in content
        assert "***REDACTED***" in content

    def test_no_repo_path_returns_empty(self):
        """If repo path can't be resolved, return empty source_code."""
        state = {
            "work_order": {"repo_name": "nonexistent", "repo_path": "/does/not/exist"},
            "localization": {"fault_files": ["app.py"]},
            "status": "",
        }
        result = read_source_node(state)
        assert result["source_code"] == {}

    def test_file_search_with_glob(self, tmp_repo):
        """Files found via rglob when exact path doesn't match."""
        state = {
            "work_order": {"repo_name": "test", "repo_path": str(tmp_repo)},
            "localization": {"fault_files": ["models.py"]},  # actually in src/models.py
            "status": "",
        }
        result = read_source_node(state)
        assert "models.py" in result["source_code"]
