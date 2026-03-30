"""
Tests for binary file safety — Phase 1.5

Ensures binary files are skipped during source reading
and UnicodeDecodeError is handled gracefully.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import _BINARY_EXTENSIONS, _read_file_safe, _redact_secrets
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


class TestReadFileSafeBinarySafety:
    """_read_file_safe skips binary files and handles errors gracefully."""

    def test_skips_binary_by_extension(self, tmp_repo):
        """Binary files return None from _read_file_safe."""
        pyc_file = tmp_repo / "compiled.pyc"
        pyc_file.write_bytes(b"fake pyc content")
        assert _read_file_safe(pyc_file) is None

        png_file = tmp_repo / "image.png"
        png_file.write_bytes(b"\x89PNG\r\n")
        assert _read_file_safe(png_file) is None

    def test_reads_valid_python_file(self, tmp_repo):
        """Normal Python files are read successfully."""
        py_file = tmp_repo / "app.py"
        py_file.write_text("def hello():\n    return 'world'\n")
        result = _read_file_safe(py_file)
        assert result is not None
        assert "def hello" in result

    def test_handles_unicode_decode_error(self, tmp_repo):
        """Files with invalid UTF-8 return None gracefully."""
        bad_file = tmp_repo / "broken.py"
        bad_file.write_bytes(b'\x80\x81\x82\x83' * 100)
        result = _read_file_safe(bad_file)
        assert result is None

    def test_truncates_long_files(self, tmp_repo):
        """Files over max_lines are truncated."""
        long_file = tmp_repo / "long.py"
        long_file.write_text("\n".join(f"line_{i} = {i}" for i in range(4000)))
        result = _read_file_safe(long_file, max_lines=500)
        assert result is not None
        lines = result.strip().split("\n")
        assert len(lines) < 4000
        assert "truncated" in result

    def test_redacts_secrets_in_source(self, tmp_repo):
        """Secrets in source files are redacted before return."""
        utils_file = tmp_repo / "utils.py"
        utils_file.write_text(
            'api_key = "sk-1234567890abcdefghij"\npassword = "SuperSecret123456789"\n'
        )
        result = _read_file_safe(utils_file)
        assert result is not None
        assert "sk-1234567890abcdefghij" not in result
        assert "SuperSecret123456789" not in result
        assert "***REDACTED***" in result

    def test_focus_lines_windowing(self, tmp_repo):
        """Focus lines produce windowed output with gap markers."""
        content = "\n".join(f"line_{i} = {i}" for i in range(1000))
        f = tmp_repo / "big.py"
        f.write_text(content)
        result = _read_file_safe(f, max_lines=100, focus_lines=[500])
        assert result is not None
        # Should contain lines near 500, not the full file
        assert "line_500" in result
