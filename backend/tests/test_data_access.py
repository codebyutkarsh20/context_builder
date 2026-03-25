"""
Unit tests for analyzer/data_access.py — data access pattern detection.

Covers:
  - _slice: line extraction (1-indexed)
  - _scan_body: read patterns (database, file, api, cache, environment)
                write patterns (database, file, api, cache, logging, event)
  - detect_data_access: integration via real temp files
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.data_access import _scan_body, _slice, detect_data_access


# ---------------------------------------------------------------------------
# _slice
# ---------------------------------------------------------------------------

class TestSlice:

    def test_extracts_correct_lines(self):
        lines = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
        result = _slice(lines, 2, 4)
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result

    def test_start_at_1(self):
        lines = ["a\n", "b\n", "c\n"]
        result = _slice(lines, 1, 2)
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_out_of_bounds_start_clamped(self):
        lines = ["a\n", "b\n"]
        result = _slice(lines, 0, 2)  # start=0 → clamped to 0
        assert "a" in result

    def test_empty_lines(self):
        assert _slice([], 1, 5) == ""


# ---------------------------------------------------------------------------
# _scan_body — database read patterns
# ---------------------------------------------------------------------------

class TestScanBodyDatabaseReads:

    def test_query_method(self):
        body = "results = session.query(User).all()"
        acc = _scan_body(body)
        assert "database" in acc["reads_from"]

    def test_filter_method(self):
        body = "users = User.objects.filter(active=True)"
        acc = _scan_body(body)
        assert "database" in acc["reads_from"]

    def test_fetchall(self):
        body = "rows = cursor.fetchall()"
        acc = _scan_body(body)
        assert "database" in acc["reads_from"]

    def test_select_cursor_execute(self):
        body = "cursor.execute('SELECT * FROM users')"
        acc = _scan_body(body)
        assert "database" in acc["reads_from"]


# ---------------------------------------------------------------------------
# _scan_body — file read patterns
# ---------------------------------------------------------------------------

class TestScanBodyFileReads:

    def test_read_text(self):
        body = "content = path.read_text()"
        acc = _scan_body(body)
        assert "file" in acc["reads_from"]

    def test_open_read_mode(self):
        body = "with open('data.txt', 'r') as f: data = f.read()"
        acc = _scan_body(body)
        assert "file" in acc["reads_from"]

    def test_json_load(self):
        body = "data = json.load(fp)"
        acc = _scan_body(body)
        assert "file" in acc["reads_from"]

    def test_yaml_safe_load(self):
        body = "config = yaml.safe_load(f)"
        acc = _scan_body(body)
        assert "file" in acc["reads_from"]


# ---------------------------------------------------------------------------
# _scan_body — API read patterns
# ---------------------------------------------------------------------------

class TestScanBodyApiReads:

    def test_requests_get(self):
        body = "resp = requests.get(url)"
        acc = _scan_body(body)
        assert "api" in acc["reads_from"]

    def test_httpx_get(self):
        body = "resp = httpx.get(url)"
        acc = _scan_body(body)
        assert "api" in acc["reads_from"]


# ---------------------------------------------------------------------------
# _scan_body — cache read patterns
# ---------------------------------------------------------------------------

class TestScanBodyCacheReads:

    def test_cache_get(self):
        body = "value = cache.get('key')"
        acc = _scan_body(body)
        assert "cache" in acc["reads_from"]

    def test_redis_get(self):
        body = "value = redis_client.get('user:123')"
        acc = _scan_body(body)
        assert "cache" in acc["reads_from"]


# ---------------------------------------------------------------------------
# _scan_body — environment reads
# ---------------------------------------------------------------------------

class TestScanBodyEnvironmentReads:

    def test_os_environ_get(self):
        body = "key = os.environ.get('API_KEY')"
        acc = _scan_body(body)
        assert "environment" in acc["reads_from"]

    def test_os_getenv(self):
        body = "val = os.getenv('DB_URL')"
        acc = _scan_body(body)
        assert "environment" in acc["reads_from"]


# ---------------------------------------------------------------------------
# _scan_body — database write patterns
# ---------------------------------------------------------------------------

class TestScanBodyDatabaseWrites:

    def test_session_add(self):
        body = "session.add(user)\nsession.commit()"
        acc = _scan_body(body)
        assert "database" in acc["writes_to"]

    def test_model_save(self):
        body = "user.save()"
        acc = _scan_body(body)
        assert "database" in acc["writes_to"]

    def test_model_create(self):
        body = "User.objects.create(name='Alice')"
        acc = _scan_body(body)
        assert "database" in acc["writes_to"]

    def test_cursor_execute_insert(self):
        body = "cursor.execute('INSERT INTO users VALUES (%s)', [name])"
        acc = _scan_body(body)
        assert "database" in acc["writes_to"]


# ---------------------------------------------------------------------------
# _scan_body — file write patterns
# ---------------------------------------------------------------------------

class TestScanBodyFileWrites:

    def test_write_text(self):
        body = "path.write_text('content')"
        acc = _scan_body(body)
        assert "file" in acc["writes_to"]

    def test_open_write_mode(self):
        body = "with open('out.txt', 'w') as f: f.write(data)"
        acc = _scan_body(body)
        assert "file" in acc["writes_to"]

    def test_json_dump(self):
        body = "json.dump(data, f)"
        acc = _scan_body(body)
        assert "file" in acc["writes_to"]


# ---------------------------------------------------------------------------
# _scan_body — API write patterns
# ---------------------------------------------------------------------------

class TestScanBodyApiWrites:

    def test_requests_post(self):
        body = "resp = requests.post(url, json=payload)"
        acc = _scan_body(body)
        assert "api" in acc["writes_to"]

    def test_requests_put(self):
        body = "resp = requests.put(url, data=data)"
        acc = _scan_body(body)
        assert "api" in acc["writes_to"]

    def test_httpx_post(self):
        body = "resp = httpx.post(url, json=data)"
        acc = _scan_body(body)
        assert "api" in acc["writes_to"]


# ---------------------------------------------------------------------------
# _scan_body — cache write patterns
# ---------------------------------------------------------------------------

class TestScanBodyCacheWrites:

    def test_cache_set(self):
        body = "cache.set('key', value, timeout=300)"
        acc = _scan_body(body)
        assert "cache" in acc["writes_to"]

    def test_redis_set(self):
        body = "redis_client.set('user:123', json.dumps(data))"
        acc = _scan_body(body)
        assert "cache" in acc["writes_to"]

    def test_redis_delete(self):
        body = "redis_client.delete('user:123')"
        acc = _scan_body(body)
        assert "cache" in acc["writes_to"]


# ---------------------------------------------------------------------------
# _scan_body — logging and event writes
# ---------------------------------------------------------------------------

class TestScanBodyLoggingAndEvents:

    def test_logger_info(self):
        body = "logger.info('user created')"
        acc = _scan_body(body)
        assert "logging" in acc["writes_to"]

    def test_logger_error(self):
        body = "logger.error('Something broke')"
        acc = _scan_body(body)
        assert "logging" in acc["writes_to"]

    def test_celery_delay(self):
        body = "send_email.delay(user_id=123)"
        acc = _scan_body(body)
        assert "event" in acc["writes_to"]

    def test_apply_async(self):
        body = "task.apply_async(args=[1, 2], countdown=10)"
        acc = _scan_body(body)
        assert "event" in acc["writes_to"]


# ---------------------------------------------------------------------------
# _scan_body — empty and no-match cases
# ---------------------------------------------------------------------------

class TestScanBodyEdgeCases:

    def test_empty_body(self):
        acc = _scan_body("")
        assert acc["reads_from"] == []
        assert acc["writes_to"] == []

    def test_no_data_access(self):
        body = "x = 1 + 2\nreturn x\n"
        acc = _scan_body(body)
        assert acc["reads_from"] == []
        assert acc["writes_to"] == []

    def test_result_lists_sorted(self):
        body = "cache.get('key')\nrequests.get(url)\njson.load(f)"
        acc = _scan_body(body)
        assert acc["reads_from"] == sorted(acc["reads_from"])


# ---------------------------------------------------------------------------
# detect_data_access — integration with real files
# ---------------------------------------------------------------------------

class TestDetectDataAccess:

    def _parsed(self, abs_path, rel_path="app.py", functions=None, classes=None):
        return {
            "path": rel_path,
            "abs_path": abs_path,
            "functions": functions or [],
            "classes": classes or [],
        }

    def test_top_level_function_detected(self, tmp_path):
        code = (
            "def get_users():\n"
            "    return User.objects.filter(active=True)\n"
        )
        f = tmp_path / "app.py"
        f.write_text(code)
        parsed = [self._parsed(
            str(f),
            functions=[{"name": "get_users", "line_start": 1, "line_end": 2}],
        )]
        result = detect_data_access(parsed)
        assert "app.py::get_users" in result
        assert "database" in result["app.py::get_users"]["reads_from"]

    def test_method_detected(self, tmp_path):
        code = (
            "class UserRepo:\n"
            "    def save(self, user):\n"
            "        session.add(user)\n"
            "        session.commit()\n"
        )
        f = tmp_path / "repo.py"
        f.write_text(code)
        parsed = [self._parsed(
            str(f),
            rel_path="repo.py",
            classes=[{
                "name": "UserRepo",
                "methods": [{"name": "save", "line_start": 2, "line_end": 4}],
            }],
        )]
        result = detect_data_access(parsed)
        assert "repo.py::UserRepo.save" in result
        assert "database" in result["repo.py::UserRepo.save"]["writes_to"]

    def test_missing_file_skipped_gracefully(self, tmp_path):
        parsed = [self._parsed(
            "/nonexistent/file.py",
            functions=[{"name": "fn", "line_start": 1, "line_end": 2}],
        )]
        result = detect_data_access(parsed)
        assert result == {}

    def test_function_with_no_access_excluded(self, tmp_path):
        code = "def add(a, b):\n    return a + b\n"
        f = tmp_path / "math.py"
        f.write_text(code)
        parsed = [self._parsed(
            str(f),
            rel_path="math.py",
            functions=[{"name": "add", "line_start": 1, "line_end": 2}],
        )]
        result = detect_data_access(parsed)
        assert "math.py::add" not in result

    def test_returns_sorted_lists(self, tmp_path):
        code = (
            "def mixed():\n"
            "    cache.get('k')\n"
            "    requests.get('url')\n"
            "    json.load(f)\n"
        )
        f = tmp_path / "mixed.py"
        f.write_text(code)
        parsed = [self._parsed(
            str(f),
            rel_path="mixed.py",
            functions=[{"name": "mixed", "line_start": 1, "line_end": 4}],
        )]
        result = detect_data_access(parsed)
        func_result = result.get("mixed.py::mixed", {})
        if func_result.get("reads_from"):
            assert func_result["reads_from"] == sorted(func_result["reads_from"])
