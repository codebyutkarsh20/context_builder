"""
Tests for secrets redaction — Phase 3.6

Ensures API keys, tokens, passwords, and credentials are stripped
before source code is sent to the LLM.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import _redact_secrets


class TestSecretsRedaction:
    """Secrets are redacted before code goes to the LLM."""

    # ── Should be redacted ───────────────────────────────────────────

    def test_api_key_single_quotes(self):
        assert "***REDACTED***" in _redact_secrets("API_KEY = 'sk-1234567890abcdef1234'")

    def test_api_key_double_quotes(self):
        assert "***REDACTED***" in _redact_secrets('API_KEY = "sk-1234567890abcdef1234"')

    def test_api_key_no_quotes(self):
        assert "***REDACTED***" in _redact_secrets("API_KEY = sk1234567890abcdef1234")

    def test_api_secret_underscore(self):
        assert "***REDACTED***" in _redact_secrets("api_secret = abcdef1234567890abcdef")

    def test_api_secret_hyphen(self):
        assert "***REDACTED***" in _redact_secrets("api-secret = abcdef1234567890abcdef")

    def test_access_token(self):
        assert "***REDACTED***" in _redact_secrets("access_token = 'ghp_abcdefghijklmnopqrst'")

    def test_auth_token(self):
        assert "***REDACTED***" in _redact_secrets("AUTH_TOKEN = Bearer_abcdefghijklmnopqr")

    def test_secret_key(self):
        assert "***REDACTED***" in _redact_secrets("SECRET_KEY = 'django-insecure-abcdefgh12345678'")

    def test_password_field(self):
        assert "***REDACTED***" in _redact_secrets("password: SuperSecretPassword123456")

    def test_passwd_field(self):
        assert "***REDACTED***" in _redact_secrets("PASSWD = my_secure_password_12345678")

    def test_private_key(self):
        assert "***REDACTED***" in _redact_secrets("private_key = 'MIIEvgIBADANBgkqhkiG9w0BAQ'")

    def test_credentials_field(self):
        assert "***REDACTED***" in _redact_secrets("credentials = base64encodedstring1234")

    def test_colon_separator(self):
        assert "***REDACTED***" in _redact_secrets("api_key: sk_live_abcdefghijklmnop")

    def test_equals_separator(self):
        assert "***REDACTED***" in _redact_secrets("API_KEY=sk_live_abcdefghijklmnop")

    def test_case_insensitive(self):
        assert "***REDACTED***" in _redact_secrets("Api_Key = sk_live_abcdefghijklmnop")
        assert "***REDACTED***" in _redact_secrets("API_KEY = sk_live_abcdefghijklmnop")
        assert "***REDACTED***" in _redact_secrets("api_key = sk_live_abcdefghijklmnop")

    def test_multiline_redaction(self):
        code = (
            "import os\n"
            "API_KEY = 'sk_live_1234567890abcdef'\n"
            "def foo():\n"
            "    password = 'Hunter2IsNotSecure!1234'\n"
            "    return True\n"
        )
        result = _redact_secrets(code)
        assert result.count("***REDACTED***") == 2
        assert "import os" in result
        assert "def foo():" in result
        assert "return True" in result

    # ── Should NOT be redacted ───────────────────────────────────────

    def test_normal_variable(self):
        assert "***REDACTED***" not in _redact_secrets("name = 'John Doe'")

    def test_comment(self):
        assert "***REDACTED***" not in _redact_secrets("# This is a comment about api keys")

    def test_function_definition(self):
        assert "***REDACTED***" not in _redact_secrets("def calculate_api_key_hash():")

    def test_import_statement(self):
        assert "***REDACTED***" not in _redact_secrets("from config import settings")

    def test_short_value_not_redacted(self):
        """Values shorter than 16 chars should not match (too short to be a real key)."""
        assert "***REDACTED***" not in _redact_secrets("api_key = 'short'")

    def test_number_assignment(self):
        assert "***REDACTED***" not in _redact_secrets("max_retries = 3")

    def test_boolean_assignment(self):
        assert "***REDACTED***" not in _redact_secrets("password_required = True")

    def test_empty_string(self):
        assert _redact_secrets("") == ""

    def test_preserves_surrounding_code(self):
        code = "x = 1\nAPI_KEY = 'abcdefghijklmnopqrstuvwxyz'\ny = 2"
        result = _redact_secrets(code)
        assert "x = 1" in result
        assert "y = 2" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result
