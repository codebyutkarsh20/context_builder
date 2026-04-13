"""
test_web_tools.py — Tests for web_fetch and web_search tools.

Network calls are mocked — these tests run offline and are safe in CI.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.web_tools import (
    WEB_TOOLS,
    _html_to_text,
    _host,
    clear_web_cache,
    web_fetch,
    web_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    clear_web_cache()
    yield
    clear_web_cache()


@pytest.fixture
def web_tools_enabled(monkeypatch):
    """Force-enable web tools for the test."""
    monkeypatch.setenv("ENABLE_WEB_TOOLS", "1")
    # The flag is read at module import time. Patch the module-level constant.
    import agent.web_tools as wt
    monkeypatch.setattr(wt, "WEB_TOOLS_ENABLED", True)
    yield


# ---------------------------------------------------------------------------
# HTML stripper
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        assert "Hello" in _html_to_text(html)
        assert "world" in _html_to_text(html)
        assert "<p>" not in _html_to_text(html)
        assert "<b>" not in _html_to_text(html)

    def test_drops_script_blocks(self):
        html = "<p>Visible</p><script>alert('x')</script><p>Also visible</p>"
        result = _html_to_text(html)
        assert "Visible" in result
        assert "Also visible" in result
        assert "alert" not in result

    def test_drops_style_blocks(self):
        html = "<style>body { color: red; }</style><p>text</p>"
        result = _html_to_text(html)
        assert "text" in result
        assert "color: red" not in result

    def test_decodes_html_entities(self):
        html = "<p>5 &lt; 10 &amp;&amp; 10 &gt; 5</p>"
        result = _html_to_text(html)
        assert "5 < 10" in result
        assert "&&" in result
        assert "10 > 5" in result

    def test_handles_nbsp(self):
        html = "<p>Hello&nbsp;world</p>"
        assert "Hello world" in _html_to_text(html)

    def test_collapses_blank_lines(self):
        html = "a\n\n\n\n\n\n\nb"
        result = _html_to_text(html)
        # Multiple blank lines collapsed to at most one blank line
        assert "\n\n\n" not in result


class TestHostExtractor:
    def test_basic_https(self):
        assert _host("https://docs.python.org/3/library/index.html") == "docs.python.org"

    def test_basic_http(self):
        assert _host("http://example.com/path") == "example.com"

    def test_no_path(self):
        assert _host("https://github.com") == "github.com"

    def test_lowercased(self):
        assert _host("https://DOCS.Python.ORG/page") == "docs.python.org"

    def test_no_scheme_returns_empty(self):
        assert _host("example.com/foo") == ""


# ---------------------------------------------------------------------------
# Disabled-by-default behavior
# ---------------------------------------------------------------------------

class TestDisabledByDefault:
    def test_web_fetch_disabled_returns_error(self):
        # Don't enable — default state
        result = web_fetch.invoke({"url": "https://docs.python.org", "prompt": "anything"})
        assert "ERROR" in result
        assert "disabled" in result.lower()

    def test_web_search_disabled_returns_error(self):
        result = web_search.invoke({"query": "anything"})
        assert "ERROR" in result
        assert "disabled" in result.lower()


# ---------------------------------------------------------------------------
# web_fetch — input validation
# ---------------------------------------------------------------------------

class TestWebFetchValidation:
    def test_empty_url_rejected(self, web_tools_enabled):
        result = web_fetch.invoke({"url": "", "prompt": "x"})
        assert "ERROR" in result and "url" in result.lower()

    def test_empty_prompt_rejected(self, web_tools_enabled):
        result = web_fetch.invoke({"url": "https://x.com", "prompt": ""})
        assert "ERROR" in result and "prompt" in result.lower()

    def test_non_http_url_rejected(self, web_tools_enabled):
        result = web_fetch.invoke({"url": "ftp://x.com/file", "prompt": "x"})
        assert "ERROR" in result
        assert "HTTP" in result or "http" in result

    def test_http_auto_upgraded_to_https(self, web_tools_enabled):
        """http:// URLs are silently upgraded to https://. Verify by asserting
        the HTTPS variant is what gets passed to requests.get.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/page"
        mock_response.content = b"<p>hello</p>"
        mock_response.encoding = "utf-8"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="answer")

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch.object(
                __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
                "ChatAnthropic", return_value=mock_llm,
             ):
            web_fetch.invoke({"url": "http://example.com/page", "prompt": "what's here?"})

        # First positional arg to requests.get was the upgraded URL
        called_url = mock_get.call_args.args[0]
        assert called_url.startswith("https://")


# ---------------------------------------------------------------------------
# web_fetch — fetch + extract happy path
# ---------------------------------------------------------------------------

class TestWebFetchHappyPath:
    def _setup_response(self, body: bytes, url: str = "https://example.com/page"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = url
        mock_response.content = body
        mock_response.encoding = "utf-8"
        return mock_response

    def test_returns_extractor_answer(self, web_tools_enabled):
        page = b"<html><body><h1>Werkzeug API</h1><p>url_quote(string) returns escaped URL.</p></body></html>"
        mock_response = self._setup_response(page)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="url_quote(string) returns the URL-escaped form."
        )

        with patch("requests.get", return_value=mock_response), \
             patch.object(
                __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
                "ChatAnthropic", return_value=mock_llm,
             ):
            result = web_fetch.invoke({
                "url": "https://example.com/page",
                "prompt": "What does url_quote do?",
            })

        assert "url_quote" in result
        assert "WebFetch result" in result

    def test_non_200_returns_error(self, web_tools_enabled):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/missing"
        mock_response.content = b""
        mock_response.encoding = "utf-8"

        with patch("requests.get", return_value=mock_response):
            result = web_fetch.invoke({
                "url": "https://example.com/missing",
                "prompt": "anything",
            })
        assert "ERROR" in result
        assert "404" in result

    def test_redirect_to_different_host_returns_redirect_message(self, web_tools_enabled):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://different-host.com/foo"
        mock_response.content = b"<p>x</p>"
        mock_response.encoding = "utf-8"

        with patch("requests.get", return_value=mock_response):
            result = web_fetch.invoke({
                "url": "https://example.com/page",
                "prompt": "anything",
            })
        assert "REDIRECT" in result
        assert "different-host.com" in result

    def test_request_exception_returns_error(self, web_tools_enabled):
        with patch("requests.get", side_effect=RuntimeError("DNS lookup failed")):
            result = web_fetch.invoke({
                "url": "https://flaky.com/page",
                "prompt": "x",
            })
        assert "ERROR" in result
        assert "DNS" in result or "fetch failed" in result.lower()


# ---------------------------------------------------------------------------
# web_fetch — caching
# ---------------------------------------------------------------------------

class TestWebFetchCache:
    def test_second_call_within_ttl_uses_cache(self, web_tools_enabled):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/page"
        mock_response.content = b"<p>hello</p>"
        mock_response.encoding = "utf-8"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="hello")

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch.object(
                __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
                "ChatAnthropic", return_value=mock_llm,
             ):
            web_fetch.invoke({"url": "https://example.com/page", "prompt": "x"})
            second = web_fetch.invoke({"url": "https://example.com/page", "prompt": "x"})

        # requests.get called once (first call), cached on second call
        assert mock_get.call_count == 1
        # Second result mentions [cached]
        assert "[cached]" in second

    def test_clear_cache_helper(self, web_tools_enabled):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/page"
        mock_response.content = b"<p>hello</p>"
        mock_response.encoding = "utf-8"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="hello")

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch.object(
                __import__("langchain_anthropic", fromlist=["ChatAnthropic"]),
                "ChatAnthropic", return_value=mock_llm,
             ):
            web_fetch.invoke({"url": "https://example.com/page", "prompt": "x"})
            clear_web_cache()
            web_fetch.invoke({"url": "https://example.com/page", "prompt": "x"})

        # Cache cleared → second call re-fetched
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# web_search — mocked Anthropic client
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_empty_query_rejected(self, web_tools_enabled):
        result = web_search.invoke({"query": ""})
        assert "ERROR" in result

    def test_returns_text_summary(self, web_tools_enabled):
        # Build a fake Anthropic message response with text blocks
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = (
            "Werkzeug 2.0 removed url_quote. Use urllib.parse.quote instead.\n\n"
            "Sources:\n- [PR](https://github.com/pallets/werkzeug/pull/123)"
        )
        fake_message = MagicMock()
        fake_message.content = [text_block]

        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        with patch.object(
            __import__("anthropic", fromlist=["Anthropic"]),
            "Anthropic", return_value=fake_client,
        ):
            result = web_search.invoke({"query": "werkzeug url_quote removed"})

        assert "WebSearch result" in result
        assert "url_quote" in result
        assert "Sources:" in result

    def test_anthropic_exception_returns_error(self, web_tools_enabled):
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = RuntimeError("rate limited")

        with patch.object(
            __import__("anthropic", fromlist=["Anthropic"]),
            "Anthropic", return_value=fake_client,
        ):
            result = web_search.invoke({"query": "anything"})

        assert "ERROR" in result
        assert "rate limited" in result

    def test_no_text_blocks_returns_error(self, web_tools_enabled):
        fake_message = MagicMock()
        fake_message.content = []  # No blocks

        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        with patch.object(
            __import__("anthropic", fromlist=["Anthropic"]),
            "Anthropic", return_value=fake_client,
        ):
            result = web_search.invoke({"query": "x"})

        assert "ERROR" in result
        assert "no text content" in result.lower()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_web_tools_collection_has_both(self):
        names = [t.name for t in WEB_TOOLS]
        assert "web_fetch" in names
        assert "web_search" in names

    def test_tools_have_docstrings(self):
        assert web_fetch.description and len(web_fetch.description) > 100
        assert web_search.description and len(web_search.description) > 100

    def test_react_loop_imports_web_tools(self):
        import agent.react_loop as react_loop
        import inspect
        src = inspect.getsource(react_loop.react_loop)
        assert "WEB_TOOLS" in src
        assert "WEB_TOOLS_ENABLED" in src
