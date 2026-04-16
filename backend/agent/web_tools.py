"""
web_tools.py — WebFetch + WebSearch tools for the agent.

Ports the WebFetchTool and WebSearchTool patterns from Claude Code
(tools/WebFetchTool, tools/WebSearchTool).

Use cases for our autonomous bug-fix agent:
- Look up library docs when the agent encounters an unfamiliar API
  (e.g., "what does werkzeug.url_quote do?" → fetch werkzeug docs)
- Find the upstream fix for a known error (e.g., "django InvalidBasesError
  workaround" → search → read the Stack Overflow answer)
- Read GitHub issues / PR discussions for bugs already filed upstream

Design notes:
- WebFetch uses Python `requests` + a regex-based HTML stripper, then a
  Haiku call to extract relevant info (matches Claude Code's pattern of
  running a small model over fetched content)
- WebSearch is a thin wrapper that delegates to Anthropic's native server-
  side web_search_20250305 tool, surfaced via a single LLM call
- Both tools have a 15-minute in-memory cache to avoid re-fetching identical
  URLs (matches Claude Code's WebFetchTool cache TTL)
- Hard caps: 30s timeout, 200KB max page size, 10 search results max
"""

from __future__ import annotations

import logging
import os
import re
import time
import threading
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEB_FETCH_TIMEOUT_SECONDS = 30
WEB_FETCH_MAX_BYTES = 200_000          # 200 KB cap
WEB_FETCH_CACHE_TTL_SECONDS = 15 * 60  # 15-minute cache (matches Claude Code)
WEB_SEARCH_MAX_RESULTS = 10
WEB_SEARCH_MODEL = os.environ.get("WEB_SEARCH_MODEL", "claude-haiku-4-5-20251001")
WEB_FETCH_EXTRACTOR_MODEL = os.environ.get("WEB_FETCH_EXTRACTOR_MODEL", "claude-haiku-4-5-20251001")

# Tools are disabled in eval/CI by default to avoid network flakiness +
# bandwidth costs. Set ENABLE_WEB_TOOLS=1 to opt in.
WEB_TOOLS_ENABLED = os.environ.get("ENABLE_WEB_TOOLS", "0") not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# Per-process URL cache (thread-safe)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
# Cache keyed by URL for raw page content (shared across prompts)
_page_cache: dict[str, tuple[float, str]] = {}  # url → (timestamp, markdown)
# Cache keyed by (url, prompt_hash) for extractor answers
_answer_cache: dict[str, tuple[float, str]] = {}  # "url|hash" → (timestamp, answer)


def _cache_get_page(url: str) -> Optional[str]:
    """Return cached page markdown for url if still fresh, else None."""
    now = time.time()
    with _cache_lock:
        entry = _page_cache.get(url)
        if entry is None:
            return None
        ts, md = entry
        if now - ts > WEB_FETCH_CACHE_TTL_SECONDS:
            _page_cache.pop(url, None)
            return None
        return md


def _cache_put_page(url: str, markdown: str) -> None:
    with _cache_lock:
        _page_cache[url] = (time.time(), markdown)


def _cache_get_answer(url: str, prompt: str) -> Optional[str]:
    """Return cached extractor answer for (url, prompt) if still fresh."""
    import hashlib
    key = f"{url}|{hashlib.md5(prompt.encode()).hexdigest()}"
    now = time.time()
    with _cache_lock:
        entry = _answer_cache.get(key)
        if entry is None:
            return None
        ts, ans = entry
        if now - ts > WEB_FETCH_CACHE_TTL_SECONDS:
            _answer_cache.pop(key, None)
            return None
        return ans


def _cache_put_answer(url: str, prompt: str, answer: str) -> None:
    import hashlib
    key = f"{url}|{hashlib.md5(prompt.encode()).hexdigest()}"
    with _cache_lock:
        _answer_cache[key] = (time.time(), answer)


def clear_web_cache() -> None:
    """Clear all URL caches — useful for tests."""
    with _cache_lock:
        _page_cache.clear()
        _answer_cache.clear()


# ---------------------------------------------------------------------------
# HTML → markdown stripper (regex-based, no extra deps)
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|#39|nbsp);")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")

_ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
}


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text. Lightweight, no external dependencies.

    Good enough for docs pages, GitHub issues, SO answers. NOT a full
    html-to-markdown converter — link URLs are dropped, images are dropped.
    The downstream Haiku extractor handles the semantics.
    """
    # 1. Drop script/style blocks entirely
    text = _SCRIPT_STYLE_RE.sub("", html)
    # 2. Strip remaining tags
    text = _TAG_RE.sub("\n", text)
    # 3. Decode common HTML entities
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    # 4. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# WebFetch tool
# ---------------------------------------------------------------------------

@tool
def web_fetch(url: str, prompt: str) -> str:
    """Fetch a URL and use a small LLM to extract information answering `prompt`.

    Use this for documentation lookups (e.g., "what does werkzeug.url_quote do?"),
    GitHub issue/PR discussions, Stack Overflow answers, or any web content
    that would help you understand an unfamiliar API or error.

    Cache: identical (url, prompt) pairs within 15 minutes return cached
    markdown without a re-fetch — only the extractor LLM call is repeated.

    Args:
        url: Fully-formed HTTP/HTTPS URL. http:// is upgraded to https://
            automatically.
        prompt: What information you want extracted from the page. Be specific:
            "What's the function signature of werkzeug.url_quote?" beats
            "Tell me about werkzeug."

    Returns:
        The extractor LLM's response (typically 1-5 paragraphs), or an
        ERROR string if the fetch / extraction failed.
    """
    if not WEB_TOOLS_ENABLED:
        return (
            "ERROR: web tools are disabled. Set ENABLE_WEB_TOOLS=1 to enable. "
            "(Disabled by default in eval/CI to avoid network flakiness.)"
        )
    if not url or not url.strip():
        return "ERROR: url cannot be empty."
    if not prompt or not prompt.strip():
        return "ERROR: prompt cannot be empty (describe what to extract)."

    # Auto-upgrade http → https
    fetch_url = url.strip()
    if fetch_url.startswith("http://"):
        fetch_url = "https://" + fetch_url[len("http://"):]
    if not fetch_url.startswith("https://"):
        return f"ERROR: url must be HTTP/HTTPS — got '{url[:80]}'"

    # SSRF protection — block requests to internal/private networks
    if _is_private_url(fetch_url):
        return f"ERROR: URL targets a private/internal network address — blocked for security."

    # Check answer cache first (keyed by url+prompt), then page cache (url only)
    cached_answer = _cache_get_answer(fetch_url, prompt)
    if cached_answer is not None:
        return f"=== WebFetch result for {fetch_url} [cached] ===\n\n{cached_answer}"

    markdown = _cache_get_page(fetch_url)
    cache_hit = markdown is not None

    if not cache_hit:
        try:
            import requests
            response = requests.get(
                fetch_url,
                timeout=WEB_FETCH_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ContextBuilderAgent/1.0)",
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                },
                allow_redirects=True,
                stream=False,
            )
        except Exception as e:
            return f"ERROR: fetch failed for {fetch_url}: {type(e).__name__}: {e}"

        if response.status_code != 200:
            return f"ERROR: HTTP {response.status_code} from {fetch_url}"

        # Detect redirects to a different host (matches Claude Code's pattern of
        # surfacing the new URL so the agent can re-fetch with explicit consent)
        if response.url != fetch_url and \
                _host(response.url) != _host(fetch_url):
            return (
                f"REDIRECT: {fetch_url} → {response.url} (different host). "
                f"Call web_fetch again with the new URL if you trust the redirect target."
            )

        # Cap response size BEFORE decoding to avoid OOM on huge pages
        body = response.content[:WEB_FETCH_MAX_BYTES]
        try:
            html = body.decode(response.encoding or "utf-8", errors="replace")
        except Exception:
            html = body.decode("utf-8", errors="replace")

        markdown = _html_to_text(html)
        # Cap markdown too — extractor doesn't need 1MB of text
        if len(markdown) > 60_000:
            markdown = markdown[:60_000] + "\n[... page truncated to 60K chars]"
        _cache_put_page(fetch_url, markdown)

    # Run the extractor LLM
    extractor_prompt = (
        "You are a web-content extractor. The user fetched a web page and "
        "wants specific information from it.\n\n"
        f"=== USER'S QUESTION ===\n{prompt}\n\n"
        f"=== PAGE CONTENT ({fetch_url}) ===\n{markdown}\n\n"
        "Answer the user's question concisely (1-5 paragraphs). Quote relevant "
        "snippets, code blocks, or function signatures verbatim. If the page "
        "doesn't contain the answer, say so explicitly — do not hallucinate."
    )

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        llm = ChatAnthropic(
            model=WEB_FETCH_EXTRACTOR_MODEL,
            max_tokens=2000,
            timeout=60.0,
            max_retries=1,
        )
        response = llm.invoke([HumanMessage(content=extractor_prompt)])
        answer = str(response.content) if not isinstance(response.content, list) \
            else " ".join(str(b.get("text", "")) for b in response.content if isinstance(b, dict))
    except Exception as e:
        return f"ERROR: extractor LLM failed: {type(e).__name__}: {e}"

    # Cache the extractor answer keyed by (url, prompt)
    _cache_put_answer(fetch_url, prompt, answer)
    cache_note = " [cached page]" if cache_hit else ""
    return f"=== WebFetch result for {fetch_url}{cache_note} ===\n\n{answer}"


def _host(url: str) -> str:
    """Extract host from a URL without importing urllib (cheap)."""
    if "://" not in url:
        return ""
    rest = url.split("://", 1)[1]
    return rest.split("/", 1)[0].lower()


def _is_private_url(url: str) -> bool:
    """Check if a URL targets a private/internal network (SSRF protection).

    Blocks: localhost, 127.x.x.x, 10.x.x.x, 172.16-31.x.x, 192.168.x.x,
    169.254.x.x (AWS metadata), [::1], 0.0.0.0.
    """
    import ipaddress
    host = _host(url)
    # Strip port if present
    if ":" in host and not host.startswith("["):
        host = host.rsplit(":", 1)[0]
    host = host.strip("[]").lower()

    # Direct hostname checks
    if host in ("localhost", "0.0.0.0", "[::]", ""):
        return True

    # Try parsing as IP address
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass

    # Hostname-based checks (e.g., "metadata.google.internal")
    if host.endswith(".internal") or host.endswith(".local"):
        return True

    return False


# ---------------------------------------------------------------------------
# WebSearch tool
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for current information using Anthropic's native web search.

    Use this when:
      - You need information beyond your training cutoff (e.g., a library
        version released after Aug 2025)
      - You're debugging a specific error message and want to find others'
        reports / fixes
      - You need to find documentation pages without knowing the exact URL

    The result includes URL citations. Follow up with web_fetch(url, prompt)
    to read a specific page in depth.

    Args:
        query: Search query. Be specific. Include version numbers and exact
            error messages in quotes when relevant.

    Returns:
        The search results summary (with markdown links), or an ERROR
        string if the search failed.
    """
    if not WEB_TOOLS_ENABLED:
        return (
            "ERROR: web tools are disabled. Set ENABLE_WEB_TOOLS=1 to enable. "
            "(Disabled by default in eval/CI to avoid network flakiness.)"
        )
    if not query or not query.strip():
        return "ERROR: query cannot be empty."

    try:
        from anthropic import Anthropic
    except ImportError:
        return "ERROR: anthropic SDK not available — web_search requires it."

    try:
        client = Anthropic()
        # Anthropic's server-side web_search tool: the model decides when to
        # invoke it. We pass the user's query as a simple message and let the
        # model run searches as needed.
        message = client.messages.create(
            model=WEB_SEARCH_MODEL,
            max_tokens=2000,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": WEB_SEARCH_MAX_RESULTS,
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Search the web for: {query}\n\n"
                    "Summarize the top results in 3-6 bullet points. For each "
                    "bullet, include the source URL as a markdown link. End "
                    "with a 'Sources:' section listing every URL referenced."
                ),
            }],
        )
    except Exception as e:
        return f"ERROR: web_search failed: {type(e).__name__}: {e}"

    # Concatenate all text blocks in the response
    parts: list[str] = []
    for block in message.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
    summary = "\n\n".join(p for p in parts if p).strip()
    if not summary:
        return f"ERROR: web_search returned no text content for query '{query[:80]}'"
    return f"=== WebSearch result for '{query[:120]}' ===\n\n{summary}"


# Tool collection for registration
WEB_TOOLS = [web_fetch, web_search]
