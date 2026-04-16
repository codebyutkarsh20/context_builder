"""
llm.py — LLM calling utilities extracted from pipeline.py.

Provides structured output calls, cost estimation, secret redaction,
and model configuration. Used by the ReAct pipeline.
"""

from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)

INTAKE_MODEL = "claude-haiku-4-5-20251001"

_MODEL_PRICING = {
    # USD per 1M tokens: (input, output)
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

# Secrets patterns
_SECRETS_RE = re.compile(
    r'(?i)((?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|'
    r'secret[_-]?key|password|passwd|private[_-]?key|credentials)'
    r'\s*[=:]\s*["\']?)[A-Za-z0-9+/=_\-]{16,}["\']?'
)
_ADDITIONAL_SECRET_PATTERNS = [
    re.compile(r'AKIA[A-Z0-9]{16}'),
    re.compile(r'(?:Bearer|token)\s+[A-Za-z0-9\-._~+/]+=*', re.I),
    re.compile(r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+'),
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'ghp_[A-Za-z0-9]{36}'),
]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts and model pricing."""
    pricing = _MODEL_PRICING.get(model, (3.0, 15.0))
    return round((input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000, 6)


def redact_secrets(text: str) -> str:
    """Redact potential secrets/tokens from source code before sending to LLM."""
    text = _SECRETS_RE.sub(r'\1***REDACTED***', text)
    for pat in _ADDITIONAL_SECRET_PATTERNS:
        text = pat.sub('***REDACTED***', text)
    return text


def _get_chat_anthropic():
    """Lazy import so the module can be loaded without langchain installed."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic

# Module-level attribute for test patching: @patch("agent.llm.ChatAnthropic")
ChatAnthropic = None


def structured_call(model: str, max_tokens: int, schema: type, prompt: str, retries: int = 1):
    """Call LLM with structured output (tool use). Returns a Pydantic model instance."""
    global ChatAnthropic
    if ChatAnthropic is None:
        ChatAnthropic = _get_chat_anthropic()

    approx_tokens = len(prompt) // 4
    logger.info("LLM call: model=%s schema=%s ~%d input tokens", model, schema.__name__, approx_tokens)

    llm = ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120.0, max_retries=2)
    structured = llm.with_structured_output(schema)

    t0 = time.monotonic()
    try:
        result = structured.invoke(prompt)
        return result
    except Exception as first_err:
        if retries <= 0:
            raise
        err_str = str(first_err)
        is_truncation = "max_tokens" in err_str.lower() or "incomplete" in err_str.lower()
        logger.warning("Structured output failed (%s), retrying%s",
                        first_err, " (truncation — increasing max_tokens)" if is_truncation else "")

        # If truncated, retry with more tokens and a shorter prompt
        retry_llm = llm
        if is_truncation:
            retry_llm = ChatAnthropic(
                model=model, max_tokens=max_tokens + 1000, timeout=120.0, max_retries=1,
            )
            structured = retry_llm.with_structured_output(schema)
            # Trim the prompt to leave room for the response
            if len(prompt) > 20000:
                prompt = prompt[:20000] + "\n[... prompt trimmed for retry]"

        error_msg = err_str[:500]
        retry_prompt = (
            f"Your previous response was incomplete: {error_msg}\n"
            "Please respond with ALL required fields. Be concise.\n\n"
            + prompt
        )
        result = structured.invoke(retry_prompt)
        return result


def simple_call(model: str, prompt: str, max_tokens: int = 500) -> str:
    """Simple text-in, text-out LLM call (no structured output).

    Used for lightweight tasks like skeleton narrowing, ensemble voting,
    and vague-description generation where we just need free-form text.
    """
    global ChatAnthropic
    if ChatAnthropic is None:
        ChatAnthropic = _get_chat_anthropic()

    llm = ChatAnthropic(model=model, max_tokens=max_tokens, timeout=60.0, max_retries=1)
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)
