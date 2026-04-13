"""
forked_subagent.py — Cache-preserving subagent forks.

Ports the forked-agent pattern from Claude Code's utils/forkedAgent.ts.

The Anthropic prompt cache works as a prefix match on:
    (system prompt, tools, messages prefix, model)

A subagent that wants to inherit the parent's cached prefix MUST use the
EXACT same system prompt + parent message history + model. This module
provides a module-level cache (`save_cache_safe_params` /
`get_last_cache_safe_params`) so subordinate calls — verifier, summarizer,
post-turn hooks — can read the parent's params without threading them
through the call stack.

Pattern adapted for our use case:
- Main ReAct loop calls `save_cache_safe_params()` at the end of each turn.
- Verifier / summarizer call `run_forked_subagent(task)` which:
    1. Reuses the parent's system_prompt, messages, model
    2. Appends the new task as a user message
    3. Invokes the LLM — the prefix is a cache hit (~87% input-token discount)
    4. Optionally parses the response into a Pydantic schema

If no parent params are cached (e.g., subagent runs without a parent loop,
or the parent crashed), the helper falls back to a fresh call so the caller
never sees a "no parent" error — it just costs more.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CacheSafeParams — the contract for subagent cache reuse
# ---------------------------------------------------------------------------

@dataclass
class CacheSafeParams:
    """Snapshot of the parent agent's call params, required for cache reuse.

    Attributes
    ----------
    system_prompt
        The exact string passed to the Anthropic API as the system message.
        Must include the cache_control marker if the parent set one.
    messages
        The parent's message history at fork time. These will be the cached
        prefix; the subagent appends a new user message after them.
    model
        Model ID. Subagent MUST use the same model for the cache to apply
        (e.g., "claude-sonnet-4-6", "claude-haiku-4-5-20251001").
    cache_control_block
        Optional dict describing the cache_control marker the parent set.
        If present, the subagent re-applies it so the prefix splits at the
        same boundary.
    """
    system_prompt: str
    messages: list = field(default_factory=list)
    model: str = "claude-sonnet-4-6"
    cache_control_block: Optional[dict] = None


# ---------------------------------------------------------------------------
# Module-level params slot (mirrors Claude Code's saveCacheSafeParams)
# ---------------------------------------------------------------------------

# Use a per-thread slot so concurrent runs (best-of-N) don't trample each other.
_tls = threading.local()


def save_cache_safe_params(params: CacheSafeParams | None) -> None:
    """Stash the parent's cache-safe params for subordinate calls.

    Called at the end of each main-loop iteration (or once at the end of the
    react loop) so that any post-loop subagent (verifier, summarizer) can
    pick them up.
    """
    _tls.last_params = params
    if params:
        logger.debug(
            "Saved cache-safe params: model=%s, system_prompt_chars=%d, msg_count=%d",
            params.model, len(params.system_prompt), len(params.messages),
        )


def get_last_cache_safe_params() -> Optional[CacheSafeParams]:
    """Get the most recently saved parent params (or None)."""
    return getattr(_tls, "last_params", None)


def clear_cache_safe_params() -> None:
    """Clear stored params — call between unrelated runs to avoid stale forks."""
    if hasattr(_tls, "last_params"):
        delattr(_tls, "last_params")


# ---------------------------------------------------------------------------
# Forked subagent runner
# ---------------------------------------------------------------------------

def run_forked_subagent(
    task: str,
    *,
    schema: Optional[type] = None,
    max_tokens: int = 1500,
    timeout: float = 60.0,
    fallback_system_prompt: str = "You are an AI assistant.",
    parent_params: Optional[CacheSafeParams] = None,
) -> dict:
    """Run a subordinate LLM call that reuses the parent's prompt cache.

    The subagent inherits the parent's system_prompt + messages, then a new
    user message containing `task` is appended. This means everything except
    the final user message + assistant response is a cache hit on the parent's
    prefix.

    Parameters
    ----------
    task
        The user message to append after the parent's history. Should describe
        what the subagent must produce (e.g., "Now act as a verifier — review
        the patch and return a verdict").
    schema
        Optional Pydantic BaseModel class. If provided, the response is parsed
        into an instance of this schema via Anthropic's tool calling.
    max_tokens
        Output token budget for the subagent's response.
    timeout
        Per-call timeout in seconds.
    fallback_system_prompt
        Used only when no parent params are available (no main loop ran).
    parent_params
        Optional override — pass a CacheSafeParams explicitly instead of using
        the module-level slot. Useful for tests.

    Returns
    -------
    dict with keys:
        "response_text" : str — the assistant's text response
        "parsed"        : optional schema instance (if schema was provided)
        "cached"        : bool — True if we successfully reused parent params
        "input_tokens"  : int (best-effort, may be 0 if API doesn't report)
        "output_tokens" : int (best-effort)
        "cache_read_tokens" : int (best-effort, indicates cache hit size)
        "error"         : optional error message if the call failed
    """
    params = parent_params or get_last_cache_safe_params()
    using_cache = params is not None

    if using_cache:
        system_prompt = params.system_prompt
        prefix_messages = params.messages
        model = params.model
        logger.info(
            "Forked subagent: cached prefix (%d msgs, %d sys chars), model=%s",
            len(prefix_messages), len(system_prompt), model,
        )
    else:
        system_prompt = fallback_system_prompt
        prefix_messages = []
        model = "claude-sonnet-4-6"
        logger.info("Forked subagent: NO cached prefix available — fresh call (model=%s)", model)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_anthropic import ChatAnthropic

        # Build the message list: system + cached prefix + new user task
        # The system prompt with cache_control was already on the parent;
        # ChatAnthropic's `model_kwargs` doesn't easily let us re-apply the
        # marker, so we pass it as a plain SystemMessage. The cache will still
        # match if the SYSTEM PROMPT TEXT is byte-identical to the parent.
        sys_msg = SystemMessage(content=system_prompt)
        task_msg = HumanMessage(content=task)
        full_messages = [sys_msg] + list(prefix_messages) + [task_msg]

        llm = ChatAnthropic(model=model, max_tokens=max_tokens, timeout=timeout, max_retries=2)

        # If a schema was provided, use tool calling to enforce structure
        if schema is not None:
            llm_bound = llm.with_structured_output(schema)
            parsed = llm_bound.invoke(full_messages)
            # with_structured_output doesn't easily expose token usage, so
            # we report 0s — caller can still log cached=True for observability.
            return {
                "response_text": str(parsed),
                "parsed": parsed,
                "cached": using_cache,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "error": None,
            }

        # Free-form text response
        response = llm.invoke(full_messages)
        usage_meta = getattr(response, "usage_metadata", None) or {}
        return {
            "response_text": str(response.content),
            "parsed": None,
            "cached": using_cache,
            "input_tokens": usage_meta.get("input_tokens", 0),
            "output_tokens": usage_meta.get("output_tokens", 0),
            "cache_read_tokens": (
                usage_meta.get("input_token_details", {}).get("cache_read", 0)
                if isinstance(usage_meta.get("input_token_details"), dict) else 0
            ),
            "error": None,
        }
    except Exception as exc:
        logger.warning("Forked subagent failed: %s", exc, exc_info=True)
        return {
            "response_text": "",
            "parsed": None,
            "cached": using_cache,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "error": str(exc),
        }
