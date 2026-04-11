"""
tool_metadata.py — Tool metadata registry for the ReAct agent.

Attaches per-tool metadata (concurrency safety, read-only flag, output caps,
activity descriptions) directly to tool functions. This replaces the separate
TOOL_OUTPUT_CAPS dict in context_manager.py and enables concurrent tool batching
in react_loop.py.

Inspired by production AI coding tool patterns:
- Tools declare is_read_only and is_concurrent_safe (fail-closed defaults)
- Output caps live with tools, not in a separate dict
- Activity descriptions for human-readable progress
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata registry
# ---------------------------------------------------------------------------

# {tool_name: ToolMeta}
_TOOL_META: dict[str, "ToolMeta"] = {}


class ToolMeta:
    """Metadata for a single tool."""

    __slots__ = (
        "name", "is_read_only", "is_concurrent_safe", "max_output_chars",
        "activity_description", "phase", "skip_for_task_types",
    )

    def __init__(
        self,
        name: str,
        *,
        is_read_only: bool = False,
        is_concurrent_safe: bool = False,
        max_output_chars: int = 4000,
        activity_description: str = "",
        phase: str = "explore",
        skip_for_task_types: tuple[str, ...] = (),
    ):
        self.name = name
        self.is_read_only = is_read_only
        self.is_concurrent_safe = is_concurrent_safe
        self.max_output_chars = max_output_chars
        self.activity_description = activity_description
        self.phase = phase
        self.skip_for_task_types = skip_for_task_types


def register_tool_meta(meta: ToolMeta) -> None:
    _TOOL_META[meta.name] = meta


def get_tool_meta(tool_name: str) -> ToolMeta:
    return _TOOL_META.get(tool_name, ToolMeta(tool_name))


def get_output_cap(tool_name: str) -> int:
    return get_tool_meta(tool_name).max_output_chars


def is_concurrent_safe(tool_name: str) -> bool:
    return get_tool_meta(tool_name).is_concurrent_safe


def is_read_only(tool_name: str) -> bool:
    return get_tool_meta(tool_name).is_read_only


# ---------------------------------------------------------------------------
# Register all tools — single source of truth for metadata
# ---------------------------------------------------------------------------

def _register_all() -> None:
    """Register metadata for every tool in the system."""

    # ── Explore tools (read-only, concurrent-safe) ────────────────────────
    for meta in [
        ToolMeta("grep_repo", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=8000, activity_description="Searching codebase",
                 phase="explore"),
        ToolMeta("read_file", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=10000, activity_description="Reading file",
                 phase="explore"),
        ToolMeta("read_function", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=15000, activity_description="Reading function",
                 phase="explore"),
        ToolMeta("list_files", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=3000, activity_description="Listing directory",
                 phase="explore"),
        ToolMeta("get_function_info", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=3000, activity_description="Looking up function info",
                 phase="explore"),
        ToolMeta("get_file_structure", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=8000, activity_description="Getting file structure",
                 phase="explore"),
        ToolMeta("get_blast_radius", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=3000, activity_description="Analyzing blast radius",
                 phase="explore"),
    ]:
        register_tool_meta(meta)

    # ── Edit tools (write, NOT concurrent-safe) ───────────────────────────
    for meta in [
        ToolMeta("string_replace", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=1000, activity_description="Editing code",
                 phase="edit"),
        ToolMeta("check_syntax", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=1000, activity_description="Checking syntax",
                 phase="edit"),
        ToolMeta("create_file", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=1000, activity_description="Creating file",
                 phase="edit", skip_for_task_types=("bug_fix",)),
    ]:
        register_tool_meta(meta)

    # ── Sandbox tools ─────────────────────────────────────────────────────
    for meta in [
        ToolMeta("create_sandbox", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=1000, activity_description="Creating sandbox",
                 phase="edit"),
        ToolMeta("run_tests", is_read_only=True, is_concurrent_safe=False,
                 max_output_chars=4000, activity_description="Running tests",
                 phase="test"),
        ToolMeta("run_brt", is_read_only=True, is_concurrent_safe=False,
                 max_output_chars=4000, activity_description="Running BRT",
                 phase="test"),
    ]:
        register_tool_meta(meta)

    # ── Multi-file tools ──────────────────────────────────────────────────
    register_tool_meta(
        ToolMeta("get_callers", is_read_only=True, is_concurrent_safe=True,
                 max_output_chars=3000, activity_description="Finding callers",
                 phase="edit"),
    )

    # ── Completion tools ──────────────────────────────────────────────────
    for meta in [
        ToolMeta("record_localization", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=500, activity_description="Recording localization",
                 phase="explore"),
        ToolMeta("request_review", is_read_only=True, is_concurrent_safe=False,
                 max_output_chars=3000, activity_description="Requesting review",
                 phase="review"),
        ToolMeta("submit_fix", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=500, activity_description="Submitting fix",
                 phase="submit"),
        ToolMeta("escalate", is_read_only=False, is_concurrent_safe=False,
                 max_output_chars=500, activity_description="Escalating to human",
                 phase="submit"),
    ]:
        register_tool_meta(meta)


_register_all()
