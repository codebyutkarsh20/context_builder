"""
trace.py — Pipeline run tracing / observability.

Captures LLM prompts, tool calls, stage timings, patch candidates,
and test output into a thread-safe in-memory store that can be:
  1. Streamed live via SSE (subscribe/unsubscribe)
  2. Polled incrementally (events_since)
  3. Saved to disk as a JSON report (save_report)
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ── Secret redaction (matches pipeline.py pattern) ──────────────────────────

_SECRETS_RE = re.compile(
    r'(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|'
    r'secret[_-]?key|password|passwd|private[_-]?key|credentials)'
    r'\s*[=:]\s*["\']?[A-Za-z0-9+/=_\-]{16,}["\']?'
)


def _redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _SECRETS_RE.sub("[REDACTED]", text)


def _redact_dict(d: dict) -> dict:
    """Deep-redact string values in a dict."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = _redact(v)
        elif isinstance(v, dict):
            out[k] = _redact_dict(v)
        elif isinstance(v, list):
            out[k] = [_redact(i) if isinstance(i, str) else i for i in v]
        else:
            out[k] = v
    return out


# ── Types ───────────────────────────────────────────────────────────────────

TraceEventType = Literal[
    "stage_start",
    "stage_end",
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "guardrail_event",
    "state_transition",
    "context_compaction",
    "prompt_build",
    "run_outcome",
    "patch_candidate",
    "test_output",
    "error",
    "info",
]


@dataclass
class TraceEvent:
    timestamp: float          # time.monotonic() offset from run start
    wall_time: str            # ISO 8601
    event_type: TraceEventType
    stage: str                # e.g. "intake", "repair", "exploration"
    data: dict[str, Any] = field(default_factory=dict)
    index: int = 0            # auto-set by RunTrace.emit()

    def to_dict(self) -> dict:
        return asdict(self)


# ── Sentinel ────────────────────────────────────────────────────────────────

_DONE_SENTINEL = None  # pushed into subscriber queues when trace completes


# ── RunTrace ────────────────────────────────────────────────────────────────

class RunTrace:
    """Thread-safe per-run trace collector with live SSE support."""

    def __init__(self, job_id: str, enabled: bool = True):
        self.job_id = job_id
        self.enabled = enabled
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._start_mono = time.monotonic()
        self._start_wall = datetime.now(timezone.utc).isoformat()
        self._completed = False
        self._stage_starts: dict[str, float] = {}  # stage -> monotonic start

    # ── Emit ────────────────────────────────────────────────────────────────

    def emit(self, event_type: TraceEventType, stage: str, data: dict[str, Any] | None = None):
        """Record a trace event and notify all SSE subscribers."""
        if not self.enabled:
            return

        safe_data = _redact_dict(data or {})

        evt = TraceEvent(
            timestamp=round(time.monotonic() - self._start_mono, 3),
            wall_time=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            stage=stage,
            data=safe_data,
        )

        with self._lock:
            evt.index = len(self._events)
            self._events.append(evt)
            for q in self._subscribers:
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    pass  # subscriber is slow — skip

    # ── Stage timing helpers ────────────────────────────────────────────────

    def stage_start(self, stage: str):
        """Emit stage_start and record start time for duration calc."""
        self._stage_starts[stage] = time.monotonic()
        self.emit("stage_start", stage, {"stage": stage})

    def stage_end(self, stage: str):
        """Emit stage_end with duration_ms."""
        start = self._stage_starts.pop(stage, None)
        duration_ms = round((time.monotonic() - start) * 1000) if start else 0
        self.emit("stage_end", stage, {"stage": stage, "duration_ms": duration_ms})

    # ── SSE subscribe/unsubscribe ───────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        """Create a new subscriber queue for SSE streaming."""
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def events_since(self, idx: int) -> list[dict]:
        """Return events from index onward (for SSE catchup)."""
        with self._lock:
            return [e.to_dict() for e in self._events[idx:]]

    # ── Completion ──────────────────────────────────────────────────────────

    def complete(self):
        """Mark trace as done — sends sentinel to all subscribers."""
        self._completed = True
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(_DONE_SENTINEL)
                except queue.Full:
                    pass

    @property
    def is_completed(self) -> bool:
        return self._completed

    # ── Report ──────────────────────────────────────────────────────────────

    def to_report(self) -> dict:
        """Build a full JSON-serializable run report."""
        with self._lock:
            events = [e.to_dict() for e in self._events]

        # Compute stage timings
        stage_timings: dict[str, dict] = {}
        for evt in events:
            if evt["event_type"] == "stage_end":
                stage = evt["data"].get("stage", evt["stage"])
                duration = evt["data"].get("duration_ms", 0)
                if stage not in stage_timings:
                    stage_timings[stage] = {"duration_ms": 0, "llm_calls": 0, "tool_calls": 0}
                stage_timings[stage]["duration_ms"] = duration

        # Count LLM/tool calls per stage
        for evt in events:
            stage = evt["stage"]
            if stage not in stage_timings:
                stage_timings[stage] = {"duration_ms": 0, "llm_calls": 0, "tool_calls": 0}
            if evt["event_type"] == "llm_request":
                stage_timings[stage]["llm_calls"] += 1
            elif evt["event_type"] == "tool_call":
                stage_timings[stage]["tool_calls"] += 1

        total_llm = sum(1 for e in events if e["event_type"] == "llm_request")
        total_tool = sum(1 for e in events if e["event_type"] == "tool_call")
        total_tokens_approx = sum(
            e["data"].get("prompt_tokens_approx", 0)
            for e in events if e["event_type"] == "llm_request"
        )

        # Actual token usage from llm_response events
        total_input_tokens = sum(
            e["data"].get("input_tokens", 0)
            for e in events if e["event_type"] == "llm_response"
        )
        total_output_tokens = sum(
            e["data"].get("output_tokens", 0)
            for e in events if e["event_type"] == "llm_response"
        )
        total_cost_usd = sum(
            e["data"].get("cost_usd", 0.0)
            for e in events if e["event_type"] == "llm_response"
        )

        # Per-stage token breakdown
        stage_tokens: dict[str, dict] = {}
        for e in events:
            if e["event_type"] == "llm_response":
                stage = e.get("stage", e["data"].get("stage", "unknown"))
                if stage not in stage_tokens:
                    stage_tokens[stage] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
                stage_tokens[stage]["input_tokens"] += e["data"].get("input_tokens", 0)
                stage_tokens[stage]["output_tokens"] += e["data"].get("output_tokens", 0)
                stage_tokens[stage]["cost_usd"] += e["data"].get("cost_usd", 0.0)
                stage_tokens[stage]["calls"] += 1

        last_ts = events[-1]["timestamp"] if events else 0

        # Phase breakdown: classify tool calls into explore/edit/test/review/submit
        phase_stats = _compute_phase_stats(events)

        # Context window timeline: token count at each LLM call
        context_timeline = [
            {"call": e["data"].get("tool_call_count", i), "tokens": e["data"].get("context_tokens", 0)}
            for i, e in enumerate(events)
            if e["event_type"] == "llm_request" and e["data"].get("context_tokens")
        ]

        # Wasted call detection
        wasted = _detect_wasted_calls(events)

        # Guardrail events
        guardrail_blocks = [
            e for e in events if e["event_type"] == "guardrail_event"
        ]

        # Run outcome (if emitted)
        outcome_events = [e for e in events if e["event_type"] == "run_outcome"]
        run_outcome = outcome_events[-1]["data"] if outcome_events else {}

        return {
            "job_id": self.job_id,
            "started_at": self._start_wall,
            "total_duration_ms": round(last_ts * 1000),
            "stage_timings": stage_timings,
            "summary": {
                "total_llm_calls": total_llm,
                "total_tool_calls": total_tool,
                "total_tokens_approx": total_tokens_approx,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_cost_usd": round(total_cost_usd, 6),
                "total_events": len(events),
            },
            "token_usage_by_stage": stage_tokens,
            "phase_breakdown": phase_stats,
            "context_timeline": context_timeline,
            "wasted_calls": wasted,
            "guardrail_events": guardrail_blocks,
            "run_outcome": run_outcome,
            "events": events,
        }

    def save_report(self, path: Path):
        """Write full JSON report to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.to_report()
        path.write_text(json.dumps(report, indent=2, default=str))


# ── Report helpers ─────────────────────────────────────────────────────────

# Tool → phase mapping
_PHASE_MAP = {
    # Explore
    "grep_repo": "explore", "read_file": "explore", "read_function": "explore",
    "list_files": "explore", "search_code": "explore", "get_function_info": "explore",
    "get_file_structure": "explore", "get_file_summary": "explore",
    "record_localization": "explore",
    # Edit
    "create_sandbox": "edit", "string_replace": "edit", "check_syntax": "edit",
    "create_file": "edit", "get_callers": "edit", "get_blast_radius": "edit",
    # Test
    "run_tests": "test",
    # Review
    "request_review": "review",
    # Submit
    "submit_fix": "submit", "escalate": "submit",
}


def _compute_phase_stats(events: list[dict]) -> dict:
    """Compute per-phase tool call counts and cost."""
    phases: dict[str, dict] = {}
    for e in events:
        if e["event_type"] != "tool_call":
            continue
        tool_name = e["data"].get("tool_name", "")
        phase = _PHASE_MAP.get(tool_name, "other")
        if phase not in phases:
            phases[phase] = {"tool_calls": 0, "tools_used": []}
        phases[phase]["tool_calls"] += 1
        if tool_name not in phases[phase]["tools_used"]:
            phases[phase]["tools_used"].append(tool_name)

    # Add first-call-in-phase timestamps
    seen_phases: set[str] = set()
    for e in events:
        if e["event_type"] != "tool_call":
            continue
        tool_name = e["data"].get("tool_name", "")
        phase = _PHASE_MAP.get(tool_name, "other")
        if phase not in seen_phases:
            seen_phases.add(phase)
            if phase in phases:
                phases[phase]["first_call_at"] = e["timestamp"]
                phases[phase]["first_call_number"] = e["data"].get("call_number", 0)

    return phases


def _detect_wasted_calls(events: list[dict]) -> dict:
    """Detect patterns that suggest wasted tool calls."""
    tool_calls = [e for e in events if e["event_type"] == "tool_call"]

    # Repeated reads of the same file
    read_targets: dict[str, int] = {}
    for tc in tool_calls:
        name = tc["data"].get("tool_name", "")
        if name in ("read_file", "read_function"):
            target = tc["data"].get("args", {}).get("file_path", "")
            if target:
                read_targets[target] = read_targets.get(target, 0) + 1
    repeated_reads = {f: c for f, c in read_targets.items() if c > 2}

    # Grep spam (5+ consecutive greps)
    grep_streaks = 0
    max_grep_streak = 0
    for tc in tool_calls:
        if tc["data"].get("tool_name") == "grep_repo":
            grep_streaks += 1
            max_grep_streak = max(max_grep_streak, grep_streaks)
        else:
            grep_streaks = 0

    # Test retry loops
    test_count = sum(1 for tc in tool_calls if tc["data"].get("tool_name") == "run_tests")

    return {
        "repeated_reads": repeated_reads,
        "max_grep_streak": max_grep_streak,
        "test_attempts": test_count,
        "total_tool_calls": len(tool_calls),
    }
