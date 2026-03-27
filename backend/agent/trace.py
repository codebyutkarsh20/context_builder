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
        total_tokens = sum(
            e["data"].get("prompt_tokens_approx", 0)
            for e in events if e["event_type"] == "llm_request"
        )

        last_ts = events[-1]["timestamp"] if events else 0

        return {
            "job_id": self.job_id,
            "started_at": self._start_wall,
            "total_duration_ms": round(last_ts * 1000),
            "stage_timings": stage_timings,
            "summary": {
                "total_llm_calls": total_llm,
                "total_tool_calls": total_tool,
                "total_tokens_approx": total_tokens,
                "total_events": len(events),
            },
            "events": events,
        }

    def save_report(self, path: Path):
        """Write full JSON report to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.to_report()
        path.write_text(json.dumps(report, indent=2, default=str))
