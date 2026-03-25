"""
mock_jira.py — Load bug tickets from JSON files instead of real Jira API.

No default sample tickets — use custom tickets from the UI or load from a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tickets(source: str | Path | None = None) -> list[dict]:
    """
    Load bug tickets from a JSON file.
    Returns empty list if no source provided.
    """
    if source:
        path = Path(source)
        if path.exists():
            return json.loads(path.read_text())
    return []


def get_ticket(ticket_id: str, source: str | Path | None = None) -> dict | None:
    """Get a specific ticket by ID."""
    for ticket in load_tickets(source):
        if ticket.get("ticket_id") == ticket_id:
            return ticket
    return None
