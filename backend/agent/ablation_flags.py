"""ablation_flags.py — single source of truth for which harness components are
disabled during an ablation run.

The harness's value comes from the *system* around the LLM: Scout localization,
the knowledge graph, the per-repo learning loop, the BRT generator, and the
forked verifier. To measure what each one is actually worth, we need to turn
them off one at a time and re-run the eval. That requires a toggle that every
gate site can read — but the gates live in different modules (`react_pipeline`
and `react_tools`), each with its own ``threading.local``.

This module centralizes that state. The pipeline entry point sets the disabled
set at the start of a run; each component checks ``is_disabled(name)`` at its
gate; the entry point clears the set when the run ends. State is thread-local so
parallel runs (best-of-N, concurrent eval cases) never leak flags into one
another.

Usage
-----
    from agent import ablation_flags

    ablation_flags.set_disabled({"verifier"})   # start of run
    ...
    if not ablation_flags.is_disabled("scout"): # at a gate site
        run_scout(...)
    ...
    ablation_flags.clear()                        # end of run (finally:)
"""

from __future__ import annotations

import threading

# The ablatable components. Keep this in sync with the gate sites and with
# eval/ablation.py's arm definitions.
COMPONENTS: tuple[str, ...] = ("scout", "brt", "graph", "lessons", "verifier")

_HUMAN_LABELS = {
    "scout": "Scout localization",
    "brt": "Bug-reproduction tests",
    "graph": "Knowledge graph context",
    "lessons": "Per-repo learning loop",
    "verifier": "Forked verifier",
}

_tls = threading.local()


def set_disabled(components: set[str] | frozenset[str] | list[str] | None) -> None:
    """Set the components disabled for the current thread's run.

    Unknown component names are ignored (with no error) so callers can pass a
    superset without coupling to this module's exact membership.
    """
    valid = {c for c in (components or set()) if c in COMPONENTS}
    _tls.disabled = valid


def clear() -> None:
    """Clear all ablation flags for the current thread."""
    _tls.disabled = set()


def is_disabled(name: str) -> bool:
    """Return True if ``name`` is disabled for the current thread's run."""
    return name in getattr(_tls, "disabled", frozenset())


def disabled_set() -> set[str]:
    """Return a copy of the currently-disabled components (for tracing/debug)."""
    return set(getattr(_tls, "disabled", frozenset()))


def label(name: str) -> str:
    """Human-readable label for a component name."""
    return _HUMAN_LABELS.get(name, name)
