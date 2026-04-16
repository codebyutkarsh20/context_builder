"""
vaguify.py — Convert technical SWE-bench descriptions into vague product-ticket style.

A user/PM filing a bug doesn't know file paths, function names, or internal
terminology. They describe symptoms. This converter rewrites technical bug
reports to match that real-world style — testing whether our agent can
localize from symptoms alone.

What gets stripped:
- File paths (src/foo.py, astropy/modeling/separable.py)
- Function/class names (separability_matrix, resolve_redirects)
- Code blocks (```python ... ```)
- Stack traces
- Specific error messages that leak internal names
- Technical jargon that a non-dev wouldn't use

What stays:
- The observable symptom from the user's perspective
- What they did, what they expected, what they got
- Relevant domain context (no leak of internals)

Usage:
    python -m agent.eval.vaguify --input eval/swebench_50.json --output eval/swebench_50_vague.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_VAGUIFY_MODEL = os.environ.get("VAGUIFY_MODEL", "claude-haiku-4-5-20251001")

_VAGUIFY_PROMPT = """You are rewriting a technical bug report as a vague, realistic product ticket \
that a non-developer (PM, support agent, or end-user) would file.

Your rewrite MUST:
- Describe the OBSERVABLE SYMPTOM from the user's perspective
- Use plain product/business language
- NOT mention any file paths, function names, class names, method names, or module names
- NOT include code blocks, stack traces, or specific error messages
- NOT use library/framework-specific terminology (e.g. don't say "CompoundModel" or "Session.resolve_redirects")
- Feel like a real Jira/GitHub ticket from a user who doesn't know the codebase
- Be 2-5 sentences, ending with what they expected vs what they got
- Keep enough DOMAIN context that a skilled engineer could still investigate (e.g. "when I do X, Y happens")

Format your response as JUST the rewritten description. No preamble, no headers, no markdown — just the text.

=== ORIGINAL BUG REPORT ===
Title: {title}

{description}

=== YOUR REWRITTEN VAGUE TICKET ==="""


def vaguify_one(title: str, description: str, timeout: float = 30.0) -> str:
    """Rewrite one bug description as vague. Returns the rewritten text or empty on failure."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        prompt = _VAGUIFY_PROMPT.format(title=title[:200], description=description[:3000])
        llm = ChatAnthropic(
            model=_VAGUIFY_MODEL,
            max_tokens=500,
            timeout=timeout,
            max_retries=1,
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = str(resp.content) if not isinstance(resp.content, list) else " ".join(
            str(b.get("text", "")) for b in resp.content if isinstance(b, dict)
        )
        return (text or "").strip()
    except Exception as e:
        logger.warning("Vaguify failed for '%s': %s", title[:50], e)
        return ""


def vaguify_dataset(input_path: Path, output_path: Path, limit: int = 0) -> None:
    """Process a bugs.json, adding nl_description (vague) to each bug."""
    bugs = json.loads(input_path.read_text())
    if limit:
        bugs = bugs[:limit]

    total = len(bugs)
    logger.info("Vaguifying %d bugs (model=%s)", total, _VAGUIFY_MODEL)

    for i, bug in enumerate(bugs, 1):
        # Skip if already has nl_description
        if bug.get("nl_description"):
            logger.info("[%d/%d] Skipping %s (already has nl_description)", i, total, bug["ticket_id"])
            continue

        title = bug.get("title", "")
        description = bug.get("description", "")
        if not description:
            continue

        vague = vaguify_one(title, description)
        if vague:
            bug["nl_description"] = vague
            # Also add _natural_lang flag so the agent knows to start from structure
            bug.setdefault("_natural_lang", True)
            logger.info("[%d/%d] %s → %s", i, total, bug["ticket_id"], vague[:80])
        else:
            logger.warning("[%d/%d] %s: vaguify returned empty", i, total, bug["ticket_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bugs, indent=2))
    logger.info("Wrote %d bugs to %s", len(bugs), output_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Vaguify SWE-bench descriptions")
    parser.add_argument("--input", required=True, help="Input bugs JSON")
    parser.add_argument("--output", required=True, help="Output bugs JSON (with nl_description)")
    parser.add_argument("--limit", type=int, default=0, help="Max bugs (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Only show 1 example, don't write")
    args = parser.parse_args()

    if args.dry_run:
        bugs = json.loads(Path(args.input).read_text())
        bug = bugs[0]
        print(f"Original ({bug['ticket_id']}):")
        print(bug.get("description", "")[:500])
        print("\n" + "=" * 70)
        print("Vague version:")
        vague = vaguify_one(bug.get("title", ""), bug.get("description", ""))
        print(vague)
        sys.exit(0)

    vaguify_dataset(Path(args.input), Path(args.output), limit=args.limit)
