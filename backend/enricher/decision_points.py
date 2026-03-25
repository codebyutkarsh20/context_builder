"""
decision_points.py — Extract and classify decision points from parsed code.

Pass 1 (heuristic): Classifies conditionals from code_parser output by pattern.
Pass 2 (LLM-enhanced, optional): Uses Claude to refine classification and generate
explanations + human review questions.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic classification patterns
# ---------------------------------------------------------------------------

_ROLE_PATTERNS = re.compile(
    r"\b(role|is_admin|is_superuser|is_staff|permission|has_perm|"
    r"is_authenticated|is_anonymous|user_type|access_level|privilege)\b",
    re.IGNORECASE,
)

_STATUS_PATTERNS = re.compile(
    r"\b(status|state|is_active|is_enabled|is_deleted|is_archived|"
    r"is_verified|is_approved|is_published|is_draft|is_locked|phase)\b",
    re.IGNORECASE,
)

_FEATURE_FLAG_PATTERNS = re.compile(
    r"\b(feature_flag|is_enabled|toggle|flag|experiment|ab_test|"
    r"feature_enabled|FEATURE_|is_feature)\b",
    re.IGNORECASE,
)

_ERROR_GUARD_PATTERNS = re.compile(
    r"\b(is None|is not None|is_none|not None|is True|is False|"
    r"is_valid|has_error|error|exception|raise|assert)\b",
)

_CONSTANT_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9_]*_(?:LIMIT|MAX|MIN|TIMEOUT|RATE|THRESHOLD|CAP|QUOTA|"
    r"WINDOW|PERIOD|SIZE|COUNT|RETRIES|ATTEMPTS|DELAY|INTERVAL|AGE|DAYS|HOURS)\b"
)

_NUMERIC_LITERAL = re.compile(r"\b\d+\.?\d*\b")


def classify_condition(condition_text: str) -> str:
    """Classify a conditional expression into a decision type."""
    if _CONSTANT_PATTERN.search(condition_text):
        return "threshold"
    if _ROLE_PATTERNS.search(condition_text):
        return "role_check"
    if _STATUS_PATTERNS.search(condition_text):
        return "status_check"
    if _FEATURE_FLAG_PATTERNS.search(condition_text):
        return "feature_flag"
    if _ERROR_GUARD_PATTERNS.search(condition_text):
        return "error_guard"
    # Check for hardcoded numeric thresholds (magic numbers)
    if _NUMERIC_LITERAL.search(condition_text) and any(
        op in condition_text for op in (">", "<", ">=", "<=", "==", "!=")
    ):
        return "threshold"
    return "logic_branch"


# ---------------------------------------------------------------------------
# Pass 1: Heuristic extraction
# ---------------------------------------------------------------------------

def extract_decision_points(parsed_files: list[dict]) -> list[dict]:
    """
    Extract decision points from parsed files' conditionals.

    Takes the output of CodeParser.parse_all() (which now includes
    conditionals on each function) and produces DecisionPoint records.

    Returns list of:
        {id, line, condition, condition_type, file, function_id, references_constant}
    """
    points: list[dict] = []

    for pf in parsed_files:
        rel = pf["path"]

        # Top-level functions
        for fn in pf.get("functions", []):
            func_id = f"{rel}::{fn['name']}"
            for cond in fn.get("conditionals", []):
                dp = _make_decision_point(rel, func_id, cond)
                if dp:
                    points.append(dp)

        # Class methods
        for cls in pf.get("classes", []):
            for method in cls.get("methods", []):
                func_id = f"{rel}::{cls['name']}.{method['name']}"
                for cond in method.get("conditionals", []):
                    dp = _make_decision_point(rel, func_id, cond)
                    if dp:
                        points.append(dp)

    logger.info("Extracted %d decision points from %d files", len(points), len(parsed_files))
    return points


def _make_decision_point(file_path: str, function_id: str, cond: dict) -> Optional[dict]:
    """Create a DecisionPoint record from a conditional dict."""
    condition_text = cond.get("condition_text", "")
    if not condition_text or len(condition_text) < 3:
        return None

    condition_type = classify_condition(condition_text)

    # Skip trivial error guards (too noisy)
    if condition_type == "error_guard" and cond.get("branch_count", 1) <= 1:
        return None

    dp_id = f"{function_id}::L{cond.get('line', 0)}"

    return {
        "id": dp_id,
        "line": cond.get("line", 0),
        "condition": condition_text,
        "condition_type": condition_type,
        "file": file_path,
        "function_id": function_id,
        "references_constant": cond.get("references_constant", False),
        # Populated by Pass 2 (LLM):
        "explanation": None,
        "question_for_human": None,
    }


# ---------------------------------------------------------------------------
# Pass 2: LLM-enhanced classification (optional)
# ---------------------------------------------------------------------------

def enhance_with_llm(
    decision_points: list[dict],
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 20,
) -> int:
    """
    Use Claude to refine decision point classifications and generate
    explanations + questions for human review.

    Returns the number of decision points enhanced.
    """
    import anthropic

    # Only enhance interesting decision points (thresholds, role checks)
    interesting = [
        dp for dp in decision_points
        if dp["condition_type"] in ("threshold", "role_check", "status_check", "feature_flag")
        and dp.get("explanation") is None
    ]

    if not interesting:
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    enhanced = 0

    for i in range(0, len(interesting), batch_size):
        batch = interesting[i:i + batch_size]
        prompt_lines = []
        for dp in batch:
            prompt_lines.append(
                f"- [{dp['id']}] `{dp['condition']}` "
                f"(type: {dp['condition_type']}, file: {dp['file']}, line: {dp['line']})"
            )

        prompt = (
            "You are a senior developer analyzing code decision points.\n\n"
            "For each conditional below, provide:\n"
            "1. A one-sentence explanation of what this decision controls (business impact)\n"
            "2. A question a developer should ask before changing this code\n\n"
            "Decision points:\n" + "\n".join(prompt_lines) + "\n\n"
            "Respond in this exact format for each:\n"
            "[id] explanation: <one sentence> | question: <one question>\n"
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            # Parse response
            for line in text.strip().splitlines():
                line = line.strip()
                if not line.startswith("["):
                    continue
                # Extract [id] explanation: ... | question: ...
                bracket_end = line.find("]")
                if bracket_end == -1:
                    continue
                dp_id = line[1:bracket_end]
                rest = line[bracket_end + 1:].strip()

                explanation = ""
                question = ""
                if "explanation:" in rest and "| question:" in rest:
                    parts = rest.split("| question:")
                    explanation = parts[0].replace("explanation:", "").strip()
                    question = parts[1].strip() if len(parts) > 1 else ""

                # Find and update the matching decision point
                for dp in batch:
                    if dp["id"] == dp_id:
                        dp["explanation"] = explanation[:300]
                        dp["question_for_human"] = question[:300]
                        enhanced += 1
                        break

        except Exception as e:
            logger.warning("LLM decision point enhancement failed: %s", e)
            continue

    logger.info("Enhanced %d decision points with LLM explanations", enhanced)
    return enhanced
