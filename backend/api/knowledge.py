"""
knowledge.py — API for the self-enriching knowledge loop.

Steps 7, 8, 21 from the implementation guide:
- List decision point questions that need human answers
- Submit answers → create BusinessRule nodes
- Escalation feedback → specific questions → permanent storage

This is the compounding advantage: every human answer permanently enriches the agent.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QuestionItem(BaseModel):
    """A decision point question for human review."""
    id: str
    function_id: str = ""
    file: str = ""
    condition: str = ""
    condition_type: str = ""
    explanation: str = ""
    question: str = ""
    suggested_answers: list[str] = Field(default_factory=list)
    answered: bool = False
    answer: str = ""
    rule_id: str = ""


class AnswerSubmission(BaseModel):
    """Human answer to a decision point question."""
    question_id: str
    answer: str
    rule_type: str = "policy"  # legal, contractual, policy, architectural
    severity: str = "medium"   # critical, high, medium, low
    answered_by: str = ""


class BusinessRule(BaseModel):
    """A permanent business rule created from a human answer."""
    id: str = ""
    description: str
    rule_type: str = "policy"
    severity: str = "medium"
    source: str = ""
    function_id: str = ""
    file: str = ""
    constraint: str = ""
    created_at: str = ""


class EscalationQuestion(BaseModel):
    """A specific question generated from an agent escalation."""
    ticket_id: str = ""
    question: str
    context: str = ""
    function_ids: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Storage helpers (JSON-based, works without Neo4j)
# ---------------------------------------------------------------------------

def _rules_path(repo: str) -> Path:
    return _DATA_DIR / repo / "business_rules.json"


def _load_rules(repo: str) -> list[dict]:
    p = _rules_path(repo)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def _save_rules(repo: str, rules: list[dict]):
    p = _rules_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rules, indent=2, default=str))


def _answers_path(repo: str) -> Path:
    return _DATA_DIR / repo / "answered_questions.json"


def _load_answers(repo: str) -> dict[str, dict]:
    p = _answers_path(repo)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_answers(repo: str, answers: dict[str, dict]):
    p = _answers_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(answers, indent=2, default=str))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/knowledge/{repo}/questions")
def list_questions(
    repo: str,
    unanswered_only: bool = Query(False, description="Only show unanswered questions"),
    limit: int = Query(50, le=200),
):
    """List decision point questions that need human answers (Step 7)."""
    graph_path = _DATA_DIR / repo / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Repository '{repo}' not found")

    try:
        data = json.loads(graph_path.read_text())
    except Exception:
        raise HTTPException(status_code=422, detail=f"Repository '{repo}' has a corrupted graph.json file")
    decision_points = data.get("decision_points", [])
    answers = _load_answers(repo)

    def _make_question(dp: dict) -> str:
        """Generate a human-readable question from a decision point."""
        q = dp.get("question_for_human") or dp.get("question", "")
        if q:
            return q
        condition = dp.get("condition", "")
        ctype = dp.get("condition_type", "")
        func = dp.get("function_id", "").split("::")[-1] if "::" in dp.get("function_id", "") else ""
        if ctype == "threshold":
            return f"What is the business reason for the threshold check `{condition}` in `{func}`?"
        if ctype == "role_check":
            return f"What access control policy does `{condition}` enforce in `{func}`?"
        if ctype == "null_check":
            return f"What should happen when `{condition}` is true in `{func}`? Is this an error, a valid state, or something else?"
        if condition:
            return f"What is the business intent behind `{condition}` in `{func}`?"
        return ""

    # Prioritise meaningful condition types
    _PRIORITY = {"threshold": 0, "role_check": 1, "null_check": 2, "logic_branch": 3}

    questions: list[dict] = []
    for dp in decision_points:
        q = _make_question(dp)
        if not q:
            continue

        qid = dp.get("id", f"{dp.get('function_id', '')}:{dp.get('line', 0)}")
        is_answered = qid in answers

        if unanswered_only and is_answered:
            continue

        item = {
            "id": qid,
            "function_id": dp.get("function_id", ""),
            "file": dp.get("file", dp.get("function_id", "").split("::")[0] if "::" in dp.get("function_id", "") else ""),
            "condition": dp.get("condition", ""),
            "condition_type": dp.get("condition_type", dp.get("type", "")),
            "explanation": dp.get("explanation", ""),
            "question": q,
            "suggested_answers": dp.get("suggested_answers", []),
            "answered": is_answered,
            "answer": answers.get(qid, {}).get("answer", ""),
            "rule_id": answers.get(qid, {}).get("rule_id", ""),
        }
        questions.append(item)

    # Sort: unanswered first, then by priority type, then by file
    questions.sort(key=lambda x: (x["answered"], _PRIORITY.get(x["condition_type"], 9), x["file"]))
    return questions[:limit]


@router.post("/knowledge/{repo}/answer")
def submit_answer(repo: str, submission: AnswerSubmission):
    """Submit a human answer to a decision point question (Step 8).

    This creates a permanent BusinessRule and links it to the code.
    The agent will see this rule in every future fix involving this code.
    """
    graph_path = _DATA_DIR / repo / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Repository '{repo}' not found")

    # Find the decision point
    data = json.loads(graph_path.read_text())
    decision_points = data.get("decision_points", [])
    dp = None
    for d in decision_points:
        qid = d.get("id", f"{d.get('function_id', '')}:{d.get('line', 0)}")
        if qid == submission.question_id:
            dp = d
            break

    if not dp:
        raise HTTPException(status_code=404, detail=f"Question '{submission.question_id}' not found")

    # Create BusinessRule
    rule_id = f"rule_{uuid.uuid4().hex[:8]}"
    func_id = dp.get("function_id", "")
    file_path = func_id.split("::")[0] if "::" in func_id else ""
    condition = dp.get("condition", "")
    explanation = dp.get("explanation", "")

    rule = {
        "id": rule_id,
        "description": f"{explanation} — {submission.answer}",
        "rule_type": submission.rule_type,
        "severity": submission.severity,
        "source": submission.answered_by or "human_answer",
        "function_id": func_id,
        "file": file_path,
        "constraint": condition,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question_id": submission.question_id,
    }

    # Save rule
    rules = _load_rules(repo)
    rules.append(rule)
    _save_rules(repo, rules)

    # Mark question as answered
    answers = _load_answers(repo)
    answers[submission.question_id] = {
        "answer": submission.answer,
        "rule_id": rule_id,
        "answered_by": submission.answered_by,
        "answered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_answers(repo, answers)

    # Also write rule to Neo4j if connected
    try:
        from graph.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            neo4j_client.run(
                """
                MERGE (br:BusinessRule {id: $id})
                SET br.description = $description,
                    br.rule_type = $rule_type,
                    br.severity = $severity,
                    br.source = $source,
                    br.constraint = $constraint,
                    br.repo = $repo
                WITH br
                OPTIONAL MATCH (f:Function {id: $func_id})
                FOREACH (_ IN CASE WHEN f IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:GOVERNED_BY]->(br)
                )
                """,
                {
                    "id": rule_id,
                    "description": rule["description"],
                    "rule_type": rule["rule_type"],
                    "severity": rule["severity"],
                    "source": rule["source"],
                    "constraint": rule["constraint"],
                    "repo": repo,
                    "func_id": func_id,
                },
            )
            logger.info("Created BusinessRule %s in Neo4j for %s", rule_id, func_id)
    except Exception as e:
        logger.debug("Could not write BusinessRule to Neo4j: %s", e)

    # Update enriched_nodes.json so the agent picks up the new rule
    _inject_rule_into_enriched(repo, rule)

    logger.info("Answer submitted for %s → rule %s (%s)", submission.question_id, rule_id, submission.severity)
    return {"rule_id": rule_id, "status": "stored", "description": rule["description"]}


class AddRuleRequest(BaseModel):
    description: str
    rule_type: str = "policy"
    severity: str = "medium"
    file: str = ""
    constraint: str = ""
    added_by: str = ""


@router.post("/knowledge/{repo}/rules")
def add_rule(repo: str, body: AddRuleRequest):
    """Manually add a business rule directly (no question needed)."""
    rule_id = f"rule_manual_{uuid.uuid4().hex[:8]}"
    rule = {
        "id": rule_id,
        "description": body.description,
        "rule_type": body.rule_type,
        "severity": body.severity,
        "source": f"manual:{body.added_by or 'developer'}",
        "function_id": "",
        "file": body.file,
        "constraint": body.constraint or body.description,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rules = _load_rules(repo)
    rules.append(rule)
    _save_rules(repo, rules)
    _inject_rule_into_enriched(repo, rule)
    logger.info("Manual rule %s added for repo '%s'", rule_id, repo)
    return {"rule_id": rule_id, "status": "stored"}


@router.get("/knowledge/{repo}/rules")
def list_rules(repo: str, severity: Optional[str] = None):
    """List all business rules for a repo."""
    rules = _load_rules(repo)
    if severity:
        rules = [r for r in rules if r.get("severity") == severity]
    rules.sort(key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(r.get("severity", ""), 4))
    return rules


@router.post("/knowledge/{repo}/escalation-answer")
def submit_escalation_answer(repo: str, escalation: EscalationQuestion):
    """Store a human answer from an agent escalation (Step 21).

    When the agent escalates with a specific question, and the human answers,
    this endpoint stores the answer as a permanent BusinessRule.
    The agent will never need to ask this question again.
    """
    rule_id = f"rule_esc_{uuid.uuid4().hex[:8]}"
    rule = {
        "id": rule_id,
        "description": escalation.question + " — " + escalation.context,
        "rule_type": "policy",
        "severity": "high",
        "source": f"escalation:{escalation.ticket_id}",
        "function_id": escalation.function_ids[0] if escalation.function_ids else "",
        "file": escalation.files[0] if escalation.files else "",
        "constraint": escalation.question,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    rules = _load_rules(repo)
    rules.append(rule)
    _save_rules(repo, rules)

    _inject_rule_into_enriched(repo, rule)

    logger.info("Escalation answer stored as rule %s for ticket %s", rule_id, escalation.ticket_id)
    return {"rule_id": rule_id, "status": "stored"}


@router.get("/knowledge/{repo}/stats")
def knowledge_stats(repo: str):
    """Get knowledge base stats for a repo."""
    graph_path = _DATA_DIR / repo / "graph.json"
    if not graph_path.exists():
        return {"questions": 0, "answered": 0, "rules": 0, "coverage": 0}

    data = json.loads(graph_path.read_text())
    dps = [d for d in data.get("decision_points", []) if d.get("question_for_human") or d.get("question")]
    answers = _load_answers(repo)
    rules = _load_rules(repo)

    total_q = len(dps)
    answered = len(answers)
    coverage = round(answered / total_q * 100, 1) if total_q > 0 else 0

    return {
        "questions": total_q,
        "answered": answered,
        "unanswered": total_q - answered,
        "rules": len(rules),
        "coverage": coverage,
        "by_severity": {
            "critical": sum(1 for r in rules if r.get("severity") == "critical"),
            "high": sum(1 for r in rules if r.get("severity") == "high"),
            "medium": sum(1 for r in rules if r.get("severity") == "medium"),
            "low": sum(1 for r in rules if r.get("severity") == "low"),
        },
    }


@router.get("/knowledge/{repo}/test-enrichments")
def get_test_enrichments(repo: str, severity: Optional[str] = None):
    """Return business-intent test enrichments for a repo (Step 18).

    Each entry maps a test function to the business rule it verifies,
    the severity, and what it means if the test fails.
    """
    from enricher.test_enricher import load_enrichments

    enrichments = load_enrichments(repo)
    if severity:
        enrichments = [e for e in enrichments if e.get("severity") == severity]
    return enrichments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_rule_into_enriched(repo: str, rule: dict):
    """Add a business rule to enriched_nodes.json so the agent sees it."""
    enriched_path = _DATA_DIR / repo / "enriched_nodes.json"
    if not enriched_path.exists():
        return
    try:
        enriched = json.loads(enriched_path.read_text())
        enriched[rule["id"]] = {
            "id": rule["id"],
            "type": "business_rule",
            "name": rule["description"][:100],
            "file": rule.get("file", ""),
            "function_id": rule.get("function_id", ""),
            "content": rule["description"],
            "rule_type": rule["rule_type"],
            "severity": rule["severity"],
        }
        enriched_path.write_text(json.dumps(enriched, default=str))
    except Exception as e:
        logger.debug("Could not inject rule into enriched_nodes: %s", e)
