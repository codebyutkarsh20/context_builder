"""
query_analyzer.py — Lightweight query intent extraction using regex patterns.
No LLM calls — keeps retrieval fast (<1ms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QueryIntent:
    """Parsed intent from a user question."""
    entity_types: list[str] = field(default_factory=lambda: ["file", "function", "class"])
    mentioned_names: list[str] = field(default_factory=list)
    scope: str = "broad"  # "broad" | "specific" | "architectural"
    relationship_focus: list[str] = field(default_factory=list)


# Patterns that suggest specific entity types
_FILE_HINTS = re.compile(r"\b(file|module|script|\.py|directory|folder|path)\b", re.IGNORECASE)
_CLASS_HINTS = re.compile(r"\b(class|model|schema|service|repository|handler|controller|entity)\b", re.IGNORECASE)
_FUNC_HINTS = re.compile(r"\b(function|method|endpoint|handler|api|route|def )\b", re.IGNORECASE)
_RULE_HINTS = re.compile(r"\b(rule|business|policy|constraint|validation|limit|threshold|requirement)\b", re.IGNORECASE)
_ARCH_HINTS = re.compile(r"\b(architecture|design|pattern|flow|pipeline|structure|overview|how does.*work)\b", re.IGNORECASE)
_CALL_HINTS = re.compile(r"\b(call|invoke|depend|import|use|connect|interact)\b", re.IGNORECASE)

# Extract quoted names or CamelCase/snake_case identifiers
_QUOTED_NAME = re.compile(r'["`\']([\w._]+)["`\']')
_CAMEL_NAME = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_SNAKE_NAME = re.compile(r"\b([a-z]+(?:_[a-z]+){2,})\b")


def analyze_query(question: str) -> QueryIntent:
    """Extract intent from a natural language question about code."""
    intent = QueryIntent()

    # Determine entity types of interest
    types = set()
    if _FILE_HINTS.search(question):
        types.add("file")
    if _CLASS_HINTS.search(question):
        types.add("class")
    if _FUNC_HINTS.search(question):
        types.add("function")
    if _RULE_HINTS.search(question):
        types.add("business_rule")
        types.add("decision_point")

    if types:
        intent.entity_types = sorted(types)
    # else: keep default ["file", "function", "class"]

    # Extract mentioned names
    names = set()
    for m in _QUOTED_NAME.finditer(question):
        names.add(m.group(1))
    for m in _CAMEL_NAME.finditer(question):
        names.add(m.group(1))
    for m in _SNAKE_NAME.finditer(question):
        names.add(m.group(1))

    # Also extract simple keywords (2+ chars, not stop words)
    _STOP = {"the", "and", "for", "how", "does", "what", "which", "where", "this", "that",
             "with", "from", "about", "are", "was", "were", "been", "have", "has", "will",
             "can", "could", "should", "would", "all", "any", "each", "every", "some",
             "not", "but", "or", "if", "then", "else", "when", "than", "too", "very",
             "just", "only", "also", "here", "there", "now", "into", "over", "more"}
    for word in question.lower().split():
        clean = re.sub(r"[^a-z0-9_]", "", word)
        if len(clean) >= 3 and clean not in _STOP:
            names.add(clean)

    intent.mentioned_names = sorted(names)

    # Determine scope.
    # Architectural hints take priority: a question like "how does the pipeline
    # architecture work?" has long names but is clearly architectural in intent.
    if _ARCH_HINTS.search(question):
        intent.scope = "architectural"
    elif names and any(len(n) > 5 for n in names):
        intent.scope = "specific"
    else:
        intent.scope = "broad"

    # Relationship focus
    if _CALL_HINTS.search(question):
        intent.relationship_focus = ["CALLS", "IMPORTS"]

    return intent
