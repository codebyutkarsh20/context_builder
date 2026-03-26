"""
domain_concepts.py — Extract domain concepts from class names, module names, and docstrings.

Pass 1 (heuristic): CamelCase splitting, module/directory name analysis.
Pass 2 (LLM-enhanced, optional): Uses Claude for semantic domain identification.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise words to skip when extracting domain concepts
# ---------------------------------------------------------------------------

_NOISE_WORDS = {
    "base", "abstract", "mixin", "model", "schema", "serializer", "view",
    "viewset", "router", "handler", "controller", "service", "repository",
    "factory", "builder", "manager", "util", "utils", "helper", "helpers",
    "config", "configuration", "settings", "test", "tests", "mock", "fake",
    "stub", "fixture", "migration", "admin", "api", "app", "main", "init",
    "setup", "common", "core", "generic", "custom", "default", "internal",
    "external", "client", "server", "request", "response", "middleware",
    "exception", "error", "errors", "enum", "enums", "type", "types",
    "interface", "protocol", "abc", "meta", "celery", "task", "tasks",
    "command", "commands", "signal", "signals", "form", "forms", "field",
    "fields", "widget", "widgets", "template", "templates", "context",
    "processor", "pipeline", "step", "stage", "phase",
}

# Suffixes that indicate the word is a concept modifier, not a concept itself
_TYPE_SUFFIXES = {
    "Service": "process",
    "Repository": "entity",
    "Controller": "process",
    "Handler": "process",
    "Manager": "process",
    "Factory": "process",
    "Builder": "process",
    "Validator": "process",
    "Serializer": "entity",
    "Schema": "entity",
    "Model": "entity",
    "DTO": "value_object",
    "Enum": "value_object",
    "Event": "event",
    "Command": "process",
    "Query": "process",
}


# ---------------------------------------------------------------------------
# CamelCase splitting
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_camel(name: str) -> list[str]:
    """Split CamelCase into individual words."""
    return [w for w in _CAMEL_RE.sub(" ", name).split() if w]


# ---------------------------------------------------------------------------
# Pass 1: Heuristic extraction
# ---------------------------------------------------------------------------

def extract_domain_concepts(parsed_files: list[dict]) -> list[dict]:
    """
    Extract domain concepts from class names and module paths.

    Returns list of:
        {id, name, type, description, related_classes, related_files}
    """
    # Collect all class names and their files
    class_info: list[dict] = []
    for pf in parsed_files:
        rel = pf.get("path")
        if not rel:
            continue
        for cls in pf.get("classes", []):
            cls_name = cls.get("name")
            if not cls_name:
                continue
            class_info.append({
                "name": cls_name,
                "file": rel,
                "docstring": cls.get("docstring", ""),
                "bases": cls.get("bases", []),
                "methods": [m.get("name") for m in cls.get("methods", []) if m.get("name")],
            })

    # Extract concept candidates from class names
    concept_counts: Counter = Counter()
    concept_classes: defaultdict[str, list[str]] = defaultdict(list)
    concept_files: defaultdict[str, set] = defaultdict(set)
    concept_type: dict[str, str] = {}

    for ci in class_info:
        words = _split_camel(ci["name"])
        if not words:
            continue

        # Determine the domain concept vs the type suffix
        suffix = words[-1] if len(words) > 1 else None
        inferred_type = _TYPE_SUFFIXES.get(suffix, "entity") if suffix else "entity"

        # The concept is the non-suffix words joined
        if suffix and suffix in _TYPE_SUFFIXES:
            concept_words = words[:-1]
        else:
            concept_words = words

        concept_name = "".join(concept_words)

        # Skip noise
        if concept_name.lower() in _NOISE_WORDS or len(concept_name) < 3:
            continue

        concept_counts[concept_name] += 1
        concept_classes[concept_name].append(ci["name"])
        concept_files[concept_name].add(ci["file"])

        if concept_name not in concept_type:
            concept_type[concept_name] = inferred_type

    # Also extract from module/directory names
    module_concepts: Counter = Counter()
    for pf in parsed_files:
        path = pf.get("path")
        if not path:
            continue
        parts = PurePosixPath(path).parts
        for part in parts:
            name = part.replace(".py", "").replace("_", " ").title().replace(" ", "")
            if name.lower() not in _NOISE_WORDS and len(name) >= 3:
                module_concepts[name] += 1

    # Merge: boost concepts that appear in both class names and module names
    for name in module_concepts:
        if name in concept_counts:
            concept_counts[name] += module_concepts[name]
        elif module_concepts[name] >= 2:
            # Module-only concept (appears in 2+ file paths)
            concept_counts[name] = module_concepts[name]
            concept_type[name] = "entity"

    # Build result — only concepts referenced by 2+ classes or appearing in 2+ files
    concepts: list[dict] = []
    for name, count in concept_counts.most_common(50):
        classes = concept_classes.get(name, [])
        files = concept_files.get(name, set())
        if count < 2 and len(classes) < 2 and len(files) < 2:
            continue

        concepts.append({
            "id": f"domain::{name.lower()}",
            "name": name,
            "type": concept_type.get(name, "entity"),
            "description": None,  # populated by LLM pass
            "related_classes": sorted(set(classes))[:10],
            "related_files": sorted(files)[:10],
        })

    logger.info("Extracted %d domain concepts from %d classes", len(concepts), len(class_info))
    return concepts


# ---------------------------------------------------------------------------
# Pass 2: LLM-enhanced (optional)
# ---------------------------------------------------------------------------

def enhance_with_llm(
    concepts: list[dict],
    parsed_files: list[dict],
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> int:
    """
    Use Claude to generate descriptions for domain concepts.

    Returns the number of concepts enhanced.
    """
    import anthropic

    to_enhance = [c for c in concepts if c.get("description") is None]
    if not to_enhance:
        return 0

    # Build a compact summary of all classes for context
    class_summary_lines = []
    for pf in parsed_files:
        for cls in pf.get("classes", []):
            doc = (cls.get("docstring") or "")[:100]
            methods = ", ".join(m["name"] for m in cls.get("methods", [])[:5])
            class_summary_lines.append(
                f"- {cls['name']} ({pf['path']}): {doc} | methods: {methods}"
            )

    prompt = (
        "You are a domain expert analyzing a codebase.\n\n"
        "Given these classes:\n"
        + "\n".join(class_summary_lines[:100])  # cap at 100 classes
        + "\n\nIdentify the business domain concepts. "
        "For each concept below, write ONE sentence describing what it represents in the business domain.\n\n"
        "Concepts to describe:\n"
    )
    for c in to_enhance[:30]:  # cap at 30
        prompt += f"- {c['name']} (classes: {', '.join(c['related_classes'][:5])})\n"

    prompt += "\nRespond in format:\nConceptName: description\n"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""

        enhanced = 0
        for line in text.strip().splitlines():
            if ":" not in line:
                continue
            name, _, desc = line.partition(":")
            name = name.strip()
            desc = desc.strip()
            # Match to concept
            for c in to_enhance:
                if c["name"].lower() == name.lower():
                    c["description"] = desc[:300]
                    enhanced += 1
                    break

        logger.info("Enhanced %d domain concepts with LLM descriptions", enhanced)
        return enhanced

    except Exception as e:
        logger.warning("LLM domain concept enhancement failed: %s", e)
        return 0
