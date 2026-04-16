"""
scoring.py — Unified scoring for eval pipeline runs.

Ports 5 existing metrics from eval_suite.py and adds:
  - patch_correctness (file-level overlap with ground truth)
  - multi_file_complete (all expected_patch_files covered)
  - test_pass (repo tests passed after patch)
  - patch_hits_target (patches touch expected files)
  - cost, duration, tool_call tracking
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_test_file(path: str) -> bool:
    """Return True if path looks like a test file."""
    parts = path.lower().split("/")
    basename = parts[-1] if parts else ""
    return (
        basename.startswith("test_")
        or basename.endswith("_test.py")
        or "tests/" in path.lower()
        or "/test/" in path.lower()
        or basename.startswith("test.")
    )


# ---------------------------------------------------------------------------
# Individual scoring functions (ported from eval_suite.py:59-101)
# ---------------------------------------------------------------------------

def _score_localization_hit(result: dict, bug: dict) -> bool:
    """Did the agent find the right file(s)?

    Primary: reads result["localization"]["fault_files"] (set by record_localization).
    v4 fallback: infers from edited files when record_localization not called.
    """
    localization = result.get("localization") or {}
    fault_files = localization.get("fault_files") or []

    # v4 fallback: infer from edited files when record_localization not called
    if not fault_files:
        patches = (result.get("repair") or {}).get("patches") or []
        edited_files = [p.get("file_path", "") for p in patches if p.get("file_path")]
        # Filter out test files — localization means finding the BUG, not the test
        fault_files = [
            f for f in edited_files
            if not _is_test_file(f)
        ]

    if not fault_files:
        return False

    expected = bug.get("expected_files") or []
    if not expected:
        return bool(fault_files)  # If no expected files, any edit counts

    # Check overlap: any fault file matches any expected file
    # Uses suffix matching to tolerate path prefix differences
    # (e.g. "full/path/to/src/module.py" matches "src/module.py")
    for exp in expected:
        exp_lower = exp.strip("/").lower()
        for found in fault_files:
            found_lower = found.strip("/").lower()
            if exp_lower in found_lower or found_lower.endswith(exp_lower):
                return True
    return False


def _score_root_cause_match(result: dict, expected_root_cause: str) -> bool:
    """Does the localization hypothesis mention the expected root-cause keywords?

    Requires at least 40% of keywords to appear in the hypothesis.
    """
    localization = result.get("localization") or {}
    hypothesis = (localization.get("root_cause_hypothesis") or "").lower()
    if not hypothesis or not expected_root_cause:
        return False

    keywords = expected_root_cause.lower().split()
    matches = sum(1 for kw in keywords if kw in hypothesis)
    threshold = max(1, len(keywords) * 0.4)
    return matches >= threshold


def _score_fix_generated(result: dict) -> bool:
    """Were patches produced?"""
    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    return len(patches) > 0


def _score_review_approved(result: dict) -> bool:
    """Did the reviewer approve?"""
    review = result.get("review") or {}
    return review.get("verdict", "").upper() == "APPROVE"


def _get_review_confidence(result: dict) -> float:
    """Extract the review confidence score.

    Falls back to verifier_confidence (set by the pipeline forked verifier)
    if the review dict's confidence is missing or 0 — this guards against
    the request_review tool not parsing confidence into the review dict.
    """
    review = result.get("review") or {}
    conf = float(review.get("confidence", 0.0) or 0.0)
    if conf == 0.0:
        conf = float(result.get("verifier_confidence", 0.0) or 0.0)
    return conf


# ---------------------------------------------------------------------------
# New scoring functions
# ---------------------------------------------------------------------------

def _score_patch_correctness(result: dict, bug: dict) -> float:
    """File-level overlap between agent's patches and ground truth.

    Score 0.0-1.0:
      1.0 = agent patched exactly the right files
      0.5 = partial overlap (some right, some extra or missing)
      0.0 = no overlap with expected files

    Deliberately uses file-level, not line-level comparison.
    Correct fixes can differ syntactically from ground truth.
    """
    expected = set(f.lower() for f in bug.get("expected_patch_files", bug.get("expected_files", [])))
    if not expected:
        return 1.0  # No ground truth to compare against

    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    patched = set()
    for p in patches:
        fp = (p.get("file_path") or "").lower()
        if fp:
            # Normalize: strip test files from the patched set for correctness scoring
            # (agent may add tests, which is good but not a "fix" file)
            if not fp.startswith("test") and "/test" not in fp:
                patched.add(fp)

    if not patched:
        return 0.0

    # Jaccard-like: intersection / union
    intersection = 0
    for exp in expected:
        for pat in patched:
            if exp in pat or pat.endswith(exp) or pat in exp:
                intersection += 1
                break

    union = len(expected | patched)
    if union == 0:
        return 0.0

    return round(intersection / max(len(expected), len(patched)), 4)


def _score_ground_truth_file_match(result: dict, bug: dict) -> dict:
    """Detailed ground truth comparison at file level.

    Returns a dict with:
      - matched_files: files correctly identified and patched
      - missing_files: expected files not patched
      - extra_files: files patched but not in ground truth
      - file_precision: matched / (matched + extra)
      - file_recall: matched / (matched + missing)
      - file_f1: harmonic mean of precision and recall
    """
    expected = set(f.lower() for f in bug.get("expected_patch_files", bug.get("expected_files", [])))

    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    patched = set()
    for p in patches:
        fp = (p.get("file_path") or "").lower()
        if fp and not fp.startswith("test") and "/test" not in fp:
            patched.add(fp)

    matched = set()
    for exp in expected:
        for pat in patched:
            if exp in pat or pat.endswith(exp) or pat in exp:
                matched.add(exp)
                break

    missing = expected - matched
    extra = patched - {p for p in patched if any(
        e in p or p.endswith(e) or p in e for e in expected
    )}

    precision = len(matched) / (len(matched) + len(extra)) if (len(matched) + len(extra)) > 0 else 0.0
    recall = len(matched) / (len(matched) + len(missing)) if (len(matched) + len(missing)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "matched_files": sorted(matched),
        "missing_files": sorted(missing),
        "extra_files": sorted(extra),
        "file_precision": round(precision, 4),
        "file_recall": round(recall, 4),
        "file_f1": round(f1, 4),
    }


def _score_multi_file_complete(result: dict, bug: dict) -> bool:
    """For multi-file bugs: did the agent modify ALL expected files?

    Returns True for single-file bugs (not applicable).
    """
    if bug.get("difficulty") != "multi-file":
        return True

    expected = set(f.lower() for f in bug.get("expected_patch_files", bug.get("expected_files", [])))
    if not expected:
        return True

    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    patched = set()
    for p in patches:
        fp = (p.get("file_path") or "").lower()
        if fp:
            patched.add(fp)

    # Check that every expected file has a matching patch
    for exp in expected:
        found = any(exp in pat or pat.endswith(exp) for pat in patched)
        if not found:
            return False
    return True


def _score_test_pass(result: dict) -> bool:
    """Did the target repo's test suite pass after applying the fix?"""
    test_result = result.get("test_result") or ""
    if not test_result:
        return False
    return test_result.strip().lower().startswith("passed")


def _score_brt_pass(result: dict) -> bool:
    """Did all confirmed Bug Reproduction Tests pass after the fix?"""
    brts = result.get("brts", [])
    if not brts:
        return False  # No BRTs = can't verify via BRT (not a failure)
    brt_pass = result.get("brt_pass_count", 0)
    brt_total = result.get("brt_total", len(brts))
    return brt_total > 0 and brt_pass == brt_total


def _get_epr_score(result: dict) -> float:
    """Get the Ensemble Pass Rate score (fraction of BRTs that pass)."""
    return float(result.get("epr_score", 0.0))


def _score_patch_hits_target(result: dict, expected_files: list[str]) -> bool:
    """Do the agent's patches target at least one expected file?

    Ported from run_experiment.py:146-149.
    """
    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    patch_files = [(p.get("file_path") or "").lower() for p in patches]

    if not patch_files:
        return False

    expected_lower = [f.lower() for f in expected_files]
    return any(
        any(exp in pf or pf.endswith(exp) for exp in expected_lower)
        for pf in patch_files
    )


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_case(result: dict, bug: dict, pipeline: str) -> dict:
    """Score a single pipeline run against ground truth.

    Parameters
    ----------
    result : dict
        Full AgentState or ReactAgentState from run_ticket[_react].
    bug : dict
        EvalBug with expected_files, expected_root_cause, etc.
    pipeline : str
        "fixed" or "react".

    Returns
    -------
    dict
        All scored metrics for this case.
    """
    expected_files = bug.get("expected_files", [])

    loc_hit = _score_localization_hit(result, bug)
    fix_gen = _score_fix_generated(result)
    approved = _score_review_approved(result)
    patch_correctness = _score_patch_correctness(result, bug)
    hits_target = _score_patch_hits_target(result, expected_files)
    gt_match = _score_ground_truth_file_match(result, bug)

    test_ok = _score_test_pass(result)
    brt_ok = _score_brt_pass(result)
    epr = _get_epr_score(result)

    # Classify test infrastructure status
    test_result_raw = (result.get("test_result") or "").strip().lower()
    infra_blocked = (
        test_result_raw.startswith("skipped") or
        test_result_raw.startswith("error") or
        not test_result_raw  # no test result at all
    )

    # full_pass (strict): localization + fix + target files + passing tests
    full_pass = loc_hit and fix_gen and hits_target and test_ok

    # brt_full_pass: like full_pass but uses BRTs as the test signal (more precise)
    # When BRTs exist, this is the most reliable "did the agent fix the actual bug" signal
    brt_full_pass = loc_hit and fix_gen and hits_target and brt_ok if result.get("brts") else False

    # agent_quality_pass (diagnostic): did the agent do its job correctly?
    # Ignores test infra failures the agent can't control.
    agent_quality_pass = loc_hit and fix_gen and hits_target

    return {
        "ticket_id": bug["ticket_id"],
        "title": bug.get("title", ""),
        "pipeline": pipeline,

        # Ported metrics
        "localization_hit": loc_hit,
        "root_cause_match": _score_root_cause_match(result, bug.get("expected_root_cause", "")),
        "fix_generated": fix_gen,
        "review_approved": approved,
        "confidence": _get_review_confidence(result),

        # New metrics
        "patch_correctness": patch_correctness,
        "multi_file_complete": _score_multi_file_complete(result, bug),
        "test_pass": test_ok,
        "patch_hits_target": hits_target,

        # Ground truth comparison
        "gt_file_precision": gt_match["file_precision"],
        "gt_file_recall": gt_match["file_recall"],
        "gt_file_f1": gt_match["file_f1"],
        "gt_matched_files": gt_match["matched_files"],
        "gt_missing_files": gt_match["missing_files"],
        "gt_extra_files": gt_match["extra_files"],

        # Resource tracking
        "cost_usd": result.get("cost_usd", 0.0) or 0.0,
        "tool_call_count": result.get("tool_call_count", 0) or 0,

        # Status
        "status": str(result.get("status", "unknown")),
        "error": result.get("error", ""),

        # Derived — strict (requires passing tests)
        "full_pass": full_pass,
        # BRT-based pass (most reliable when BRTs exist)
        "brt_full_pass": brt_full_pass,
        "brt_pass": brt_ok,
        "epr_score": epr,
        "brt_count": len(result.get("brts", [])),
        # Diagnostic — agent behavior quality (ignores test infra)
        "agent_quality_pass": agent_quality_pass,
        "infra_blocked": infra_blocked,
    }


def build_summary(scores: list[dict], pipeline: str | None = None) -> dict:
    """Compute aggregate metrics from individual scores.

    Parameters
    ----------
    scores : list[dict]
        List of scored cases (from score_case).
    pipeline : str or None
        If set, filter to scores for this pipeline only.

    Returns
    -------
    dict
        Aggregate metrics: pass_rate, localization_accuracy, etc.
    """
    if pipeline:
        scores = [s for s in scores if s.get("pipeline") == pipeline]

    total = len(scores)
    if total == 0:
        return {
            "total": 0,
            "pass_rate": 0.0,
            "localization_accuracy": 0.0,
            "root_cause_accuracy": 0.0,
            "fix_rate": 0.0,
            "approval_rate": 0.0,
            "patch_correctness_avg": 0.0,
            "multi_file_complete_rate": 0.0,
            "test_pass_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_cost_usd": 0.0,
            "avg_duration_seconds": 0.0,
            "avg_tool_calls": 0.0,
            "failures": [],
        }

    passes = sum(1 for s in scores if s.get("full_pass"))
    quality_passes = sum(1 for s in scores if s.get("agent_quality_pass"))
    infra_blocked_count = sum(1 for s in scores if s.get("infra_blocked"))
    loc_hits = sum(1 for s in scores if s.get("localization_hit"))
    root_matches = sum(1 for s in scores if s.get("root_cause_match"))
    fixes = sum(1 for s in scores if s.get("fix_generated"))
    approvals = sum(1 for s in scores if s.get("review_approved"))
    test_passes = sum(1 for s in scores if s.get("test_pass"))

    multi_file_scores = [s for s in scores if not s.get("multi_file_complete", True) or
                         any(b_diff == "multi-file" for b_diff in [])]
    multi_complete = sum(1 for s in scores if s.get("multi_file_complete"))

    avg_conf = sum(s.get("confidence", 0) for s in scores) / total
    avg_correctness = sum(s.get("patch_correctness", 0) for s in scores) / total
    avg_cost = sum(s.get("cost_usd", 0) for s in scores) / total
    avg_duration = sum(s.get("duration_seconds", 0) for s in scores) / total
    avg_tools = sum(s.get("tool_call_count", 0) for s in scores) / total

    # Categorized failures
    failures: list[dict] = []
    for s in scores:
        if s.get("full_pass"):
            continue
        reasons = []
        if not s.get("localization_hit"):
            reasons.append("localization_miss")
        if not s.get("fix_generated"):
            reasons.append("no_fix")
        if not s.get("review_approved"):
            reasons.append("not_approved")
        if s.get("infra_blocked"):
            reasons.append("infra_blocked")
        elif not s.get("test_pass"):
            reasons.append("test_failed")
        if s.get("error"):
            reasons.append("error")
        failures.append({
            "ticket_id": s["ticket_id"],
            "reasons": reasons,
            "error": s.get("error", ""),
        })

    return {
        "total": total,
        "pass_rate": round(passes / total, 4),
        "agent_quality_rate": round(quality_passes / total, 4),
        "infra_blocked_count": infra_blocked_count,
        "localization_accuracy": round(loc_hits / total, 4),
        "root_cause_accuracy": round(root_matches / total, 4),
        "fix_rate": round(fixes / total, 4),
        "approval_rate": round(approvals / total, 4),
        "patch_correctness_avg": round(avg_correctness, 4),
        "multi_file_complete_rate": round(multi_complete / total, 4),
        "test_pass_rate": round(test_passes / total, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_cost_usd": round(avg_cost, 4),
        "avg_duration_seconds": round(avg_duration, 2),
        "avg_tool_calls": round(avg_tools, 1),
        "failures": failures,
    }
