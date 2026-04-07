"""
Unified evaluation pipeline for the AI Deploy Agent.

Provides A/B comparison of fixed vs ReAct pipelines, auto-clone repo management,
enhanced scoring, regression detection, and GitHub PR review tracking.
"""

from .dataset import load_eval_dataset, EvalBug
from .scoring import score_case
from .runner import EvalRunner, EvalCaseResult, EvalRunReport
from .report import generate_markdown_report, generate_json_report
from .regression import check_regression_gate, detect_regressions
from .repo_manager import RepoManager
from .pr_tracker import PRTracker
from .ab_eval import run_ab_eval, format_comparison_table

__all__ = [
    "load_eval_dataset",
    "EvalBug",
    "score_case",
    "EvalRunner",
    "EvalCaseResult",
    "EvalRunReport",
    "generate_markdown_report",
    "generate_json_report",
    "check_regression_gate",
    "detect_regressions",
    "RepoManager",
    "PRTracker",
    "run_ab_eval",
    "format_comparison_table",
]
