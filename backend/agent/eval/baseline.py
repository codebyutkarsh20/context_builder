"""
baseline.py — Dumb-loop baseline diagnostic for measuring infrastructure value.

Single-shot Claude API call per bug. No ReAct loop, no tools, no retries, no
graph context, no localization (file is given). Measures the floor: what does
model + test output + single shot achieve on the same dataset?

Spec from CEO plan (2026-04-08):
  For each bug: clone at base_sha → run test_command (capture failing output) →
  read fault_file → send to Claude (sonnet, temp=0, single-shot) → write file →
  rerun test_command. Report: pass count / total, cost per attempt.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent.eval.dataset import load_eval_dataset
from agent.eval.repo_manager import RepoManager

logger = logging.getLogger(__name__)

BASELINE_MODEL = "claude-sonnet-4-6"
BASELINE_TIMEOUT = 300  # 5 min per bug (test execution)
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", "eval/results"))

# Pricing per 1M tokens (input, output) — same as react_loop.py
_PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0),
}


@dataclass
class BaselineCaseResult:
    ticket_id: str
    fault_file: str
    test_before: str  # "passed" / "failed" / "error"
    test_after: str   # "passed" / "failed" / "error"
    fixed: bool
    pre_test_failed: bool = True  # False = test didn't fail at buggy SHA (dataset issue)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class BaselineReport:
    run_id: str
    timestamp: float
    model: str
    total_bugs: int
    results: list[BaselineCaseResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    commit_sha: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "total_bugs": self.total_bugs,
            "results": [asdict(r) for r in self.results],
            "summary": self.summary,
            "commit_sha": self.commit_sha,
        }


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = _PRICING[BASELINE_MODEL]
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


def _run_test_command(repo_dir: Path, bug: dict) -> str:
    """Run the bug's test_command and return raw output."""
    test_cmd = bug.get("test_command", "")
    if not test_cmd:
        return "error: no test_command in bug spec"

    # Check for .agent_config.json — if it has a pytest_path, use it
    config_path = repo_dir / ".agent_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            pytest_path = config.get("pytest_path", "")
            if pytest_path and test_cmd.startswith("pytest "):
                test_cmd = pytest_path + test_cmd[len("pytest"):]
        except Exception:
            pass

    # Split command — handle pytest and python specially
    if test_cmd.startswith("pytest ") or test_cmd == "pytest":
        parts = [sys.executable, "-m", "pytest"] + test_cmd.split()[1:]
    elif test_cmd.startswith("python "):
        parts = [sys.executable] + test_cmd.split()[1:]
    else:
        parts = test_cmd.split()

    try:
        result = subprocess.run(
            parts,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=BASELINE_TIMEOUT,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode == 0:
            return f"passed\n{output[-1000:]}"
        return f"failed (exit code {result.returncode})\n{output[-3000:]}"
    except subprocess.TimeoutExpired:
        return f"error: timed out after {BASELINE_TIMEOUT}s"
    except Exception as e:
        return f"error: {e}"


def _run_single_baseline(
    bug: dict,
    repo_manager: RepoManager,
    client: anthropic.Anthropic,
) -> BaselineCaseResult:
    """Run one bug through the dumb baseline loop."""
    ticket_id = bug["ticket_id"]
    start = time.time()

    # 1. Clone repo at buggy SHA
    try:
        repo_dir = repo_manager.ensure_repo(bug)
    except Exception as e:
        return BaselineCaseResult(
            ticket_id=ticket_id,
            fault_file="",
            test_before="error",
            test_after="error",
            fixed=False,
            error=f"Clone failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )

    # Set up venv for external repos
    if not bug.get("local_repo_path"):
        try:
            repo_manager.setup_venv(repo_dir, bug)
        except Exception as e:
            logger.warning("Venv setup failed for %s: %s", ticket_id, e)
    else:
        # Local repos still need setup_commands (e.g., pip install -e backend/)
        for cmd in bug.get("setup_commands", []):
            if not cmd.strip():
                continue
            try:
                subprocess.run(
                    cmd, shell=True, cwd=str(repo_dir),
                    capture_output=True, text=True, timeout=120,
                )
            except Exception as e:
                logger.warning("Setup command failed for %s: %s", ticket_id, e)

    # 2. Run test_command — should fail (this is the buggy SHA)
    fault_file = bug["expected_files"][0]
    test_before = _run_test_command(repo_dir, bug)
    pre_test_failed = test_before.startswith("failed")
    logger.info("%s pre-test: %s (failed=%s)", ticket_id, test_before[:80], pre_test_failed)

    if not pre_test_failed:
        logger.warning(
            "%s: pre-test did NOT fail at buggy SHA — test may not exist or "
            "env is wrong. Baseline result is unreliable for this bug.",
            ticket_id,
        )

    # 3. Read the fault file
    fault_path = repo_dir / fault_file
    if not fault_path.exists():
        return BaselineCaseResult(
            ticket_id=ticket_id,
            fault_file=fault_file,
            test_before=test_before[:100],
            test_after="error",
            fixed=False,
            error=f"Fault file not found: {fault_file}",
            duration_seconds=round(time.time() - start, 2),
        )

    file_contents = fault_path.read_text()

    # 4. Send to Claude API — single shot, temperature=0
    system_prompt = (
        "You are a bug fixer. You will receive a failing test output and the "
        "source file that contains the bug. Output ONLY the complete corrected "
        "file contents. No explanations, no markdown fences, no commentary. "
        "Just the fixed source code."
    )
    user_prompt = (
        f"This test fails:\n\n{test_before[-3000:]}\n\n"
        f"Here is the file ({fault_file}):\n\n{file_contents}\n\n"
        f"Fix the bug. Output ONLY the corrected file contents."
    )

    try:
        response = client.messages.create(
            model=BASELINE_MODEL,
            max_tokens=8000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        fixed_content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = _estimate_cost(input_tokens, output_tokens)
    except Exception as e:
        return BaselineCaseResult(
            ticket_id=ticket_id,
            fault_file=fault_file,
            test_before=test_before[:100],
            test_after="error",
            fixed=False,
            error=f"API call failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )

    # 5. Write the fixed file
    # Strip markdown fences if the model wraps output despite instructions
    cleaned = _strip_markdown_fences(fixed_content)
    fault_path.write_text(cleaned)

    # 6. Rerun tests
    test_after = _run_test_command(repo_dir, bug)
    fixed = test_after.startswith("passed")
    logger.info(
        "%s post-test: %s (fixed=%s, $%.3f)",
        ticket_id, test_after[:80], fixed, cost,
    )

    # 7. Reset repo for next run
    try:
        repo_manager._reset_repo(repo_dir)
    except Exception:
        pass

    return BaselineCaseResult(
        ticket_id=ticket_id,
        fault_file=fault_file,
        test_before=test_before[:200],
        test_after=test_after[:200],
        fixed=fixed,
        pre_test_failed=pre_test_failed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 4),
        duration_seconds=round(time.time() - start, 2),
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if the model wraps its output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def run_baseline(
    dataset_path: str | Path = "eval/bugs.json",
    bug_filter: str | None = None,
    repo_cache_dir: Path | None = None,
) -> BaselineReport:
    """Run the dumb-loop baseline on the eval dataset.

    Parameters
    ----------
    dataset_path : path to bugs.json
    bug_filter : optional ticket_id to run a single bug
    repo_cache_dir : optional repo cache directory

    Returns
    -------
    BaselineReport with per-bug results and aggregate summary.
    """
    # Load .env from project root (same as eval runner child process)
    root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()

    bugs = load_eval_dataset(Path(dataset_path))

    if bug_filter:
        bugs = [b for b in bugs if b["ticket_id"] == bug_filter]
        if not bugs:
            raise ValueError(f"No bug found with ticket_id={bug_filter}")

    run_id = str(uuid.uuid4())[:8]
    commit_sha = ""
    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    report = BaselineReport(
        run_id=run_id,
        timestamp=time.time(),
        model=BASELINE_MODEL,
        total_bugs=len(bugs),
        commit_sha=commit_sha,
    )

    client = anthropic.Anthropic(max_retries=3)
    repo_manager = RepoManager(cache_dir=repo_cache_dir)

    logger.info(
        "Starting baseline run %s: %d bugs, model=%s",
        run_id, len(bugs), BASELINE_MODEL,
    )

    for i, bug in enumerate(bugs):
        logger.info(
            "=" * 60 + "\nBaseline %d/%d: %s — %s",
            i + 1, len(bugs), bug["ticket_id"], bug.get("title", "")[:60],
        )
        result = _run_single_baseline(bug, repo_manager, client)
        report.results.append(result)

    # Build summary
    total = len(report.results)
    fixed = sum(1 for r in report.results if r.fixed)
    errors = sum(1 for r in report.results if r.error)
    valid = sum(1 for r in report.results if r.pre_test_failed)
    valid_fixed = sum(1 for r in report.results if r.fixed and r.pre_test_failed)
    total_cost = sum(r.cost_usd for r in report.results)
    total_input = sum(r.input_tokens for r in report.results)
    total_output = sum(r.output_tokens for r in report.results)

    report.summary = {
        "total_bugs": total,
        "valid_bugs": valid,  # pre-test actually failed at buggy SHA
        "fixed": fixed,
        "valid_fixed": valid_fixed,  # fixed AND pre-test was valid
        "errors": errors,
        "fix_rate": round(fixed / total, 4) if total else 0,
        "valid_fix_rate": round(valid_fixed / valid, 4) if valid else 0,
        "error_rate": round(errors / total, 4) if total else 0,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_usd": round(total_cost / total, 4) if total else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }

    # Persist
    _persist_baseline_report(report)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BASELINE RESULTS (run {run_id})")
    print(f"{'=' * 60}")
    print(f"Model:       {BASELINE_MODEL}")
    print(f"Total bugs:  {total}")
    print(f"Valid bugs:  {valid}/{total} (pre-test failed at buggy SHA)")
    print(f"Fixed:       {fixed}/{total} ({report.summary['fix_rate']:.1%})")
    if valid:
        print(f"Valid fix:   {valid_fixed}/{valid} ({report.summary['valid_fix_rate']:.1%})")
    print(f"Errors:      {errors}")
    print(f"Cost:        ${total_cost:.2f} total, ${report.summary['avg_cost_usd']:.3f}/bug avg")
    print(f"{'=' * 60}")

    # Per-bug breakdown
    print(f"\n{'Ticket':<25} {'Pre':<6} {'Fix':<6} {'Cost':>8} {'Time':>8}  Error")
    print("-" * 80)
    for r in report.results:
        pre = "FAIL" if r.pre_test_failed else "pass"
        fix = "PASS" if r.fixed else ("ERR" if r.error else "FAIL")
        err_short = r.error[:30] if r.error else ""
        print(f"{r.ticket_id:<25} {pre:<6} {fix:<6} ${r.cost_usd:>7.3f} {r.duration_seconds:>7.1f}s  {err_short}")

    return report


def _persist_baseline_report(report: BaselineReport) -> None:
    """Save baseline report to eval/results/."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    report_file = results_dir / f"baseline_{report.run_id}_{int(report.timestamp)}.json"
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("Baseline report written to %s", report_file)

    # Latest baseline symlink
    latest_file = results_dir / "baseline_latest.json"
    with open(latest_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
