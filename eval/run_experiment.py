#!/usr/bin/env python3
"""
Eval Experiment: Run real open-source bugs through the pipeline.

Usage:
    cd /path/to/context_builder
    source .venv/bin/activate
    cd backend && python ../eval/run_experiment.py
    cd backend && python ../eval/run_experiment.py --bug FLASK-2651
    cd backend && python ../eval/run_experiment.py --skip-build
    cd backend && python ../eval/run_experiment.py --no-graph
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "backend"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
BUGS_FILE = EVAL_DIR / "bugs.json"
REPOS_DIR = EVAL_DIR / "repos"
RESULTS_DIR = EVAL_DIR / "results"


def load_bugs(bug_filter: str | None = None) -> list[dict]:
    with open(BUGS_FILE) as f:
        bugs = json.load(f)
    if bug_filter:
        bugs = [b for b in bugs if b["ticket_id"] == bug_filter]
        if not bugs:
            logger.error("No bug found with ticket_id=%s", bug_filter)
            sys.exit(1)
    return bugs


def clone_repo(bug: dict) -> Path:
    repo_url = bug["repo_url"]
    repo_sha = bug["repo_sha"]
    repo_dir = REPOS_DIR / bug["ticket_id"].lower()

    if repo_dir.exists():
        try:
            current_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
            ).strip()
            if current_sha.startswith(repo_sha[:8]):
                logger.info("Repo already at %s: %s", repo_sha[:8], repo_dir)
                return repo_dir
        except subprocess.CalledProcessError:
            pass
        subprocess.run(["rm", "-rf", str(repo_dir)], check=True)

    logger.info("Cloning %s at %s...", repo_url, repo_sha[:8])
    subprocess.run(["git", "clone", "--quiet", repo_url, str(repo_dir)], check=True)
    subprocess.run(["git", "checkout", "--quiet", repo_sha], cwd=repo_dir, check=True)
    return repo_dir


def build_graph(bug: dict, repo_dir: Path) -> None:
    repo_name = bug["ticket_id"].lower()
    logger.info("Building graph for %s...", repo_name)
    result = subprocess.run(
        [sys.executable, "cli.py", "build", str(repo_dir), "--name", repo_name, "--no-neo4j"],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        logger.error("Graph build failed: %s", result.stderr[-500:])
    else:
        logger.info("Graph built for %s", repo_name)


def run_pipeline(bug: dict, repo_dir: Path) -> dict:
    from agent.trace import RunTrace

    work_order = {
        "ticket_id": bug["ticket_id"],
        "title": bug["title"],
        "description": bug["description"],
        "repo_name": bug["ticket_id"].lower(),
        "repo_path": str(repo_dir),
        "priority": bug.get("priority", "medium"),
        "comments": bug.get("comments", []),
    }

    trace = RunTrace(job_id=bug["ticket_id"], enabled=True)
    logger.info("Running ReAct pipeline for %s: %s", bug["ticket_id"], bug["title"][:60])

    start = time.time()
    try:
        from agent.react_pipeline import run_ticket_react
        result = run_ticket_react(work_order, trace=trace, dry_run=True)
    except Exception as e:
        logger.exception("Pipeline crashed for %s", bug["ticket_id"])
        result = {"status": "failed", "error": str(e)}

    result["_duration"] = round(time.time() - start, 2)
    result["_trace"] = trace.to_report()
    return result


def score_result(result: dict, bug: dict) -> dict:
    localization = result.get("localization") or {}
    found_files = [f.lower() for f in localization.get("fault_files", [])]
    expected_files = [f.lower() for f in bug.get("expected_files", [])]

    loc_hit = any(
        any(exp in found or found.endswith(exp) for found in found_files)
        for exp in expected_files
    )

    hypothesis = (localization.get("root_cause_hypothesis") or "").lower()
    keywords = bug.get("expected_root_cause", "").lower().split()
    kw_matches = sum(1 for kw in keywords if kw in hypothesis) if keywords else 0
    root_match = kw_matches >= max(1, len(keywords) * 0.4) if keywords else False

    repair = result.get("repair") or {}
    patches = repair.get("patches") or []
    fix_generated = len(patches) > 0

    patch_files = [p.get("file_path", "").lower() for p in patches]
    patch_hits_target = any(
        any(exp in pf or pf.endswith(exp) for exp in expected_files)
        for pf in patch_files
    ) if patch_files else False

    review = result.get("review") or {}
    verdict = review.get("verdict", "").upper()
    approved = verdict == "APPROVE"
    confidence = float(review.get("confidence", 0.0))

    return {
        "ticket_id": bug["ticket_id"],
        "title": bug["title"],
        "localization_hit": loc_hit,
        "found_files": found_files,
        "expected_files": expected_files,
        "root_cause_match": root_match,
        "fix_generated": fix_generated,
        "patch_count": len(patches),
        "patch_hits_target": patch_hits_target,
        "review_verdict": verdict,
        "review_approved": approved,
        "review_confidence": confidence,
        "full_pass": loc_hit and fix_generated and approved,
        "duration_seconds": result.get("_duration", 0),
        "status": str(result.get("status", "unknown")),
        "error": result.get("error", ""),
    }


def print_summary(scores: list[dict]) -> None:
    total = len(scores)
    if not total:
        print("\nNo results.")
        return

    passes = sum(1 for s in scores if s["full_pass"])
    loc_hits = sum(1 for s in scores if s["localization_hit"])
    fixes = sum(1 for s in scores if s["fix_generated"])
    approvals = sum(1 for s in scores if s["review_approved"])
    target_hits = sum(1 for s in scores if s["patch_hits_target"])

    print("\n" + "=" * 70)
    print("EVAL RESULTS SUMMARY")
    print("=" * 70)

    for s in scores:
        icon = "PASS" if s["full_pass"] else "FAIL"
        print(f"\n  [{icon}] {s['ticket_id']}: {s['title'][:50]}")
        print(f"         Loc: {'HIT' if s['localization_hit'] else 'MISS'}  "
              f"Fix: {'YES' if s['fix_generated'] else 'NO'} ({s['patch_count']}p)  "
              f"Review: {s['review_verdict']} ({s['review_confidence']:.0%})  "
              f"Time: {s['duration_seconds']:.0f}s")
        if s["error"]:
            print(f"         Error: {s['error'][:80]}")

    print("\n" + "-" * 70)
    print(f"  PASS RATE:             {passes}/{total} ({passes/total*100:.0f}%)")
    print(f"  Localization:          {loc_hits}/{total}")
    print(f"  Fix rate:              {fixes}/{total}")
    print(f"  Correct file patched:  {target_hits}/{total}")
    print(f"  Approval rate:         {approvals}/{total}")
    total_dur = sum(s["duration_seconds"] for s in scores)
    print(f"  Total time:            {total_dur:.0f}s ({total_dur/60:.1f}min)")
    print("=" * 70)

    if passes / total >= 0.8:
        print("\n  TARGET MET: 80%+ pass rate.")
    else:
        modes = {}
        for s in scores:
            if not s["full_pass"]:
                if not s["localization_hit"]: modes["localization"] = modes.get("localization", 0) + 1
                elif not s["fix_generated"]: modes["no_fix"] = modes.get("no_fix", 0) + 1
                elif not s["review_approved"]: modes["review_rejected"] = modes.get("review_rejected", 0) + 1
        print(f"\n  TARGET NOT MET. Failure modes: {modes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug", help="Run only this ticket_id")
    parser.add_argument("--sentinel", action="store_true", help="Run first 5 bugs only")
    parser.add_argument("--timeout", type=int, default=600, help="Per-case timeout in seconds")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Results output directory")
    parser.add_argument("--create-prs", action="store_true", help="Create real PRs (default: dry-run)")
    # Legacy flags retained for compatibility; no longer used by EvalRunner.
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()

    if args.skip_build or args.skip_clone or args.no_graph:
        logger.warning(
            "Flags --skip-build/--skip-clone/--no-graph are legacy and ignored. "
            "Use backend/cli.py eval run options for modern eval controls."
        )

    from agent.eval.runner import EvalRunner
    from agent.eval.report import generate_markdown_report

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = EvalRunner(
        dataset_path=BUGS_FILE,
        pipelines=["react"],
        timeout_per_case=args.timeout,
        create_prs=args.create_prs,
        results_dir=output_dir,
    )
    eval_report = runner.run(bug_filter=args.bug, sentinel=args.sentinel)

    print(generate_markdown_report(eval_report))
    print(f"\nLatest report: {output_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
