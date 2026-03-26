"""
patch_utils.py — Fuzzy patch matching, verification, and deduplication.

Extracted from pipeline.py to keep each module focused on a single concern.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def fuzzy_match_replace(content: str, original: str, patched: str) -> str | None:
    """Try multiple matching strategies, from strict to fuzzy.

    Strategies (in order):
    1. Exact substring match
    2. Whitespace-normalized line-by-line match
    3. Stripped-whitespace match (ignores leading indentation differences)
    4. Best sliding-window match (tolerates minor differences like variable names)
    5. Anchor-based matching with adaptive region sizing
    """
    if not original or not original.strip():
        return None

    # Strategy 1: Exact match
    if original in content:
        return content.replace(original, patched, 1)

    def normalize_line(s: str) -> str:
        return s.rstrip().expandtabs(4)

    orig_lines = [normalize_line(l) for l in original.splitlines()]
    content_lines = content.splitlines()
    norm_content_lines = [normalize_line(l) for l in content_lines]

    if not orig_lines:
        return None

    # Strategy 2: Whitespace-normalized exact match
    for i in range(len(norm_content_lines) - len(orig_lines) + 1):
        if norm_content_lines[i:i + len(orig_lines)] == orig_lines:
            new_lines = content_lines[:i] + patched.splitlines() + content_lines[i + len(orig_lines):]
            return '\n'.join(new_lines)

    # Strategy 3: Stripped match (ignores leading whitespace differences entirely)
    stripped_orig = [l.strip() for l in orig_lines if l.strip()]
    stripped_content = [l.strip() for l in content_lines]

    if len(stripped_orig) >= 2:
        for i in range(len(stripped_content) - len(stripped_orig) + 1):
            window = [l for l in stripped_content[i:i + len(stripped_orig) + 5] if l][:len(stripped_orig)]
            if window == stripped_orig:
                matched = 0
                j = i
                start_j = None
                while j < len(content_lines) and matched < len(stripped_orig):
                    if content_lines[j].strip() == stripped_orig[matched]:
                        if start_j is None:
                            start_j = j
                        matched += 1
                    elif content_lines[j].strip():
                        break
                    j += 1
                if matched == len(stripped_orig) and start_j is not None:
                    new_lines = content_lines[:start_j] + patched.splitlines() + content_lines[j:]
                    return '\n'.join(new_lines)

    # Strategy 4: Best sliding-window match with similarity scoring
    if len(orig_lines) >= 3:
        import difflib
        best_score = 0.0
        best_pos = -1
        window_size = len(orig_lines)

        for i in range(len(norm_content_lines) - window_size + 1):
            window = norm_content_lines[i:i + window_size]
            ratio = difflib.SequenceMatcher(None,
                '\n'.join(orig_lines), '\n'.join(window)).ratio()
            if ratio > best_score:
                best_score = ratio
                best_pos = i

        if best_score >= 0.85 and best_pos >= 0:
            new_lines = content_lines[:best_pos] + patched.splitlines() + content_lines[best_pos + window_size:]
            return '\n'.join(new_lines)

    # Strategy 5: Anchor-based matching with adaptive region sizing
    if len(orig_lines) >= 1:
        def _anchor_score(line: str) -> float:
            s = line.strip()
            if not s or s in ('{', '}', 'pass', 'return', 'else:', 'try:', 'except:'):
                return 0
            score = len(s)
            score += len(re.findall(r'\w+\.\w+', s)) * 20
            score += len(re.findall(r'await |return_exceptions|raise |async ', s)) * 15
            return score

        scored = [(i, _anchor_score(l)) for i, l in enumerate(orig_lines)]
        scored.sort(key=lambda x: -x[1])

        for anchor_idx, anchor_score in scored[:3]:
            if anchor_score < 10:
                continue
            anchor = orig_lines[anchor_idx].strip()
            if len(anchor) < 8:
                continue

            candidates = []
            dotted = re.findall(r'\w+\.\w+', anchor)
            for ci, cl in enumerate(content_lines):
                if anchor in cl.strip() or cl.strip() in anchor:
                    candidates.append(ci)
                elif dotted and any(d in cl for d in dotted):
                    if ci not in candidates:
                        candidates.append(ci)

            for ci in candidates:
                anchor_indent = len(content_lines[ci]) - len(content_lines[ci].lstrip())
                stmt_start = ci
                for k in range(ci - 1, max(ci - 10, -1), -1):
                    line_k = content_lines[k]
                    if not line_k.strip():
                        continue
                    indent_k = len(line_k) - len(line_k.lstrip())
                    if indent_k <= anchor_indent:
                        stmt_start = k
                        break
                    elif indent_k > anchor_indent:
                        stmt_start = k
                    else:
                        break

                stmt_end = ci + 1
                region_text = '\n'.join(content_lines[stmt_start:stmt_end])
                open_count = region_text.count('(') - region_text.count(')')
                open_count += region_text.count('[') - region_text.count(']')

                while open_count > 0 and stmt_end < min(len(content_lines), ci + 20):
                    line_text = content_lines[stmt_end]
                    open_count += line_text.count('(') - line_text.count(')')
                    open_count += line_text.count('[') - line_text.count(']')
                    stmt_end += 1

                if stmt_start < 0 or stmt_end > len(content_lines):
                    continue

                region = content_lines[stmt_start:stmt_end]
                region_text = '\n'.join(region)
                if dotted and not any(d in region_text for d in dotted):
                    continue
                if 'asyncio.gather' not in anchor and 'await' not in anchor:
                    import difflib
                    ratio = difflib.SequenceMatcher(None,
                        '\n'.join(l.strip() for l in orig_lines),
                        '\n'.join(l.strip() for l in region)).ratio()
                    if ratio < 0.35:
                        continue

                actual_indent = len(content_lines[stmt_start]) - len(content_lines[stmt_start].lstrip())
                patch_lines = patched.splitlines()
                if patch_lines:
                    patch_indent = len(patch_lines[0]) - len(patch_lines[0].lstrip())
                    indent_diff = actual_indent - patch_indent
                    if indent_diff > 0:
                        patch_lines = [(' ' * indent_diff + l) if l.strip() else l for l in patch_lines]
                    elif indent_diff < 0:
                        patch_lines = [l[-indent_diff:] if l[:(-indent_diff)].strip() == '' else l for l in patch_lines]

                new_lines = content_lines[:stmt_start] + patch_lines + content_lines[stmt_end:]
                return '\n'.join(new_lines)

    return None


def check_syntax(file_path: Path) -> str | None:
    """Check Python file for syntax errors. Returns error message or None if OK."""
    if file_path.suffix != ".py":
        return None
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return str(e)
    return None


def deduplicate_patches(patches: list[dict]) -> list[dict]:
    """Deduplicate patches: one patch per unique (file_path, original_code) pair."""
    seen: dict[str, dict] = {}
    for p in patches:
        key = f"{p.get('file_path', '')}::{p.get('original_code', '').strip()[:200]}"
        seen[key] = p
    return list(seen.values())


def pick_best_patch_per_file(
    patches: list[dict],
    repo_path: Path | None,
    find_file_fn,
    read_file_fn,
) -> list[dict]:
    """When multiple patches target the same file, pick the most complete one."""
    if not repo_path:
        return patches

    by_file: dict[str, list[dict]] = {}
    for p in patches:
        by_file.setdefault(p.get("file_path", ""), []).append(p)

    result: list[dict] = []
    for file_path, file_patches in by_file.items():
        if len(file_patches) == 1:
            result.append(file_patches[0])
            continue

        resolved = find_file_fn(repo_path, file_path)
        content = read_file_fn(resolved, max_lines=5000) if resolved else None

        best = None
        best_score = -1
        for p in file_patches:
            original = p.get("original_code", "")
            patched_code = p.get("patched_code", "")
            score = len(patched_code) - len(original)
            if content and fuzzy_match_replace(content, original, patched_code) is not None:
                score += 1000
            if score > best_score:
                best_score = score
                best = p

        if best:
            result.append(best)
            logger.info("Picked best patch for %s (%d candidates, score=%d)", file_path, len(file_patches), best_score)

    return result
