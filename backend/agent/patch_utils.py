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


def apply_patch_by_line_number(
    content: str,
    patch: dict,
) -> str | None:
    """Strategy 0: Replace lines [line_start, line_end] with patched_code.

    Uses 1-indexed line numbers. line_end is inclusive.
    Returns None if line numbers are absent, out of range, or if the result
    has syntax errors (Python only, best-effort).
    """
    line_start = patch.get("line_start")
    line_end = patch.get("line_end")
    patched_code = patch.get("patched_code", "")

    if line_start is None or line_end is None:
        return None  # No line numbers — skip this strategy

    try:
        line_start = int(line_start)
        line_end = int(line_end)
    except (TypeError, ValueError):
        return None

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    if line_start < 1 or line_end < line_start or line_end > total_lines:
        logger.warning(
            "Line-number patch out of range: line_start=%d line_end=%d total=%d",
            line_start, line_end, total_lines,
        )
        return None

    # Preserve trailing newline consistency
    patched_lines = patched_code.splitlines(keepends=True)
    if patched_lines and not patched_lines[-1].endswith("\n"):
        patched_lines[-1] += "\n"

    # Build new content
    new_lines = lines[: line_start - 1] + patched_lines + lines[line_end:]
    new_content = "".join(new_lines)

    logger.debug(
        "Strategy 0 (line-number): lines %d-%d replaced (%d lines → %d lines)",
        line_start, line_end,
        line_end - line_start + 1,
        len(patched_lines),
    )

    return new_content


def apply_patch(
    content: str,
    patch: dict,
    file_path: str | None = None,
) -> str | None:
    """Apply a patch dict to *content*, trying strategies in order.

    Strategy 0 (line-number based) is tried first when ``line_start`` /
    ``line_end`` are present in the patch dict — it is the most reliable
    because it does not require exact substring matching.  If line numbers are
    absent, or if the result produces a syntax error in a Python file, the
    function falls through to the fuzzy-matching strategies (1-5) provided by
    :func:`fuzzy_match_replace`.

    Args:
        content:   Full text of the file to be patched.
        patch:     Patch dict with keys ``original_code``, ``patched_code``,
                   and optionally ``line_start`` / ``line_end``.
        file_path: Optional file path used for Python syntax validation.

    Returns:
        Patched file content, or ``None`` if no strategy succeeded.
    """
    original = patch.get("original_code", "")
    patched_code = patch.get("patched_code", "")

    # Strategy 0: Line-number patching (most reliable when available)
    result = apply_patch_by_line_number(content, patch)
    if result is not None:
        if file_path and file_path.endswith(".py"):
            try:
                ast.parse(result)
            except SyntaxError as e:
                logger.warning(
                    "Strategy 0: line-number patch produced invalid Python: %s", e
                )
                # Fall through to fuzzy strategies
            else:
                logger.debug("Strategy 0 succeeded for %s", file_path or "<unknown>")
                return result
        else:
            logger.debug("Strategy 0 succeeded for %s", file_path or "<unknown>")
            return result

    # Strategies 1-5: Fuzzy / string-based matching
    result = fuzzy_match_replace(content, original, patched_code)
    if result is not None:
        logger.debug(
            "Fuzzy strategies (1-5) succeeded for %s", file_path or "<unknown>"
        )
    return result


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

        if best_score >= 0.92 and best_pos >= 0:
            # Function-name guard: if original starts with a def/async def, verify
            # the matched region also defines the same function name.
            first_orig_line = orig_lines[0].strip()
            fn_def_match = re.match(r'^(?:async\s+)?def\s+(\w+)\s*\(', first_orig_line)
            if fn_def_match:
                expected_name = fn_def_match.group(1)
                first_matched_line = norm_content_lines[best_pos].strip()
                matched_fn_match = re.match(r'^(?:async\s+)?def\s+(\w+)\s*\(', first_matched_line)
                if not matched_fn_match or matched_fn_match.group(1) != expected_name:
                    logger.debug(
                        "Strategy 4: function-name guard rejected match at line %d "
                        "(expected def %s, got %s)",
                        best_pos,
                        expected_name,
                        matched_fn_match.group(1) if matched_fn_match else "<none>",
                    )
                    return None

            matched_region = '\n'.join(content_lines[best_pos:best_pos + window_size])
            logger.debug(
                "Strategy 4: matched region at line %d (score=%.4f):\n%s",
                best_pos,
                best_score,
                matched_region,
            )

            new_lines = content_lines[:best_pos] + patched.splitlines() + content_lines[best_pos + window_size:]
            new_content = '\n'.join(new_lines)

            # Verify the result still parses (Python files only)
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                logger.warning("Strategy 4: fuzzy match produced invalid Python: %s", e)
                return None

            return new_content

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


def patches_overlap(p1: dict, p2: dict, content: str) -> bool:
    """Return True if two patches target overlapping regions of content."""
    loc1 = content.find(p1.get("original_code", ""))
    loc2 = content.find(p2.get("original_code", ""))
    if loc1 == -1 or loc2 == -1:
        return True  # Can't verify — assume overlap (safe default)
    end1 = loc1 + len(p1.get("original_code", ""))
    end2 = loc2 + len(p2.get("original_code", ""))
    # Overlap if ranges intersect
    return not (end1 <= loc2 or end2 <= loc1)


def _compose_patches_for_file(patches: list[dict], content: str) -> list[dict]:
    """Return a list of non-overlapping patches that can be applied together."""
    # Sort by position in file (highest offset first to avoid position shifts)
    positioned = []
    for p in patches:
        loc = content.find(p.get("original_code", ""))
        if loc >= 0:
            positioned.append((loc, p))
    positioned.sort(key=lambda x: x[0], reverse=True)

    selected = []
    covered_ranges: list[tuple[int, int]] = []
    for loc, p in positioned:
        end = loc + len(p.get("original_code", ""))
        # Check against already selected ranges
        overlap = any(not (end <= s or e <= loc) for s, e in covered_ranges)
        if not overlap:
            selected.append(p)
            covered_ranges.append((loc, end))

    return selected


def pick_best_patch_per_file(
    patches: list[dict],
    repo_path: Path | None,
    find_file_fn,
    read_file_fn,
) -> list[dict]:
    """When multiple patches target the same file, compose non-overlapping ones;
    fall back to picking the single best patch only when patches overlap."""
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

        if content:
            # Try to compose non-overlapping patches that target different regions
            composed = _compose_patches_for_file(file_patches, content)
            if len(composed) > 1:
                logger.info(
                    "Composing %d non-overlapping patches for %s (out of %d candidates)",
                    len(composed),
                    file_path,
                    len(file_patches),
                )
                result.extend(composed)
                continue

        # Fall back: patches overlap or content unavailable — pick the single best
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
            logger.info(
                "Picked best patch for %s (%d candidates, score=%d)",
                file_path,
                len(file_patches),
                best_score,
            )

    return result
