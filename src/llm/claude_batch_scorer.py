"""
Batch Claude scorer — scores up to BATCH_SIZE posts in a single subprocess call.

~5-15x faster than claude_scorer.py for large backlogs.
Set CURATOR_BACKEND=claude-batch to enable.

Truncation modes (Phase 49 TRUNC-02): set CURATOR_TRUNCATION_MODE env to:
  - "prefix" (default): take first _MAX_TEXT_CHARS chars
  - "head_tail": take head 1K + tail 1K with '[middle elided]' marker;
    targets boilerplate intros (YT sponsor reads, newsletter unsubscribe
    headers, forwarded-message preambles)
Activate after Phase 48 audit verdict CHAIN_OPEN (≥10% migration).
"""
import asyncio
import json
import os
import re
import subprocess

BATCH_SIZE = 20
# Phase 48 (TRUNC-01): bumped 800 → 4000 (~1000 tokens/post). See
# data/research/truncation_audit_999_54.md for chain gate verdict.
_MAX_TEXT_CHARS = 4000

# Phase 49 (TRUNC-02): head+tail composition. Head + tail + separator
# total must fit inside _MAX_TEXT_CHARS budget.
_HEAD_CHARS = 1000
_TAIL_CHARS = 1000
_TRUNCATION_SEPARATOR = "\n\n...[middle elided]\n\n..."
_PARA_BOUNDARY_TOLERANCE = 100  # search for \n\n within ±100 chars of target cut


def _truncate_head_tail(
    text: str,
    head: int = _HEAD_CHARS,
    tail: int = _TAIL_CHARS,
    sep: str = _TRUNCATION_SEPARATOR,
    tolerance: int = _PARA_BOUNDARY_TOLERANCE,
) -> str:
    """Phase 49: head+tail composition with paragraph-boundary snap.

    If text fits within head+tail+sep, return unchanged. Otherwise:
      [head chars] + sep + [tail chars]
    Cut points snap to nearest \\n\\n within ±tolerance for word integrity.
    """
    min_size_for_truncation = head + tail + len(sep)
    if not text or len(text) <= min_size_for_truncation:
        return text

    # Head cut: target index = head; snap to nearest \n\n within ±tolerance
    head_target = head
    head_lo = max(0, head_target - tolerance)
    head_hi = min(len(text), head_target + tolerance)
    head_window = text[head_lo:head_hi]
    head_cut = head_target
    # Find nearest \n\n in window to original target
    nl_positions = [head_lo + i for i in range(len(head_window) - 1) if head_window[i:i + 2] == "\n\n"]
    if nl_positions:
        head_cut = min(nl_positions, key=lambda p: abs(p - head_target))

    # Tail cut: start = len(text) - tail; snap to nearest \n\n within ±tolerance
    tail_target = len(text) - tail
    tail_lo = max(0, tail_target - tolerance)
    tail_hi = min(len(text), tail_target + tolerance)
    tail_window = text[tail_lo:tail_hi]
    tail_cut = tail_target
    nl_positions_tail = [tail_lo + i + 2 for i in range(len(tail_window) - 1) if tail_window[i:i + 2] == "\n\n"]
    if nl_positions_tail:
        tail_cut = min(nl_positions_tail, key=lambda p: abs(p - tail_target))

    # Defense: ensure head_cut < tail_cut
    if head_cut >= tail_cut:
        return text[:head] + sep + text[-tail:]

    return text[:head_cut] + sep + text[tail_cut:]


def _truncate(text: str) -> str:
    """Dispatch on CURATOR_TRUNCATION_MODE env. Fail-soft: bad value → prefix mode."""
    mode = os.environ.get("CURATOR_TRUNCATION_MODE", "prefix").lower().strip()
    if mode == "head_tail":
        return _truncate_head_tail(text)
    # Default: prefix truncation (legacy behavior)
    return text[:_MAX_TEXT_CHARS]


def _build_prompt(context: str, texts: list[str]) -> str:
    posts_block = "\n".join(
        f"--- POST {i} ---\n{_truncate(t)}"
        for i, t in enumerate(texts)
    )
    return (
        f"You are a content relevance scorer.\n\n"
        f"PROFILE CONTEXT:\n{context}\n\n"
        f"Score each post 0-100 for relevance to this profile.\n"
        f"Return ONLY a JSON array with exactly {len(texts)} objects, no other text:\n"
        f'[{{"index":0,"score":85,"reason":"brief"}}, ...]\n\n'
        f"POSTS:\n{posts_block}"
    )


def _parse_response(raw: str, n: int) -> list[dict]:
    """Parse JSON array from Claude output. Falls back to regex per-item."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        items = json.loads(raw)
        if isinstance(items, list) and len(items) == n:
            return [{"score": int(x.get("score", 0)), "reason": x.get("reason", "")} for x in items]
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback: find all score/reason pairs in order
    scores = [int(m) for m in re.findall(r'"score"\s*:\s*(\d+)', raw)]
    reasons = re.findall(r'"reason"\s*:\s*"([^"]*)"', raw)
    if len(scores) == n:
        return [
            {"score": scores[i], "reason": reasons[i] if i < len(reasons) else ""}
            for i in range(n)
        ]

    raise RuntimeError(f"batch scorer: expected {n} results, got {len(scores)}. raw={raw[:300]}")


async def score_batch(context: str, texts: list[str], timeout: int = 180) -> list[dict]:
    """Score a batch of texts in one Claude CLI call.

    Returns list of {"score": int, "reason": str} dicts, same length as texts.
    """
    prompt = _build_prompt(context, texts)

    def _run():
        return subprocess.run(
            ["claude", "--output-format", "text", "--model", "claude-sonnet-4-6"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )

    result = await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, _run),
        timeout=timeout + 5,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}: {result.stderr[:200]}")

    return _parse_response(result.stdout.strip(), len(texts))
