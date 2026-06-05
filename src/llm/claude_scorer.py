"""
Claude CLI scorer — thin adapter matching ollama.score_content() return shape.

Uses subprocess stdin to avoid Windows 32k command-line length limit.
Identical subprocess pattern to src/llm/claude_headless.py.

Usage:
    Set CURATOR_BACKEND=claude in .env to enable.
    Warning: one subprocess per post — slow for large batches (>100 posts).
    Use only for small batches or quality comparisons with Ollama.
"""
import asyncio
import json
import re
import subprocess

SCORE_PROMPT_TEMPLATE = (
    "Context: {context}\n"
    "Post: {text}\n"
    "Rate the relevance of this post from 0 to 100. "
    'Return ONLY valid JSON with no other text: {{"score": 85, "reason": "explanation"}}'
)

# Phase 52 / DIVERGE-03: rate-limit detection markers. Case-insensitive scan
# of stderr (on non-zero returncode) AND stdout (on returncode=0 — Claude Max
# sometimes returns 0 with garbage body when quota exhausted; see
# reference_curator_routing_claude_limits.md).
#
# WR-01 fix (2026-05-18): tightened markers to phrases that are context-bound
# (multi-word or prefix-bound). Previously bare "429" and "quota" produced
# false positives on legitimately scored posts whose `reason` field merely
# *mentioned* HTTP error codes, port numbers, API quotas, disk quotas, etc.
# False positives polluted `claude_calls_skipped` — the operator's only
# signal for tuning SCORER_SAMPLING_RATE (per `_format_ok_line` docstring).
#
# Markers retained:
#   - "rate_limit"        — Anthropic's `rate_limit_error`, `rate_limit reached`
#   - "rate limit"        — variants like "rate limit exceeded"
#   - "http 429"          — bounded http-status idiom
#   - "status 429"        — Anthropic-style status code
#   - "usage limit"       — "Usage limit reached for org"
#   - "anthropic-ratelimit" — Anthropic response header marker
#
# Markers dropped:
#   - "429"   — too broad (matches port numbers, dates, post bodies)
#   - "quota" — too broad (matches API/disk/dev-tool quota discussions)
_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "http 429",
    "status 429",
    "usage limit",
    "anthropic-ratelimit",
)


async def generate_score(context: str, text: str, timeout: int = 120) -> dict:
    """
    Score content using Claude CLI via stdin subprocess.

    Returns:
        dict with keys "score" (int 0-100) and "reason" (str)

    Raises:
        RuntimeError: if claude CLI exits with non-zero return code
    """
    prompt = SCORE_PROMPT_TEMPLATE.format(context=context, text=text)

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
        # Phase 52 / DIVERGE-03: scan stderr for rate-limit signals before
        # raising RuntimeError. Plan 02 routes ClaudeRateLimitError to the
        # claude_calls_skipped counter in shadow_router.
        stderr_lower = result.stderr.lower()
        if any(marker in stderr_lower for marker in _RATE_LIMIT_MARKERS):
            from src.llm.shadow_router import ClaudeRateLimitError
            raise ClaudeRateLimitError(
                f"Claude rate-limit detected in stderr: {result.stderr[:200]}"
            )
        raise RuntimeError(
            f"claude exited {result.returncode}: {result.stderr[:200]}"
        )

    # Phase 52 / DIVERGE-03: also scan stdout — Claude Max CLI occasionally
    # returns 0 + a rate-limited body. Scan happens BEFORE JSON parse.
    stdout_lower = result.stdout.lower()
    if any(marker in stdout_lower for marker in _RATE_LIMIT_MARKERS):
        from src.llm.shadow_router import ClaudeRateLimitError
        raise ClaudeRateLimitError(
            f"Claude rate-limit signal in stdout: {result.stdout[:200]}"
        )

    raw = result.stdout.strip()

    # Try direct JSON parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Regex fallback for partial/wrapped JSON output
    score_match = re.search(r'"score"\s*:\s*(\d+)', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    if score_match:
        return {
            "score": int(score_match.group(1)),
            "reason": reason_match.group(1) if reason_match else "parse error",
        }

    raise RuntimeError(
        f"claude scorer returned unparseable output: {raw[:200]}"
    )
