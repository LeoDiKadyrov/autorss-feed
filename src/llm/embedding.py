"""Phase 65 / DEDUP-BGE-01 — bge-m3 embedding wrapper (Ollama, fail-soft).

POSTs to {OLLAMA_HOST}/api/embeddings with {"model": "bge-m3", "prompt": text}.
Returns np.float32(1024,) on success, None on ANY backend failure (never raises).

The dedup_rejected counter is process-global and surfaces on the OK-log line
per `reference_ok_log_field_accretion.md` 6-file lockstep (P65). Reset at
run start in run_pipeline.main(); read fail-soft at OK-log emit site.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import requests

log = logging.getLogger(__name__)

# Process-global counter — mirrors src.editor.spec_conformance and
# src.curator.{snippet_quality,giveaway_filter}.
_DEDUP_REJECTED: int = 0

# WR-04 (P65 review): emit a one-shot WARNING the FIRST time the wrapper hits
# any exception so OLLAMA_HOST typos / Ollama-down don't masquerade as silent
# "no dedup hits". After the warning fires once per process we stay silent to
# avoid log spam.
_WARNED_ONCE: bool = False

# Default Ollama embeddings endpoint constants. Read OLLAMA_HOST INSIDE
# get_embedding (module-level env binding gotcha per
# reference_module_level_env_binding.md).
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_EMBEDDING_MODEL = "bge-m3"
_EMBEDDING_TIMEOUT = 10.0


def get_embedding(text: str) -> np.ndarray | None:
    """Embed `text` via Ollama bge-m3. Return np.float32(1024,) or None.

    Fail-soft on:
      - empty/whitespace-only text (no HTTP call)
      - any HTTP/connection/timeout error
      - malformed JSON or missing 'embedding' key
      - numpy conversion errors

    NEVER raises. Caller (insert_raw_post) treats None as "skip bge-m3 dedup,
    fall through to existing simhash path".
    """
    if not text or not text.strip():
        return None
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST).rstrip("/")
    url = f"{host}/api/embeddings"
    try:
        resp = requests.post(
            url,
            json={"model": _EMBEDDING_MODEL, "prompt": text},
            timeout=_EMBEDDING_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        vec = payload["embedding"]
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr
    except (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        KeyError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        global _WARNED_ONCE
        if not _WARNED_ONCE:
            _WARNED_ONCE = True
            log.warning(
                "bge-m3 embedding call failed (url=%s, err=%s); "
                "dedup will fall through to simhash. Suppressing further warnings.",
                url,
                exc,
            )
        return None


def _reset_warning_state() -> None:
    """Test hook: clear the one-shot warning latch."""
    global _WARNED_ONCE
    _WARNED_ONCE = False


def get_dedup_rejected() -> int:
    """Return current process-global bge-m3 dedup-rejected count."""
    return _DEDUP_REJECTED


def increment_dedup_rejected() -> None:
    """Bump the bge-m3 dedup-rejected counter by 1."""
    global _DEDUP_REJECTED
    _DEDUP_REJECTED += 1


def reset_dedup_rejected() -> None:
    """Zero the bge-m3 dedup-rejected counter (called at run_pipeline start)."""
    global _DEDUP_REJECTED
    _DEDUP_REJECTED = 0
