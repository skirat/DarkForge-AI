"""Rotate through Gemini API clients on quota / rate limits (same pattern as image/TTS)."""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from google import genai

from config import MAX_RETRIES, RETRY_BACKOFF, RETRY_RATE_LIMIT_WAIT

logger = logging.getLogger("pipeline")

# Brief pause before trying the next key after 429 (aligns with image_generator)
TEXT_ROTATE_KEY_WAIT_SEC = 12.0

T = TypeVar("T")


def is_rate_limit_error(exc: BaseException) -> bool:
    s = str(exc)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def max_text_rotation_attempts(n_clients: int) -> int:
    """Enough full passes over all keys plus retries (e.g. 4 keys * 6 rounds = 24)."""
    n = max(1, n_clients)
    return max(24, MAX_RETRIES * max(n, 4))


def with_gemini_client_rotation(
    clients: list[genai.Client],
    operation_name: str,
    fn: Callable[[genai.Client], T],
    *,
    max_attempts: int | None = None,
    openai_fallback: Callable[[], T] | None = None,
) -> T:
    """Call *fn(client)* rotating *clients* on failure until success or *max_attempts*.

    On 429 / RESOURCE_EXHAUSTED with multiple keys, waits TEXT_ROTATE_KEY_WAIT_SEC then tries
    the next key. With one key, uses longer rate-limit wait. Other errors use exponential backoff.

    If *clients* is empty or all attempts fail and *openai_fallback* is set, calls it instead.
    """
    if not clients:
        if openai_fallback is not None:
            logger.info("No Gemini API keys; using OpenAI fallback for %s", operation_name)
            return openai_fallback()
        raise ValueError("At least one Gemini client is required (or set OPENAI_API_KEY for fallback)")
    n = len(clients)
    if max_attempts is None:
        max_attempts = max_text_rotation_attempts(n)
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        client = clients[(attempt - 1) % n]
        try:
            return fn(client)
        except Exception as exc:
            last_err = exc
            is_rl = is_rate_limit_error(exc)
            if n > 1 and is_rl:
                wait = TEXT_ROTATE_KEY_WAIT_SEC
                hint = "next key"
            elif is_rl:
                wait = RETRY_RATE_LIMIT_WAIT
                hint = "rate limit"
            else:
                wait = RETRY_BACKOFF ** min(attempt, 8)
                hint = "retry"
            logger.warning(
                "%s failed (attempt %d/%d, key %d/%d): %s – %s in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                (attempt - 1) % n + 1,
                n,
                type(exc).__name__,
                hint,
                wait,
            )
            time.sleep(wait)
    assert last_err is not None
    if openai_fallback is not None:
        logger.warning(
            "Gemini exhausted for %s after %d attempt(s) (%s); using OpenAI fallback",
            operation_name,
            max_attempts,
            type(last_err).__name__,
        )
        return openai_fallback()
    raise last_err
