"""OpenAI Chat Completions fallback when Gemini text quota fails."""
from __future__ import annotations

import logging

from config import OPENAI_API_KEY, OPENAI_TEXT_MODEL

logger = logging.getLogger("pipeline")


def openai_chat(
    system_instruction: str,
    user_content: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Run a chat completion; optional JSON object mode."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs: dict = {
        "model": OPENAI_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned empty content")
    logger.debug("OpenAI chat ok (model=%s, json=%s)", OPENAI_TEXT_MODEL, json_mode)
    return text
