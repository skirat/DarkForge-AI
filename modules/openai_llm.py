"""OpenAI Chat Completions fallback when Gemini text quota fails."""
from __future__ import annotations

import logging
from typing import Any

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


def openai_chat_json_schema(
    system_instruction: str,
    user_content: str,
    *,
    json_schema: dict[str, Any],
    schema_name: str = "response",
    temperature: float = 0.45,
    max_tokens: int | None = 8192,
) -> str:
    """Chat completion with Structured Outputs (json_schema). Falls back to json_object if unsupported."""
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
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            },
        },
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned empty content")
        logger.debug("OpenAI structured output ok (model=%s)", OPENAI_TEXT_MODEL)
        return text
    except Exception as exc:
        logger.warning(
            "OpenAI json_schema mode failed (%s); retrying with json_object",
            exc,
        )
        kwargs.pop("response_format", None)
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned empty content")
        return text
