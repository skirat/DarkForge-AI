from __future__ import annotations

import logging
from google import genai
from google.genai import types

from config import OPENAI_ENABLED, TEXT_MODEL
from modules.openai_llm import openai_chat
from utils.file_utils import parse_json_response
from utils.gemini_retry import with_gemini_client_rotation

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = """\
You are a YouTube SEO and content strategist specialising in ethical hacking, \
darknet stories, and cyberpunk horror. Given a video idea, produce metadata that \
maximises clicks and watch-time.

Return ONLY valid JSON with these keys:
{
  "title": "eye-catching YouTube title (60 chars max)",
  "hook": "a gripping 2-sentence opening hook for the video (plain English, readable for non-experts; avoid jargon)",
  "description": "YouTube description (~150 words, include keywords)",
  "tags": ["exactly", "five", "or", "six", "strings"],
  "thumbnail_prompt": "a short image-generation prompt for a clickable thumbnail"
}

For "tags": output EXACTLY 5 or 6 strings (no more, no fewer). Each tag must be \
directly tied to this video's specific topic. Prioritise high-search-volume and \
trending keywords in this niche (ethical hacking, dark web, cyber horror, etc.). \
Prefer 1–3 words per tag; keep them short and punchy. Mix a few broad reach terms \
with a few laser-specific terms. Do NOT use vague tags like "video", "youtube", \
"trending", "subscribe", or anything unrelated to the story. Do not number tags \
or add explanations—only the tag strings in the JSON array.
"""


def normalize_youtube_tags(raw: object) -> list[str]:
    """Dedupe, trim, and cap at 6 tags. Accepts list or a single comma-separated string."""
    items: list[str] = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, list):
        for t in raw:
            if isinstance(t, str):
                s = " ".join(t.split()).strip()
                if s:
                    items.append(s)
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
        if len(out) >= 6:
            break
    return out


def generate_metadata(clients: list[genai.Client], prompt: str) -> dict:
    """Call Gemini to produce title, hook, description, tags, and thumbnail prompt."""
    logger.info("Generating YouTube metadata …")

    def _call(c: genai.Client) -> str:
        response = c.models.generate_content(
            model=TEXT_MODEL,
            contents=f"Video idea: {prompt}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.9,
            ),
        )
        return response.text

    text = with_gemini_client_rotation(
        clients,
        "YouTube metadata",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                SYSTEM_PROMPT,
                f"Video idea: {prompt}",
                json_mode=True,
                temperature=0.9,
                max_tokens=4096,
            )
        )
        if OPENAI_ENABLED
        else None,
    )
    data = parse_json_response(text)
    data["tags"] = normalize_youtube_tags(data.get("tags"))
    logger.info("Title: %s", data.get("title"))
    return data
