from __future__ import annotations

import logging
from google import genai
from google.genai import types

from config import TEXT_MODEL
from utils.file_utils import parse_json_response

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = """\
You are a YouTube SEO and content strategist specialising in ethical hacking, \
darknet stories, and cyberpunk horror. Given a video idea, produce metadata that \
maximises clicks and watch-time.

Return ONLY valid JSON with these keys:
{
  "title": "eye-catching YouTube title (60 chars max)",
  "hook": "a gripping 2-sentence opening hook for the video",
  "description": "YouTube description (~150 words, include keywords)",
  "tags": ["list", "of", "relevant", "tags"],
  "thumbnail_prompt": "a short image-generation prompt for a clickable thumbnail"
}
"""


def generate_metadata(client: genai.Client, prompt: str) -> dict:
    """Call Gemini to produce title, hook, description, tags, and thumbnail prompt."""
    logger.info("Generating YouTube metadata …")

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=f"Video idea: {prompt}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.9,
        ),
    )

    data = parse_json_response(response.text)
    logger.info("Title: %s", data.get("title"))
    return data
