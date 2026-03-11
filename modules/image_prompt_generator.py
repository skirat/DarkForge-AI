from __future__ import annotations

import json
import logging
from google import genai
from google.genai import types

from config import TEXT_MODEL, IMAGE_STYLE_SUFFIX
from utils.file_utils import parse_json_response

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = f"""\
You are a prompt engineer for AI image generation (Imagen). Convert raw scene \
descriptions into vivid, cinematic image prompts.

Style keywords to incorporate: {IMAGE_STYLE_SUFFIX}

Rules:
- Each prompt must be a single paragraph, 1-3 sentences.
- Focus on composition, lighting, colour palette, and mood.
- Never mention text, letters, or watermarks.
- Return ONLY a JSON array of objects: [{{"scene_id": 1, "image_prompt": "..."}}]
"""


def generate_image_prompts(client: genai.Client, scenes: list[dict]) -> list[dict]:
    """Refine visual_prompt for each scene into a polished Imagen prompt."""
    logger.info("Generating cinematic image prompts …")

    scene_descriptions = json.dumps(
        [{"scene_id": s["scene_id"], "visual_prompt": s["visual_prompt"]} for s in scenes],
        indent=2,
    )

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=f"Scene descriptions:\n{scene_descriptions}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )

    prompts = parse_json_response(response.text)
    logger.info("Refined %d image prompts", len(prompts))
    return prompts
