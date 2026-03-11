from __future__ import annotations

import json
import logging
from google import genai
from google.genai import types

from config import (
    TEXT_MODEL,
    SCENE_DURATION_DEFAULT,
    SCENE_COUNT_MIN,
    SCENE_COUNT_MAX,
)
from utils.file_utils import parse_json_response

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = f"""\
You are a video scene planner. Given a narration script, break it into many sequential \
scenes so the final video is visually dynamic (frequent cuts and varied visuals).

Rules:
- Create between {SCENE_COUNT_MIN} and {SCENE_COUNT_MAX} scenes. Prefer more scenes for visual variety.
- Each scene = one clear story beat or paragraph. Keep narration per scene concise (1–4 sentences).
- Default duration per scene: {SCENE_DURATION_DEFAULT} seconds (adjust 10–25s based on text length).
- For each scene provide:
  1. scene_id (integer starting at 1)
  2. narration — the exact text the narrator will read for this scene
  3. visual_prompt — a vivid, specific description of what the viewer should see (different for each scene)
  4. duration_seconds — estimated screen time
- CRITICAL: Output a single, complete, valid JSON array. Escape any quotes inside strings. Do not truncate.

Return ONLY a JSON array:
[
  {{ "scene_id": 1, "narration": "...", "visual_prompt": "...", "duration_seconds": 12 }},
  ...
]
"""


def _call_scene_api(client: genai.Client, script: str) -> str:
    return client.models.generate_content(
        model=TEXT_MODEL,
        contents=f"Here is the full narration script:\n\n{script}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.5,
            max_output_tokens=16384,
        ),
    ).text


def generate_scenes(client: genai.Client, script: str) -> list[dict]:
    """Split a narration script into scene objects."""
    logger.info("Breaking script into scenes …")

    text = _call_scene_api(client, script)
    try:
        scenes = parse_json_response(text)
    except json.JSONDecodeError as e:
        logger.warning("Scene JSON parse failed (%s), retrying once …", e)
        text = _call_scene_api(client, script)
        scenes = parse_json_response(text)

    if not isinstance(scenes, list):
        raise ValueError("Expected a JSON array of scenes")
    logger.info("Generated %d scenes", len(scenes))
    return scenes
