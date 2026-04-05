from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from config import (
    OPENAI_ENABLED,
    TEXT_MODEL,
    SCENE_DURATION_DEFAULT,
    SCENE_COUNT_MIN,
    SCENE_COUNT_MAX,
    HERO_VEO_CLIP_SEC,
    MAX_SCENE_DURATION,
    WORDS_PER_MINUTE,
)
from modules.openai_llm import openai_chat
from utils.file_utils import parse_json_response
from utils.gemini_retry import with_gemini_client_rotation

logger = logging.getLogger("pipeline")


def allowed_duration_seconds(
    clip_sec: float | None = None,
    max_sec: float | None = None,
) -> list[int]:
    """Multiples of Veo clip length up to MAX_SCENE_DURATION (e.g. 8, 16, 24 for 8s clips)."""
    c = float(clip_sec if clip_sec is not None else HERO_VEO_CLIP_SEC)
    step = max(1, int(round(c)))
    cap = float(max_sec if max_sec is not None else MAX_SCENE_DURATION)
    out: list[int] = []
    x = step
    while x <= cap + 0.001:
        out.append(x)
        x += step
    return out if out else [step]


_ALLOWED_DURATIONS = allowed_duration_seconds()
_ALLOWED_DURATION_STR = ", ".join(str(x) for x in _ALLOWED_DURATIONS)
_BASE_CLIP_SEC = _ALLOWED_DURATIONS[0]

SYSTEM_PROMPT = f"""\
You are a video scene planner. Given a narration script, break it into many sequential \
scenes so the final video is visually dynamic (frequent cuts and varied visuals).

Rules:
- Create between {SCENE_COUNT_MIN} and {SCENE_COUNT_MAX} scenes. Prefer more scenes for visual variety.
- Each scene = one clear story beat or paragraph. Keep narration per scene concise (1–4 sentences).
- narration must stay in plain, simple English (same accessibility as the main script): no unnecessary jargon; \
if a technical word appears, keep it rare and clear from context.
- duration_seconds (critical): MUST be exactly one of: {_ALLOWED_DURATION_STR}. \
These values align with Veo hero clips (base segment {_BASE_CLIP_SEC}s per clip; configurable via HERO_VEO_CLIP_SEC). \
Pick shorter values for brief narration, longer for heavier beats — match spoken length, not arbitrary numbers.
- For each scene provide:
  1. scene_id (integer starting at 1)
  2. narration — the exact text the narrator will read for this scene
  3. visual_prompt — a vivid, specific description of what the viewer should see (different for each scene)
  4. duration_seconds — one of: {_ALLOWED_DURATION_STR}
  5. characters_present — optional array of character ids from the character bible. Use **[]** or omit entirely \
  when there are **no visible people** (pure environment, monitors-only, corridors, abstract, establishing shots \
  without bodies, POV with no face/body of a named character). Include **protagonist** and/or supporting ids **only** \
  when that character is **clearly visible** (face or unambiguous silhouette/body matching the bible).
- When characters_present is non-empty and lists a character, visual_prompt MUST reuse the same physical keywords \
  from the bible (face, hair, outfit) — do not invent a different-looking person.
- CRITICAL: Output one complete valid JSON array. Escape double quotes inside string values. Do not stop until the array is closed.
- Keep each visual_prompt concise (about 40 words or fewer) so the full array fits in one response.

Return ONLY a JSON array:
[
  {{ "scene_id": 1, "narration": "...", "visual_prompt": "...", "duration_seconds": 16, "characters_present": ["protagonist"] }},
  ...
]
"""


def normalize_scene_durations(
    scenes: list[dict],
    *,
    clip_sec: float | None = None,
    max_sec: float | None = None,
) -> list[dict]:
    """Snap each duration_seconds to the nearest allowed Veo-aligned value."""
    allowed = allowed_duration_seconds(clip_sec, max_sec)
    if not allowed:
        return scenes
    for s in scenes:
        if not isinstance(s, dict):
            continue
        raw = s.get("duration_seconds")
        try:
            d = float(raw)
        except (TypeError, ValueError):
            d = float(SCENE_DURATION_DEFAULT)
        nearest = min(allowed, key=lambda x: abs(float(x) - d))
        s["duration_seconds"] = int(nearest)
    return scenes


def _bible_preamble(character_bible: dict[str, Any] | None) -> str:
    if not character_bible:
        return ""
    try:
        slim = {
            "visual_style": character_bible.get("visual_style"),
            "continuity_rules": character_bible.get("continuity_rules"),
            "protagonist": character_bible.get("protagonist"),
            "supporting_characters": character_bible.get("supporting_characters"),
        }
        return (
            "Character bible (MUST keep the same faces and outfits when these characters appear):\n"
            + json.dumps(slim, indent=2, ensure_ascii=False)
            + "\n\n"
        )
    except Exception:
        return ""


def _timing_preamble(target_total_seconds: float | None) -> str:
    if target_total_seconds is None or target_total_seconds <= 0:
        return ""
    return (
        "Planning targets (voiceover length):\n"
        f"- Approximate target sum of all duration_seconds ≈ {target_total_seconds:.0f} seconds "
        f"(total spoken length at ~{WORDS_PER_MINUTE} words per minute).\n"
        f"- Choose a scene count between {SCENE_COUNT_MIN} and {SCENE_COUNT_MAX} so the sum of duration_seconds "
        "is within about 10–15% of that target.\n"
        "- If the target would require more scenes than the maximum at 8s each, prefer longer beats (16s or 24s) "
        "and fewer scenes instead of only 8s clips.\n\n"
    )


def _call_scene_api(
    clients: list[genai.Client],
    script: str,
    character_bible: dict[str, Any] | None = None,
    target_total_seconds: float | None = None,
) -> str:
    prefix = _timing_preamble(target_total_seconds) + _bible_preamble(character_bible)
    contents = f"{prefix}Here is the full narration script:\n\n{script}"

    def _call(c: genai.Client) -> str:
        return c.models.generate_content(
            model=TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.5,
                # Large: many scenes × (narration + visual_prompt) must not be cut off mid-JSON.
                max_output_tokens=65536,
            ),
        ).text

    return with_gemini_client_rotation(
        clients,
        "Scene planning",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                SYSTEM_PROMPT,
                contents,
                json_mode=True,
                temperature=0.5,
                max_tokens=16384,
            )
        )
        if OPENAI_ENABLED
        else None,
    )


def _repair_scenes_json(clients: list[genai.Client], broken_text: str) -> str:
    """Ask the model to fix truncated or invalid scene JSON."""
    snippet = broken_text[:100000]
    allowed = _ALLOWED_DURATION_STR
    repair_contents = (
        "The text below was meant to be a JSON array only. Each element must have: "
        "scene_id (int), narration (string), visual_prompt (string), duration_seconds (number); "
        f"duration_seconds must be one of: {allowed}. "
        "Optionally characters_present (array of strings, or empty array). "
        "It may be truncated or malformed. Return ONLY a complete valid JSON array. "
        "Escape double quotes inside strings. Recover as many complete scenes as possible.\n\n"
        + snippet
    )

    def _call(c: genai.Client) -> str:
        return c.models.generate_content(
            model=TEXT_MODEL,
            contents=repair_contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=65536,
            ),
        ).text

    repair_system = (
        "You repair truncated or invalid JSON. Return ONLY a valid JSON array of scene objects, no prose."
    )
    return with_gemini_client_rotation(
        clients,
        "Scene JSON repair",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                repair_system,
                repair_contents,
                json_mode=True,
                temperature=0.2,
                max_tokens=16384,
            )
        )
        if OPENAI_ENABLED
        else None,
    )


def generate_scenes(
    clients: list[genai.Client],
    script: str,
    *,
    character_bible: dict[str, Any] | None = None,
    target_total_seconds: float | None = None,
) -> list[dict]:
    """Split a narration script into scene objects."""
    logger.info("Breaking script into scenes …")

    text = _call_scene_api(clients, script, character_bible, target_total_seconds)
    try:
        scenes = parse_json_response(text)
    except json.JSONDecodeError as e:
        logger.warning("Scene JSON parse failed (%s), retrying once …", e)
        text = _call_scene_api(clients, script, character_bible, target_total_seconds)
        try:
            scenes = parse_json_response(text)
        except json.JSONDecodeError as e2:
            logger.warning(
                "Scene JSON still invalid (%s), attempting repair pass …", e2
            )
            text = _repair_scenes_json(clients, text)
            scenes = parse_json_response(text)

    if not isinstance(scenes, list):
        raise ValueError("Expected a JSON array of scenes")
    normalize_scene_durations(scenes)
    total_d = sum(
        float(s.get("duration_seconds") or 0)
        for s in scenes
        if isinstance(s, dict)
    )
    if target_total_seconds:
        logger.info(
            "Scenes: %d, sum(duration_seconds)=%.0fs (target ~%.0fs)",
            len(scenes),
            total_d,
            target_total_seconds,
        )
    else:
        logger.info("Generated %d scenes", len(scenes))
    return scenes
