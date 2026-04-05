from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from config import OPENAI_ENABLED, TEXT_MODEL, IMAGE_STYLE_SUFFIX
from modules.character_bible import compact_style_for_prompt
from modules.openai_llm import openai_chat
from utils.file_utils import parse_json_response
from utils.gemini_retry import with_gemini_client_rotation

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = f"""\
You are a prompt engineer for AI image generation (Imagen). Convert raw scene \
descriptions into vivid, cinematic image prompts.

Style keywords to incorporate: {IMAGE_STYLE_SUFFIX}

Rules:
- Each prompt must be a single paragraph, 1-3 sentences.
- Focus on composition, lighting, colour palette, and mood.
- When characters_present lists a character, merge that character's image_prompt_fragment from the bible into the final image_prompt (same face/clothes as bible).
- Never mention text, letters, or watermarks.
- Return ONLY a JSON array of objects: [{{"scene_id": 1, "image_prompt": "..."}}]
"""


def _character_lookup(bible: dict[str, Any] | None) -> dict[str, str]:
    """Map character id -> image_prompt_fragment."""
    if not bible or not isinstance(bible, dict):
        return {}
    out: dict[str, str] = {}
    p = bible.get("protagonist")
    if isinstance(p, dict):
        pid = str(p.get("id") or "protagonist")
        frag = (p.get("image_prompt_fragment") or "").strip()
        if frag:
            out[pid] = frag
    for c in bible.get("supporting_characters") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        frag = (c.get("image_prompt_fragment") or "").strip()
        if cid and frag:
            out[cid] = frag
    return out


def generate_image_prompts(
    clients: list[genai.Client],
    scenes: list[dict],
    *,
    character_bible: dict[str, Any] | None = None,
) -> list[dict]:
    """Refine visual_prompt for each scene into a polished Imagen prompt."""
    logger.info("Generating cinematic image prompts …")

    style_line = compact_style_for_prompt(
        (character_bible or {}).get("visual_style") or {}
    )
    continuity = ((character_bible or {}).get("continuity_rules") or "").strip()

    scene_payload = []
    for s in scenes:
        row: dict[str, Any] = {
            "scene_id": s["scene_id"],
            "visual_prompt": s["visual_prompt"],
        }
        cp = s.get("characters_present")
        if isinstance(cp, list) and cp:
            row["characters_present"] = cp
        scene_payload.append(row)

    bible_hint = ""
    if character_bible:
        bible_hint = (
            "Global visual style line (prepend to each image_prompt when relevant):\n"
            + (style_line + "\n" if style_line else "")
            + (continuity + "\n" if continuity else "")
            + "Character fragments by id (use when that id is in characters_present):\n"
            + json.dumps(_character_lookup(character_bible), indent=2, ensure_ascii=False)
            + "\n\n"
        )

    scene_descriptions = json.dumps(scene_payload, indent=2)
    contents = f"{bible_hint}Scene descriptions:\n{scene_descriptions}"

    def _call(c: genai.Client) -> str:
        response = c.models.generate_content(
            model=TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        return response.text

    text = with_gemini_client_rotation(
        clients,
        "Image prompt refinement",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                SYSTEM_PROMPT,
                contents,
                json_mode=True,
                temperature=0.7,
                max_tokens=8192,
            )
        )
        if OPENAI_ENABLED
        else None,
    )
    prompts = parse_json_response(text)
    logger.info("Refined %d image prompts", len(prompts))
    return prompts
