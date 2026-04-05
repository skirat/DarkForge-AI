"""Per-story character bible for consistent stills and Veo hero clips."""
from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from config import OPENAI_ENABLED, TEXT_MODEL
from modules.openai_llm import openai_chat
from utils.file_utils import parse_json_response
from utils.gemini_retry import with_gemini_client_rotation

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = """\
You are a casting director and visual continuity supervisor for darknet / cyber-horror YouTube stories.

Given the video concept and full narration script, return ONLY valid JSON with this exact structure:
{
  "schema_version": 1,
  "visual_style": {
    "palette": "short string: dominant colors and contrast",
    "lighting": "short string: key lighting mood",
    "film_look": "short string: grain, lens, aspect feel",
    "motifs": "short string: recurring visual motifs"
  },
  "continuity_rules": "one sentence: e.g. same actor likeness and wardrobe for recurring characters unless story says otherwise",
  "protagonist": {
    "id": "protagonist",
    "name": "first name or nickname",
    "role": "one line who they are in this story",
    "personality_traits": ["2-5 short traits"],
    "physical": {
      "age_band": "e.g. late 20s",
      "build": "e.g. lean, broad",
      "hair": "color, cut",
      "skin": "tone and any note for consistent rendering",
      "distinctive": "one memorable visual detail",
      "default_outfit": "clothes they wear across most scenes"
    },
    "image_prompt_fragment": "ONE compact line, max ~200 characters, copy-paste safe: physical look + outfit + vibe for image/video models; no quotes inside; same wording whenever this character appears on screen"
  },
  "supporting_characters": [
    {
      "id": "slug_id",
      "name": "...",
      "role": "...",
      "personality_traits": ["..."],
      "physical": { same keys as protagonist.physical },
      "image_prompt_fragment": "same rules as protagonist, max ~200 chars"
    }
  ]
}

Rules:
- The viewer is the male protagonist in second-person scripts; define ONE clear male lead in protagonist.
- Only include supporting_characters who actually appear or are clearly implied in the script; use [] if none.
- image_prompt_fragment fields must be concrete and repeatable (not vague).
- Escape double quotes inside strings; no trailing commas.
"""


def generate_character_bible(
    clients: list[genai.Client],
    video_prompt: str,
    script: str,
    *,
    title: str = "",
    hook: str = "",
) -> dict[str, Any]:
    """Produce a character bible JSON dict for the story."""
    logger.info("Generating character bible …")
    extras = []
    if title:
        extras.append(f"Working title: {title}")
    if hook:
        extras.append(f"Hook: {hook}")
    extra_block = "\n".join(extras) + "\n\n" if extras else ""

    user_msg = (
        f"{extra_block}"
        f"Video concept:\n{video_prompt}\n\n"
        f"Full narration script:\n\n{script}"
    )

    def _call(c: genai.Client) -> str:
        response = c.models.generate_content(
            model=TEXT_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.65,
            ),
        )
        return response.text

    text = with_gemini_client_rotation(
        clients,
        "Character bible",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                SYSTEM_PROMPT,
                user_msg,
                json_mode=True,
                temperature=0.65,
                max_tokens=8192,
            )
        )
        if OPENAI_ENABLED
        else None,
    )
    data = parse_json_response(text)
    data = _normalize_bible(data)
    logger.info(
        "Character bible: protagonist=%s, supporting=%d",
        (data.get("protagonist") or {}).get("name", "?"),
        len(data.get("supporting_characters") or []),
    )
    return data


def _normalize_bible(data: Any) -> dict[str, Any]:
    """Ensure expected keys exist for downstream code."""
    if not isinstance(data, dict):
        return _empty_bible()
    data.setdefault("schema_version", 1)
    vs = data.get("visual_style")
    if not isinstance(vs, dict):
        data["visual_style"] = {
            "palette": "",
            "lighting": "",
            "film_look": "",
            "motifs": "",
        }
    else:
        for k in ("palette", "lighting", "film_look", "motifs"):
            vs.setdefault(k, "")
    data.setdefault("continuity_rules", "")
    if not isinstance(data.get("protagonist"), dict):
        data["protagonist"] = {}
    p = data["protagonist"]
    p.setdefault("id", "protagonist")
    p.setdefault("name", "")
    p.setdefault("role", "")
    p.setdefault("personality_traits", [])
    if not isinstance(p.get("physical"), dict):
        p["physical"] = {}
    phys = p["physical"]
    for k in ("age_band", "build", "hair", "skin", "distinctive", "default_outfit"):
        phys.setdefault(k, "")
    p.setdefault("image_prompt_fragment", "")
    sup = data.get("supporting_characters")
    if not isinstance(sup, list):
        data["supporting_characters"] = []
    else:
        for c in sup:
            if isinstance(c, dict):
                c.setdefault("id", "supporting")
                c.setdefault("image_prompt_fragment", "")
                if not isinstance(c.get("physical"), dict):
                    c["physical"] = {}
                for k in ("age_band", "build", "hair", "skin", "distinctive", "default_outfit"):
                    c["physical"].setdefault(k, "")
    return data


def _empty_bible() -> dict[str, Any]:
    return _normalize_bible({})


def compact_style_for_prompt(visual_style: dict[str, Any]) -> str:
    """Single line from visual_style for injection into prompts."""
    if not visual_style:
        return ""
    parts = [
        visual_style.get("palette", ""),
        visual_style.get("lighting", ""),
        visual_style.get("film_look", ""),
        visual_style.get("motifs", ""),
    ]
    return "; ".join(p for p in parts if p).strip()
