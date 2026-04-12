from __future__ import annotations

import json
import logging
import math
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
    MAX_HERO_PARTS_PER_SCENE,
    MAX_SCENE_DURATION,
    WORDS_PER_MINUTE,
)
from modules.hero_video_generator import _intra_scene_shot_directive
from modules.openai_llm import openai_chat, openai_chat_json_schema
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
  5. characters_present — optional array of character ids from the character bible. Prefer **many** scenes with \
  **[]** (no on-screen people) for variety — only tag **protagonist** or supporting ids when that beat **needs** a \
  visible face or body. Use **[]** when there are **no visible people** (pure environment, monitors-only, corridors, \
  abstract, establishing shots without bodies, POV with no face/body of a named character). Include ids **only** \
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


def _split_duration_seconds(total: int, n: int) -> list[int]:
    """Split total into n positive integers summing to total (even distribution)."""
    total = max(1, int(total))
    n = max(1, n)
    if n == 1:
        return [total]
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


def _veo_part_count(duration_seconds: float, clip_sec: float, max_parts: int) -> int:
    d = max(1.0, float(duration_seconds))
    n = int(math.ceil(d / max(clip_sec, 1.0)))
    return max(1, min(max_parts, n))


def _heuristic_narration_chunks(narration: str, chunks: list[int]) -> list[str]:
    """Split narration into len(chunks) segments by word count proportional to chunk duration."""
    words = (narration or "").split()
    if not words:
        return [""] * len(chunks)
    tot = float(sum(chunks))
    n = len(chunks)
    out: list[str] = []
    wi = 0
    for i, ch in enumerate(chunks):
        if i == n - 1:
            seg = words[wi:]
        else:
            nw = max(1, int(round(len(words) * float(ch) / tot)))
            seg = words[wi : wi + nw]
            wi += len(seg)
        out.append(" ".join(seg).strip())
    return out


def _heuristic_visual_segments(scene: dict, n: int) -> list[str]:
    base = (scene.get("visual_prompt") or "").strip()
    return [
        _fit_segment_visual(base, part, n, scene) for part in range(n)
    ]


def _fit_segment_visual(base: str, part_index: int, total_parts: int, scene: dict) -> str:
    intra = _intra_scene_shot_directive(part_index, total_parts, scene=scene)
    if not base:
        return intra
    if intra:
        return f"{base}\n\n{intra}"
    return base


def _veo_segment_split_json_schema(n: int) -> dict[str, Any]:
    """JSON Schema for API-enforced output: exactly *n* segment objects (Gemini + OpenAI structured)."""
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "narration": {"type": "string"},
                        "visual_prompt": {"type": "string"},
                    },
                    "required": ["narration", "visual_prompt"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }


def _segments_list_from_parsed(data: Any) -> list | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        v = data["segments"]
        return v if isinstance(v, list) else None
    return None


def _parse_segments_response(text: str) -> Any:
    """Parse model JSON; try object span if the whole buffer fails."""
    try:
        return parse_json_response(text)
    except json.JSONDecodeError:
        pass
    t = (text or "").strip()
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start : end + 1])
    raise json.JSONDecodeError("Could not parse segments JSON", text or "", 0)


def _items_to_segment_dicts(items: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "narration": str(item.get("narration", "")).strip(),
                "visual_prompt": str(item.get("visual_prompt", "")).strip(),
            }
        )
    return out


def _align_segment_count(segments: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    """Trim to n or pad with empty dicts so downstream heuristic can fill gaps."""
    if len(segments) > n:
        logger.debug("Trimming segment list from %d to %d", len(segments), n)
        return segments[:n]
    if len(segments) < n:
        return segments + [{"narration": "", "visual_prompt": ""} for _ in range(n - len(segments))]
    return segments


def _expand_scene_segments_llm(
    clients: list[genai.Client],
    scene: dict,
    chunks: list[int],
    character_bible: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Ask the model to split one scene into N narration + visual_prompt rows.

    Uses Gemini/OpenAI **structured JSON** (schema) so output is valid in one shot — no repair loop.
    Returns exactly N segment dicts, or [] on failure (caller uses full heuristic).
    """
    n = len(chunks)
    schema = _veo_segment_split_json_schema(n)
    preamble = _bible_preamble(character_bible)
    payload = {
        "N": n,
        "duration_seconds_list": chunks,
        "original_narration": scene.get("narration", ""),
        "original_visual_prompt": scene.get("visual_prompt", ""),
        "characters_present": scene.get("characters_present"),
    }
    user = preamble + json.dumps(payload, ensure_ascii=False, indent=2)

    system = f"""You split ONE video scene into exactly {n} sequential segments for AI video generation \
(~{HERO_VEO_CLIP_SEC:g}s of voiceover each; durations in duration_seconds_list).

The response must follow the JSON schema: an object with key "segments" containing exactly {n} items. \
Each item has "narration" and "visual_prompt" strings.

Rules:
- Split original_narration at natural sentence boundaries; segments in order must cover the full text without duplication or gaps.
- Each visual_prompt is a distinct cinematic shot for that segment only; together they cover the full visual beat.
- When characters appear, keep the same look as implied by original_visual_prompt and characters_present."""

    def _call(c: genai.Client) -> str:
        r = c.models.generate_content(
            model=TEXT_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.45,
                max_output_tokens=8192,
            ),
        )
        return r.text or ""

    text = with_gemini_client_rotation(
        clients,
        "Scene split for Veo segments",
        _call,
        openai_fallback=(
            lambda: openai_chat_json_schema(
                system,
                user,
                json_schema=schema,
                schema_name="VeoSegmentSplit",
                temperature=0.45,
                max_tokens=8192,
            )
        )
        if OPENAI_ENABLED
        else None,
    )

    data: Any = None
    try:
        data = _parse_segments_response(text)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Scene segment JSON parse failed after structured output (%s); heuristic split", e)
        return []

    raw_list = _segments_list_from_parsed(data)
    if raw_list is None:
        logger.warning("Scene segment response had no list or segments key; using heuristic split")
        return []

    segments = _items_to_segment_dicts(raw_list)
    segments = _align_segment_count(segments, n)
    return segments


def expand_scenes_to_veo_segments(
    clients: list[genai.Client],
    scenes: list[dict],
    *,
    character_bible: dict[str, Any] | None = None,
    clip_sec: float | None = None,
    max_parts: int | None = None,
) -> list[dict]:
    """Split any scene longer than *clip_sec* into multiple rows (~8s each) with distinct narration and visual_prompt.

    Each output row is one Veo-sized segment; renumbered scene_id 1..N. Adds source_scene_id, sub_part_index,
    sub_parts_total, and video_prompt (same as visual_prompt) for downstream video tools.
    """
    csec = float(clip_sec if clip_sec is not None else HERO_VEO_CLIP_SEC)
    cap = int(max_parts if max_parts is not None else MAX_HERO_PARTS_PER_SCENE)
    ordered = sorted(scenes, key=lambda s: int(s.get("scene_id", 0)))
    out: list[dict] = []

    for scene in ordered:
        if not isinstance(scene, dict):
            continue
        orig_id = int(scene.get("scene_id", 0))
        d = float(scene.get("duration_seconds") or csec)

        if d <= csec + 1e-6:
            row = dict(scene)
            row["duration_seconds"] = max(1, int(round(d)))
            row["source_scene_id"] = orig_id
            row["sub_part_index"] = 0
            row["sub_parts_total"] = 1
            vp = (row.get("visual_prompt") or "").strip()
            row["video_prompt"] = vp
            out.append(row)
            continue

        n = _veo_part_count(d, csec, cap)
        tot = max(1, int(round(d)))
        chunks = _split_duration_seconds(tot, n)

        try:
            segments = _expand_scene_segments_llm(
                clients, scene, chunks, character_bible
            )
        except Exception as exc:
            logger.warning(
                "LLM scene split failed for source scene %s (%s); using heuristic",
                orig_id,
                exc,
            )
            segments = []
        if not segments:
            logger.debug(
                "Source scene %s: no LLM segments (repair failed or empty); heuristic narration/visuals",
                orig_id,
            )
            segments = [{"narration": "", "visual_prompt": ""} for _ in range(n)]
        elif len(segments) != n:
            segments = _align_segment_count(segments, n)

        narrations = _heuristic_narration_chunks(
            str(scene.get("narration") or ""), chunks
        )
        visuals = _heuristic_visual_segments(scene, n)

        for i, ch in enumerate(chunks):
            seg_n = ""
            seg_v = ""
            if i < len(segments) and segments[i].get("narration"):
                seg_n = segments[i]["narration"]
            if i < len(segments) and segments[i].get("visual_prompt"):
                seg_v = segments[i]["visual_prompt"]
            if not seg_n:
                seg_n = narrations[i] if i < len(narrations) else ""
            if not seg_v:
                seg_v = visuals[i] if i < len(visuals) else ""

            cp = scene.get("characters_present")
            row: dict[str, Any] = {
                "narration": seg_n,
                "visual_prompt": seg_v,
                "video_prompt": seg_v,
                "duration_seconds": int(ch),
                "characters_present": cp if isinstance(cp, list) else cp,
                "source_scene_id": orig_id,
                "sub_part_index": i,
                "sub_parts_total": len(chunks),
            }
            out.append(row)

    for i, row in enumerate(out, start=1):
        row["scene_id"] = i

    logger.info(
        "Expanded scenes for ~%.0fs Veo segments: %d planner row(s) → %d segment row(s)",
        csec,
        len(ordered),
        len(out),
    )
    return out


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
