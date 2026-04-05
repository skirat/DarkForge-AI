"""Optional background music via Google Lyria 3 (Gemini API).

See: https://ai.google.dev/gemini-api/docs/music-generation

Requires billing / preview access; falls back to file-based music on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from google.genai import types

from config import LYRIA_MODEL, LYRIA_BGM

logger = logging.getLogger("pipeline")


def _build_prompt(
    metadata: dict | None,
    scenes: list[dict] | None,
    character_bible: dict | None = None,
) -> str:
    title = (metadata or {}).get("title") or "dark horror story"
    hook = (metadata or {}).get("hook") or ""
    tags = (metadata or {}).get("tags") or []
    tag_str = ", ".join(str(t) for t in tags[:8]) if tags else ""
    narr_snip = ""
    if scenes:
        for s in scenes[:4]:
            narr_snip += (s.get("narration") or "")[:180] + " "
    narr_snip = narr_snip.strip()[:600]
    pro_line = ""
    if character_bible:
        p = character_bible.get("protagonist")
        if isinstance(p, dict):
            pro_line = f" Lead character mood: {p.get('name', '')} — {p.get('role', '')}."

    # Instrumental-only: API may still return sparse vocals; prompt minimizes it.
    return (
        "Create a 30-second instrumental underscore only — no singing, no spoken words. "
        "Dark cinematic horror mood: low drones, subtle tension, sparse pulses, wide stereo. "
        "Suitable as background under voice narration for a YouTube horror video. "
        f"Title theme: {title}. Hook: {hook}. Keywords: {tag_str}.{pro_line} Story excerpt: {narr_snip}"
    )[:8000]


def try_generate_lyria_bed(
    clients: list[genai.Client],
    metadata: dict | None,
    scenes: list[dict] | None,
    out_path: Path,
    *,
    character_bible: dict | None = None,
) -> bool:
    """Generate one Lyria clip (30s MP3), save to *out_path*. Returns True on success."""
    if not LYRIA_BGM or not clients:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1000:
        logger.info("Using cached Lyria bed: %s", out_path.name)
        return True

    prompt = _build_prompt(metadata, scenes, character_bible)
    last_err: Exception | None = None

    for i, client in enumerate(clients):
        try:
            response = client.models.generate_content(
                model=LYRIA_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO", "TEXT"],
                ),
            )
            cand = response.candidates
            if not cand:
                raise ValueError("No candidates in Lyria response")
            parts = cand[0].content.parts
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    out_path.write_bytes(inline.data)
                    logger.info(
                        "Lyria background music saved → %s (key %d/%d)",
                        out_path.name,
                        i + 1,
                        len(clients),
                    )
                    return True
            raise ValueError("Lyria response had no audio inline_data")
        except Exception as exc:
            last_err = exc
            is_rl = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            logger.warning(
                "Lyria generation failed (key %d/%d): %s%s",
                i + 1,
                len(clients),
                exc,
                " — trying next key" if is_rl and i < len(clients) - 1 else "",
            )
            if i < len(clients) - 1 and is_rl:
                continue
            if not is_rl:
                break

    logger.warning("Lyria unavailable (%s); using file/fallback music.", last_err)
    return False
