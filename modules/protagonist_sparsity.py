"""After scene expansion, limit on-screen characters to a fraction of Veo rows for visual variety.

Story narration is unchanged; only characters_present and visual/video prompts are adjusted.
"""
from __future__ import annotations

import hashlib
import logging
from copy import deepcopy
from typing import Any

from config import PROTAGONIST_CLIP_RATIO
from modules.character_bible import compact_style_for_prompt

logger = logging.getLogger("pipeline")


def _seed_string(metadata: dict[str, Any] | None, script: str) -> str:
    title = (metadata or {}).get("title") or ""
    hook = (metadata or {}).get("hook") or ""
    head = (script or "")[:800]
    raw = f"{title}\n{hook}\n{head}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pick_keep_indices(n: int, k: int, seed: str) -> set[int]:
    """Deterministically pick exactly *k* indices in [0, n) for rows that keep planner character tags."""
    if n <= 0:
        return set()
    k = max(0, min(n, k))
    scored: list[tuple[str, int]] = [
        (hashlib.sha256(f"{seed}:{i}".encode("utf-8")).hexdigest(), i) for i in range(n)
    ]
    scored.sort(key=lambda x: x[0])
    return {i for _, i in scored[:k]}


def _no_people_visual(
    narration: str,
    character_bible: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> str:
    nar = (narration or "").strip()
    if len(nar) > 420:
        nar = nar[:417] + "..."
    vs = compact_style_for_prompt((character_bible or {}).get("visual_style") or {})
    hook = ((metadata or {}).get("hook") or "").strip()
    if len(hook) > 220:
        hook = hook[:217] + "..."
    parts = [
        "Cinematic 16:9, no visible people or identifiable faces in frame.",
        "Match the voiceover beat with environment, light, weather, architecture, objects, props,",
        "screens (glow or UI only, no faces), hands-free details, abstract textures, or symbolic imagery.",
        f"Voiceover: {nar}",
    ]
    if hook:
        parts.append(f"Series tone: {hook}")
    if vs:
        parts.append(f"Visual style: {vs}")
    return " ".join(parts).strip()


def apply_protagonist_sparsity(
    scenes: list[dict[str, Any]],
    character_bible: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    *,
    script: str = "",
    ratio: float | None = None,
) -> list[dict[str, Any]]:
    """Keep character tags + planner visuals on a fraction of rows; others become no-people, narration-led.

    Selection is deterministic given *script* + *metadata* and row order so re-runs and cached scenes.json stay stable.
    """
    r = float(PROTAGONIST_CLIP_RATIO if ratio is None else ratio)
    r = max(0.05, min(0.5, r))

    ordered = sorted(
        [s for s in scenes if isinstance(s, dict)],
        key=lambda s: int(s.get("scene_id", 0)),
    )
    n = len(ordered)
    if n == 0:
        return []

    k = max(1, int(round(n * r)))
    if k >= n:
        logger.info(
            "Protagonist/character clip ratio %.0f%%: keeping planner characters on all %d segment row(s)",
            100 * r,
            n,
        )
        return [deepcopy(s) for s in ordered]

    seed = _seed_string(metadata, script)
    keep_idx = _pick_keep_indices(n, k, seed)

    out: list[dict[str, Any]] = []
    for i, row in enumerate(ordered):
        s = deepcopy(row)
        if i in keep_idx:
            out.append(s)
            continue

        s["characters_present"] = []
        narr = str(s.get("narration") or "")
        nv = _no_people_visual(narr, character_bible, metadata)
        s["visual_prompt"] = nv
        s["video_prompt"] = nv
        out.append(s)

    logger.info(
        "Protagonist/character clip ratio ~%.0f%%: character-capable rows %d/%d (rest = no on-screen characters)",
        100 * r,
        k,
        n,
    )
    return out
