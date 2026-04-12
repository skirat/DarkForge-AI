"""Generate short hero-scene videos with Veo (Google Labs Flow)."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import (
    VEO_MODELS,
    HERO_VIDEOS_DIR,
    HERO_VEO_CLIP_SEC,
    MAX_HERO_PARTS_PER_SCENE,
    SCENE_DURATION_DEFAULT,
    VEO_KEY_COOLDOWN_SEC,
    VEO_MAX_ROUNDS,
    VEO_POLL_INTERVAL_SEC,
    VEO_POLL_TIMEOUT_SEC,
    VEO_RETRY_ROUND_WAIT_SEC,
)

from modules.character_bible import compact_style_for_prompt

logger = logging.getLogger("pipeline")

VEO_PROMPT_MAX = 1000


def _ordered_scenes(scenes: list[dict]) -> list[dict]:
    return sorted(scenes, key=lambda s: int(s.get("scene_id", 0)))


def _neighbor_scenes(scenes: list[dict], scene_id: int) -> tuple[dict | None, dict | None]:
    ordered = _ordered_scenes(scenes)
    ids = [int(s["scene_id"]) for s in ordered]
    try:
        i = ids.index(int(scene_id))
    except ValueError:
        return None, None
    prev_s = ordered[i - 1] if i > 0 else None
    next_s = ordered[i + 1] if i + 1 < len(ordered) else None
    return prev_s, next_s


def _snippet_visual(scene: dict | None, max_len: int) -> str:
    if not scene:
        return ""
    vp = (scene.get("visual_prompt") or "").strip()
    if len(vp) <= max_len:
        return vp
    return vp[: max_len - 1] + "…"


def _scene_has_any_character(scene: dict | None) -> bool:
    """True when characters_present lists at least one id (visible cast on screen)."""
    if not scene:
        return False
    cp = scene.get("characters_present")
    if not isinstance(cp, list):
        return False
    return len(cp) > 0


def _include_protagonist_lock(scene: dict) -> bool:
    """True only when protagonist is explicitly listed in characters_present."""
    cp = scene.get("characters_present")
    if cp is None:
        return False
    if not isinstance(cp, list):
        return False
    if len(cp) == 0:
        return False
    return any(str(x).strip().lower() == "protagonist" for x in cp)


def _intra_scene_shot_directive(
    part_index: int,
    total_parts: int,
    scene: dict | None = None,
) -> str:
    """Strong, distinct cinematography per clip within ONE scene — avoids repetitive hero parts.

    Character appearance stays identical via bible locks; only camera, lens, motion, and beat change.
    For no-character scenes, directives avoid faces/bodies and emphasize environment and inserts.
    """
    if total_parts <= 1:
        return ""
    i = part_index
    n = total_parts
    no_people = scene is not None and not _scene_has_any_character(scene)
    anti = (
        "ANTI-REPEAT: This clip must NOT reuse the same framing, height, distance, or motion as "
        "the other clips for this scene — each segment is different coverage of the same moment."
    )
    if no_people:
        if n == 2:
            pair = [
                (
                    "INTRA-SCENE SHOT 1/2 — ESTABLISHING: wide; architecture, weather, or room geography; "
                    "no people; slow drift or tripod."
                ),
                (
                    "INTRA-SCENE SHOT 2/2 — ALT COVERAGE: new angle or focal plane; props, screen glow, "
                    "texture, or symbolic object — NOT a reshoot of shot 1."
                ),
            ]
            return anti + "\n" + pair[min(i, 1)]
        if n == 3:
            triple = [
                (
                    "INTRA-SCENE SHOT 1/3 — MASTER: wide/medium-wide; space and light; establish mood; no figures."
                ),
                (
                    "INTRA-SCENE SHOT 2/3 — COVERAGE: medium; NEW angle; emphasize objects, corridor depth, "
                    "or monitor/UI glow — distinct from shot 1."
                ),
                (
                    "INTRA-SCENE SHOT 3/3 — DETAIL: close insert of object, rain on glass, cable, LED, "
                    "or abstract texture — shallow DOF; NOT matching 1 or 2."
                ),
            ]
            return anti + "\n" + triple[min(i, 2)]
        cycle = [
            "INTRA-SCENE — WIDE: environment only; establish.",
            "INTRA-SCENE — MEDIUM: new angle; objects and space; no bodies.",
            "INTRA-SCENE — CLOSE/INSERT: prop, screen edge, texture, or symbolic detail.",
            "INTRA-SCENE — ACCENT: low/high angle, foreground frame, slow slide — fresh beat.",
        ]
        return (
            anti
            + "\n"
            + cycle[i % len(cycle)]
            + f" (segment {i + 1}/{n}; must differ from all other segments)."
        )
    if n == 2:
        pair = [
            (
                "INTRA-SCENE SHOT 1/2 — ESTABLISHING: wide or medium-wide; show room and figure; "
                "slow drift or tripod; environmental read."
            ),
            (
                "INTRA-SCENE SHOT 2/2 — ALT COVERAGE: different camera side or height (e.g. 3/4, "
                "over-shoulder, opposite eyeline); medium shot; new focal point — NOT a reshoot of shot 1."
            ),
        ]
        return anti + "\n" + pair[min(i, 1)]
    if n == 3:
        triple = [
            (
                "INTRA-SCENE SHOT 1/3 — MASTER: wide/medium-wide; blocking and space clear; "
                "minimal move; establish geography."
            ),
            (
                "INTRA-SCENE SHOT 2/3 — COVERAGE: medium; NEW angle (not master repeat); "
                "emphasize desk, hands toward keyboard, or profile — distinct silhouette from shot 1."
            ),
            (
                "INTRA-SCENE SHOT 3/3 — DETAIL/EMOTION: close or medium-close; shallow DOF; "
                "face, hands, or monitor reflection; slow push or rack focus — NOT matching 1 or 2."
            ),
        ]
        return anti + "\n" + triple[min(i, 2)]
    # 4+ parts: cycle through four distinct coverage types
    cycle = [
        "INTRA-SCENE — WIDE MASTER: environment + figure; establish.",
        "INTRA-SCENE — MEDIUM A: new angle/side; action at workstation.",
        "INTRA-SCENE — CLOSE/INSERT: hands, eyes, or screen; shallow focus.",
        "INTRA-SCENE — ACCENT: low/high angle, silhouette, foreground frame, or slow orbit — fresh beat.",
    ]
    return (
        anti
        + "\n"
        + cycle[i % len(cycle)]
        + f" (segment {i + 1}/{n}; must differ from all other segments)."
    )


def _identity_variation_reminder(total_parts: int, scene: dict | None = None) -> str:
    if total_parts <= 1:
        return ""
    if scene is not None and not _scene_has_any_character(scene):
        return ""
    return (
        "Identity: same face, hair, skin, and wardrobe as bible — only camera, lens, motion, "
        "and composition change between clips."
    )


def _character_lock_lines(
    character_bible: dict[str, Any] | None,
    scene: dict,
) -> list[str]:
    lines: list[str] = []
    if not character_bible:
        return lines
    cr = (character_bible.get("continuity_rules") or "").strip()
    if cr:
        lines.append(cr)
    st = compact_style_for_prompt(character_bible.get("visual_style") or {})
    if st:
        lines.append(st)
    p = character_bible.get("protagonist")
    if isinstance(p, dict) and _include_protagonist_lock(scene):
        pfrag = (p.get("image_prompt_fragment") or "").strip()
        if pfrag:
            lines.append("Lead character (match exactly): " + pfrag[:280])
    cp = scene.get("characters_present")
    ids: list[str] = []
    if isinstance(cp, list):
        ids = [str(x).strip() for x in cp]
    by_id = {
        str(c.get("id")): c
        for c in (character_bible.get("supporting_characters") or [])
        if isinstance(c, dict) and c.get("id")
    }
    pid = str((character_bible.get("protagonist") or {}).get("id") or "protagonist")
    for cid in ids:
        if cid.lower() == "protagonist" or cid == pid:
            continue
        c = by_id.get(cid)
        if isinstance(c, dict):
            frag = (c.get("image_prompt_fragment") or "").strip()
            if frag:
                lines.append(f"Character {cid} (match exactly): {frag[:220]}")
    return lines


def _fit_veo_prompt(
    head: str,
    neighbor_block: str,
    visual: str,
    max_len: int,
) -> str:
    """Keep head (locks + shot variation) intact; trim visual then neighbors if over budget."""
    visual = (visual or "").strip()
    sep = "\n\n"
    # Try full assembly
    def assemble(v: str, nb: str) -> str:
        chunks: list[str] = []
        if head:
            chunks.append(head)
        if nb:
            chunks.append(nb)
        if v:
            chunks.append(v)
        return sep.join(chunks)

    nb = neighbor_block.strip()
    body = assemble(visual, nb)
    if len(body) <= max_len:
        return body
    # Drop neighbor text first for space (intra-scene continuity matters more than cross-scene)
    body = assemble(visual, "")
    if len(body) <= max_len:
        return body[:max_len]
    # Truncate visual from the end (keep opening — usually subject + action)
    v = visual
    while len(assemble(v, "")) > max_len and len(v) > 120:
        v = v[: max(0, len(v) - 40)].rstrip()
        if len(v) > 0 and v[-1] not in " .,;:":
            cut = v.rfind(" ")
            v = v[: cut if cut > 0 else len(v) - 1]
    out = assemble(v, "")
    return out[:max_len]


def build_veo_prompt(
    image_prompt: str,
    character_bible: dict[str, Any] | None,
    scene: dict,
    prev_scene: dict | None,
    next_scene: dict | None,
    *,
    part_index: int,
    total_parts: int,
) -> str:
    """Assemble Veo text: identity locks, then intra-scene variation (multi-part), then story beat.

    Order ensures shot directives are never pushed past the 1000-char cut (previously the weak
    suffix sat last and was often truncated, producing duplicate-looking clips).
    """
    lock_lines = _character_lock_lines(character_bible, scene)
    intra = _intra_scene_shot_directive(part_index, total_parts, scene=scene)
    ident = _identity_variation_reminder(total_parts, scene=scene)

    head_chunks: list[str] = []
    if lock_lines:
        head_chunks.append("\n".join(lock_lines))
    if intra:
        head_chunks.append(intra)
    if ident:
        head_chunks.append(ident)
    head = "\n\n".join(head_chunks)

    # Shorter cross-scene hints when multiple parts compete for the same char budget
    n_prev = 160 if total_parts <= 1 else 72
    n_next = 160 if total_parts <= 1 else 72
    prev_sn = _snippet_visual(prev_scene, n_prev)
    next_sn = _snippet_visual(next_scene, n_next)
    neigh_lines: list[str] = []
    if prev_sn:
        neigh_lines.append("Previous scene (story): " + prev_sn)
    if next_sn:
        neigh_lines.append("Next scene (story): " + next_sn)
    neighbor_block = "\n".join(neigh_lines)

    visual = (image_prompt or "").strip()
    return _fit_veo_prompt(head, neighbor_block, visual, VEO_PROMPT_MAX)


def _pick_hero_indices(num_scenes: int, count: int) -> list[int]:
    """Return scene indices (1-based) for hero scenes. count=0 means all scenes (Veo-first)."""
    if num_scenes <= 0:
        return []
    if count <= 0 or count >= num_scenes:
        return list(range(1, num_scenes + 1))
    indices: list[int] = []
    indices.append(1)  # first
    if count >= 2:
        indices.append(num_scenes)  # last
    if count >= 3 and num_scenes > 2:
        mid = num_scenes // 2
        if mid not in indices:
            indices.insert(1, mid)
    return sorted(indices)[:count]


def pick_hero_scene_ids(scenes: list[dict], hero_count: int) -> list[int]:
    """Return list of scene_id values (1-based) to use as hero (Veo) scenes."""
    n = len(scenes)
    return _pick_hero_indices(n, hero_count)


def hero_scene_ids_with_complete_hero_files(
    scenes: list[dict],
    hero_scene_ids: list[int],
    output_dir: Path | None = None,
) -> set[int]:
    """Scene IDs in *hero_scene_ids* whose expected hero MP4s already exist on disk (all parts)."""
    output_dir = output_dir or HERO_VIDEOS_DIR
    by_id = {
        int(s["scene_id"]): s
        for s in scenes
        if isinstance(s, dict) and s.get("scene_id") is not None
    }
    complete: set[int] = set()
    for sid in hero_scene_ids:
        sid = int(sid)
        scene = by_id.get(sid)
        if not scene:
            continue
        n_parts = _parts_for_scene_duration(_scene_duration_sec(scene))
        ok = True
        for part in range(n_parts):
            pth = _hero_dest_path(output_dir, sid, part)
            if not pth.is_file():
                ok = False
                break
        if ok:
            complete.add(sid)
    return complete


def _is_quota_exhausted(exc: BaseException) -> bool:
    """True if the error is 429 / RESOURCE_EXHAUSTED (quota exhausted)."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg


def _should_try_next_veo_call(exc: BaseException) -> bool:
    """True if another model/key might succeed (quota, billing gate, precondition)."""
    msg = str(exc).lower()
    if _is_quota_exhausted(exc):
        return True
    if "failed_precondition" in msg:
        return True
    if "billing" in msg and ("gcp" in msg or "google cloud" in msg):
        return True
    return False


def _veo_operation_error_retryable(op_err) -> bool:
    s = str(op_err).lower() if op_err is not None else ""
    if not s:
        return True
    return any(
        x in s
        for x in (
            "quota",
            "429",
            "resource",
            "rate",
            "unavailable",
            "deadline",
            "timeout",
            "try",
            "billing",
        )
    )


def _hero_dest_path(output_dir: Path, scene_id: int, part_index: int) -> Path:
    return output_dir / f"hero_scene_{scene_id:03d}_p{part_index:02d}.mp4"


def generate_hero_video(
    clients: list[genai.Client],
    scene_id: int,
    prompt: str,
    output_dir: Path | None = None,
    *,
    part_index: int = 0,
    total_parts: int = 1,
) -> Path | None:
    """Generate one short video with Veo from a text prompt.

    *part_index* / *total_parts* name the file ``hero_scene_{id}_p{part}.mp4`` and
    append a variation hint so each segment is a distinct shot (no duplicated loops).

    Retries across all keys and Veo models, then up to VEO_MAX_ROUNDS full sweeps
    with VEO_RETRY_ROUND_WAIT_SEC between rounds (quota/rate recovery). Poll timeouts
    and operation errors trigger the next model/key rather than immediate failure.
    Returns path or None only after all rounds exhausted.
    """
    output_dir = output_dir or HERO_VIDEOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = _hero_dest_path(output_dir, scene_id, part_index)

    if dest.exists():
        logger.info("Using cached hero video for scene %d part %d", scene_id, part_index)
        return dest

    prompt = (prompt or "")[:VEO_PROMPT_MAX]

    n_clients = len(clients)
    n_models = len(VEO_MODELS)
    last_exc: BaseException | None = None

    for round_idx in range(VEO_MAX_ROUNDS):
        if round_idx > 0:
            logger.info(
                "Veo scene %d part %d: retry round %d/%d — pausing %.1fs for quota/rate recovery …",
                scene_id,
                part_index,
                round_idx + 1,
                VEO_MAX_ROUNDS,
                VEO_RETRY_ROUND_WAIT_SEC,
            )
            time.sleep(VEO_RETRY_ROUND_WAIT_SEC)

        for key_index, client in enumerate(clients):
            for model_index, model in enumerate(VEO_MODELS):
                key_label = key_index + 1
                model_label = model_index + 1
                logger.info(
                    "Veo scene %d part %d/%d round %d/%d | %s [key %d/%d, model %d/%d] …",
                    scene_id,
                    part_index + 1,
                    total_parts,
                    round_idx + 1,
                    VEO_MAX_ROUNDS,
                    model,
                    key_label,
                    n_clients,
                    model_label,
                    n_models,
                )
                try:
                    operation = client.models.generate_videos(
                        model=model,
                        source=types.GenerateVideosSource(prompt=prompt[:VEO_PROMPT_MAX]),
                        config=types.GenerateVideosConfig(
                            number_of_videos=1,
                            aspect_ratio="16:9",
                        ),
                    )
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "Veo start failed scene %d part %d (key %d/%d, model %s): %s — next after %.1fs",
                        scene_id,
                        part_index,
                        key_label,
                        n_clients,
                        model,
                        e,
                        VEO_KEY_COOLDOWN_SEC,
                    )
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                deadline = time.monotonic() + VEO_POLL_TIMEOUT_SEC
                poll_timed_out = False
                while not operation.done:
                    if time.monotonic() > deadline:
                        poll_timed_out = True
                        logger.warning(
                            "Veo poll timeout scene %d part %d (>%ds) — will try another model/key/round",
                            scene_id,
                            part_index,
                            VEO_POLL_TIMEOUT_SEC,
                        )
                        break
                    time.sleep(VEO_POLL_INTERVAL_SEC)
                    operation = client.operations.get(operation=operation)

                if poll_timed_out:
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                if operation.error:
                    last_exc = operation.error
                    if _veo_operation_error_retryable(operation.error):
                        logger.warning(
                            "Veo op error scene %d part %d: %s — trying next",
                            scene_id,
                            part_index,
                            operation.error,
                        )
                    else:
                        logger.warning(
                            "Veo op error scene %d part %d: %s — still trying other keys/models",
                            scene_id,
                            part_index,
                            operation.error,
                        )
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                if not operation.result or not operation.result.generated_videos:
                    logger.warning(
                        "Veo empty result scene %d part %d — trying next",
                        scene_id,
                        part_index,
                    )
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                video_out = operation.result.generated_videos[0].video
                if not video_out:
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                try:
                    if video_out.video_bytes:
                        dest.write_bytes(video_out.video_bytes)
                    elif video_out.uri:
                        data = client.files.download(file=video_out)
                        dest.write_bytes(data)
                    else:
                        logger.warning(
                            "Veo video no bytes/uri scene %d part %d — trying next",
                            scene_id,
                            part_index,
                        )
                        time.sleep(VEO_KEY_COOLDOWN_SEC)
                        continue
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "Veo save failed scene %d part %d: %s — trying next",
                        scene_id,
                        part_index,
                        e,
                    )
                    time.sleep(VEO_KEY_COOLDOWN_SEC)
                    continue

                logger.info(
                    "Hero video saved → %s (model %s, key %d, round %d)",
                    dest.name,
                    model,
                    key_label,
                    round_idx + 1,
                )
                return dest

    if last_exc is not None:
        logger.warning(
            "Veo exhausted after %d round(s) × %d key(s) × %d model(s) for scene %d part %d: %s (fallback: Remotion/image)",
            VEO_MAX_ROUNDS,
            n_clients,
            n_models,
            scene_id,
            part_index,
            last_exc,
        )
    return None


def _parts_for_scene_duration(duration_seconds: float) -> int:
    """How many distinct Veo clips to request so total length ≈ narration (no looping same clip)."""
    d = max(1.0, float(duration_seconds))
    n = int(math.ceil(d / max(HERO_VEO_CLIP_SEC, 1.0)))
    return max(1, min(MAX_HERO_PARTS_PER_SCENE, n))


def _scene_duration_sec(scene: dict) -> float:
    raw = scene.get("duration_seconds")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return float(SCENE_DURATION_DEFAULT)


def generate_hero_videos(
    clients: list[genai.Client],
    image_prompts: list[dict],
    hero_scene_ids: list[int],
    scenes: list[dict],
    output_dir: Path | None = None,
    character_bible: dict[str, Any] | None = None,
) -> dict[int, list[Path]]:
    """Generate Veo videos for hero scene IDs — **multiple distinct clips per scene** when
    the planned scene duration exceeds ~HERO_VEO_CLIP_SEC, so the final edit does not
    loop the same MP4.

    Returns dict scene_id -> list of paths (ordered). Empty lists omitted.
    """
    output_dir = output_dir or HERO_VIDEOS_DIR
    if not clients:
        logger.warning("No API clients for hero videos; skipping Veo generation")
        return {}
    by_id_prompt = {p["scene_id"]: p["image_prompt"] for p in image_prompts}
    by_id_scene = {s["scene_id"]: s for s in scenes}
    result: dict[int, list[Path]] = {}

    for scene_id in hero_scene_ids:
        base_prompt = by_id_prompt.get(scene_id, "")
        if not base_prompt:
            continue
        scene = by_id_scene.get(scene_id, {})
        prev_scene, next_scene = _neighbor_scenes(scenes, scene_id)
        n_parts = _parts_for_scene_duration(_scene_duration_sec(scene))
        if n_parts > 1:
            logger.info(
                "Scene %d: requesting %d distinct Veo clip(s) (~%.1fs planned duration, ~%.1fs per clip)",
                scene_id,
                n_parts,
                _scene_duration_sec(scene),
                HERO_VEO_CLIP_SEC,
            )
        paths: list[Path] = []
        for part in range(n_parts):
            veo_prompt = build_veo_prompt(
                base_prompt,
                character_bible,
                scene,
                prev_scene,
                next_scene,
                part_index=part,
                total_parts=n_parts,
            )
            path = generate_hero_video(
                clients,
                scene_id,
                veo_prompt,
                output_dir,
                part_index=part,
                total_parts=n_parts,
            )
            if path is not None:
                paths.append(path)
        if paths:
            result[scene_id] = paths
    return result


def get_hero_paths_for_scene(
    hero_video_paths: dict[int, Path | list[Path]] | None,
    scene_id: int,
) -> list[Path]:
    """Normalize hero dict values to an ordered list of existing paths."""
    if not hero_video_paths:
        return []
    v = hero_video_paths.get(scene_id)
    if v is None:
        return []
    if isinstance(v, Path):
        return [v] if v.exists() else []
    return [p for p in v if isinstance(p, Path) and p.exists()]


def scene_has_hero_video(
    hero_video_paths: dict[int, Path | list[Path]] | None,
    scene_id: int,
) -> bool:
    return len(get_hero_paths_for_scene(hero_video_paths, scene_id)) > 0
