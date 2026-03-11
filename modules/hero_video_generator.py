"""Generate short hero-scene videos with Veo (Google Labs Flow)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import (
    VEO_MODELS,
    HERO_VIDEOS_DIR,
    VEO_POLL_INTERVAL_SEC,
    VEO_POLL_TIMEOUT_SEC,
)

logger = logging.getLogger("pipeline")


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


def _is_quota_exhausted(exc: BaseException) -> bool:
    """True if the error is 429 / RESOURCE_EXHAUSTED (quota exhausted)."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg


def generate_hero_video(
    clients: list[genai.Client],
    scene_id: int,
    prompt: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Generate one short video with Veo from a text prompt.

    On 429 RESOURCE_EXHAUSTED, tries the next (key, model) pair: cycles through
    all Veo models for each key, then the next key. Only falls back to image
    after trying all keys × all models. Returns path or None on failure.
    """
    output_dir = output_dir or HERO_VIDEOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"hero_scene_{scene_id:03d}.mp4"

    if dest.exists():
        logger.info("Using cached hero video for scene %d", scene_id)
        return dest

    n_clients = len(clients)
    n_models = len(VEO_MODELS)
    last_exc: BaseException | None = None

    for key_index, client in enumerate(clients):
        for model_index, model in enumerate(VEO_MODELS):
            key_label = key_index + 1
            model_label = model_index + 1
            logger.info(
                "Generating hero video for scene %d with Veo (%s) [key %d/%d, model %d/%d] …",
                scene_id, model, key_label, n_clients, model_label, n_models,
            )
            try:
                operation = client.models.generate_videos(
                    model=model,
                    source=types.GenerateVideosSource(prompt=prompt[:1000]),  # cap prompt length
                    config=types.GenerateVideosConfig(
                        number_of_videos=1,
                        aspect_ratio="16:9",
                    ),
                )
            except Exception as e:
                last_exc = e
                if _is_quota_exhausted(e):
                    logger.warning(
                        "Veo quota exhausted for scene %d (key %d/%d, model %s): %s — trying next key/model",
                        scene_id, key_label, n_clients, model, e,
                    )
                    continue
                logger.warning("Veo generation failed for scene %d: %s (will use image)", scene_id, e)
                return None

            # Poll with the same client that started the operation
            deadline = time.monotonic() + VEO_POLL_TIMEOUT_SEC
            while not operation.done:
                if time.monotonic() > deadline:
                    logger.warning("Veo poll timeout for scene %d (will use image)", scene_id)
                    return None
                time.sleep(VEO_POLL_INTERVAL_SEC)
                operation = client.operations.get(operation=operation)

            if operation.error:
                logger.warning("Veo operation error for scene %d: %s (will use image)", scene_id, operation.error)
                return None
            if not operation.result or not operation.result.generated_videos:
                logger.warning("Veo returned no video for scene %d (will use image)", scene_id)
                return None

            video_out = operation.result.generated_videos[0].video
            if not video_out:
                return None

            try:
                if video_out.video_bytes:
                    dest.write_bytes(video_out.video_bytes)
                elif video_out.uri:
                    data = client.files.download(file=video_out)
                    dest.write_bytes(data)
                else:
                    logger.warning("Veo video has no bytes or uri for scene %d", scene_id)
                    return None
            except Exception as e:
                logger.warning("Failed to save Veo video for scene %d: %s", scene_id, e)
                return None

            logger.info("Hero video saved → %s (model %s, key %d)", dest.name, model, key_label)
            return dest

    # All keys × models tried (quota exhausted or failed for each)
    if last_exc is not None:
        logger.warning(
            "Veo generation failed for scene %d after trying all %d key(s) × %d model(s): %s (will use image)",
            scene_id, n_clients, n_models, last_exc,
        )
    return None


def generate_hero_videos(
    clients: list[genai.Client],
    image_prompts: list[dict],
    hero_scene_ids: list[int],
    output_dir: Path | None = None,
) -> dict[int, Path]:
    """Generate Veo videos for hero scene IDs. Tries all keys on 429 before falling back to image.
    Returns dict scene_id -> path (only successful)."""
    output_dir = output_dir or HERO_VIDEOS_DIR
    if not clients:
        logger.warning("No API clients for hero videos; skipping Veo generation")
        return {}
    by_id = {p["scene_id"]: p["image_prompt"] for p in image_prompts}
    result: dict[int, Path] = {}
    for scene_id in hero_scene_ids:
        prompt = by_id.get(scene_id, "")
        if not prompt:
            continue
        path = generate_hero_video(clients, scene_id, prompt, output_dir)
        if path is not None:
            result[scene_id] = path
    return result
