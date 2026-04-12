from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Set

import requests
from google import genai
from google.genai import types

import config
from config import (
    IMAGES_DIR,
    IMAGE_WORKERS,
    IMAGE_GEN_MAX_ATTEMPTS,
    IMAGE_RETRY_ROUND_WAIT_SEC,
    MAX_RETRIES,
    OPENAI_ENABLED,
    RETRY_BACKOFF,
    RETRY_RATE_LIMIT_WAIT,
    NANOBANANA_API_KEY,
)

logger = logging.getLogger("pipeline")

# When rotating to another key after 429, wait briefly to avoid hammering
ROTATE_KEY_WAIT_SEC = 10.0
NANOBANANA_MAX_TRIES = 4
NANOBANANA_RETRY_WAIT_SEC = 8.0


def _is_transient_image_error(exc: BaseException) -> bool:
    """Quota, rate limits, overload — worth another key or later attempt."""
    msg = str(exc).lower()
    if "no image in response" in msg or "no image part in response" in msg:
        return True
    return any(
        x in msg
        for x in (
            "429",
            "resource_exhausted",
            "503",
            "unavailable",
            "quota",
            "rate",
            "deadline",
            "timeout",
            "overloaded",
            "try again",
            "too many requests",
        )
    )


def _gemini_image_error_skip_to_openai(exc: BaseException) -> bool:
    """Plan/access errors that won't fix by retrying Imagen/Gemini with the same keys."""
    msg = str(exc).lower()
    if "api_key_service_blocked" in msg:
        return True
    if "only available on paid plans" in msg:
        return True
    if "permission_denied" in msg and "blocked" in msg:
        return True
    if "imagen returned no image" in msg or "imagen returned empty" in msg:
        return True
    return False


_IMAGEM_DOWNGRADE_LOCK = threading.Lock()


def _downgrade_from_imagen_if_needed(exc: BaseException) -> bool:
    """Once per process: switch from Imagen to Gemini native image when Imagen cannot be used."""
    if not config.IMAGE_MODEL.startswith("imagen-"):
        return False
    if not _gemini_image_error_skip_to_openai(exc):
        return False
    with _IMAGEM_DOWNGRADE_LOCK:
        if not config.IMAGE_MODEL.startswith("imagen-"):
            return False
        config.IMAGE_MODEL = "gemini-2.5-flash-image"
        logger.warning(
            "Switched IMAGE_MODEL to %s — Imagen not usable for this key/account (retrying with native image).",
            config.IMAGE_MODEL,
        )
        return True


def _save_imagen_response_to_file(response, dest: Path) -> None:
    """Save first image from generate_images response; raise if empty or malformed."""
    imgs = getattr(response, "generated_images", None) or []
    if not imgs or imgs[0] is None:
        raise ValueError("Imagen returned no image")
    first = imgs[0]
    img = getattr(first, "image", None)
    if img is None:
        raise ValueError("Imagen returned empty image object")
    img.save(str(dest))

NANOBANANA_BASE = "https://api.nanobananaapi.ai"
NANOBANANA_POLL_INTERVAL = 3.0
NANOBANANA_POLL_MAX_WAIT = 120.0


def generate_image_nanobanana(prompt: str, dest: Path) -> bool:
    """Generate a single image via NanoBanana 2 API (e.g. for thumbnails). Returns True if saved, False otherwise."""
    return _try_nanobanana(prompt, dest)


def generate_single_image_gemini(
    clients: list[genai.Client],
    prompt: str,
    dest: Path,
) -> bool:
    """Generate a single image (e.g. thumbnail) using Gemini with the given clients. Tries each key on failure. Returns True if saved."""
    if not clients:
        if OPENAI_ENABLED:
            from modules.openai_image import generate_openai_image

            return generate_openai_image(prompt, dest, label="thumbnail")
        return False
    n_clients = len(clients)
    for attempt in range(1, MAX_RETRIES + 1):
        client = clients[(attempt - 1) % n_clients]
        use_imagen = config.IMAGE_MODEL.startswith("imagen-")
        try:
            if use_imagen:
                response = client.models.generate_images(
                    model=config.IMAGE_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                    ),
                )
                _save_imagen_response_to_file(response, dest)
            else:
                response = client.models.generate_content(
                    model=config.IMAGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                    ),
                )
                _save_image_from_content_response(response, dest)
            logger.info("Thumbnail from Gemini → %s", dest.name)
            return True
        except Exception as exc:
            if _downgrade_from_imagen_if_needed(exc):
                continue
            err_msg = str(exc).strip() or type(exc).__name__
            err_safe = err_msg.replace("%", "%%")[:200]
            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            wait = ROTATE_KEY_WAIT_SEC if (is_rate_limit and n_clients > 1) else (RETRY_RATE_LIMIT_WAIT if is_rate_limit else RETRY_BACKOFF ** attempt)
            logger.warning(
                "Thumbnail gen failed (attempt %d/%d, key %d/%d): %s – %s in %.1fs",
                attempt, MAX_RETRIES, (attempt - 1) % n_clients + 1, n_clients, err_safe,
                "trying next key" if n_clients > 1 else "retrying",
                wait,
            )
            time.sleep(wait)
            if OPENAI_ENABLED and _gemini_image_error_skip_to_openai(exc):
                from modules.openai_image import generate_openai_image

                if generate_openai_image(prompt, dest, label="thumbnail"):
                    return True
    if OPENAI_ENABLED:
        from modules.openai_image import generate_openai_image

        if generate_openai_image(prompt, dest, label="thumbnail"):
            return True
    return False


def _try_nanobanana(prompt: str, dest: Path) -> bool:
    """Try to generate image via NanoBanana 2 API. Returns True if image saved, False otherwise."""
    if not NANOBANANA_API_KEY:
        return False
    try:
        r = requests.post(
            f"{NANOBANANA_BASE}/api/v1/nanobanana/generate-2",
            headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}"},
            json={
                "prompt": prompt[:20000],
                "imageUrls": [],
                "aspectRatio": "16:9",
                "resolution": "1K",
                "outputFormat": "png",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 200 or not data.get("data", {}).get("taskId"):
            logger.debug("NanoBanana: no taskId in response")
            return False
        task_id = data["data"]["taskId"]
        deadline = time.monotonic() + NANOBANANA_POLL_MAX_WAIT
        while time.monotonic() < deadline:
            time.sleep(NANOBANANA_POLL_INTERVAL)
            tr = requests.get(
                f"{NANOBANANA_BASE}/api/v1/nanobanana/record-info",
                params={"taskId": task_id},
                headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}"},
                timeout=15,
            )
            tr.raise_for_status()
            td = tr.json()
            if td.get("code") != 200 or not td.get("data"):
                continue
            success_flag = td["data"].get("successFlag", 0)
            if success_flag == 1:
                resp = (td["data"] or {}).get("response") or {}
                url = resp.get("resultImageUrl") or resp.get("originImageUrl")
                if url:
                    img_r = requests.get(url, timeout=60)
                    img_r.raise_for_status()
                    dest.write_bytes(img_r.content)
                    logger.info("Scene image from NanoBanana → %s", dest.name)
                    return True
                return False
            if success_flag in (2, 3):
                logger.debug("NanoBanana task failed: %s", td["data"].get("errorMessage"))
                return False
        logger.debug("NanoBanana poll timeout")
        return False
    except Exception as e:
        logger.debug("NanoBanana fallback failed: %s", e)
        return False


def _save_image_from_content_response(response, dest: Path) -> None:
    """Extract first image from generate_content response and save to dest."""
    cands = getattr(response, "candidates", None) or []
    if not cands:
        raise ValueError("No image in response (no candidates)")
    first = cands[0]
    if first is None:
        raise ValueError("No image in response (empty candidate)")
    content = getattr(first, "content", None)
    if content is None:
        raise ValueError("No image in response (no content)")
    parts = getattr(content, "parts", None) or []
    if not parts:
        raise ValueError("No image in response (no parts)")
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if data:
            dest.write_bytes(data)
            return
    raise ValueError("No image part in response")


def _generate_single(
    clients: list[genai.Client],
    key_index: int,
    scene_id: int,
    prompt: str,
    output_dir: Path,
    *,
    skip_api: bool = False,
) -> Path:
    """Generate one image with retry logic. On 429, rotates to the next API key. Returns the saved file path."""
    filename = f"scene_{scene_id:03d}.png"
    dest = output_dir / filename

    if dest.exists():
        logger.debug("Cached image for scene %d", scene_id)
        return dest

    if skip_api:
        logger.info(
            "Skipping image API for scene %d (hero video already on disk) → placeholder",
            scene_id,
        )
        return _create_placeholder(dest)

    # 1. Try NanoBanana first (if configured) — a few attempts with backoff
    if NANOBANANA_API_KEY:
        for nb_try in range(1, NANOBANANA_MAX_TRIES + 1):
            if _try_nanobanana(prompt, dest):
                logger.info(
                    "Scene %d image saved (NanoBanana) → %s (try %d/%d)",
                    scene_id,
                    filename,
                    nb_try,
                    NANOBANANA_MAX_TRIES,
                )
                return dest
            if nb_try < NANOBANANA_MAX_TRIES:
                logger.warning(
                    "NanoBanana failed for scene %d (try %d/%d) — retrying in %.1fs",
                    scene_id,
                    nb_try,
                    NANOBANANA_MAX_TRIES,
                    NANOBANANA_RETRY_WAIT_SEC,
                )
                time.sleep(NANOBANANA_RETRY_WAIT_SEC)

    # 2. Gemini / Imagen: rotate keys many times; pause between full key cycles on quota
    last_err: Exception | None = None
    n_clients = len(clients)

    if n_clients == 0:
        if OPENAI_ENABLED:
            from modules.openai_image import generate_openai_image

            if generate_openai_image(prompt, dest, label=f"scene {scene_id}"):
                return dest
        return _create_placeholder(dest)
    max_attempts = IMAGE_GEN_MAX_ATTEMPTS
    attempt = 0

    while attempt < max_attempts:
        use_imagen = config.IMAGE_MODEL.startswith("imagen-")
        client = clients[(key_index + attempt) % n_clients]
        key_slot = (key_index + attempt) % n_clients + 1
        try:
            if use_imagen:
                response = client.models.generate_images(
                    model=config.IMAGE_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                    ),
                )
                _save_imagen_response_to_file(response, dest)
            else:
                response = client.models.generate_content(
                    model=config.IMAGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                    ),
                )
                _save_image_from_content_response(response, dest)
            logger.info("Scene %d image saved → %s (after %d attempt(s))", scene_id, filename, attempt + 1)
            return dest
        except Exception as exc:
            last_err = exc
            if OPENAI_ENABLED and _gemini_image_error_skip_to_openai(exc):
                from modules.openai_image import generate_openai_image

                if generate_openai_image(prompt, dest, label=f"scene {scene_id}"):
                    logger.info(
                        "Scene %d image saved via OpenAI (Gemini Imagen/plan blocked for this key)",
                        scene_id,
                    )
                    return dest
            if _downgrade_from_imagen_if_needed(exc):
                continue
            attempt += 1
            err_msg = str(exc).strip() or type(exc).__name__
            if len(err_msg) > 200:
                err_msg = err_msg[:197] + "..."
            err_msg_safe = err_msg.replace("%", "%%")
            transient = _is_transient_image_error(exc)
            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg

            if attempt >= max_attempts:
                break

            full_key_cycle = (
                n_clients > 1 and attempt > 0 and attempt % n_clients == 0
            )
            single_key_breather = (
                n_clients == 1 and attempt > 0 and attempt % 12 == 0
            )
            if full_key_cycle or single_key_breather:
                logger.warning(
                    "Image gen scene %d: quota pause after %d/%d attempts — %.1fs (%s)",
                    scene_id,
                    attempt,
                    max_attempts,
                    IMAGE_RETRY_ROUND_WAIT_SEC,
                    err_msg_safe[:120],
                )
                time.sleep(IMAGE_RETRY_ROUND_WAIT_SEC)
            elif transient and n_clients > 1:
                wait = min(ROTATE_KEY_WAIT_SEC, 60.0)
                logger.warning(
                    "Image gen failed scene %d (attempt %d/%d, key %d/%d): %s — next key in %.1fs",
                    scene_id,
                    attempt,
                    max_attempts,
                    key_slot,
                    n_clients,
                    err_msg_safe,
                    wait,
                )
                time.sleep(wait)
            else:
                wait = min(
                    RETRY_RATE_LIMIT_WAIT if is_rate_limit else RETRY_BACKOFF ** min(attempt, 10),
                    180.0,
                )
                logger.warning(
                    "Image gen failed scene %d (attempt %d/%d): %s — retry in %.1fs",
                    scene_id,
                    attempt,
                    max_attempts,
                    err_msg_safe,
                    wait,
                )
                time.sleep(wait)

    if OPENAI_ENABLED:
        from modules.openai_image import generate_openai_image

        if generate_openai_image(prompt, dest, label=f"scene {scene_id}"):
            return dest

    logger.error(
        "Image gen exhausted after %d attempts for scene %d: %s — using placeholder",
        max_attempts,
        scene_id,
        str(last_err).replace("%", "%%") if last_err else "unknown",
    )
    dest = _create_placeholder(dest)
    return dest


def _create_placeholder(dest: Path) -> Path:
    """Create a black 1920x1080 placeholder image."""
    from PIL import Image

    img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
    img.save(str(dest))
    logger.info("Created placeholder image → %s", dest.name)
    return dest


def generate_images(
    clients: list[genai.Client],
    image_prompts: list[dict],
    output_dir: Path | None = None,
    *,
    skip_api_for_scene_ids: Set[int] | None = None,
) -> list[Path]:
    """Generate images in parallel. Uses one client (API key) per worker to avoid rate limits.

    When *skip_api_for_scene_ids* contains a scene_id, skip Gemini/OpenAI/NanoBanana for that
    scene and use a cached PNG if present, else a black placeholder (e.g. hero MP4 already exists).
    """
    output_dir = output_dir or IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    clients = clients or []  # allow single-client callers to pass [client]
    skip_api_for_scene_ids = skip_api_for_scene_ids or set()

    needs_api = any(
        int(p["scene_id"]) not in skip_api_for_scene_ids for p in image_prompts
    )
    if needs_api and not clients and not OPENAI_ENABLED and not NANOBANANA_API_KEY:
        raise ValueError(
            "Need at least one GEMINI_API_KEY, or set OPENAI_API_KEY / NANOBANANA_API_KEY for images"
        )

    n_skip = sum(
        1 for p in image_prompts if int(p["scene_id"]) in skip_api_for_scene_ids
    )
    if n_skip:
        logger.info(
            "Skipping image API for %d/%d scene(s) with hero video(s) already on disk",
            n_skip,
            len(image_prompts),
        )

    n_keys = max(1, len(clients))
    workers = IMAGE_WORKERS if clients else min(IMAGE_WORKERS, 4)
    logger.info(
        "Generating %d scene images (workers=%d, gemini_keys=%d, openai_fallback=%s) …",
        len(image_prompts),
        workers,
        len(clients),
        OPENAI_ENABLED,
    )
    paths: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, p in enumerate(image_prompts):
            key_index = i % n_keys if clients else 0
            sid = int(p["scene_id"])
            fut = pool.submit(
                _generate_single,
                clients,
                key_index,
                sid,
                p["image_prompt"],
                output_dir,
                skip_api=sid in skip_api_for_scene_ids,
            )
            futures[fut] = sid
        for future in as_completed(futures):
            sid = futures[future]
            paths[sid] = future.result()

    ordered = [paths[p["scene_id"]] for p in sorted(image_prompts, key=lambda x: x["scene_id"])]
    logger.info("All images ready (%d files)", len(ordered))
    return ordered
