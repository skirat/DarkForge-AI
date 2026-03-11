from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from google import genai
from google.genai import types

from config import (
    IMAGE_MODEL,
    IMAGES_DIR,
    IMAGE_WORKERS,
    MAX_RETRIES,
    RETRY_BACKOFF,
    RETRY_RATE_LIMIT_WAIT,
    NANOBANANA_API_KEY,
)

logger = logging.getLogger("pipeline")

# When rotating to another key after 429, wait briefly to avoid hammering
ROTATE_KEY_WAIT_SEC = 10.0

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
        return False
    use_imagen = IMAGE_MODEL.startswith("imagen-")
    n_clients = len(clients)
    for attempt in range(1, MAX_RETRIES + 1):
        client = clients[(attempt - 1) % n_clients]
        try:
            if use_imagen:
                response = client.models.generate_images(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                    ),
                )
                response.generated_images[0].image.save(str(dest))
            else:
                response = client.models.generate_content(
                    model=IMAGE_MODEL,
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
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("No image in response")
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            dest.write_bytes(part.inline_data.data)
            return
    raise ValueError("No image part in response")


def _generate_single(
    clients: list[genai.Client],
    key_index: int,
    scene_id: int,
    prompt: str,
    output_dir: Path,
) -> Path:
    """Generate one image with retry logic. On 429, rotates to the next API key. Returns the saved file path."""
    filename = f"scene_{scene_id:03d}.png"
    dest = output_dir / filename

    if dest.exists():
        logger.debug("Cached image for scene %d", scene_id)
        return dest

    # 1. Try NanoBanana first (if configured)
    if NANOBANANA_API_KEY and _try_nanobanana(prompt, dest):
        logger.info("Scene %d image saved (NanoBanana) → %s", scene_id, filename)
        return dest

    # 2. Fall back to Gemini (gemini-2.5-flash-image or Imagen) with key rotation
    use_imagen = IMAGE_MODEL.startswith("imagen-")
    last_err: Exception | None = None
    n_clients = len(clients)

    for attempt in range(1, MAX_RETRIES + 1):
        # Rotate key on each attempt: try key_index, then key_index+1, ... so exhausted keys are skipped
        client = clients[(key_index + attempt - 1) % n_clients]
        try:
            if use_imagen:
                response = client.models.generate_images(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                    ),
                )
                response.generated_images[0].image.save(str(dest))
            else:
                response = client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                    ),
                )
                _save_image_from_content_response(response, dest)
            logger.info("Scene %d image saved → %s", scene_id, filename)
            return dest
        except Exception as exc:
            last_err = exc
            err_msg = str(exc).strip() or type(exc).__name__
            if len(err_msg) > 200:
                err_msg = err_msg[:197] + "..."
            err_msg_safe = err_msg.replace("%", "%%")  # avoid % breaking format
            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            if is_rate_limit and n_clients > 1:
                # Try next key soon; avoid long wait when we have other keys
                wait = ROTATE_KEY_WAIT_SEC
                logger.warning(
                    "Image gen failed for scene %d (attempt %d/%d, key %d/%d): %s – trying next key in %.1fs",
                    scene_id, attempt, MAX_RETRIES, (key_index + attempt - 1) % n_clients + 1, n_clients,
                    err_msg_safe, wait,
                )
            else:
                wait = RETRY_RATE_LIMIT_WAIT if is_rate_limit else RETRY_BACKOFF ** attempt
                logger.warning(
                    "Image gen failed for scene %d (attempt %d/%d): %s – retrying in %.1fs",
                    scene_id, attempt, MAX_RETRIES, err_msg_safe, wait,
                )
            time.sleep(wait)

    logger.error("Image gen failed permanently for scene %d: %s", scene_id, str(last_err).replace("%", "%%"))
    # 3. Final fallback: placeholder
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
) -> list[Path]:
    """Generate images in parallel. Uses one client (API key) per worker to avoid rate limits."""
    output_dir = output_dir or IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    clients = clients or []  # allow single-client callers to pass [client]
    if not clients:
        raise ValueError("At least one Gemini client is required")

    logger.info(
        "Generating %d scene images (workers=%d, keys=%d) …",
        len(image_prompts), IMAGE_WORKERS, len(clients),
    )
    paths: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        futures = {}
        for i, p in enumerate(image_prompts):
            key_index = i % len(clients)
            fut = pool.submit(
                _generate_single,
                clients,
                key_index,
                p["scene_id"],
                p["image_prompt"],
                output_dir,
            )
            futures[fut] = p["scene_id"]
        for future in as_completed(futures):
            sid = futures[future]
            paths[sid] = future.result()

    ordered = [paths[p["scene_id"]] for p in sorted(image_prompts, key=lambda x: x["scene_id"])]
    logger.info("All images ready (%d files)", len(ordered))
    return ordered
