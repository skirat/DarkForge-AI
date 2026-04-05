"""Create YouTube title, description, tags, and thumbnail from pipeline output."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from config import NANOBANANA_API_KEY, OUTPUT_DIR
from modules import image_generator
from modules.youtube_metadata import normalize_youtube_tags

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger("pipeline")

YOUTUBE_TITLE_FILE = "youtube_title.txt"
YOUTUBE_DESCRIPTION_FILE = "youtube_description.txt"
YOUTUBE_TAGS_FILE = "youtube_tags.txt"
THUMBNAIL_FILE = "thumbnail.png"


def create_youtube_assets(
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    clients: list[genai.Client] | None = None,
) -> dict[str, Path | None]:
    """Create YouTube title, description, tags, and thumbnail from pipeline metadata.

    Reads metadata.json (title, description, tags, thumbnail_prompt) and writes:
    - youtube_title.txt
    - youtube_description.txt
    - youtube_tags.txt (comma-separated tags, 5–6 items)
    - thumbnail.png (Gemini with *clients* if provided, else NanoBanana if NANOBANANA_API_KEY set)

    When *clients* is provided (e.g. your 4 Gemini API key clients), thumbnail is generated
    with Gemini first (trying each key on failure); NanoBanana is used as fallback if Gemini fails.

    Returns dict with keys: title_path, description_path, tags_path, thumbnail_path.
    """
    output_dir = output_dir or OUTPUT_DIR
    metadata_path = metadata_path or (output_dir / "metadata.json")
    if not metadata_path.exists():
        logger.warning("No metadata.json at %s – run pipeline first", metadata_path)
        return {
            "title_path": None,
            "description_path": None,
            "tags_path": None,
            "thumbnail_path": None,
        }

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = normalize_youtube_tags(data.get("tags") or [])
    thumbnail_prompt = (data.get("thumbnail_prompt") or "").strip()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Title
    title_path = output_dir / YOUTUBE_TITLE_FILE
    title_path.write_text(title, encoding="utf-8")
    logger.info("YouTube title → %s", title_path.name)

    # Description (optionally add a header/footer for YT)
    desc_lines = [description]
    if tags:
        desc_lines.append("")
        desc_lines.append("---")
        desc_lines.append("Tags: " + ", ".join(tags))
    description_path = output_dir / YOUTUBE_DESCRIPTION_FILE
    description_path.write_text("\n".join(desc_lines), encoding="utf-8")
    logger.info("YouTube description → %s", description_path.name)

    # Tags (comma-separated for YouTube paste; no extra lines)
    tags_path = output_dir / YOUTUBE_TAGS_FILE
    tags_path.write_text(", ".join(tags), encoding="utf-8")
    logger.info("YouTube tags → %s", tags_path.name)

    # Thumbnail: try Gemini (same 4 keys) first, then NanoBanana fallback
    thumbnail_out = output_dir / THUMBNAIL_FILE
    thumbnail_saved: Path | None = None
    if not thumbnail_prompt:
        logger.warning("No thumbnail_prompt in metadata; skipping thumbnail")
    else:
        if clients:
            logger.info("Generating thumbnail with Gemini (%d keys) …", len(clients))
            if image_generator.generate_single_image_gemini(clients, thumbnail_prompt, thumbnail_out):
                thumbnail_saved = thumbnail_out
        if thumbnail_saved is None and NANOBANANA_API_KEY:
            logger.info("Thumbnail via NanoBanana fallback …")
            if image_generator.generate_image_nanobanana(thumbnail_prompt, thumbnail_out):
                thumbnail_saved = thumbnail_out
        if thumbnail_saved is None and clients:
            logger.warning("Gemini thumbnail failed; NanoBanana not configured or failed")
        elif thumbnail_saved is None:
            logger.warning("Set GEMINI_API_KEY (or pass clients) or NANOBANANA_API_KEY to generate thumbnail")

    return {
        "title_path": title_path,
        "description_path": description_path,
        "tags_path": tags_path,
        "thumbnail_path": thumbnail_saved,
    }