#!/usr/bin/env python3
"""Create YouTube title, description, tags, and thumbnail from pipeline output.

Run after the pipeline (or when output/metadata.json exists):
    python create_youtube_assets.py

Uses the same 4 Gemini API keys (GEMINI_API_KEY, GEMINI_API_KEY_2, etc.) to generate
the thumbnail with Gemini first; falls back to NanoBanana if NANOBANANA_API_KEY is set.

Reads output/metadata.json and writes:
    output/youtube_title.txt
    output/youtube_description.txt
    output/youtube_tags.txt
    output/thumbnail.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai  # noqa: E402
from config import GEMINI_API_KEYS, OUTPUT_DIR  # noqa: E402
from utils.logger import setup_logger  # noqa: E402
from modules.youtube_assets import create_youtube_assets  # noqa: E402


def main() -> None:
    logger = setup_logger("youtube_assets", PROJECT_ROOT / "logs")
    clients = [genai.Client(api_key=k) for k in GEMINI_API_KEYS] if GEMINI_API_KEYS else []
    result = create_youtube_assets(
        OUTPUT_DIR / "metadata.json",
        OUTPUT_DIR,
        clients=clients if clients else None,
    )
    if result["title_path"]:
        print("Title:     ", result["title_path"])
    if result["description_path"]:
        print("Description:", result["description_path"])
    if result["tags_path"]:
        print("Tags:      ", result["tags_path"])
    if result["thumbnail_path"]:
        print("Thumbnail: ", result["thumbnail_path"])
    else:
        print("Thumbnail: not generated (add Gemini API keys in .env or NANOBANANA_API_KEY)")


if __name__ == "__main__":
    main()
