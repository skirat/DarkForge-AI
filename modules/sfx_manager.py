"""Sound-effect matching based on scene keywords.

Looks for audio files in assets/sfx/ and returns positioned clips.
Degrades gracefully when the directory is empty.
"""
from __future__ import annotations

import logging
from pathlib import Path

from moviepy import AudioFileClip

from config import SFX_DIR, SFX_VOLUME

logger = logging.getLogger("pipeline")

_KEYWORD_MAP: dict[str, list[str]] = {
    "typing":   ["hack", "keyboard", "typing", "terminal", "code", "keys"],
    "glitch":   ["glitch", "distort", "corrupt", "static", "anomal"],
    "beep":     ["beep", "alert", "notification", "ping", "signal"],
    "ambient":  ["room", "ambient", "hum", "server", "fan"],
}


def _available_sfx() -> dict[str, Path]:
    """Map category stems to files found in SFX_DIR."""
    if not SFX_DIR.exists():
        return {}
    files: dict[str, Path] = {}
    for f in SFX_DIR.iterdir():
        if f.suffix.lower() in (".mp3", ".wav", ".ogg", ".m4a") and not f.name.startswith("."):
            files[f.stem.lower()] = f
    return files


def get_sfx_for_scene(
    scene: dict,
    scene_offset: float,
    scene_duration: float,
) -> list[tuple[float, Path]]:
    """Return list of (global_offset, sfx_path) for matching SFX.

    At most one SFX per category per scene, placed at the scene start.
    """
    available = _available_sfx()
    if not available:
        return []

    combined = f"{scene.get('visual_prompt', '')} {scene.get('narration', '')}".lower()
    hits: list[tuple[float, Path]] = []
    used_cats: set[str] = set()

    for category, keywords in _KEYWORD_MAP.items():
        if category in used_cats:
            continue
        if any(kw in combined for kw in keywords):
            for stem, path in available.items():
                if category in stem or stem in category:
                    hits.append((scene_offset, path))
                    used_cats.add(category)
                    break

    return hits


def load_sfx_clip(sfx_path: Path, volume: float | None = None) -> AudioFileClip | None:
    """Load an SFX file and apply volume. Returns None on failure."""
    try:
        clip = AudioFileClip(str(sfx_path))
        clip = clip.with_volume_scaled(volume or SFX_VOLUME)
        return clip
    except Exception as exc:
        logger.warning("Could not load SFX %s: %s", sfx_path.name, exc)
        return None
