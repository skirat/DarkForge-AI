"""OpenAI Images API fallback (e.g. DALL-E 3) when Gemini image gen fails."""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from config import OPENAI_API_KEY, OPENAI_IMAGE_MODEL

logger = logging.getLogger("pipeline")

try:
    import openai  # noqa: F401

    OPENAI_PACKAGE_AVAILABLE = True
except ImportError:
    OPENAI_PACKAGE_AVAILABLE = False

_openai_missing_logged = False


def generate_openai_image(prompt: str, dest: Path, *, label: str = "image") -> bool:
    """Generate one 16:9 image via OpenAI Images API; save PNG to *dest*. Returns True on success."""
    global _openai_missing_logged
    if not OPENAI_PACKAGE_AVAILABLE:
        if not _openai_missing_logged:
            logger.warning(
                "OpenAI image fallback skipped: install the package with pip install openai (see requirements.txt)."
            )
            _openai_missing_logged = True
        return False
    if not OPENAI_API_KEY:
        return False
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_IMAGE_MODEL
        # DALL-E 3: 1792x1024 is closest to 16:9
        size = "1792x1024" if model.startswith("dall-e") else "1024x1024"
        r = client.images.generate(
            model=model,
            prompt=prompt[:4000],
            size=size,
            quality="standard",
            n=1,
        )
        url = r.data[0].url
        if not url:
            return False
        img = requests.get(url, timeout=120)
        img.raise_for_status()
        dest.write_bytes(img.content)
        logger.info("OpenAI image (%s) → %s", model, dest.name)
        return True
    except Exception as e:
        logger.warning("OpenAI image gen failed (%s): %s", label, e)
        return False
