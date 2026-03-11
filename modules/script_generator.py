from __future__ import annotations

import logging
from google import genai
from google.genai import types

from config import TEXT_MODEL, SCRIPT_WORD_COUNT_MIN, SCRIPT_WORD_COUNT_MAX

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = f"""\
You are an elite scriptwriter for YouTube videos in the ethical-hacking and darknet \
horror niche. Your scripts are gripping, atmospheric, and cinematic.

Rules:
- Write between {SCRIPT_WORD_COUNT_MIN} and {SCRIPT_WORD_COUNT_MAX} words (about 5–10 minutes when read aloud).
- Structure: Hook → Introduction → Investigation → Dark Twist → Conclusion.
- Tone: dark storytelling, cyberpunk atmosphere, suspenseful narration.
- Write in second person ("you") to immerse the viewer.
- Do NOT include stage directions, scene headings, or formatting markup.
- Output ONLY the narration text, nothing else.
"""


def generate_script(client: genai.Client, prompt: str, hook: str) -> str:
    """Return a full narration script as plain text."""
    logger.info("Generating storytelling script …")

    user_msg = (
        f"Video concept: {prompt}\n\n"
        f"Opening hook to incorporate: {hook}\n\n"
        "Write the full narration script now."
    )

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.85,
            max_output_tokens=4096,
        ),
    )

    script = response.text.strip()
    word_count = len(script.split())
    logger.info("Script generated: %d words", word_count)
    return script
