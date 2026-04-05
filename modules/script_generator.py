from __future__ import annotations

import logging
from google import genai
from google.genai import types

from config import OPENAI_ENABLED, TEXT_MODEL, SCRIPT_WORD_COUNT_MIN, SCRIPT_WORD_COUNT_MAX
from modules.openai_llm import openai_chat
from utils.gemini_retry import with_gemini_client_rotation

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = f"""\
You are an elite scriptwriter for YouTube videos in the ethical-hacking and darknet \
horror niche. Your scripts are gripping, atmospheric, and cinematic.

Rules:
- Write between {SCRIPT_WORD_COUNT_MIN} and {SCRIPT_WORD_COUNT_MAX} words (about 5–10 minutes when read aloud).
- Structure: Hook → Introduction → Investigation → Dark Twist → Conclusion.
- Tone: dark storytelling, cyberpunk atmosphere, suspenseful narration.
- Write in second person ("you") to immerse the viewer.
- Language (critical): Use simple, clear English anyone can follow—general audience, not only tech workers. \
Short sentences where possible. Avoid jargon dumps. If you use a technical term (e.g. exploit, VPN, metadata), \
use it sparingly and make meaning obvious from context, or add a brief plain-English gloss once. \
Do not write like a manual, research paper, or CVE write-up.
- Do NOT include stage directions, scene headings, or formatting markup.
- Output ONLY the narration text, nothing else.
"""


def generate_script(clients: list[genai.Client], prompt: str, hook: str) -> str:
    """Return a full narration script as plain text."""
    logger.info("Generating storytelling script …")

    user_msg = (
        f"Video concept: {prompt}\n\n"
        f"Opening hook to incorporate: {hook}\n\n"
        "Write the full narration script now. Keep vocabulary accessible; explain or avoid heavy jargon."
    )

    def _call(c: genai.Client) -> str:
        response = c.models.generate_content(
            model=TEXT_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.85,
                max_output_tokens=4096,
            ),
        )
        return response.text.strip()

    script = with_gemini_client_rotation(
        clients,
        "Script generation",
        _call,
        openai_fallback=(
            lambda: openai_chat(
                SYSTEM_PROMPT,
                user_msg,
                json_mode=False,
                temperature=0.85,
                max_tokens=8192,
            )
        )
        if OPENAI_ENABLED
        else None,
    )
    word_count = len(script.split())
    logger.info("Script generated: %d words", word_count)
    return script
