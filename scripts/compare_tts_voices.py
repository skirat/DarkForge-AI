#!/usr/bin/env python3
"""Generate short horror narration samples for Charon vs Algenib (Gemini TTS).

Run from repo root:
  .venv/bin/python scripts/compare_tts_voices.py

Outputs (same text, two voices):
  output/audio/tts_sample_charon.wav
  output/audio/tts_sample_algenib.wav

Requires GEMINI_API_KEY in .env (or environment). On 429, rotates through
GEMINI_API_KEY_2 … _4 like the pipeline voiceover step.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEYS,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    TTS_SAMPLE_WIDTH,
    TTS_CHANNELS,
    AUDIO_DIR,
)

# Match modules/voiceover_generator.py — brief pause before trying the next key after 429
TTS_ROTATE_KEY_WAIT_SEC = 15.0

# ~10–15s at typical narration pace (adjust if clips are shorter/longer)
SAMPLE_TEXT = """\
They said the archive was sealed for good. But when the file finished downloading, \
the timestamp on the footage was tomorrow—and the figure walking toward the camera \
was already wearing your face. You closed the laptop. The footsteps in the hall \
did not stop.\
"""


def _write_wav(path: Path, pcm_data: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(TTS_CHANNELS)
        wf.setsampwidth(TTS_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm_data)


def _render_voice(
    clients: list[genai.Client], voice_name: str, out_path: Path
) -> None:
    """TTS with round-robin keys on 429 RESOURCE_EXHAUSTED (same idea as voiceover)."""
    n = len(clients)
    last_err: Exception | None = None
    for attempt in range(n):
        client = clients[attempt]
        try:
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=SAMPLE_TEXT,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                ),
            )
            pcm = response.candidates[0].content.parts[0].inline_data.data
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(out_path, pcm)
            return
        except Exception as exc:
            last_err = exc
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if is_rate_limit and attempt < n - 1:
                print(
                    f"  Rate limited on key {attempt + 1}/{n} — "
                    f"trying next key in {TTS_ROTATE_KEY_WAIT_SEC:.0f}s …",
                    flush=True,
                )
                time.sleep(TTS_ROTATE_KEY_WAIT_SEC)
                continue
            raise
    raise RuntimeError(f"TTS failed for {voice_name} after {n} key(s): {last_err}")


def main() -> None:
    if not GEMINI_API_KEYS:
        print("Set GEMINI_API_KEY in .env (or environment).", file=sys.stderr)
        sys.exit(1)

    clients = [genai.Client(api_key=k) for k in GEMINI_API_KEYS]
    out_dir = AUDIO_DIR

    pairs = [
        ("Charon", out_dir / "tts_sample_charon.wav"),
        ("Algenib", out_dir / "tts_sample_algenib.wav"),
    ]

    print("Model:", TTS_MODEL)
    print(f"API keys: {len(clients)} (429 → rotate with {TTS_ROTATE_KEY_WAIT_SEC:.0f}s delay)")
    print("Sample text length:", len(SAMPLE_TEXT), "chars\n")

    for voice, path in pairs:
        print(f"Rendering {voice} → {path.relative_to(_ROOT)} …")
        _render_voice(clients, voice, path)
        with wave.open(str(path), "rb") as wf:
            actual = wf.getnframes() / float(wf.getframerate())
        print(f"  Done (~{actual:.1f}s)\n")

    print("Listen with:")
    print(f"  open {out_dir / 'tts_sample_charon.wav'}")
    print(f"  open {out_dir / 'tts_sample_algenib.wav'}")
    print("\nPick one in .env: TTS_VOICE=Charon  or  TTS_VOICE=Algenib")


if __name__ == "__main__":
    main()
