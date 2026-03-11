from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types
from pydub import AudioSegment

from config import (
    TTS_MODEL,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
    TTS_SAMPLE_WIDTH,
    TTS_CHANNELS,
    AUDIO_DIR,
    MAX_RETRIES,
    RETRY_BACKOFF,
    RETRY_RATE_LIMIT_WAIT,
)

logger = logging.getLogger("pipeline")


def _write_wav(path: Path, pcm_data: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(TTS_CHANNELS)
        wf.setsampwidth(TTS_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm_data)


# When rotating to another key after 429, wait briefly before trying next key
TTS_ROTATE_KEY_WAIT_SEC = 15.0


def _generate_scene_audio(
    clients: list[genai.Client],
    key_index: int,
    scene_id: int,
    narration: str,
    output_dir: Path,
) -> Path:
    """Generate TTS for a single scene with retry. On 429, tries next API key. Returns WAV path."""
    dest = output_dir / f"scene_{scene_id:03d}.wav"

    if dest.exists():
        logger.debug("Cached audio for scene %d", scene_id)
        return dest

    last_err: Exception | None = None
    n_clients = len(clients)
    for attempt in range(1, MAX_RETRIES + 1):
        client = clients[(key_index + attempt - 1) % n_clients]
        try:
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=narration,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=TTS_VOICE,
                            )
                        )
                    ),
                ),
            )
            pcm = response.candidates[0].content.parts[0].inline_data.data
            _write_wav(dest, pcm)
            logger.info("Scene %d audio saved → %s", scene_id, dest.name)
            return dest
        except Exception as exc:
            last_err = exc
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if is_rate_limit and n_clients > 1:
                wait = TTS_ROTATE_KEY_WAIT_SEC
                logger.warning(
                    "TTS failed for scene %d (attempt %d/%d, key %d/%d): %s – trying next key in %.1fs",
                    scene_id, attempt, MAX_RETRIES, (key_index + attempt - 1) % n_clients + 1, n_clients,
                    type(exc).__name__, wait,
                )
            else:
                wait = RETRY_RATE_LIMIT_WAIT if is_rate_limit else RETRY_BACKOFF ** attempt
                logger.warning(
                    "TTS failed for scene %d (attempt %d/%d): %s – retrying in %.1fs",
                    scene_id, attempt, MAX_RETRIES, type(exc).__name__, wait,
                )
            time.sleep(wait)

    raise RuntimeError(f"TTS failed for scene {scene_id} after {MAX_RETRIES} retries: {last_err}")


def generate_voiceover(
    clients: list[genai.Client],
    scenes: list[dict],
    output_dir: Path | None = None,
) -> tuple[Path, list[Path]]:
    """Generate per-scene WAVs and a combined narration file.

    Uses clients in round-robin (scene i -> clients[i % len(clients)]) for rate-limit spread.
    Returns (combined_wav_path, list_of_per_scene_wavs).
    """
    output_dir = output_dir or AUDIO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not clients:
        raise ValueError("At least one Gemini client is required")

    logger.info("Generating voiceover for %d scenes (%d keys) …", len(scenes), len(clients))
    scene_wavs: list[Path] = []
    sorted_scenes = sorted(scenes, key=lambda s: s["scene_id"])

    for i, scene in enumerate(sorted_scenes):
        key_index = i % len(clients)
        wav = _generate_scene_audio(clients, key_index, scene["scene_id"], scene["narration"], output_dir)
        scene_wavs.append(wav)

    combined = output_dir / "narration.wav"
    if not combined.exists() or len(scene_wavs) > 0:
        logger.info("Concatenating %d audio clips …", len(scene_wavs))
        segments = [AudioSegment.from_wav(str(w)) for w in scene_wavs]
        full = segments[0]
        for seg in segments[1:]:
            full += seg
        full.export(str(combined), format="wav")

    logger.info("Narration saved → %s (%.1fs)", combined.name, len(AudioSegment.from_wav(str(combined))) / 1000)
    return combined, scene_wavs
