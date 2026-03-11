from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from config import OUTPUT_DIR

logger = logging.getLogger("pipeline")


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _chunk_narration(text: str, max_chars: int = 80) -> list[str]:
    """Split narration into subtitle-friendly chunks."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        if length + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def generate_subtitles(
    scenes: list[dict],
    scene_wavs: list[Path],
    output_path: Path | None = None,
) -> Path:
    """Build an SRT file from scenes, using actual audio durations for timing."""
    output_path = output_path or (OUTPUT_DIR / "subtitles.srt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating subtitles …")

    srt_entries: list[str] = []
    index = 1
    global_offset = 0.0

    for scene, wav_path in zip(
        sorted(scenes, key=lambda s: s["scene_id"]),
        scene_wavs,
    ):
        audio = AudioSegment.from_wav(str(wav_path))
        scene_duration = len(audio) / 1000.0

        chunks = _chunk_narration(scene["narration"])
        if not chunks:
            global_offset += scene_duration
            continue

        chunk_duration = scene_duration / len(chunks)

        for chunk in chunks:
            start = global_offset
            end = global_offset + chunk_duration
            srt_entries.append(
                f"{index}\n"
                f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{chunk}\n"
            )
            index += 1
            global_offset = end

    output_path.write_text("\n".join(srt_entries), encoding="utf-8")
    logger.info("Subtitles saved → %s (%d entries)", output_path.name, index - 1)
    return output_path
