from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("pipeline")


def probe_duration(file: Path) -> float:
    """Return duration in seconds of an audio/video file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def convert_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)],
        capture_output=True,
        check=True,
    )
    return mp3_path


def color_grade(input_path: Path, output_path: Path, vf_filter: str) -> Path:
    """Apply an FFmpeg video filter for cinematic color grading."""
    logger.info("Applying color grading → %s", output_path.name)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "medium",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path
