"""Background music selection, looping, and narration-aware volume ducking."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
from moviepy.audio import fx as afx
from pydub import AudioSegment

from config import (
    MUSIC_DIR,
    BG_MUSIC_VOLUME,
    BG_MUSIC_DUCKED,
    DUCK_RAMP_SECONDS,
)

logger = logging.getLogger("pipeline")

_EXTENSIONS = ("*.mp3", "*.wav", "*.ogg", "*.m4a")
_FALLBACK_FILENAME = "fallback_ambient.wav"
_FALLBACK_DURATION_SEC = 90
_SAMPLE_RATE = 44100
# Preferred horror tracks: use all matching stems for multi-track bed (cut/stitch per story).
_PREFERRED_HORROR_STEM = "dark_lurker_protocol"
_HORROR_KEYWORDS = ("horror", "dark", "darknet", "lurker", "creepy", "scary")


def _list_tracks(*, include_fallback: bool = False) -> list[Path]:
    tracks: list[Path] = []
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    for ext in _EXTENSIONS:
        tracks.extend(MUSIC_DIR.glob(ext))
    if not include_fallback:
        tracks = [p for p in tracks if p.name != _FALLBACK_FILENAME]
    return sorted(tracks)


def _build_theme_text(scenes: list[dict] | None, metadata: dict | None) -> str:
    """Combine scene content and metadata for theme keyword matching."""
    parts: list[str] = []
    if metadata:
        parts.append(metadata.get("title", ""))
        parts.append(metadata.get("description", ""))
        for tag in metadata.get("tags") or []:
            parts.append(str(tag))
    if scenes:
        for s in scenes:
            parts.append(s.get("visual_prompt", ""))
            parts.append(s.get("narration", ""))
    return " ".join(parts).lower()


def _score_track_for_theme(track_path: Path, theme_text: str) -> int:
    """Higher = better match. Filename stem words that appear in theme_text."""
    stem = track_path.stem.lower().replace("_", " ").replace("-", " ")
    words = [w for w in stem.split() if len(w) > 1]
    return sum(1 for w in words if w in theme_text)


def _is_horror_theme(theme_text: str) -> bool:
    """True if theme suggests horror/darknet content."""
    t = theme_text.lower()
    return any(k in t for k in _HORROR_KEYWORDS)


def _preferred_horror_tracks(tracks: list[Path]) -> list[Path]:
    """All tracks whose stem contains the preferred horror stem (e.g. dark_lurker_protocol)."""
    out = [p for p in tracks if _PREFERRED_HORROR_STEM in p.stem.lower()]
    return sorted(out)


def select_track(
    scenes: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path | list[Path] | None:
    """Pick background music from assets/music/.

    For horror/darknet content, prefers Dark Lurker Protocol tracks and may return
    multiple paths so the pipeline can build a music bed from segments of each.
    Otherwise uses theme (metadata + scenes) to choose the best-matching file.
    """
    tracks = _list_tracks()
    if not tracks:
        logger.info("No background music found in %s", MUSIC_DIR)
        return None

    theme_text = _build_theme_text(scenes, metadata)
    preferred = _preferred_horror_tracks(tracks)

    if preferred and _is_horror_theme(theme_text):
        logger.info(
            "Selected horror music (Dark Lurker): %s",
            [p.name for p in preferred],
        )
        return preferred if len(preferred) > 1 else preferred[0]

    if len(tracks) == 1 or not theme_text.strip():
        logger.info("Selected background music: %s", tracks[0].name)
        return tracks[0]

    best = max(tracks, key=lambda t: _score_track_for_theme(t, theme_text))
    if _score_track_for_theme(best, theme_text) > 0:
        logger.info("Selected background music (theme match): %s", best.name)
    else:
        logger.info("Selected background music (default): %s", best.name)
    return best


def _generate_fallback_ambient() -> Path:
    """Generate a soft dark-ambient pad and save to assets/music/fallback_ambient.wav."""
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MUSIC_DIR / _FALLBACK_FILENAME
    if out_path.exists():
        logger.info("Using existing fallback ambient: %s", out_path.name)
        return out_path

    logger.info("Generating fallback ambient music (dark pad) → %s", out_path.name)
    t = np.linspace(0, _FALLBACK_DURATION_SEC, int(_FALLBACK_DURATION_SEC * _SAMPLE_RATE), dtype=np.float32)
    # Low, soft pad: mix of low frequencies with gentle envelope
    envelope = np.exp(-0.5 * (t / _FALLBACK_DURATION_SEC)) * (0.3 + 0.7 * (1 - t / _FALLBACK_DURATION_SEC))
    pad = (
        np.sin(2 * np.pi * 55 * t) * 0.15
        + np.sin(2 * np.pi * 82 * t) * 0.1
        + np.sin(2 * np.pi * 110 * t) * 0.08
    ) * envelope
    pad = pad / (np.max(np.abs(pad)) + 1e-8) * 0.25
    pad_int16 = (np.clip(pad, -1, 1) * 32767).astype(np.int16)
    segment = AudioSegment(
        data=pad_int16.tobytes(),
        sample_width=2,
        frame_rate=_SAMPLE_RATE,
        channels=1,
    )
    segment.export(str(out_path), format="wav", parameters=["-ac", "1"])
    logger.info("Fallback ambient saved (%.1fs)", _FALLBACK_DURATION_SEC)
    return out_path


def ensure_background_music(
    scenes: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path | list[Path]:
    """Return music track path(s): from assets/music if any, otherwise generated fallback.

    For horror content may return a list of paths (e.g. both Dark Lurker tracks)
    so the pipeline can cut/stitch segments per story. Call this so the video
    always has background music (voiceover + music).
    """
    chosen = select_track(scenes=scenes, metadata=metadata)
    if chosen is not None:
        return chosen
    return _generate_fallback_ambient()


def _prepare_music_single(track_path: Path, total_duration: float) -> AudioFileClip:
    """Load one track, loop/trim to total_duration, volume and fades."""
    music = AudioFileClip(str(track_path))
    if music.duration < total_duration:
        loops = int(total_duration / music.duration) + 1
        music = concatenate_audioclips([music] * loops)
    music = music.subclipped(0, total_duration)
    music = music.with_volume_scaled(BG_MUSIC_VOLUME)
    music = music.with_effects([afx.AudioFadeIn(2.0), afx.AudioFadeOut(3.0)])
    return music


def prepare_music(
    track_path: Path | list[Path], total_duration: float
) -> AudioFileClip:
    """Load, loop/trim, and apply head/tail fades.

    If *track_path* is a list, builds a single bed by concatenating segments
    from each track in order (cycling as needed) so music varies by story length;
    then trims to *total_duration* and applies volume/fades.
    """
    paths = [track_path] if isinstance(track_path, Path) else list(track_path)
    if not paths:
        raise ValueError("prepare_music requires at least one track path")

    if len(paths) == 1:
        music = _prepare_music_single(paths[0], total_duration)
        logger.info(
            "Music prepared: %s (%.1fs)",
            paths[0].name, total_duration,
        )
        return music

    # Multi-track: concatenate segments from each track until we cover total_duration
    clips: list[AudioFileClip] = []
    total_so_far = 0.0
    index = 0
    while total_so_far < total_duration:
        p = paths[index % len(paths)]
        clip = AudioFileClip(str(p))
        if total_so_far + clip.duration <= total_duration:
            clips.append(clip.with_volume_scaled(BG_MUSIC_VOLUME))
            total_so_far += clip.duration
        else:
            need = total_duration - total_so_far
            sub = clip.subclipped(0, need).with_volume_scaled(BG_MUSIC_VOLUME)
            clips.append(sub)
            total_so_far += need
        index += 1

    music = concatenate_audioclips(clips)
    music = music.with_effects([afx.AudioFadeIn(2.0), afx.AudioFadeOut(3.0)])
    logger.info(
        "Music prepared: %d tracks (%.1fs): %s",
        len(paths), total_duration, [x.name for x in paths],
    )
    return music


def duck_under_narration(
    music: AudioFileClip,
    narration_segments: list[tuple[float, float]],
) -> AudioFileClip:
    """Reduce music volume during narration segments.

    *narration_segments* is a list of (start_sec, end_sec) tuples where
    narration is active.  Between those segments music plays at full
    configured volume; during them it drops to BG_MUSIC_DUCKED.
    """
    if not narration_segments:
        return music

    duck_ratio = BG_MUSIC_DUCKED / max(BG_MUSIC_VOLUME, 0.001)
    total = music.duration
    segments: list[AudioFileClip] = []
    cursor = 0.0

    for start, end in sorted(narration_segments):
        start = max(start, cursor)
        end = min(end, total)
        if start <= cursor:
            start = cursor

        if start > cursor:
            segments.append(music.subclipped(cursor, start))

        ramp_in = min(DUCK_RAMP_SECONDS, (end - start) / 2)
        ramp_out = ramp_in
        ducked = music.subclipped(start, end).with_volume_scaled(duck_ratio)
        ducked = ducked.with_effects([afx.AudioFadeIn(ramp_in), afx.AudioFadeOut(ramp_out)])
        segments.append(ducked)
        cursor = end

    if cursor < total:
        segments.append(music.subclipped(cursor, total))

    if not segments:
        return music

    return concatenate_audioclips(segments)
