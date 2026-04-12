"""Cinematic video builder with multi-layer compositing, transitions,
animated subtitles, audio ducking, SFX, and color grading.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from moviepy import (
    VideoClip,
    AudioFileClip,
    VideoFileClip,
    TextClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    concatenate_audioclips,
    vfx,
)
from PIL import Image

from config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    VIDEO_CODEC,
    AUDIO_CODEC,
    VIDEO_DIR,
    FONTS_DIR,
    CROSSFADE_DURATION,
    TRANSITION_TYPES,
    GLITCH_CUT_DURATION,
    COLOR_GRADE_FILTER,
    OUTPUT_DIR,
    VIGNETTE_STRENGTH,
    REMOTION_SKIP_OVERLAY,
    SCENE_NARRATION_GAP_SEC,
)
from modules import effects as fx
from modules import music_manager
from modules import sfx_manager
from modules.hero_video_generator import get_hero_paths_for_scene
from utils.ffmpeg_utils import color_grade

logger = logging.getLogger("pipeline")


# -------------------------------------------------------------------
#  Image helpers
# -------------------------------------------------------------------

def _resize_cover(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_h = img.height
        new_w = int(new_h * target_ratio)
    else:
        new_w = img.width
        new_h = int(new_w / target_ratio)

    left = (img.width - new_w) // 2
    top = (img.height - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
    return np.array(img)


# -------------------------------------------------------------------
#  Animated subtitles with glow + fade
# -------------------------------------------------------------------

def _chunk_narration(text: str, max_chars: int = 80) -> list[str]:
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


def _find_font() -> str | None:
    for ext in ("*.ttf", "*.otf"):
        fonts = list(FONTS_DIR.glob(ext))
        if fonts:
            return str(fonts[0])
    return None


def _animated_subtitle_clips(
    narration: str,
    duration: float,
    font_path: str | None,
) -> list[VideoClip]:
    """Subtitle chunks with glow backing and per-chunk fade in/out."""
    chunks = _chunk_narration(narration)
    if not chunks:
        return []

    chunk_dur = duration / len(chunks)
    fade = min(0.15, chunk_dur / 4)
    clips: list[VideoClip] = []

    for i, text in enumerate(chunks):
        base_kw: dict = {
            "text": text,
            "font_size": 42,
            "color": "white",
            "stroke_color": "black",
            "stroke_width": 2,
            "method": "caption",
            "size": (VIDEO_WIDTH - 200, None),
            "text_align": "center",
        }
        if font_path:
            base_kw["font"] = font_path

        glow_kw = {**base_kw, "color": "#00ccff", "stroke_color": "#003344", "font_size": 44}

        glow = (
            TextClip(**glow_kw)
            .with_opacity(0.35)
            .with_position(("center", VIDEO_HEIGHT - 162))
            .with_start(i * chunk_dur)
            .with_duration(chunk_dur)
            .with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])
        )

        main = (
            TextClip(**base_kw)
            .with_position(("center", VIDEO_HEIGHT - 160))
            .with_start(i * chunk_dur)
            .with_duration(chunk_dur)
            .with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])
        )

        clips.extend([glow, main])
    return clips


# -------------------------------------------------------------------
#  Transition helpers
# -------------------------------------------------------------------

def _apply_transition(clip: VideoClip, ttype: str) -> VideoClip:
    if ttype == "crossfade":
        return clip.with_effects([vfx.CrossFadeIn(CROSSFADE_DURATION)])
    elif ttype == "fade_black":
        return clip.with_effects([
            vfx.FadeIn(CROSSFADE_DURATION, initial_color=(0, 0, 0)),
        ])
    elif ttype == "glitch_cut":
        return clip.with_effects([vfx.CrossFadeIn(GLITCH_CUT_DURATION)])
    return clip


def _silent_audio_clip(duration_sec: float, ref: AudioFileClip) -> AudioFileClip:
    """Gap padding; temp WAV cleaned up by OS after process exit."""
    from pydub import AudioSegment

    ms = max(1, int(duration_sec * 1000))
    seg = AudioSegment.silent(duration=ms)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    seg.export(path, format="wav")
    return AudioFileClip(path)


def _extend_video_to_speech_duration(bg: VideoClip, speech_dur: float) -> VideoClip:
    """Trim long clips to narration length.

    If the clip is longer than *speech_dur*, use the opening segment only.

    If shorter: start at the midpoint, play to the end, then play from the start to the
    midpoint (full footage in a new order, no frozen frame). Repeat that cycle until
    *speech_dur* is covered, then trim.
    """
    if speech_dur <= 0:
        return bg
    D = float(bg.duration)
    if D <= 0:
        return bg
    if D >= speech_dur:
        return bg.subclipped(0, speech_dur)

    mid = D / 2.0
    part_a = bg.subclipped(mid, D)
    part_b = bg.subclipped(0, mid)
    cycle = concatenate_videoclips([part_a, part_b])
    c_len = float(cycle.duration)
    if c_len <= 1e-6:
        return bg.subclipped(0, min(D, speech_dur))

    n = max(1, int(math.ceil(speech_dur / c_len)))
    long_bg = concatenate_videoclips([cycle] * n)
    return long_bg.subclipped(0, speech_dur)


def _append_freeze_last_frame(clip: VideoClip, gap: float, fps: int) -> VideoClip:
    """Hold last frame for *gap* seconds (paired with silent narration tail)."""
    if gap <= 0:
        return clip
    t = max(0.0, float(clip.duration) - 1.0 / max(fps, 1))
    frame = clip.get_frame(t)
    frozen = ImageClip(frame).with_duration(gap).with_fps(fps)
    return concatenate_videoclips([clip, frozen])


# -------------------------------------------------------------------
#  SRT side-effect writer
# -------------------------------------------------------------------

def _write_srt(
    scenes: list[dict],
    scene_wavs: list[Path],
    scene_start_times: list[float],
    srt_path: Path,
) -> None:
    """Write SRT file synced to the final rendered video timeline (with crossfades).

    scene_start_times[i] = start time of scene i in the final video (seconds).
    Use this file when uploading to YouTube; captions are not burned into the video.
    """
    from pydub import AudioSegment

    entries: list[str] = []
    idx = 1
    for scene, wav, start_time in zip(
        sorted(scenes, key=lambda s: s["scene_id"]), scene_wavs, scene_start_times
    ):
        audio = AudioSegment.from_wav(str(wav))
        dur = len(audio) / 1000.0
        chunks = _chunk_narration(scene["narration"])
        if not chunks:
            continue
        cdur = dur / len(chunks)
        for chunk in chunks:
            s = start_time
            e = start_time + cdur
            entries.append(
                f"{idx}\n"
                f"{_fmt(s)} --> {_fmt(e)}\n"
                f"{chunk}\n"
            )
            idx += 1
            start_time = e
    srt_path.write_text("\n".join(entries), encoding="utf-8")
    logger.info("SRT saved → %s (%d entries, synced to video)", srt_path.name, idx - 1)


def _fmt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# -------------------------------------------------------------------
#  Main builder
# -------------------------------------------------------------------

def build_video(
    scenes: list[dict],
    image_paths: list[Path],
    scene_wavs: list[Path],
    bg_music_path: Path | list[Path] | None = None,
    hero_video_paths: dict[int, Path | list[Path]] | None = None,
    remotion_video_paths: dict[int, Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or (VIDEO_DIR / "video.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hero_video_paths = hero_video_paths or {}
    remotion_video_paths = remotion_video_paths or {}

    logger.info("Building cinematic video from %d scenes …", len(scenes))

    sorted_scenes = sorted(scenes, key=lambda s: s["scene_id"])
    scene_clips: list[CompositeVideoClip] = []
    scene_start_times: list[float] = []  # start time of each scene in final video (for SRT sync)
    narration_segments: list[tuple[float, float]] = []
    all_sfx: list[tuple[float, Path]] = []
    global_offset = 0.0

    n_scenes = len(sorted_scenes)
    for idx, (scene, img_path, wav_path) in enumerate(
        zip(sorted_scenes, image_paths, scene_wavs)
    ):
        scene_start_times.append(global_offset)
        scene_id = scene["scene_id"]
        audio_clip = AudioFileClip(str(wav_path))
        speech_dur = audio_clip.duration
        gap = SCENE_NARRATION_GAP_SEC if idx < n_scenes - 1 else 0.0

        from_remotion_bg = False
        hero_paths = get_hero_paths_for_scene(hero_video_paths, scene_id)
        if hero_paths:
            # Hero: concatenate distinct Veo clips; if still shorter than VO, middle→end then start→middle, loop
            clips_v = [VideoFileClip(str(p)).with_fps(VIDEO_FPS) for p in hero_paths]
            bg = concatenate_videoclips(clips_v)
            bg = bg.with_effects([vfx.Resize((VIDEO_WIDTH, VIDEO_HEIGHT))])
            bg = _extend_video_to_speech_duration(bg, speech_dur)
        elif scene_id in remotion_video_paths and remotion_video_paths[scene_id].exists():
            rpath = remotion_video_paths[scene_id]
            bg = VideoFileClip(str(rpath)).with_fps(VIDEO_FPS)
            bg = bg.with_effects([vfx.Resize((VIDEO_WIDTH, VIDEO_HEIGHT))])
            bg = _extend_video_to_speech_duration(bg, speech_dur)
            from_remotion_bg = True
        else:
            img_array = _resize_cover(img_path)
            bg = fx.pick_random_motion(img_array, speech_dur)
            if fx.should_apply_glitch(scene):
                bg = fx.apply_glitch(bg)
            elif fx.should_apply_flicker(scene):
                bg = fx.apply_flicker(bg)

        if from_remotion_bg and REMOTION_SKIP_OVERLAY:
            overlay = None
            vignette = None
        else:
            overlay = fx.pick_random_overlay(scene, speech_dur)
            vignette = (
                fx.make_vignette_overlay(speech_dur) if VIGNETTE_STRENGTH > 0 else None
            )

        layers = [bg]
        if overlay is not None:
            layers.append(overlay)
        if vignette is not None:
            layers.append(vignette)

        composite = CompositeVideoClip(layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT)).with_duration(
            speech_dur
        )
        if gap > 0:
            composite = _append_freeze_last_frame(composite, gap, VIDEO_FPS)
        scene_dur = float(composite.duration)

        if gap > 0:
            full_audio = concatenate_audioclips(
                [audio_clip, _silent_audio_clip(gap, audio_clip)]
            )
        else:
            full_audio = audio_clip
        composite = composite.with_audio(full_audio).with_duration(scene_dur)

        curr_hero = len(get_hero_paths_for_scene(hero_video_paths, scene_id)) > 0
        if idx > 0:
            prev_id = sorted_scenes[idx - 1]["scene_id"]
            prev_hero = len(get_hero_paths_for_scene(hero_video_paths, prev_id)) > 0
        else:
            prev_hero = False
        if prev_hero and curr_hero:
            ttype = "fade_black"
        else:
            ttype = TRANSITION_TYPES[idx % len(TRANSITION_TYPES)]
        composite = _apply_transition(composite, ttype)
        if idx < n_scenes - 1:
            composite = composite.with_effects(
                [vfx.CrossFadeOut(CROSSFADE_DURATION)]
            )

        scene_clips.append(composite)

        narration_segments.append((global_offset, global_offset + speech_dur))

        sfx_hits = sfx_manager.get_sfx_for_scene(scene, global_offset, speech_dur)
        all_sfx.extend(sfx_hits)

        global_offset += (
            scene_dur - CROSSFADE_DURATION if idx > 0 else scene_dur
        )

    # Concatenate scenes
    if len(scene_clips) > 1:
        final = concatenate_videoclips(
            scene_clips, padding=-CROSSFADE_DURATION, method="compose"
        )
    else:
        final = scene_clips[0]

    # --- Audio mixing ---
    audio_layers = [final.audio]

    # Background music with ducking
    if bg_music_path:
        names = [p.name for p in (bg_music_path if isinstance(bg_music_path, list) else [bg_music_path])]
        logger.info("Mixing background music with ducking: %s", names)
        music = music_manager.prepare_music(bg_music_path, final.duration)
        music = music_manager.duck_under_narration(music, narration_segments)
        audio_layers.append(music)

    # SFX
    for offset, sfx_path in all_sfx:
        clip = sfx_manager.load_sfx_clip(sfx_path)
        if clip is not None:
            clip = clip.with_start(offset)
            audio_layers.append(clip)

    if len(audio_layers) > 1:
        final = final.with_audio(CompositeAudioClip(audio_layers))

    # Write SRT sidecar (synced to final video timeline; upload to YouTube separately)
    srt_path = OUTPUT_DIR / "subtitles.srt"
    _write_srt(sorted_scenes, scene_wavs, scene_start_times, srt_path)

    # Render to temp, then color-grade to final
    logger.info(
        "Rendering video → %s (%.1fs @ %dfps) …",
        output_path.name, final.duration, VIDEO_FPS,
    )

    tmp_path = output_path.with_suffix(".tmp.mp4")
    final.write_videofile(
        str(tmp_path),
        fps=VIDEO_FPS,
        codec=VIDEO_CODEC,
        audio_codec=AUDIO_CODEC,
        threads=4,
        logger="bar",
    )

    # Color grading pass
    try:
        color_grade(tmp_path, output_path, COLOR_GRADE_FILTER)
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Color grading failed (%s), using ungraded video", exc)
        tmp_path.rename(output_path)

    logger.info("Video saved → %s", output_path)
    return output_path
