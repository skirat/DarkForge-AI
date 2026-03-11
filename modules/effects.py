"""Visual effects library for cinematic scene animation.

All public functions accept an np.ndarray image (H, W, 3) and a duration,
returning a MoviePy VideoClip.  Overlay functions return RGBA clips meant
to be composited on top of a background layer.
"""
from __future__ import annotations

import random
import string
from typing import Any

import numpy as np
from moviepy import VideoClip, ImageClip, vfx
from PIL import Image, ImageDraw, ImageFont

from config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    ZOOM_START,
    ZOOM_END,
    PAN_OVERSCAN,
    GLITCH_INTERVAL,
    GLITCH_BURST_FRAMES,
    OVERLAY_OPACITY,
    OVERLAY_DEFAULT_CHANCE,
    VIGNETTE_STRENGTH,
)

# ---------------------------------------------------------------------------
#  Motion effects (return opaque RGB clips)
# ---------------------------------------------------------------------------

def apply_zoom_in(img: np.ndarray, duration: float) -> VideoClip:
    h, w = img.shape[:2]

    def _frame(t: float) -> np.ndarray:
        p = t / max(duration, 0.001)
        s = ZOOM_START + (ZOOM_END - ZOOM_START) * p
        cw, ch = int(w / s), int(h / s)
        x, y = (w - cw) // 2, (h - ch) // 2
        crop = img[y : y + ch, x : x + cw]
        return np.array(Image.fromarray(crop).resize((w, h), Image.LANCZOS))

    return VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)


def apply_zoom_out(img: np.ndarray, duration: float) -> VideoClip:
    h, w = img.shape[:2]

    def _frame(t: float) -> np.ndarray:
        p = t / max(duration, 0.001)
        s = ZOOM_END - (ZOOM_END - ZOOM_START) * p
        cw, ch = int(w / s), int(h / s)
        x, y = (w - cw) // 2, (h - ch) // 2
        crop = img[y : y + ch, x : x + cw]
        return np.array(Image.fromarray(crop).resize((w, h), Image.LANCZOS))

    return VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)


def apply_pan_left(img: np.ndarray, duration: float) -> VideoClip:
    """Pan from right to left across an over-scanned image."""
    h, w = img.shape[:2]
    wide_w = int(w * PAN_OVERSCAN)
    wide = np.array(Image.fromarray(img).resize((wide_w, h), Image.LANCZOS))
    max_offset = wide_w - w

    def _frame(t: float) -> np.ndarray:
        p = t / max(duration, 0.001)
        x = int(max_offset * (1.0 - p))
        return wide[:, x : x + w]

    return VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)


def apply_pan_right(img: np.ndarray, duration: float) -> VideoClip:
    h, w = img.shape[:2]
    wide_w = int(w * PAN_OVERSCAN)
    wide = np.array(Image.fromarray(img).resize((wide_w, h), Image.LANCZOS))
    max_offset = wide_w - w

    def _frame(t: float) -> np.ndarray:
        p = t / max(duration, 0.001)
        x = int(max_offset * p)
        return wide[:, x : x + w]

    return VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)


_MOTION_FNS = [apply_zoom_in, apply_zoom_out, apply_pan_left, apply_pan_right]


def pick_random_motion(img: np.ndarray, duration: float) -> VideoClip:
    fn = random.choice(_MOTION_FNS)
    return fn(img, duration)


# ---------------------------------------------------------------------------
#  Post-processing effects (applied to an existing clip's frames)
# ---------------------------------------------------------------------------

def apply_glitch(clip: VideoClip) -> VideoClip:
    """Add periodic RGB-shift + slice displacement bursts."""
    dur = clip.duration
    rng = np.random.default_rng(42)

    burst_times: set[int] = set()
    t = 0.0
    while t < dur:
        for bf in range(GLITCH_BURST_FRAMES):
            burst_times.add(int((t + bf / VIDEO_FPS) * VIDEO_FPS))
        t += GLITCH_INTERVAL

    orig_frame = clip.get_frame

    def _frame(t: float) -> np.ndarray:
        frame = orig_frame(t).copy()
        fid = int(t * VIDEO_FPS)
        if fid not in burst_times:
            return frame
        h, w = frame.shape[:2]
        shift = rng.integers(5, 20)
        frame[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1)
        frame[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)
        for _ in range(3):
            y = rng.integers(0, h - 10)
            span = rng.integers(2, 8)
            dx = rng.integers(-30, 30)
            frame[y : y + span] = np.roll(frame[y : y + span], dx, axis=1)
        return frame

    return VideoClip(_frame, duration=dur).with_fps(VIDEO_FPS)


def apply_flicker(clip: VideoClip) -> VideoClip:
    """Simulate brief CRT-style brightness dips."""
    dur = clip.duration
    rng = np.random.default_rng(7)
    flicker_frames = {int(t * VIDEO_FPS) for t in rng.uniform(0, dur, size=int(dur * 2))}
    orig_frame = clip.get_frame

    def _frame(t: float) -> np.ndarray:
        frame = orig_frame(t)
        if int(t * VIDEO_FPS) in flicker_frames:
            factor = rng.uniform(0.5, 0.8)
            return (frame * factor).astype(np.uint8)
        return frame

    return VideoClip(_frame, duration=dur).with_fps(VIDEO_FPS)


# ---------------------------------------------------------------------------
#  Overlay generators (return RGBA clips for compositing)
# ---------------------------------------------------------------------------

_CODE_CHARS = string.ascii_letters + string.digits + "{}[]<>=/\\|@#$%&*"
_NUM_COLS = 60
_CHAR_H = 18


def make_code_rain_overlay(duration: float) -> VideoClip:
    """Matrix-style falling green characters."""
    col_w = VIDEO_WIDTH // _NUM_COLS
    speeds = np.random.default_rng(0).uniform(40, 120, size=_NUM_COLS)
    offsets = np.random.default_rng(1).uniform(0, VIDEO_HEIGHT, size=_NUM_COLS)
    rng = random.Random(3)

    def _frame(t: float) -> np.ndarray:
        canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
        pil = Image.fromarray(canvas, "RGBA")
        draw = ImageDraw.Draw(pil)
        for col in range(_NUM_COLS):
            x = col * col_w
            y_head = (offsets[col] + speeds[col] * t) % (VIDEO_HEIGHT + 200)
            for row in range(18):
                y = int(y_head - row * _CHAR_H)
                if y < 0 or y > VIDEO_HEIGHT - _CHAR_H:
                    continue
                alpha = max(0, 255 - row * 15)
                ch = rng.choice(_CODE_CHARS)
                draw.text((x, y), ch, fill=(0, 255, 70, alpha))
        return np.array(pil)

    clip = VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)
    return clip.with_opacity(OVERLAY_OPACITY)


def make_scanline_overlay(duration: float) -> VideoClip:
    """Static semi-transparent horizontal scanlines."""
    canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
    for y in range(0, VIDEO_HEIGHT, 3):
        canvas[y, :, :3] = 0
        canvas[y, :, 3] = 40

    clip = ImageClip(canvas, duration=duration, is_mask=False).with_fps(VIDEO_FPS)
    return clip.with_opacity(OVERLAY_OPACITY)


def make_terminal_overlay(text: str, duration: float) -> VideoClip:
    """Typing animation of monospaced text on a dark panel (top-left)."""
    panel_w, panel_h = 700, 220
    chars_per_sec = 30

    def _frame(t: float) -> np.ndarray:
        canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
        pil = Image.fromarray(canvas, "RGBA")
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(20, 20), (20 + panel_w, 20 + panel_h)], fill=(0, 0, 0, 180))
        visible = int(t * chars_per_sec)
        shown = text[:visible]
        lines = _wrap_text(shown, 55)
        y = 30
        for line in lines[:8]:
            draw.text((30, y), line, fill=(0, 255, 70, 230))
            y += 22
        if int(t * 3) % 2 == 0:
            draw.text((30 + len(lines[-1]) * 8 if lines else 30, y - 22 if lines else 30), "█", fill=(0, 255, 70, 230))
        return np.array(pil)

    return VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)


def _wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    while len(text) > width:
        lines.append(text[:width])
        text = text[width:]
    if text:
        lines.append(text)
    return lines or [""]


# ---------------------------------------------------------------------------
#  Vignette (darkened edges for cinematic look)
# ---------------------------------------------------------------------------

def make_vignette_overlay(duration: float, strength: float | None = None) -> VideoClip:
    """Darken edges of the frame. Strength 0 = none, 0.5 = strong."""
    s = strength if strength is not None else VIGNETTE_STRENGTH
    if s <= 0:
        return None  # type: ignore
    h, w = VIDEO_HEIGHT, VIDEO_WIDTH
    yx = np.ogrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yx[1] - cx) ** 2 + (yx[0] - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    alpha = np.clip((r / max_r) ** 2 * s * 255, 0, 255).astype(np.uint8)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    canvas[:, :, 3] = alpha

    def _frame(t: float) -> np.ndarray:
        return canvas.copy()

    clip = VideoClip(_frame, duration=duration).with_fps(VIDEO_FPS)
    return clip


# ---------------------------------------------------------------------------
#  Overlay picker
# ---------------------------------------------------------------------------

_HACKER_KEYWORDS = {"hack", "hacker", "terminal", "code", "keyboard", "typing", "screen", "monitor", "cyber", "computer", "exploit"}
_GLITCH_KEYWORDS = {"glitch", "distort", "corrupt", "flicker", "static", "anomal"}
_DARK_KEYWORDS = {"dark", "shadow", "horror", "eerie", "creep", "dread", "fear", "sinister"}


def _has_keyword(text: str, keywords: set[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def pick_random_overlay(scene: dict[str, Any], duration: float) -> VideoClip | None:
    """Return a thematic overlay clip based on scene content. Prefer overlays for visual interest."""
    visual = scene.get("visual_prompt", "")
    narration = scene.get("narration", "")
    combined = f"{visual} {narration}"

    if _has_keyword(combined, _HACKER_KEYWORDS):
        choice = random.choice(["code_rain", "terminal"])
        if choice == "code_rain":
            return make_code_rain_overlay(duration)
        snippet = narration[:200] if narration else "$ sudo access granted..."
        return make_terminal_overlay(snippet, duration)

    if _has_keyword(combined, _GLITCH_KEYWORDS):
        return make_scanline_overlay(duration)

    # Default: add an overlay often for visual variety (code rain, scanlines, or terminal)
    if random.random() < OVERLAY_DEFAULT_CHANCE:
        return random.choice([
            make_code_rain_overlay(duration),
            make_scanline_overlay(duration),
        ])

    return None


def should_apply_glitch(scene: dict[str, Any]) -> bool:
    combined = f"{scene.get('visual_prompt', '')} {scene.get('narration', '')}"
    return _has_keyword(combined, _GLITCH_KEYWORDS | _HACKER_KEYWORDS)


def should_apply_flicker(scene: dict[str, Any]) -> bool:
    combined = f"{scene.get('visual_prompt', '')} {scene.get('narration', '')}"
    return _has_keyword(combined, _DARK_KEYWORDS)
