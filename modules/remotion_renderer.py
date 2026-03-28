"""Render per-scene MP4 clips with Remotion (Node) when Veo did not produce a hero clip."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import (
    BASE_DIR,
    REMOTION_CLIPS_DIR,
    REMOTION_COMPOSITION_ID,
    REMOTION_EMBED_SCENE_IMAGE,
    REMOTION_RENDER_DIR,
    REMOTION_RENDER_TIMEOUT_SEC,
    ENABLE_REMOTION,
    VIDEO_FPS,
)

logger = logging.getLogger("pipeline")


def remotion_environment_ready() -> bool:
    """True if Node is on PATH and remotion_clips is present (run npm install there)."""
    if not shutil.which("node"):
        return False
    if not (REMOTION_CLIPS_DIR / "package.json").exists():
        return False
    if not (REMOTION_CLIPS_DIR / "src" / "index.ts").exists():
        return False
    return True


def _wav_duration_sec(wav_path: Path) -> float:
    from moviepy import AudioFileClip

    clip = AudioFileClip(str(wav_path))
    try:
        return max(0.1, float(clip.duration))
    finally:
        clip.close()


def _first_line_or_words(text: str, max_len: int = 110) -> str:
    """Short on-screen caption from narration — never dump full visual_prompt."""
    text = (text or "").strip()
    if not text:
        return ""
    for sep in ".?!":
        i = text.find(sep)
        if 8 < i <= max_len + 30:
            return text[: i + 1].strip()[:max_len]
    words = text.split()
    if not words:
        return ""
    out: list[str] = []
    n = 0
    for w in words[:18]:
        if n + len(w) + 1 > max_len:
            break
        out.append(w)
        n += len(w) + 1
    s = " ".join(out)
    return (s + "…") if len(words) > len(out) else s


def _scene_text_for_props(scene: dict) -> tuple[str, str]:
    narr = (scene.get("narration") or "").strip()
    if not narr:
        visual = (scene.get("visual_prompt") or "").strip()
        h = (visual[:85] + "…") if len(visual) > 85 else (visual or "—")
        return h, ""
    first_end = -1
    for sep in ".?!":
        i = narr.find(sep)
        if i > 0 and (first_end < 0 or i < first_end):
            first_end = i
    if 0 < first_end < len(narr) - 1:
        headline = narr[: first_end + 1].strip()[:110]
        rest = narr[first_end + 1 :].strip()
        subtitle = _first_line_or_words(rest, 90) if rest else ""
        return headline, subtitle
    return _first_line_or_words(narr, 110), ""


def render_remotion_for_scenes(
    scenes: list[dict],
    image_paths: list[Path],
    scene_wavs: list[Path],
    hero_video_paths: dict[int, Path] | None = None,
) -> dict[int, Path]:
    """Render Remotion clips for scenes that lack a Veo hero video. Returns scene_id -> mp4 path."""
    hero_video_paths = hero_video_paths or {}
    if not ENABLE_REMOTION:
        logger.info("Remotion disabled (ENABLE_REMOTION=0).")
        return {}
    if not remotion_environment_ready():
        logger.warning(
            "Remotion skipped: install Node 18+ and run `cd remotion_clips && npm install` in %s",
            BASE_DIR,
        )
        return {}

    sorted_scenes = sorted(scenes, key=lambda s: s["scene_id"])
    if len(sorted_scenes) != len(image_paths) or len(sorted_scenes) != len(scene_wavs):
        logger.warning(
            "Remotion skipped: scenes/images/wavs length mismatch (%d / %d / %d)",
            len(sorted_scenes),
            len(image_paths),
            len(scene_wavs),
        )
        return {}

    REMOTION_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}

    entry = "src/index.ts"
    entry_path = REMOTION_CLIPS_DIR / entry
    if not entry_path.exists():
        logger.error("Remotion entry missing: %s", entry_path)
        return {}

    for scene, img_path, wav_path in zip(sorted_scenes, image_paths, scene_wavs):
        scene_id = scene["scene_id"]
        if scene_id in hero_video_paths and hero_video_paths[scene_id].exists():
            continue

        out_path = REMOTION_RENDER_DIR / f"scene_{scene_id:03d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 1000:
            logger.info("Using cached Remotion clip for scene %d → %s", scene_id, out_path.name)
            result[scene_id] = out_path
            continue

        duration = _wav_duration_sec(wav_path)
        duration_in_frames = max(1, min(36000, int(round(duration * VIDEO_FPS))))
        headline, subtitle = _scene_text_for_props(scene)

        image_src = ""
        if REMOTION_EMBED_SCENE_IMAGE and img_path.exists():
            # Remotion headless render requires assets under public/ + staticFile()
            public_dir = REMOTION_CLIPS_DIR / "public" / "df_render"
            public_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"scene_{scene_id:03d}{img_path.suffix.lower() or '.png'}"
            dest_file = public_dir / dest_name
            try:
                shutil.copy2(img_path, dest_file)
                image_src = f"df_render/{dest_name}"
            except OSError as exc:
                logger.warning("Could not copy image for Remotion scene %d: %s", scene_id, exc)

        props = {
            "durationInFrames": duration_in_frames,
            "headline": headline,
            "subtitle": subtitle,
            "imageSrc": image_src,
            "embedImage": 1 if (REMOTION_EMBED_SCENE_IMAGE and image_src) else 0,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(props, tmp, ensure_ascii=False)
            props_file = Path(tmp.name)

        try:
            cmd = [
                "npx",
                "--yes",
                "remotion",
                "render",
                entry,
                REMOTION_COMPOSITION_ID,
                str(out_path),
                f"--props={props_file}",
                "--log=warn",
            ]
            logger.info(
                "Remotion render scene %d (~%.1fs, %d frames) …",
                scene_id,
                duration,
                duration_in_frames,
            )
            proc = subprocess.run(
                cmd,
                cwd=str(REMOTION_CLIPS_DIR),
                capture_output=True,
                text=True,
                timeout=REMOTION_RENDER_TIMEOUT_SEC,
                check=False,
            )
            if proc.returncode != 0 or not out_path.exists():
                err = (proc.stderr or proc.stdout or "")[:2000]
                logger.warning(
                    "Remotion render failed for scene %d (exit %s): %s",
                    scene_id,
                    proc.returncode,
                    err.replace("%", "%%"),
                )
                continue
            result[scene_id] = out_path
            logger.info("Remotion clip saved → %s", out_path.name)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Remotion render timed out for scene %d after %ds",
                scene_id,
                REMOTION_RENDER_TIMEOUT_SEC,
            )
        except Exception as exc:
            logger.warning("Remotion render error for scene %d: %s", scene_id, exc)
        finally:
            try:
                props_file.unlink(missing_ok=True)
            except OSError:
                pass

    if result:
        logger.info("Remotion clips ready: %d scene(s)", len(result))
    return result
