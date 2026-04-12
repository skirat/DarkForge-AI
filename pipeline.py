#!/usr/bin/env python3
"""DarkForge AI — Automated YouTube video pipeline.

Usage:
    python pipeline.py "Darknet horror story about a hacker discovering a cursed marketplace"
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402 — mutate IMAGE_MODEL after probe; modules read config.IMAGE_MODEL at runtime
from config import (  # noqa: E402
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    OPENAI_ENABLED,
    OPENAI_IMAGE_MODEL,
    OUTPUT_DIR,
    IMAGES_DIR,
    AUDIO_DIR,
    VIDEO_DIR,
    LOGS_DIR,
    HERO_VIDEOS_DIR,
    REMOTION_RENDER_DIR,
    HERO_SCENE_COUNT,
    TEXT_MODEL,
    WORDS_PER_MINUTE,
)
from utils.logger import setup_logger  # noqa: E402
from utils.file_utils import ensure_dirs, cache_json, cache_text, save_json  # noqa: E402

from modules.youtube_metadata import generate_metadata  # noqa: E402
from modules.script_generator import generate_script  # noqa: E402
from modules.character_bible import generate_character_bible  # noqa: E402
from modules.scene_generator import (  # noqa: E402
    generate_scenes,
    expand_scenes_to_veo_segments,
)
from modules.protagonist_sparsity import apply_protagonist_sparsity  # noqa: E402
from modules.image_prompt_generator import generate_image_prompts  # noqa: E402
from modules.image_generator import generate_images  # noqa: E402
from modules.voiceover_generator import generate_voiceover  # noqa: E402
from modules.music_manager import ensure_background_music  # noqa: E402
from modules.hero_video_generator import (  # noqa: E402
    pick_hero_scene_ids,
    generate_hero_videos,
    hero_scene_ids_with_complete_hero_files,
)
from modules.remotion_renderer import render_remotion_for_scenes  # noqa: E402
from modules.video_builder import build_video  # noqa: E402
from modules.youtube_assets import create_youtube_assets  # noqa: E402
from modules.manual_video_prompts import write_manual_video_prompts  # noqa: E402

STEPS = [
    "Generate metadata",
    "Generate script",
    "Generate character bible",
    "Break into scenes",
    "Create image prompts",
    "Export manual video prompts",
    "Generate images",
    "Generate hero videos (Veo)",
    "Generate voiceover",
    "Render Remotion clips",
    "Select music",
    "Build video",
    "Create YouTube title, description, thumbnail",
]


def _exit_failed(step_number: int, step_name: str, exc: BaseException, logger: logging.Logger) -> None:
    """Log the failure, explain what went wrong, and exit with code 1."""
    err_str = str(exc).lower()
    hint = ""
    if "api key not valid" in err_str or "invalid_argument" in err_str and "key" in err_str:
        hint = "\nHint: Check GEMINI_API_KEY in .env and quota at https://aistudio.google.com/apikey"
    elif "429" in err_str or "resource_exhausted" in err_str:
        hint = "\nHint: Rate limit hit. Add more API keys (GEMINI_API_KEY_2, etc.) or retry later."

    logger.error("=" * 60)
    logger.error("PIPELINE FAILED")
    logger.error("=" * 60)
    logger.error("Step %d: %s", step_number, step_name)
    logger.error("What failed: %s%s", exc, hint)
    logger.error("Fix the issue above and re-run the pipeline.")
    logger.error("=" * 60)
    if logger.handlers:
        for h in logger.handlers:
            h.flush()
    print(
        f"\n{'=' * 60}\nPIPELINE FAILED\n{'=' * 60}\n"
        f"Step {step_number}: {step_name}\nWhat failed: {exc}{hint}\n\n"
        "Fix the issue above and re-run the pipeline.\n" + "=" * 60,
        file=sys.stderr,
    )
    sys.exit(1)


def clear_previous_output(logger: logging.Logger) -> None:
    """Remove all previous output files and cached data so each run starts fresh."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        logger.info("Cleared previous output (output/)")


def _api_key_label(index: int) -> str:
    """Return the env var name for key at index (0-based)."""
    return "GEMINI_API_KEY" if index == 0 else f"GEMINI_API_KEY_{index + 1}"


def validate_api_keys(keys: list[str], logger: logging.Logger) -> list[tuple[str, str]]:
    """Test each API key with a minimal generate_content call. Returns list of (key_label, error) for failures."""
    failures: list[tuple[str, str]] = []
    for i, api_key in enumerate(keys):
        label = _api_key_label(i)
        try:
            client = genai.Client(api_key=api_key)
            client.models.generate_content(
                model=TEXT_MODEL,
                contents="Validate",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
        except Exception as e:
            failures.append((label, str(e)))
    return failures


def validate_image_model(client: genai.Client, logger: logging.Logger) -> str | None:
    """Test that the image model works with this client. Returns error message if it fails."""
    use_imagen = config.IMAGE_MODEL.startswith("imagen-")
    try:
        if use_imagen:
            client.models.generate_images(
                model=config.IMAGE_MODEL,
                prompt="A single red apple on a white background.",
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9"),
            )
        else:
            r = client.models.generate_content(
                model=config.IMAGE_MODEL,
                contents="A single red apple on a white background.",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            if not r.candidates or not r.candidates[0].content.parts:
                return "Image model returned no image"
        return None
    except Exception as e:
        return str(e).strip() or type(e).__name__


def run_pipeline(prompt: str, *, fresh: bool = False) -> Path:
    logger = setup_logger("pipeline", LOGS_DIR)
    logger.info("=" * 60)
    logger.info("DarkForge AI Pipeline — starting")
    logger.info("Prompt: %s", prompt)
    logger.info("API keys: %d Gemini", len(GEMINI_API_KEYS))
    if OPENAI_ENABLED:
        logger.info("OpenAI fallback: enabled (text + DALL-E images when Gemini fails)")
    logger.info("=" * 60)

    if not GEMINI_API_KEYS:
        logger.error(
            "No Gemini API key set. Add to .env:\n"
            "  GEMINI_API_KEY=your_key\n"
            "  (Optional: GEMINI_API_KEY_2, …; OPENAI_API_KEY for quota fallback on text/images)\n"
            "Get keys at: https://aistudio.google.com/apikey"
        )
        sys.exit(1)

    logger.info("Validating %d API key(s) …", len(GEMINI_API_KEYS))
    failures = validate_api_keys(GEMINI_API_KEYS, logger)
    if failures:
        if OPENAI_ENABLED:
            logger.warning("=" * 60)
            logger.warning(
                "Gemini key validation reported errors — continuing (OpenAI will handle text/image if needed)"
            )
            logger.warning("=" * 60)
            for label, err in failures:
                logger.warning("%s: %s", label, err[:400] + ("..." if len(err) > 400 else ""))
            logger.warning("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("API KEY VALIDATION FAILED")
            logger.error("=" * 60)
            for label, err in failures:
                logger.error("%s: %s", label, err)
            logger.error("")
            logger.error("Fix or remove the invalid key(s) in .env and re-run.")
            logger.error("Or set OPENAI_API_KEY to continue when Gemini quota is exhausted.")
            logger.error("Get valid keys at: https://aistudio.google.com/apikey")
            logger.error("=" * 60)
            if logger.handlers:
                for h in logger.handlers:
                    h.flush()
            msg = "\n".join([f"  {label}: {err}" for label, err in failures])
            print(
                f"\n{'=' * 60}\nAPI KEY VALIDATION FAILED\n{'=' * 60}\n{msg}\n\n"
                "Fix or remove the invalid key(s) in .env and re-run.\n"
                "Or set OPENAI_API_KEY for fallback when Gemini quota is hit.\n"
                "Get valid keys at: https://aistudio.google.com/apikey\n" + "=" * 60,
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        logger.info("All API keys valid.")

    logger.info("Checking image model (%s) on every key …", config.IMAGE_MODEL)
    image_failures: list[tuple[str, str]] = []
    for i, api_key in enumerate(GEMINI_API_KEYS):
        label = _api_key_label(i)
        client = genai.Client(api_key=api_key)
        err = validate_image_model(client, logger)
        if err is None:
            logger.info("Image model OK (%s).", label)
        else:
            image_failures.append((label, err))
            logger.warning(
                "Image model check failed (%s): %s",
                label,
                err[:200] + ("..." if len(err) > 200 else ""),
            )

    if image_failures and len(image_failures) < len(GEMINI_API_KEYS):
        if config.IMAGE_MODEL.startswith("imagen-"):
            # Workers round-robin keys; Imagen must work on every key or we use native Gemini image.
            logger.warning(
                "Imagen failed on %d/%d key(s) — switching to gemini-2.5-flash-image for this run "
                "(set IMAGE_MODEL in .env to override).",
                len(image_failures),
                len(GEMINI_API_KEYS),
            )
            config.IMAGE_MODEL = "gemini-2.5-flash-image"
        else:
            logger.warning(
                "Some keys failed the image check (%d/%d); generation will rotate keys.",
                len(image_failures),
                len(GEMINI_API_KEYS),
            )

    if image_failures and len(image_failures) == len(GEMINI_API_KEYS):
        # all keys failed
        if OPENAI_ENABLED:
            logger.warning("=" * 60)
            logger.warning(
                "Gemini image model check failed on all keys — scene/thumbnail images will use OpenAI (%s) when Gemini fails per image",
                OPENAI_IMAGE_MODEL,
            )
            logger.warning("=" * 60)
            for label, err in image_failures:
                logger.warning("  %s: %s", label, err[:200] + ("..." if len(err) > 200 else ""))
            # Imagen requires a paid plan; keep running with Gemini native image + OpenAI fallback per scene.
            config.IMAGE_MODEL = "gemini-2.5-flash-image"
            logger.info(
                "Switched IMAGE_MODEL to %s for this run (set IMAGE_MODEL in .env to override).",
                config.IMAGE_MODEL,
            )
        else:
            logger.error("=" * 60)
            logger.error("IMAGE MODEL CHECK FAILED (all %d key(s))", len(GEMINI_API_KEYS))
            logger.error("=" * 60)
            logger.error("Model: %s", config.IMAGE_MODEL)
            for label, err in image_failures:
                logger.error("  %s: %s", label, err[:200] + ("..." if len(err) > 200 else ""))
            logger.error("")
            logger.error("None of your keys have image quota or access for this model.")
            logger.error("Wait for quota reset, or set IMAGE_MODEL=gemini-2.5-flash-image and try again.")
            logger.error("Or set OPENAI_API_KEY for DALL-E fallback on images.")
            logger.error("=" * 60)
            if logger.handlers:
                for h in logger.handlers:
                    h.flush()
            msg = "\n".join(
                [
                    f"  {label}: {err[:150]}..." if len(err) > 150 else f"  {label}: {err}"
                    for label, err in image_failures
                ]
            )
            print(
                f"\n{'=' * 60}\nIMAGE MODEL CHECK FAILED (all keys)\n{'=' * 60}\n"
                f"Model: {config.IMAGE_MODEL}\n\n{msg}\n\n"
                "None of your keys have image quota or access for this model.\n"
                "Wait for quota reset, set OPENAI_API_KEY, or try again later.\n" + "=" * 60,
                file=sys.stderr,
            )
            sys.exit(1)

    if fresh:
        clear_previous_output(logger)
    else:
        logger.info("Resumable run: keeping existing output (use --fresh to clear and start over)")
    ensure_dirs(
        OUTPUT_DIR,
        IMAGES_DIR,
        AUDIO_DIR,
        VIDEO_DIR,
        HERO_VIDEOS_DIR,
        REMOTION_RENDER_DIR,
        LOGS_DIR,
    )

    clients = [genai.Client(api_key=k) for k in GEMINI_API_KEYS]
    return _run_pipeline_steps(clients, prompt, logger)


def _run_pipeline_steps(
    clients: list,
    prompt: str,
    logger: logging.Logger,
) -> Path:
    progress = tqdm(STEPS, desc="Pipeline", unit="step", ncols=80)
    t0 = time.time()
    metadata = None
    script = None
    character_bible = None
    scenes = None
    image_prompts = None
    image_paths = None
    narration_wav = None
    scene_wavs = None
    remotion_video_paths: dict[int, Path] | None = None
    bg_music_path = None

    try:
        # --- Step 1: Metadata ---
        progress.set_postfix_str("metadata")
        metadata = cache_json(
            OUTPUT_DIR / "metadata.json",
            lambda: generate_metadata(clients, prompt),
        )
        progress.update(1)

        # --- Step 2: Script ---
        progress.set_postfix_str("script")
        script = cache_text(
            OUTPUT_DIR / "script.txt",
            lambda: generate_script(clients, prompt, metadata["hook"]),
        )
        progress.update(1)

        # --- Step 3: Character bible (visual continuity for stills + Veo) ---
        progress.set_postfix_str("character bible")
        character_bible = cache_json(
            OUTPUT_DIR / "characters.json",
            lambda: generate_character_bible(
                clients,
                prompt,
                script,
                title=(metadata.get("title") or ""),
                hook=(metadata.get("hook") or ""),
            ),
        )
        progress.update(1)

        # --- Step 4: Scenes (planner) then expand to ~HERO_VEO_CLIP_SEC rows with per-segment video prompts ---
        progress.set_postfix_str("scenes")
        script_words = max(1, len(script.split()))
        target_total_seconds = (script_words / float(WORDS_PER_MINUTE)) * 60.0
        scenes = cache_json(
            OUTPUT_DIR / "scenes.json",
            lambda: expand_scenes_to_veo_segments(
                clients,
                generate_scenes(
                    clients,
                    script,
                    character_bible=character_bible,
                    target_total_seconds=target_total_seconds,
                ),
                character_bible=character_bible,
            ),
        )
        scenes = apply_protagonist_sparsity(
            scenes, character_bible, metadata, script=script or ""
        )
        save_json(OUTPUT_DIR / "scenes.json", scenes)
        logger.info("Scenes: %d", len(scenes))
        progress.update(1)

        # --- Step 5: Image prompts ---
        progress.set_postfix_str("image prompts")
        image_prompts = cache_json(
            OUTPUT_DIR / "image_prompts.json",
            lambda: generate_image_prompts(
                clients, scenes, character_bible=character_bible
            ),
        )
        progress.update(1)

        # --- Step 6: Manual video prompts (JSON + TXT for external Veo / Runway / etc.) ---
        progress.set_postfix_str("manual video prompts")
        write_manual_video_prompts(
            OUTPUT_DIR, scenes, character_bible, metadata, image_prompts, logger
        )
        progress.update(1)

        # --- Step 7: Images (parallel with one key per worker) ---
        progress.set_postfix_str("images")
        hero_scene_ids = pick_hero_scene_ids(scenes, HERO_SCENE_COUNT)
        skip_img_for = hero_scene_ids_with_complete_hero_files(
            scenes, hero_scene_ids, HERO_VIDEOS_DIR
        )
        if skip_img_for:
            logger.info(
                "Hero MP4s already on disk for %d scene(s); skipping image API for those (cached PNG or placeholder)",
                len(skip_img_for),
            )
        image_paths = generate_images(
            clients, image_prompts, skip_api_for_scene_ids=skip_img_for
        )
        progress.update(1)

        # --- Step 8: Hero videos (Veo) — try for each scene when HERO_SCENE_COUNT=0 ---
        progress.set_postfix_str("hero videos")
        if hero_scene_ids:
            logger.info(
                "Generating hero videos for %d scene(s) (Veo-first; fallback to image for failures) …",
                len(hero_scene_ids),
            )
        hero_video_paths = generate_hero_videos(
            clients,
            image_prompts,
            hero_scene_ids,
            scenes,
            character_bible=character_bible,
        )
        if hero_video_paths:
            n_clips = sum(
                len(v) if isinstance(v, list) else 1 for v in hero_video_paths.values()
            )
            logger.info(
                "Hero videos ready: %d clip(s) across %d scene(s) %s",
                n_clips,
                len(hero_video_paths),
                sorted(hero_video_paths.keys()),
            )
        progress.update(1)

        # --- Step 9: Voiceover (round-robin keys) ---
        progress.set_postfix_str("voiceover")
        narration_wav, scene_wavs = generate_voiceover(clients, scenes)
        progress.update(1)

        # --- Step 10: Remotion clips for scenes without Veo (real MP4, not Ken Burns on stills) ---
        progress.set_postfix_str("remotion")
        remotion_video_paths = render_remotion_for_scenes(
            scenes, image_paths, scene_wavs, hero_video_paths or {}
        )
        if remotion_video_paths:
            logger.info(
                "Remotion clips: %d scene(s) (Veo + Remotion + image fallback as needed)",
                len(remotion_video_paths),
            )
        progress.update(1)

        # --- Step 11: Select music (theme-based; fallback ambient if no files) ---
        progress.set_postfix_str("music")
        bg_music_path = ensure_background_music(
            scenes=scenes,
            metadata=metadata,
            clients=clients,
            character_bible=character_bible,
        )
        progress.update(1)

        # --- Step 12: Build video (Veo > Remotion > image/motion, voiceover, music) ---
        progress.set_postfix_str("video")
        video_path = build_video(
            scenes, image_paths, scene_wavs,
            bg_music_path=bg_music_path,
            hero_video_paths=hero_video_paths or None,
            remotion_video_paths=remotion_video_paths or None,
        )
        progress.update(1)

        # --- Step 13: YouTube title, description, tags, thumbnail (Gemini 4 keys, then NanoBanana) ---
        progress.set_postfix_str("YouTube assets")
        create_youtube_assets(OUTPUT_DIR / "metadata.json", OUTPUT_DIR, clients=clients)
        progress.update(1)
    except Exception as exc:
        progress.close()
        # progress.n = number of steps completed so far; we failed on the next one
        step_num = min(progress.n + 1, len(STEPS))
        step_name = STEPS[step_num - 1]
        _exit_failed(step_num, step_name, exc, logger)

    progress.close()

    elapsed = time.time() - t0
    srt_path = OUTPUT_DIR / "subtitles.srt"
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Video  → %s", video_path)
    logger.info("SRT    → %s", srt_path)
    logger.info("Audio  → %s", narration_wav)
    logger.info("Meta   → %s", OUTPUT_DIR / "metadata.json")
    if (OUTPUT_DIR / "characters.json").exists():
        logger.info("Characters → %s", OUTPUT_DIR / "characters.json")
    if (OUTPUT_DIR / "manual_video_prompts.json").exists():
        logger.info("Manual video prompts (JSON) → %s", OUTPUT_DIR / "manual_video_prompts.json")
    if (OUTPUT_DIR / "manual_video_prompts.txt").exists():
        logger.info("Manual video prompts (index) → %s", OUTPUT_DIR / "manual_video_prompts.txt")
    clips_idx = OUTPUT_DIR / "manual_video_prompts" / "clips"
    if clips_idx.is_dir():
        logger.info("Manual video prompts (per-clip copy-paste) → %s", clips_idx)
    if (OUTPUT_DIR / "youtube_title.txt").exists():
        logger.info("YT title → %s", OUTPUT_DIR / "youtube_title.txt")
    if (OUTPUT_DIR / "youtube_description.txt").exists():
        logger.info("YT desc  → %s", OUTPUT_DIR / "youtube_description.txt")
    if (OUTPUT_DIR / "thumbnail.png").exists():
        logger.info("Thumbnail → %s", OUTPUT_DIR / "thumbnail.png")
    logger.info("=" * 60)

    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DarkForge AI — generate a YouTube video from a single prompt.",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The video idea / topic (e.g. 'Darknet horror story about …')",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear all previous output before running (default: keep output and resume from where it failed)",
    )
    args = parser.parse_args()
    run_pipeline(args.prompt, fresh=args.fresh)


if __name__ == "__main__":
    main()
