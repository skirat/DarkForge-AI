# DarkForge AI — Automated YouTube Video Pipeline

Generate a complete YouTube video from a single text prompt using Google Gemini.

The pipeline handles script writing, scene planning, image generation, voiceover, subtitles, and final video assembly — all automatically. Videos are **5–10 minutes** long with **many scenes** and **rich visuals** (motion, overlays, vignette, transitions). Scripts target **clear, simple English** so general viewers can follow the story; technical words are used sparingly and kept understandable.

## Niche

Ethical hacking stories, darknet horror, and cyberpunk storytelling.

## Quick Start

```bash
# 1. Clone / enter the project (DarkForge AI root)
cd "DarkForge AI"

# 2. Create a virtual environment (use python3 if python is not available)
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 3b. Remotion (Node 18+) — real MP4 clips for scenes where Veo is unavailable
cd remotion_clips && npm install && cd ..

# 4. Set your Gemini API key(s)
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key
# Optional: add GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4 for parallel image/voiceover (avoids rate limits)

# 5. Run the pipeline (use python from the venv after activating)
python pipeline.py "Darknet horror story about a hacker discovering a cursed marketplace"
```

**Without activating the venv**, you can run:
```bash
.venv/bin/python pipeline.py "Your video idea here"
```

The final video will be at `output/video/video.mp4`.

## Requirements

- Python 3.11+
- **Node.js 18+** and npm (for [Remotion](https://www.remotion.dev/docs) scene clips — run `npm install` in `remotion_clips/`)
- FFmpeg installed and on `PATH`
- A Google Gemini API key (Gemini 2.5 Flash, Gemini TTS; images use **Gemini 2.5 Flash Image** by default, free tier). For paid Imagen 4 set `IMAGE_MODEL=imagen-4.0-fast-generate-001` in `.env`.

If Node is missing or `remotion_clips` is not installed, the pipeline skips Remotion and falls back to image + motion for scenes without Veo.

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `youtube_metadata.py` | Title, hook, description, tags, thumbnail prompt |
| 2 | `script_generator.py` | 750–1500 word script (~5–10 min when read aloud) |
| 3 | `scene_generator.py` | Break into 20–35 scenes for visual variety |
| 4 | `image_prompt_generator.py` | Refine visuals into cinematic Imagen prompts |
| 5 | `image_generator.py` | Generate images in parallel (one API key per worker) |
| 6 | `hero_video_generator.py` | Veo real video per hero scene (`HERO_SCENE_COUNT=0` = all scenes) |
| 7 | `voiceover_generator.py` | TTS per scene, keys used in round-robin |
| 8 | `remotion_renderer.py` | **Remotion:** render motion-graphic MP4 for scenes without Veo (see below) |
| 9 | `music_manager.py` | Select background music |
| 10 | `video_builder.py` | Assembly: **Veo → Remotion → image+motion**, ducking, transitions, color grade |
| 11 | `youtube_assets.py` | YouTube title, description, tags, thumbnail |

### Visual priority per scene

1. **Veo** MP4 when generation succeeded.  
2. **Remotion** MP4 (animated gradient, scanlines, optional embedded scene image).  
3. **Image + Ken Burns** / glitch (last resort).

This keeps the final video **video-first** instead of a slideshow when Veo quota fails.

### Remotion + Claude Code (optional authoring)

- **Install deps:** `cd remotion_clips && npm install`
- **Preview:** `npm run dev` — opens [Remotion Studio](https://www.remotion.dev/docs) in the browser.
- **Claude Code workflow:** see [Remotion + Claude Code guide (2026)](https://docs.google.com/document/d/1dSK3P6eZsm-63Jy2q3FAQq2MHYM6moSCNLOPEq_h2Ew/mobilebasic) — optional: `npx skills add remotion-dev/skills` inside `remotion_clips/` so Claude has Remotion Agent Skills when editing compositions.
- **Env:** `ENABLE_REMOTION=0` disables Remotion. **`REMOTION_EMBED_SCENE_IMAGE=1`** overlays the scene PNG with heavy parallax motion; default **`0`** is full motion graphics (matrix/grid/orbs) so clips are not “one photo + text” slideshows. After changing the Remotion composition, delete `output/remotion_clips/*.mp4` to force re-render. `REMOTION_SKIP_OVERLAY=1` skips code-rain/vignette on Remotion-backed scenes in the final MoviePy pass (see `.env.example`).
- **YouTube Shorts (edit in Studio):** Composition **`ShortsHighlightReel`** (1080×1920 @ 30fps) uses `@remotion/media` + `@remotion/captions` (Remotion skill patterns: trim segments, TikTok-style pages, bottom red captions). Download the source with `yt-dlp` (example: `yt-dlp -f "bv*+ba/b" --merge-output-format mp4 -o "remotion_clips/public/shorts/source.mp4" <youtube-url>`), set **`placeholder`** to **`false`** in the composition props, then edit **`segments`** (hook / middle / payoff from the long video) and **`captions`** (`Caption[]` with `startMs`/`endMs` on the **final cut** timeline). Render: `npx remotion render src/index.ts ShortsHighlightReel out/shorts.mp4`.

## Multiple API keys (faster, no rate limits)

Add up to 4 keys in `.env` to run image and voiceover generation in parallel without hitting per-key quotas:

```env
GEMINI_API_KEY=your_first_key
GEMINI_API_KEY_2=your_second_key
GEMINI_API_KEY_3=your_third_key
GEMINI_API_KEY_4=your_fourth_key
```

- **Text steps** (metadata, script, character bible, scenes, image prompts): on **429 / RESOURCE_EXHAUSTED**, the pipeline **rotates through all configured keys** with short pauses and many attempts (`utils/gemini_retry.py`), same idea as images/TTS.
- **Image generation**: uses one key per worker (up to 4 workers).
- **Voiceover**: scenes are assigned to keys in round-robin.
- With 4 keys, both steps run up to 4× faster and avoid 429 rate-limit errors.

## Hero scenes (Veo / Google Labs Flow)

With **`HERO_SCENE_COUNT=0`** (default in `config.py`), the pipeline tries **Veo** for every scene. Scene planning uses **only** `duration_seconds` values that are multiples of **`HERO_VEO_CLIP_SEC`** (see `scenes.json`). For each hero scene it requests **multiple distinct clips** when the planned scene duration is longer than **`HERO_VEO_CLIP_SEC`** (~8s default), then **concatenates** them — it does **not** loop the same MP4. If clips are still shorter than final narration, the last frame is **held** (no repeat). If Veo fails (quota, etc.), **Remotion** renders a full MP4 for that scene when Node + `remotion_clips` are set up; otherwise the scene uses **image + motion**. Set `HERO_SCENE_COUNT=3` to limit Veo to a subset of scenes (first/middle/last pattern). Veo models are cycled per `VEO_MODELS` in `config.py`. Cached files use names like `hero_scene_001_p00.mp4`, `hero_scene_001_p01.mp4`. **Retries:** Veo cycles all keys/models up to **`VEO_MAX_ROUNDS`** times with pauses (`VEO_RETRY_ROUND_WAIT_SEC`) between rounds; scene images use **`IMAGE_GEN_MAX_ATTEMPTS`** with pauses after each full key cycle. Tune in `config.py` / `.env`.

## Background music

Every video includes **voiceover and background music**. Music is ducked under narration so the voice stays clear.

- **No files in `assets/music/`**: The pipeline generates a soft dark-ambient pad and uses it so the video always has background music.
- **Add your own tracks**: Put `.mp3`, `.wav`, `.ogg`, or `.m4a` files in `assets/music/`. Name files with theme keywords (e.g. `dark_ambient.mp3`, `cyberpunk_suspense.wav`) so the pipeline can pick the best match from your video’s title, description, and scene content.
- **Volume**: Controlled by `config.py` (`BG_MUSIC_VOLUME`, `BG_MUSIC_DUCKED`). Music is reduced during narration.

## Project Structure

```
DarkForge AI/                  # Project root
├── config.py                  # All settings (models, paths, video params)
├── pipeline.py                # CLI entry point
├── requirements.txt
├── .env.example
├── remotion_clips/            # Remotion project (npm install; DarkForgeScene composition)
│   ├── src/
│   ├── public/df_render/      # Copied scene images at render time (gitignored)
│   └── package.json
├── modules/
│   ├── youtube_metadata.py
│   ├── script_generator.py
│   ├── character_bible.py        # Protagonist + style JSON for visual continuity
│   ├── scene_generator.py
│   ├── image_prompt_generator.py
│   ├── image_generator.py
│   ├── hero_video_generator.py   # Veo
│   ├── remotion_renderer.py      # CLI render per scene
│   ├── voiceover_generator.py
│   ├── subtitle_generator.py
│   └── video_builder.py
├── utils/
│   ├── logger.py
│   ├── file_utils.py
│   └── ffmpeg_utils.py
├── assets/
│   ├── music/                 # Drop background music here (.mp3/.wav)
│   └── fonts/                 # Drop a .ttf/.otf font for subtitles
├── output/
│   ├── characters.json           # Story bible (protagonist, supporting, visual_style)
│   ├── images/
│   ├── hero_videos/              # Cached Veo clips
│   ├── remotion_clips/           # Cached Remotion MP4s per scene
│   ├── audio/
│   └── video/
└── logs/
    └── pipeline.log
```

## Models Used

| Purpose | Model |
|---------|-------|
| Text generation | `gemini-2.5-flash` |
| Image generation | `imagen-4.0-fast-generate-001` |
| Text-to-speech | `gemini-2.5-flash-preview-tts` |

## Configuration

All tuneable parameters live in `config.py`:

- **Video**: resolution (1920x1080), FPS (30), codec (H.264)
- **Script**: word count range (1200–1500)
- **TTS**: voice (`Algenib` by default — gravelly; set `TTS_VOICE` in `.env` for others, e.g. `Charon` narrator, `Gacrux` mature), sample rate (24 kHz)
- **Edit pacing**: `SCENE_NARRATION_GAP_SEC` (silence + frozen last frame between scenes), `CROSSFADE_DURATION` (visual overlap). Hero→hero cuts use a **fade-from-black** transition.
- **Lyria BGM** (optional): set `LYRIA_BGM=1` to generate a **30s instrumental** bed from the story via Gemini **Lyria 3 Clip** (`lyria-3-clip-preview`); falls back to `assets/music/` or synthesized ambient if the API fails or quota blocks.
- **Generation**: retry count (3), thread workers (4), backoff multiplier
- **Ken Burns**: zoom range (1.0 → 1.15)
- **Music**: background volume (8%)

## Caching

Intermediate outputs (metadata, script, **characters.json**, scenes, image prompts) are saved as JSON/text in `output/`. Re-running the pipeline skips completed generation steps automatically.

**Character bible:** `output/characters.json` holds the male protagonist, any supporting characters, and shared **visual_style** strings used for Imagen stills and Veo hero clips (plus scene-to-scene continuity hints). If you edit `script.txt` and want the cast and visuals to match the new story, delete `characters.json` and downstream artifacts (`scenes.json`, `image_prompts.json`, images, hero clips) or run with `--fresh`.

**Scene durations / Veo:** In `scenes.json`, each `duration_seconds` is aligned to multiples of **`HERO_VEO_CLIP_SEC`** (default **8** seconds; override in `.env`) up to `MAX_SCENE_DURATION`, so planned hero-clip counts match Veo segment length. The scene planner also targets a total duration near the script’s spoken length (`WORDS_PER_MINUTE`). Final edit timing still follows actual TTS audio.

To regenerate from scratch, delete the `output/` directory.

## Optional Assets

- **Background music**: place any `.mp3` or `.wav` in `assets/music/` — it will loop and mix at low volume.
- **Custom font**: place a `.ttf` or `.otf` in `assets/fonts/` for subtitle rendering.

## Error Handling

- All API calls retry up to 3 times with exponential backoff.
- If image generation fails permanently, a black placeholder is used.
- Full logs are written to `logs/pipeline.log`.

## License

MIT
