# DarkForge AI — Automated YouTube Video Pipeline

Generate a complete YouTube video from a single text prompt using Google Gemini.

The pipeline handles script writing, scene planning, image generation, voiceover, subtitles, and final video assembly — all automatically. Videos are **5–10 minutes** long with **many scenes** and **rich visuals** (motion, overlays, vignette, transitions).

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
- FFmpeg installed and on `PATH`
- A Google Gemini API key (Gemini 2.5 Flash, Gemini TTS; images use **Gemini 2.5 Flash Image** by default, free tier). For paid Imagen 4 set `IMAGE_MODEL=imagen-4.0-fast-generate-001` in `.env`.

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `youtube_metadata.py` | Title, hook, description, tags, thumbnail prompt |
| 2 | `script_generator.py` | 750–1500 word script (~5–10 min when read aloud) |
| 3 | `scene_generator.py` | Break into 20–35 scenes for visual variety |
| 4 | `image_prompt_generator.py` | Refine visuals into cinematic Imagen prompts |
| 5 | `image_generator.py` | Generate images in parallel (one API key per worker) |
| 6 | `hero_video_generator.py` | **Hybrid:** Generate a few “hero” scenes with Veo (real video); rest use images + motion |
| 7 | `voiceover_generator.py` | TTS per scene, keys used in round-robin |
| 8 | `music_manager.py` | Select background music, then build video with ducking/transitions |
| 9 | `video_builder.py` | Cinematic assembly: hero Veo clips + image/motion, overlays, subtitles, color grading |

## Multiple API keys (faster, no rate limits)

Add up to 4 keys in `.env` to run image and voiceover generation in parallel without hitting per-key quotas:

```env
GEMINI_API_KEY=your_first_key
GEMINI_API_KEY_2=your_second_key
GEMINI_API_KEY_3=your_third_key
GEMINI_API_KEY_4=your_fourth_key
```

- **Image generation**: uses one key per worker (up to 4 workers).
- **Voiceover**: scenes are assigned to keys in round-robin.
- With 4 keys, both steps run up to 4× faster and avoid 429 rate-limit errors.

## Hero scenes (Veo / Google Labs Flow)

The pipeline uses a **hybrid** approach: a few “hero” scenes are generated as real short videos with **Veo** (Google’s video model); the rest use static images + motion effects. Hero scenes are chosen automatically (first, middle, last by default). Set `HERO_SCENE_COUNT` in `config.py` (default `3`). Veo requires supported/paid access; if Veo fails for a scene, that scene falls back to the generated image + motion. Optional: set `VEO_MODEL` in `.env` (e.g. `veo-2.0-generate-001` or `veo-3.1-generate-preview`).

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
├── modules/
│   ├── youtube_metadata.py
│   ├── script_generator.py
│   ├── scene_generator.py
│   ├── image_prompt_generator.py
│   ├── image_generator.py
│   ├── hero_video_generator.py   # Veo hero scenes (hybrid)
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
│   ├── images/
│   ├── hero_videos/              # Cached Veo clips for hero scenes
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
- **TTS**: voice (`Kore`), sample rate (24 kHz)
- **Generation**: retry count (3), thread workers (4), backoff multiplier
- **Ken Burns**: zoom range (1.0 → 1.15)
- **Music**: background volume (8%)

## Caching

Intermediate outputs (metadata, script, scenes, image prompts) are saved as JSON/text in `output/`. Re-running the pipeline skips completed generation steps automatically.

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
