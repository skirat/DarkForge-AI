import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API (multi-key for parallel rate-limit-free usage) ---
def _load_api_keys() -> list[str]:
    keys: list[str] = []
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    if primary and primary.lower() not in ("", "your_key_here"):
        keys.append(primary)
    for i in range(2, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k.lower() not in ("", "your_key_here"):
            keys.append(k)
    return keys


GEMINI_API_KEYS = _load_api_keys()
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Optional: NanoBanana image API as fallback when Gemini/Imagen image gen fails.
NANOBANANA_API_KEY = (os.getenv("NANOBANANA_API_KEY") or "").strip()

# --- Models ---
TEXT_MODEL = "gemini-2.5-flash"
# Default: Gemini native image (free tier). Use imagen-4.0-fast-generate-001 for paid Imagen.
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image").strip() or "gemini-2.5-flash-image"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
# Veo for hero scenes (hybrid mode). Requires paid / supported access.
# Try these models in order across all API keys when one exhausts quota.
VEO_MODEL = os.getenv("VEO_MODEL", "veo-2.0-generate-001").strip() or "veo-2.0-generate-001"
VEO_MODELS = [
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
]

# --- TTS (Gemini prebuilt voices: https://ai.google.dev/gemini-api/docs/speech-generation#voices) ---
# Default Algenib = gravelly male (strong for horror). Override: Charon (narrator), Gacrux (mature), Orus (firm).
TTS_VOICE = (os.getenv("TTS_VOICE", "Algenib") or "Algenib").strip()
TTS_SAMPLE_RATE = 24000
TTS_SAMPLE_WIDTH = 2
TTS_CHANNELS = 1

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEO_DIR = OUTPUT_DIR / "video"
HERO_VIDEOS_DIR = OUTPUT_DIR / "hero_videos"
REMOTION_RENDER_DIR = OUTPUT_DIR / "remotion_clips"
REMOTION_CLIPS_DIR = BASE_DIR / "remotion_clips"
ASSETS_DIR = BASE_DIR / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
FONTS_DIR = ASSETS_DIR / "fonts"
LOGS_DIR = BASE_DIR / "logs"

# --- Video ---
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"

# --- Generation (5–10 min video at ~150 wpm) ---
SCRIPT_WORD_COUNT_MIN = 750   # ~5 min
SCRIPT_WORD_COUNT_MAX = 1500  # ~10 min
WORDS_PER_MINUTE = 150
# Scene count target for visual variety (more scenes = more cuts and graphics)
SCENE_COUNT_MIN = 20
SCENE_COUNT_MAX = 35
SCENE_DURATION_DEFAULT = 15  # seconds per scene (will vary by narration length)
# Veo-first: try to generate a hero video for each scene. 0 = all scenes; 3 = only 3 hero scenes.
HERO_SCENE_COUNT = max(0, int(os.getenv("HERO_SCENE_COUNT", "0") or 0))
VEO_POLL_INTERVAL_SEC = 15
VEO_POLL_TIMEOUT_SEC = 600

# --- Remotion (video clips when Veo unavailable; requires Node + npm install in remotion_clips/) ---
ENABLE_REMOTION = os.getenv("ENABLE_REMOTION", "1").strip().lower() in ("1", "true", "yes")
# Off by default: motion-graphics-only reads as video; enable to blend animated scene PNG.
REMOTION_EMBED_SCENE_IMAGE = os.getenv("REMOTION_EMBED_SCENE_IMAGE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
REMOTION_COMPOSITION_ID = (os.getenv("REMOTION_COMPOSITION_ID", "DarkForgeScene").strip() or "DarkForgeScene")
REMOTION_RENDER_TIMEOUT_SEC = max(60, int(os.getenv("REMOTION_RENDER_TIMEOUT_SEC", "900") or 900))
# Skip code-rain / vignette on Remotion-backed scenes (they already have motion graphics)
REMOTION_SKIP_OVERLAY = os.getenv("REMOTION_SKIP_OVERLAY", "0").strip().lower() in ("1", "true", "yes")

# --- Retry / Parallelism ---
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
RETRY_RATE_LIMIT_WAIT = 60.0
# Use one worker per API key when multiple keys are set (max 4)
IMAGE_WORKERS = min(4, len(GEMINI_API_KEYS)) if GEMINI_API_KEYS else 4

# --- Style ---
IMAGE_STYLE_SUFFIX = (
    "dark cyberpunk, neon lighting, hacker aesthetic, "
    "cinematic composition, 4k, dark web environment, "
    "dramatic shadows, digital art"
)

# --- Ken Burns ---
ZOOM_START = 1.0
ZOOM_END = 1.15

# --- Pan Effect ---
PAN_OVERSCAN = 1.25

# --- Transitions ---
CROSSFADE_DURATION = 0.5
TRANSITION_TYPES = ["crossfade", "glitch_cut", "fade_black"]
GLITCH_CUT_DURATION = 0.3

# --- Background Music ---
BG_MUSIC_VOLUME = 0.08
BG_MUSIC_DUCKED = 0.04
DUCK_RAMP_SECONDS = 0.3

# --- Scene Splitting ---
MAX_SCENE_DURATION = 30.0

# --- Effects (rich visuals: overlays, vignette, glitch) ---
GLITCH_INTERVAL = 2.0
GLITCH_BURST_FRAMES = 4
OVERLAY_OPACITY = 0.25
# How often to add a thematic overlay when no keyword match (0–1)
OVERLAY_DEFAULT_CHANCE = 0.7
# Vignette darkening at edges (0 = off, 0.5 = strong)
VIGNETTE_STRENGTH = 0.4

# --- Color Grading (FFmpeg) ---
COLOR_GRADE_FILTER = (
    "eq=contrast=1.15:brightness=-0.03:saturation=0.85,"
    "colorbalance=bs=0.08:ms=0.05"
)

# --- SFX ---
SFX_DIR = ASSETS_DIR / "sfx"
SFX_VOLUME = 0.15
