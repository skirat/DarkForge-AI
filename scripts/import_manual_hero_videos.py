#!/usr/bin/env python3
"""Copy manually generated hero clips into output/hero_videos with pipeline naming.

Expected workflow filenames (see modules/hero_video_generator._hero_dest_path):
  hero_scene_{scene_id:03d}_p{part_index:02d}.mp4

Source files are resolved per scene row from output/scenes.json. Clip id matches scene_id.
When a scene needs multiple Veo parts (duration > HERO_VEO_CLIP_SEC), use suffixed names.

Examples (scene_id=7, one part):
  clip7.mp4 (common export), 7.mp4, 007.mp4, clip_007.mp4, clip_7.mp4, scene_007.mp4, hero_scene_007_p00.mp4

Examples (scene_id=7, two parts):
  7_p00.mp4 / 7_p01.mp4, clip_007_p00.mp4, hero_scene_007_p00.mp4

Usage (from repo root):
  .venv/bin/python scripts/import_manual_hero_videos.py --source ./hackathon-vid-gen-final
  .venv/bin/python scripts/import_manual_hero_videos.py --source ./red-room-vid-gen-final
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import HERO_VIDEOS_DIR, OUTPUT_DIR  # noqa: E402
from modules.hero_video_generator import (  # noqa: E402
    _hero_dest_path,
    _parts_for_scene_duration,
)


VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")


def _scene_duration_sec(scene: dict) -> float:
    raw = scene.get("duration_seconds")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 8.0


def _candidate_source_names(scene_id: int, part_index: int, total_parts: int) -> list[str]:
    sid = int(scene_id)
    out: list[str] = []
    if total_parts <= 1:
        out.extend(
            [
                f"clip{sid}",
                f"clip{sid:03d}",
                f"{sid}",
                f"{sid:03d}",
                f"clip_{sid}",
                f"clip_{sid:03d}",
                f"scene_{sid:03d}",
                f"scene_{sid}",
                f"hero_scene_{sid:03d}_p{part_index:02d}",
            ]
        )
    else:
        out.extend(
            [
                f"clip{sid}_p{part_index:02d}",
                f"clip{sid:03d}_p{part_index:02d}",
                f"{sid}_p{part_index:02d}",
                f"{sid:03d}_p{part_index:02d}",
                f"clip_{sid}_p{part_index:02d}",
                f"clip_{sid:03d}_p{part_index:02d}",
                f"hero_scene_{sid:03d}_p{part_index:02d}",
            ]
        )
    return out


def _find_source_file(src_dir: Path, scene_id: int, part_index: int, total_parts: int) -> Path | None:
    for base in _candidate_source_names(scene_id, part_index, total_parts):
        for ext in VIDEO_EXTS:
            p = src_dir / f"{base}{ext}"
            if p.is_file():
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "hackathon-vid-gen-final",
        help="Folder containing manual clips (default: ./hackathon-vid-gen-final)",
    )
    ap.add_argument(
        "--scenes",
        type=Path,
        default=OUTPUT_DIR / "scenes.json",
        help="Path to scenes.json (default: output/scenes.json)",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=HERO_VIDEOS_DIR,
        help="Hero videos directory (default: output/hero_videos)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies only",
    )
    args = ap.parse_args()

    src_dir = args.source.resolve()
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        return 1

    scenes_path = args.scenes.resolve()
    if not scenes_path.is_file():
        print(f"scenes.json not found: {scenes_path}", file=sys.stderr)
        return 1

    scenes: list[dict] = json.loads(scenes_path.read_text(encoding="utf-8"))
    dest_dir = args.dest.resolve()
    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        try:
            scene_id = int(scene["scene_id"])
        except (KeyError, TypeError, ValueError):
            continue
        n_parts = _parts_for_scene_duration(_scene_duration_sec(scene))
        for part in range(n_parts):
            dest = _hero_dest_path(dest_dir, scene_id, part)
            src = _find_source_file(src_dir, scene_id, part, n_parts)
            if src is None:
                missing.append(f"scene_id={scene_id} part={part}/{n_parts} → {dest.name}")
                continue
            if args.dry_run:
                print(f"COPY {src.name} → {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"OK {dest.name} ← {src.name}")
            copied += 1

    print(
        f"Done: {copied} file(s) {'would be ' if args.dry_run else ''}written to {dest_dir}",
        file=sys.stderr,
    )
    if missing:
        print("Missing sources:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return 2 if copied == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
