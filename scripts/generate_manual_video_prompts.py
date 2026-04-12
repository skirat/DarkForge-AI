#!/usr/bin/env python3
"""Regenerate output/manual_video_prompts.{json,txt} from existing pipeline output (no API calls).

Usage (from repo root):
  .venv/bin/python scripts/generate_manual_video_prompts.py
  .venv/bin/python scripts/generate_manual_video_prompts.py --output-dir ./output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.manual_video_prompts import write_manual_video_prompts  # noqa: E402
from modules.protagonist_sparsity import apply_protagonist_sparsity  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory containing scenes.json, characters.json, metadata.json, image_prompts.json",
    )
    args = p.parse_args()
    out: Path = args.output_dir.resolve()

    required = [
        out / "scenes.json",
        out / "characters.json",
        out / "metadata.json",
        out / "image_prompts.json",
    ]
    missing = [f for f in required if not f.is_file()]
    if missing:
        print("Missing required files:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        return 1

    scenes = json.loads(required[0].read_text(encoding="utf-8"))
    character_bible = json.loads(required[1].read_text(encoding="utf-8"))
    metadata = json.loads(required[2].read_text(encoding="utf-8"))
    image_prompts = json.loads(required[3].read_text(encoding="utf-8"))

    script_text = ""
    stp = out / "script.txt"
    if stp.is_file():
        script_text = stp.read_text(encoding="utf-8")
    scenes = apply_protagonist_sparsity(
        scenes, character_bible, metadata, script=script_text
    )

    log = logging.getLogger("manual_video_prompts_cli")
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    json_path, index_path, clips_dir = write_manual_video_prompts(
        out, scenes, character_bible, metadata, image_prompts, log
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {index_path}")
    print(f"Per-clip prompts in {clips_dir}/ (clip_001.txt, ...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
