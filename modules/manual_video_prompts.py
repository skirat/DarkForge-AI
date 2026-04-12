"""Build copy-paste manual video generation prompts (external Veo, Runway, etc.)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("pipeline")

CLIPS_SUBDIR = "clips"


def _character_entry(bible: dict[str, Any], char_id: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Return ('protagonist'|'supporting', char_dict) or (None, None)."""
    pro = bible.get("protagonist") or {}
    if char_id == pro.get("id") or char_id == "protagonist":
        return "protagonist", pro
    for c in bible.get("supporting_characters") or []:
        if c.get("id") == char_id:
            return "supporting", c
    return None, None


def _global_style_plain(bible: dict[str, Any]) -> str:
    """Single paragraph for video models (no markdown)."""
    vs = bible.get("visual_style") or {}
    bits: list[str] = []
    for k in ("palette", "lighting", "film_look", "motifs"):
        v = vs.get(k)
        if v:
            bits.append(str(v).strip())
    cr = bible.get("continuity_rules")
    if cr:
        bits.append(f"Continuity: {str(cr).strip()}")
    return " ".join(bits) if bits else ""


def _format_character_plain(char: dict[str, Any]) -> str:
    """One prose sentence for copy-paste (name, role, locked look)."""
    name = (char.get("name") or "Character").strip()
    role = (char.get("role") or "").strip()
    frag = (char.get("image_prompt_fragment") or "").strip()
    phys = char.get("physical")
    phys_bits: list[str] = []
    if isinstance(phys, dict):
        for key in ("age_band", "build", "hair", "skin", "distinctive", "default_outfit"):
            v = phys.get(key)
            if v:
                phys_bits.append(str(v))
    traits = char.get("personality_traits")
    trait_s = ""
    if isinstance(traits, list) and traits:
        trait_s = " Personality: " + ", ".join(str(t) for t in traits[:8]) + "."
    role_s = f" {role}" if role else ""
    phys_s = f" Details: {', '.join(phys_bits)}." if phys_bits else ""
    frag_s = f" {frag}" if frag else ""
    return f"{name}{role_s}.{frag_s}{phys_s}{trait_s}".strip()


def _characters_plain_lines(bible: dict[str, Any], present: list[str] | None) -> list[str]:
    ids = present or []
    lines: list[str] = []
    for cid in ids:
        label, char = _character_entry(bible, cid)
        if char:
            lines.append(_format_character_plain(char))
        else:
            lines.append(
                f"Character id {cid}: keep consistent with earlier clips in this series."
            )
    return lines


def _clip_header_line(
    scene_id: int,
    source_scene_id: int,
    sub_i: int,
    sub_n: int,
    duration: float,
    title: str,
) -> str:
    part = f"part {sub_i + 1} of {sub_n}" if sub_n > 1 else "single segment"
    t = title or "Untitled"
    return (
        f"Clip {scene_id} | planner scene {source_scene_id} | {part} | ~{duration:g}s | 16:9 | {t}"
    )


def build_copy_paste_video_prompt(
    *,
    scene_id: int,
    source_scene_id: int,
    sub_i: int,
    sub_n: int,
    duration: float,
    title: str,
    global_style: str,
    narration: str,
    character_lines: list[str],
    video_shot: str,
    still_image: str,
) -> str:
    """One plain-text block to paste into a video generator (no markdown)."""
    header = _clip_header_line(scene_id, source_scene_id, sub_i, sub_n, duration, title)
    blocks: list[str] = [header, ""]

    if global_style:
        blocks.append("Series visual style:")
        blocks.append(global_style)
        blocks.append("")

    if character_lines:
        blocks.append("Characters (keep likeness and wardrobe identical across clips):")
        for line in character_lines:
            blocks.append(f"- {line}")
        blocks.append("")

    blocks.append("Shot — motion, camera, action, environment:")
    blocks.append(video_shot or "(no shot text; infer from narration and style).")
    blocks.append("")

    if narration:
        blocks.append("Voiceover / mood (optional; do not burn in as subtitles unless stylistic):")
        blocks.append(narration)
        blocks.append("")

    if still_image:
        blocks.append("Optional keyframe alignment (match mood and composition if helpful):")
        blocks.append(still_image)
        blocks.append("")

    blocks.append("Output: cinematic 16:9 video, consistent with the style and characters above.")

    return "\n".join(blocks).strip()


def build_manual_video_prompts_payload(
    scenes: list[dict[str, Any]],
    character_bible: dict[str, Any],
    metadata: dict[str, Any],
    image_prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Structured document: one entry per scene row / Veo sub-segment."""
    image_by_scene: dict[int, str] = {}
    for row in image_prompts:
        sid = row.get("scene_id")
        if sid is not None:
            image_by_scene[int(sid)] = (row.get("image_prompt") or "").strip()

    global_style_plain = _global_style_plain(character_bible)
    title = (metadata.get("title") or "").strip()
    hook = (metadata.get("hook") or "").strip()
    pro_id = (character_bible.get("protagonist") or {}).get("id") or "protagonist"

    clips: list[dict[str, Any]] = []
    for scene in scenes:
        sid = int(scene["scene_id"])
        src = int(scene.get("source_scene_id", sid))
        sub_i = int(scene.get("sub_part_index", 0))
        sub_n = int(scene.get("sub_parts_total", 1))
        duration = float(scene.get("duration_seconds", 8))
        narration = (scene.get("narration") or "").strip()
        present = scene.get("characters_present")
        if present is None:
            present = []
        elif isinstance(present, str):
            present = [present]
        else:
            present = list(present)

        vp = (scene.get("video_prompt") or scene.get("visual_prompt") or "").strip()
        vis = (scene.get("visual_prompt") or "").strip()
        still = image_by_scene.get(sid, "")

        pro_in = pro_id in present or "protagonist" in present
        char_lines = _characters_plain_lines(character_bible, present)

        copy_paste = build_copy_paste_video_prompt(
            scene_id=sid,
            source_scene_id=src,
            sub_i=sub_i,
            sub_n=sub_n,
            duration=duration,
            title=title,
            global_style=global_style_plain,
            narration=narration,
            character_lines=char_lines,
            video_shot=vp or vis,
            still_image=still,
        )

        clips.append(
            {
                "scene_id": sid,
                "source_scene_id": src,
                "sub_part_index": sub_i,
                "sub_parts_total": sub_n,
                "duration_seconds": duration,
                "narration": narration,
                "characters_present": present,
                "protagonist_in_scene": bool(pro_in),
                "visual_prompt": vis,
                "video_prompt": vp,
                "still_image_prompt": still,
                "copy_paste_video_prompt": copy_paste,
                "clip_filename": f"clip_{sid:03d}.txt",
            }
        )

    return {
        "schema_version": 2,
        "title": title,
        "hook": hook,
        "global_style_plain": global_style_plain,
        "clips_dir": CLIPS_SUBDIR,
        "clips": clips,
    }


def write_manual_video_prompts(
    output_dir: Path,
    scenes: list[dict[str, Any]],
    character_bible: dict[str, Any],
    metadata: dict[str, Any],
    image_prompts: list[dict[str, Any]],
    log: logging.Logger | None = None,
) -> tuple[Path, Path, Path]:  # json, index txt, clips directory
    """Write JSON, per-clip text files, and a short root index."""
    log = log or logger
    payload = build_manual_video_prompts_payload(
        scenes, character_bible, metadata, image_prompts
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = output_dir / "manual_video_prompts"
    clips_dir = bundle_dir / CLIPS_SUBDIR
    clips_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "manual_video_prompts.json"
    index_path = output_dir / "manual_video_prompts.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in payload.get("clips") or []:
        name = c.get("clip_filename") or f"clip_{int(c['scene_id']):03d}.txt"
        text = (c.get("copy_paste_video_prompt") or "").strip()
        (clips_dir / name).write_text(text + "\n", encoding="utf-8")

    lines: list[str] = [
        "Manual video prompts — one self-contained file per clip",
        "=" * 60,
        "",
        f"Title: {payload.get('title') or '(untitled)'}",
        f"Clips: {len(payload.get('clips') or [])}",
        "",
        "Each file below is a SINGLE copy-paste unit. Open the file, select all, paste into your video model.",
        f"Folder: manual_video_prompts/{CLIPS_SUBDIR}/",
        "",
    ]
    for c in payload.get("clips") or []:
        fn = c.get("clip_filename") or f"clip_{int(c['scene_id']):03d}.txt"
        sid = c.get("scene_id")
        src = c.get("source_scene_id")
        sub_i = int(c.get("sub_part_index", 0))
        sub_n = int(c.get("sub_parts_total", 1))
        dur = c.get("duration_seconds")
        dur_s = f"~{float(dur):g}s" if dur is not None else "~8s"
        if sub_n > 1:
            seg = f"segment {sub_i + 1}/{sub_n} of planner scene {src}"
        else:
            seg = f"planner scene {src} (single segment)"
        lines.append(f"  {fn}  |  clip {sid}  |  {seg}  |  {dur_s}")
    lines.append("")
    lines.append("JSON (all prompts in one file): manual_video_prompts.json")
    lines.append("Field per clip: copy_paste_video_prompt")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n = len(payload.get("clips") or [])
    rel_clips = clips_dir.relative_to(output_dir)
    log.info(
        "Manual video prompts → %s, %d clip file(s) in %s, index %s",
        json_path.name,
        n,
        rel_clips,
        index_path.name,
    )
    return json_path, index_path, clips_dir

