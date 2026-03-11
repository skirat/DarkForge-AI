from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


def parse_json_response(text: str) -> Any:
    """Parse JSON from a model response. Strips markdown code fences if present.
    Raises json.JSONDecodeError if invalid."""
    text = text.strip()
    # Remove optional ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def cache_json(path: Path, generator_fn: Callable[[], Any]) -> Any:
    """Return cached JSON if *path* exists, otherwise call *generator_fn*, save, and return."""
    if path.exists():
        return load_json(path)
    data = generator_fn()
    save_json(path, data)
    return data


def cache_text(path: Path, generator_fn: Callable[[], str]) -> str:
    if path.exists():
        return load_text(path)
    text = generator_fn()
    save_text(path, text)
    return text
