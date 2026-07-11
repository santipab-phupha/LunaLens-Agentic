from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


STYLE_DESCRIPTIONS: dict[str, str] = {
    "formal": "professional, objective, factual",
    "sarcastic": "dry, ironic, lightly mocking tone",
    "humorous_tech": "funny with technology or programming references",
    "humorous_non_tech": "funny everyday humor without technical jargon",
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / ".local.env",
        repo_root / ".local.env",
        repo_root.parent / ".local.env",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            load_dotenv(dotenv_path=resolved, override=False)
            log(f"loaded local environment file: {resolved}")
            return


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def safe_slug(value: str, fallback: str = "task") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80].strip("._")
    return cleaned or fallback


def write_json_atomic(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def extract_json(text: str) -> Any:
    if not text:
        raise ValueError("empty model response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end > object_start:
        return json.loads(cleaned[object_start : object_end + 1])

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end > array_start:
        return json.loads(cleaned[array_start : array_end + 1])

    raise ValueError("could not find JSON in model response")


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def clean_caption(text: Any, style: str = "formal") -> str:
    if not isinstance(text, str):
        text = ""

    caption = text.strip()
    caption = caption.replace("\n", " ")
    caption = re.sub(r"^[\s>*_-]+", "", caption)
    caption = re.sub(r"^['\"]|['\"]$", "", caption)
    caption = re.sub(rf"^(?:{re.escape(style)}|caption)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\s+", " ", caption).strip()

    if not caption:
        caption = fallback_caption(style)

    match = re.search(r"^(.+?[.!?])(?:\s+|$)", caption)
    if match:
        caption = match.group(1).strip()

    parts = caption.split()
    if len(parts) < 8:
        return fallback_caption(style)
    if len(parts) > 22:
        caption = " ".join(parts[:22]).rstrip(" ,;:") + "."

    caption = caption.rstrip(" ,;:")
    if not caption.endswith((".", "!", "?")):
        caption += "."
    return caption


def fallback_caption(style: str) -> str:
    if style == "sarcastic":
        return "The clip reveals a visible scene, bravely keeping the finer details mysterious."
    if style == "humorous_tech":
        return "The video renders a visual routine, but several scene variables remain undefined."
    if style == "humorous_non_tech":
        return "The clip shows a scene doing its best without bringing an instruction card."
    return "A short video shows visible activity, though exact subjects remain uncertain."


def fallback_candidates(style: str, caption_seed: str | None = None, count: int = 2) -> list[str]:
    seed = clean_caption(caption_seed or fallback_caption("formal"), "formal")
    templates = {
        "formal": [
            seed,
            "The video presents a brief visible scene with some details remaining uncertain.",
        ],
        "sarcastic": [
            "The scene makes its point visually, because apparently details are optional today.",
            "The clip confidently shows activity while refusing to overexplain itself.",
        ],
        "humorous_tech": [
            "The clip executes a visual routine, though several variables remain undefined.",
            "The video compiles into visible action, but the object names are still pending.",
        ],
        "humorous_non_tech": [
            "The clip shows a small scene that forgot to bring its instruction card.",
            "A brief scene unfolds, doing its best without a narrator clearing things up.",
        ],
    }
    values = templates.get(style, [fallback_caption(style), seed])
    while len(values) < count:
        values.append(fallback_caption(style))
    return values[:count]


def ensure_caption_map(captions: dict[str, Any] | None, styles: list[str]) -> dict[str, str]:
    captions = captions or {}
    safe: dict[str, str] = {}
    for style in styles:
        safe[style] = clean_caption(captions.get(style) or fallback_caption(style), style)
    return safe
