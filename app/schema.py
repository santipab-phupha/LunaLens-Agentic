from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

VIDEO_FACTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "main_subjects": {"type": "array", "items": {"type": "string"}},
        "visible_actions": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "mood": {"type": "string"},
        "important_visual_details": {"type": "array", "items": {"type": "string"}},
        "uncertain_details": {"type": "array", "items": {"type": "string"}},
        "caption_seed": {"type": "string"},
    },
    "required": [
        "main_subjects",
        "visible_actions",
        "setting",
        "mood",
        "important_visual_details",
        "uncertain_details",
        "caption_seed",
    ],
}


def candidate_schema(styles: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            style: {"type": "array", "items": {"type": "string"}}
            for style in styles
        },
        "required": styles,
    }


def final_caption_schema(styles: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "captions": {
                "type": "object",
                "additionalProperties": False,
                "properties": {style: {"type": "string"} for style in styles},
                "required": styles,
            }
        },
        "required": ["captions"],
    }


@dataclass(frozen=True)
class Task:
    task_id: str
    video_url: str
    styles: list[str]


def load_tasks(path: Path) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array")

    tasks: list[Task] = []
    for index, raw_task in enumerate(data):
        if not isinstance(raw_task, dict):
            continue

        task_id = str(raw_task.get("task_id") or f"task_{index + 1}")
        video_url = str(raw_task.get("video_url") or "").strip()
        if not video_url:
            continue

        tasks.append(
            Task(
                task_id=task_id,
                video_url=video_url,
                styles=_normalize_styles(raw_task.get("styles")),
            )
        )

    return tasks


def _normalize_styles(raw: Any) -> list[str]:
    if raw is None:
        return list(SUPPORTED_STYLES)
    if not isinstance(raw, list):
        return list(SUPPORTED_STYLES)

    styles: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        style = item.strip()
        if not style or style in seen:
            continue
        styles.append(style)
        seen.add(style)

    return styles or list(SUPPORTED_STYLES)
