from __future__ import annotations

from typing import Any

from .frame_sampler import VideoSample
from .openai_client import OpenAIConfig, OpenAIVisionClient
from .prompts import SELF_JUDGE_PROMPT, STYLE_CAPTION_PROMPT, VIDEO_FACTS_PROMPT
from .schema import VIDEO_FACTS_SCHEMA, candidate_schema, final_caption_schema
from .utils import (
    STYLE_DESCRIPTIONS,
    clean_caption,
    compact_json,
    ensure_caption_map,
    env_bool,
    env_int,
    extract_json,
    fallback_candidates,
    log,
)


class CaptionAgent:
    def __init__(self, openai_client: OpenAIVisionClient | None) -> None:
        self.openai_client = openai_client
        self.candidate_count = env_int("CAPTION_CANDIDATES", 2, minimum=1, maximum=4)
        self.self_check = env_bool("SELF_CHECK", True)

    @classmethod
    def from_env(cls) -> "CaptionAgent":
        config = OpenAIConfig.from_env()
        if config is None:
            log("OpenAI provider not configured; using safe fallback captions")
            return cls(openai_client=None)

        log(f"using OpenAI Responses API with model: {config.model}")
        return cls(openai_client=OpenAIVisionClient(config))

    def caption_video(self, sample: VideoSample, styles: list[str]) -> dict[str, str]:
        try:
            facts = self.analyze_video(sample)
        except Exception as exc:
            log(f"OpenAI visual fact extraction failed; using local facts: {exc}")
            facts = local_video_facts(sample, reason=str(exc))

        try:
            candidates = self.generate_candidates(facts, styles)
        except Exception as exc:
            log(f"OpenAI candidate generation failed; using fallback candidates: {exc}")
            candidates = {
                style: fallback_candidates(style, str(facts.get("caption_seed", "")), self.candidate_count)
                for style in styles
            }

        try:
            captions = self.self_judge(facts, candidates, styles) if self.self_check else _first_candidates(candidates, styles)
        except Exception as exc:
            log(f"OpenAI self-check failed; selecting first candidates: {exc}")
            captions = _first_candidates(candidates, styles)

        return ensure_caption_map(captions, styles)

    def analyze_video(self, sample: VideoSample) -> dict[str, Any]:
        if self.openai_client is None:
            return local_video_facts(sample, reason="OPENAI_API_KEY was not configured")

        prompt = (
            VIDEO_FACTS_PROMPT.strip()
            + "\n\nFrame/video metadata for orientation only:\n"
            + compact_json(sample.metadata())
        )
        response = self.openai_client.generate_text(
            prompt=prompt,
            frame_paths=sample.frame_paths,
            max_output_tokens=2400,
            json_schema=VIDEO_FACTS_SCHEMA,
            schema_name="video_facts",
        )
        return _normalize_facts(extract_json(response), sample)

    def generate_candidates(self, facts: dict[str, Any], styles: list[str]) -> dict[str, list[str]]:
        if self.openai_client is None:
            seed = str(facts.get("caption_seed") or "")
            return {style: fallback_candidates(style, seed, self.candidate_count) for style in styles}

        prompt = (
            STYLE_CAPTION_PROMPT.strip()
            + "\n\nRequested styles:\n"
            + compact_json(styles)
            + "\n\nCandidate count per style:\n"
            + str(self.candidate_count)
            + "\n\nStyle guide:\n"
            + compact_json({style: STYLE_DESCRIPTIONS.get(style, "match the requested style") for style in styles})
            + "\n\nVideo facts:\n"
            + compact_json(facts)
        )
        response = self.openai_client.generate_text(
            prompt=prompt,
            max_output_tokens=3000,
            json_schema=candidate_schema(styles),
            schema_name="caption_candidates",
        )
        return _normalize_candidates(extract_json(response), styles, facts, self.candidate_count)

    def self_judge(
        self,
        facts: dict[str, Any],
        candidates: dict[str, list[str]],
        styles: list[str],
    ) -> dict[str, str]:
        if self.openai_client is None:
            return _first_candidates(candidates, styles)

        prompt = (
            SELF_JUDGE_PROMPT.strip()
            + "\n\nRequested styles:\n"
            + compact_json(styles)
            + "\n\nVideo facts:\n"
            + compact_json(facts)
            + "\n\nCandidate captions:\n"
            + compact_json(candidates)
        )
        response = self.openai_client.generate_text(
            prompt=prompt,
            max_output_tokens=2400,
            json_schema=final_caption_schema(styles),
            schema_name="final_captions",
        )
        parsed = extract_json(response)
        if isinstance(parsed, dict) and isinstance(parsed.get("captions"), dict):
            parsed = parsed["captions"]
        if not isinstance(parsed, dict):
            raise ValueError("self-check response was not a caption map")
        return {style: clean_caption(parsed.get(style), style) for style in styles}


def local_video_facts(sample: VideoSample, reason: str) -> dict[str, Any]:
    metadata = sample.metadata()
    resolution = ""
    if metadata.get("source_width") and metadata.get("source_height"):
        resolution = f" at {metadata['source_width']}x{metadata['source_height']}"

    details = [
        f"{metadata.get('sampled_frame_count', 0)} frames sampled",
        f"dominant colors include {_dominant_colors_from_metadata(metadata)}",
    ]
    if metadata.get("duration_sec"):
        details.append(f"duration is about {float(metadata['duration_sec']):.1f} seconds")
    if metadata.get("motion_score") is not None:
        details.append(f"estimated frame-to-frame motion score is {metadata['motion_score']}")

    return {
        "main_subjects": ["unknown visible subjects"],
        "visible_actions": ["visible activity across sampled frames"],
        "setting": "unknown",
        "mood": "neutral",
        "important_visual_details": details,
        "uncertain_details": [
            reason,
            "semantic content cannot be identified without a working OpenAI vision call",
        ],
        "caption_seed": f"A short video{resolution} shows visible activity, though exact subjects remain uncertain.",
        "metadata": metadata,
    }


def _normalize_facts(parsed: Any, sample: VideoSample) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("facts response was not a JSON object")

    def listify(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    facts = {
        "main_subjects": listify(parsed.get("main_subjects")),
        "visible_actions": listify(parsed.get("visible_actions") or parsed.get("actions")),
        "setting": str(parsed.get("setting") or "unknown").strip(),
        "mood": str(parsed.get("mood") or "neutral").strip(),
        "important_visual_details": listify(parsed.get("important_visual_details")),
        "uncertain_details": listify(parsed.get("uncertain_details") or parsed.get("uncertain")),
        "caption_seed": str(parsed.get("caption_seed") or "").strip(),
        "metadata": sample.metadata(),
    }
    if not facts["caption_seed"]:
        facts["caption_seed"] = "A short video shows visible activity in the sampled frames."
    return facts


def _normalize_candidates(
    parsed: Any,
    styles: list[str],
    facts: dict[str, Any],
    candidate_count: int,
) -> dict[str, list[str]]:
    if not isinstance(parsed, dict):
        raise ValueError("candidate response was not a JSON object")

    candidates: dict[str, list[str]] = {}
    seed = str(facts.get("caption_seed") or "")
    for style in styles:
        raw = parsed.get(style)
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, list):
            values = [item for item in raw if isinstance(item, str)]
        else:
            values = []

        cleaned = [clean_caption(value, style) for value in values if value.strip()]
        if len(cleaned) < candidate_count:
            cleaned.extend(fallback_candidates(style, seed, candidate_count))
        candidates[style] = cleaned[:candidate_count]

    return candidates


def _first_candidates(candidates: dict[str, list[str]], styles: list[str]) -> dict[str, str]:
    return {style: clean_caption((candidates.get(style) or fallback_candidates(style))[0], style) for style in styles}


def _dominant_colors_from_metadata(metadata: dict[str, Any]) -> str:
    frames = metadata.get("frames") or []
    colors: list[str] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        color = str(frame.get("dominant_color") or "").strip()
        if color and color not in colors:
            colors.append(color)
    return ", ".join(colors[:4]) or "unknown"
