from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import image_to_data_url


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    fallback_model: str | None
    reasoning_effort: str
    image_detail: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig | None":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return None
        model = (os.getenv("OPENAI_MODEL", "gpt-5.6-luna") or "gpt-5.6-luna").strip()
        fallback = (os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5.4-mini") or "").strip()
        effort = (os.getenv("OPENAI_REASONING_EFFORT", "high") or "high").strip().lower()
        detail = (os.getenv("OPENAI_IMAGE_DETAIL", "low") or "low").strip().lower()
        if effort not in {"none", "low", "medium", "high", "xhigh"}:
            effort = "high"
        if detail not in {"low", "high", "auto"}:
            detail = "low"
        return cls(
            api_key=api_key,
            model=model,
            fallback_model=fallback if fallback and fallback != model else None,
            reasoning_effort=effort,
            image_detail=detail,
        )


class OpenAIVisionClient:
    def __init__(self, config: OpenAIConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=config.api_key, timeout=90.0, max_retries=2)

    @property
    def model(self) -> str:
        return self.config.model

    def generate_text(
        self,
        prompt: str,
        frame_paths: list[Path] | None = None,
        max_output_tokens: int = 1600,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "caption_response",
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame_path in frame_paths or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(frame_path),
                    "detail": self.config.image_detail,
                }
            )

        request: dict[str, Any] = {
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if json_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                }
            }

        response = self._create_response(request)
        text = getattr(response, "output_text", None)
        if not text:
            raise ValueError("OpenAI response did not include output_text")
        return str(text)

    def _create_response(self, request: dict[str, Any]) -> Any:
        try:
            return self.client.responses.create(model=self.config.model, **request)
        except Exception as primary_error:
            if not self.config.fallback_model or not _is_model_availability_error(primary_error):
                raise
            from .utils import log

            log(
                f"model {self.config.model} is unavailable for this account; "
                f"retrying with {self.config.fallback_model}"
            )
            return self.client.responses.create(model=self.config.fallback_model, **request)


def _is_model_availability_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    return status in {403, 404} or any(
        marker in message
        for marker in ("model_not_found", "does not have access", "not available", "unknown model")
    )
