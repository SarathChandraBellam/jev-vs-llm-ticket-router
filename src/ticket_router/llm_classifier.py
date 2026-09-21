"""OpenRouter LLM baseline (OpenAI-compatible structured outputs)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ticket_router.categories import DEPARTMENTS, LLM_SYSTEM_PROMPT, department_schema
from ticket_router.types import ClassificationResult

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
DEFAULT_APP_TITLE = "jev-vs-llm-ticket-router"


def resolve_llm_api_key() -> str:
    """Prefer OPENROUTER_API_KEY; fall back to OPENAI_API_KEY."""
    return (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()


def resolve_llm_model() -> str:
    return (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_LLM_MODEL
    ).strip()


def resolve_llm_base_url() -> str:
    return (os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")


def openrouter_headers() -> dict[str, str]:
    """Optional OpenRouter attribution headers (HTTP-Referer, X-Title)."""
    headers: dict[str, str] = {}
    referer = (
        os.environ.get("OPENROUTER_HTTP_REFERER")
        or os.environ.get("HTTP_REFERER")
        or ""
    ).strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = (os.environ.get("OPENROUTER_X_TITLE") or DEFAULT_APP_TITLE).strip()
    if title:
        headers["X-Title"] = title
    return headers


class LLMClassifier:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else resolve_llm_api_key()).strip()
        self.model = (model if model is not None else resolve_llm_model()).strip()
        self.base_url = (base_url if base_url is not None else resolve_llm_base_url()).rstrip(
            "/"
        )
        self.timeout = timeout
        self.default_headers = default_headers if default_headers is not None else openrouter_headers()
        self._client = None

    def __enter__(self) -> LLMClassifier:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (OPENAI_API_KEY is also accepted). "
                "Export one of them or pass --dry-run."
            )
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "base_url": self.base_url,
        }
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        self._client = OpenAI(**kwargs)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def classify(self, text: str) -> ClassificationResult:
        if self._client is None:
            raise RuntimeError("LLMClassifier must be used as a context manager")
        started = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "department_route",
                    "strict": True,
                    "schema": department_schema(),
                },
            },
            temperature=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        message = completion.choices[0].message
        content = message.content or "{}"
        parsed = json.loads(content)
        label = str(parsed.get("department") or "").strip().lower()
        if label not in DEPARTMENTS:
            raise ValueError(f"LLM returned unknown department {label!r}")
        usage = completion.usage
        return ClassificationResult(
            label=label,
            latency_ms=latency_ms,
            confidence=None,
            probabilities=None,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            extras={
                "backend": "openrouter",
                "base_url": self.base_url,
                "model": completion.model or self.model,
            },
        )
