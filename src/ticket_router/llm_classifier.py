"""OpenAI-compatible structured-output baseline for department routing."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ticket_router.categories import DEPARTMENTS, LLM_SYSTEM_PROMPT, department_schema
from ticket_router.types import ClassificationResult

DEFAULT_LLM_MODEL = "gpt-4o-mini"


class LLMClassifier:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_LLM_MODEL
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.timeout = timeout
        self._client = None

    def __enter__(self) -> LLMClassifier:
        if not self.api_key.strip():
            raise RuntimeError("OPENAI_API_KEY is not set. Export it or pass --dry-run.")
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
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
                "backend": "openai",
                "model": completion.model or self.model,
            },
        )
