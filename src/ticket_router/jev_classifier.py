"""TypeSafe Jev (System One) classifier using Choice + optional Noul/Score."""

from __future__ import annotations

import os
import time
from typing import Any

from ticket_router.categories import (
    DEFAULT_TAXONOMY,
    FRUSTRATION_CRITERIA,
    FRUSTRATION_INSTRUCTIONS,
    Taxonomy,
    URGENCY_INSTRUCTIONS,
)
from ticket_router.types import ClassificationResult

DEFAULT_JEV_MODEL = "jev-latest"
SYSTEMONE_PATH = "/v1/systemone"


def _questions(taxonomy: Taxonomy) -> dict[str, Any]:
    return {
        taxonomy.choice_key: {
            "type": "choice",
            "instructions": taxonomy.instructions,
            "criteria": dict(taxonomy.criteria),
        },
        "urgent": {
            "type": "noul",
            "instructions": URGENCY_INSTRUCTIONS,
        },
        "frustration": {
            "type": "score",
            "instructions": FRUSTRATION_INSTRUCTIONS,
            "criteria": list(FRUSTRATION_CRITERIA),
        },
    }


def _sdk_questions(taxonomy: Taxonomy) -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul, Score

    return {
        taxonomy.choice_key: Choice(
            instructions=taxonomy.instructions,
            criteria=dict(taxonomy.criteria),
        ),
        "urgent": Noul(instructions=URGENCY_INSTRUCTIONS),
        "frustration": Score(
            instructions=FRUSTRATION_INSTRUCTIONS,
            criteria=list(FRUSTRATION_CRITERIA),
        ),
    }


def _state(text: str) -> dict[str, str]:
    return {"ticket_text": text}


class JevClassifier:
    """Wrap TypeSafeClient.system_one; fall back to REST if the SDK is missing."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        fan_out: bool = True,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or ""
        self.model = model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_JEV_MODEL
        self.base_url = (
            base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
        ).rstrip("/")
        self.timeout = timeout
        self.fan_out = fan_out
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self._sdk_client = None
        self._http = None
        self.backend = "unset"

    def __enter__(self) -> JevClassifier:
        if not self.api_key.strip():
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Export it or pass --dry-run."
            )
        try:
            from typesafe_sdk import TypeSafeClient

            self._sdk_client = TypeSafeClient(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self.backend = "typesafe-sdk"
        except ImportError:
            import httpx

            self._http = httpx.Client(timeout=self.timeout)
            self.backend = "httpx"
        return self

    def __exit__(self, *exc: object) -> None:
        if self._sdk_client is not None:
            self._sdk_client.close()
            self._sdk_client = None
        if self._http is not None:
            self._http.close()
            self._http = None

    def classify(self, text: str) -> ClassificationResult:
        if self._sdk_client is not None:
            return self._classify_sdk(text)
        if self._http is not None:
            return self._classify_http(text)
        raise RuntimeError("JevClassifier must be used as a context manager")

    def _classify_sdk(self, text: str) -> ClassificationResult:
        all_questions = _sdk_questions(self.taxonomy)
        questions = all_questions if self.fan_out else {
            self.taxonomy.choice_key: all_questions[self.taxonomy.choice_key]
        }
        started = time.perf_counter()
        response = self._sdk_client.system_one(
            state=_state(text),
            questions=questions,
            model=self.model,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[self.taxonomy.choice_key]
        extras: dict[str, Any] = {
            "backend": self.backend,
            "model": getattr(response, "model", self.model),
        }
        if self.fan_out:
            nouls = getattr(response, "nouls", {}) or {}
            scores = getattr(response, "scores", {}) or {}
            if "urgent" in nouls:
                extras["urgent"] = nouls["urgent"].noul
            if "frustration" in scores:
                extras["frustration"] = scores["frustration"].score
        usage = getattr(response, "usage", None)
        return ClassificationResult(
            label=str(choice.choice),
            latency_ms=latency_ms,
            confidence=float(choice.confidence) if choice.confidence is not None else None,
            probabilities=dict(choice.probabilities) if choice.probabilities else None,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            extras=extras,
        )

    def _classify_http(self, text: str) -> ClassificationResult:
        all_questions = _questions(self.taxonomy)
        questions = all_questions if self.fan_out else {
            self.taxonomy.choice_key: all_questions[self.taxonomy.choice_key]
        }
        payload = {
            "model": self.model,
            "state": _state(text),
            "questions": questions,
        }
        started = time.perf_counter()
        response = self._http.post(
            f"{self.base_url}{SYSTEMONE_PATH}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        answers = body.get("answers") or {}
        department = answers.get(self.taxonomy.choice_key) or {}
        extras: dict[str, Any] = {
            "backend": self.backend,
            "model": body.get("model", self.model),
        }
        if "urgent" in answers:
            extras["urgent"] = answers["urgent"].get("noul")
        if "frustration" in answers:
            extras["frustration"] = answers["frustration"].get("score")
        usage = body.get("usage") or {}
        return ClassificationResult(
            label=str(department.get("choice") or ""),
            latency_ms=latency_ms,
            confidence=_as_float(department.get("confidence")),
            probabilities=department.get("probabilities"),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            extras=extras,
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
