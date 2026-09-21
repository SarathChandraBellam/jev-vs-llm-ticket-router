from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    label: str


@dataclass
class ClassificationResult:
    label: str
    latency_ms: float
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
