"""Accuracy, latency, and cost evaluation for ticket-routing classifiers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ticket_router.categories import DEPARTMENTS
from ticket_router.types import ClassificationResult, Ticket

# Jev: $0.042 / million input tokens; output is free.
JEV_INPUT_USD_PER_MTOK = 0.042
JEV_OUTPUT_USD_PER_MTOK = 0.0

# Rough $/MTok used only for the cost estimate. OpenRouter bills per model;
# look up the live rate at https://openrouter.ai/models. Cached-input ignored.
LLM_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
DEFAULT_LLM_INPUT_USD_PER_MTOK = 0.15
DEFAULT_LLM_OUTPUT_USD_PER_MTOK = 0.60

HIGH_CONFIDENCE_DEFAULT = 0.75


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    n: int


@dataclass
class CostStats:
    input_tokens: int
    output_tokens: int
    usd: float
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    notes: str


@dataclass
class ConfidenceStats:
    mean_confidence: float | None
    high_confidence_threshold: float
    high_confidence_count: int
    high_confidence_accuracy: float | None


@dataclass
class ModelReport:
    name: str
    n: int
    n_errors: int
    accuracy: float
    per_class: list[ClassMetrics]
    confusion: list[list[int]]
    latency: LatencyStats
    cost: CostStats
    confidence: ConfidenceStats | None
    predictions: list[ClassificationResult] = field(repr=False)
    mistakes: list[tuple[Ticket, ClassificationResult]] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    models: list[ModelReport]
    labels: tuple[str, ...] = DEPARTMENTS
    n_tickets: int = 0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for truth, guess in zip(gold, pred):
        if truth not in index or guess not in index:
            continue
        matrix[index[truth]][index[guess]] += 1
    return matrix


def per_class_metrics(
    matrix: Sequence[Sequence[int]], labels: Sequence[str]
) -> list[ClassMetrics]:
    metrics: list[ClassMetrics] = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(labels)) if col != i)
        support = sum(matrix[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics.append(
            ClassMetrics(
                label=label,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return metrics


def llm_rates(model: str) -> tuple[float, float]:
    key = model.strip().lower()
    if key in LLM_PRICING_USD_PER_MTOK:
        return LLM_PRICING_USD_PER_MTOK[key]
    for known, rates in LLM_PRICING_USD_PER_MTOK.items():
        if known in key:
            return rates
    return DEFAULT_LLM_INPUT_USD_PER_MTOK, DEFAULT_LLM_OUTPUT_USD_PER_MTOK


def estimate_cost(
    *,
    name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> CostStats:
    lowered = name.lower()
    if "jev" in lowered or "typesafe" in lowered:
        in_rate, out_rate = JEV_INPUT_USD_PER_MTOK, JEV_OUTPUT_USD_PER_MTOK
        notes = "Jev list price: $0.042 / M input tokens; output free."
    elif "heuristic" in lowered:
        in_rate, out_rate = 0.0, 0.0
        notes = "Local keyword baseline; no API cost."
    else:
        in_rate, out_rate = llm_rates(model)
        notes = (
            f"Estimated OpenRouter cost for {model} at ${in_rate:.2f} / M input, "
            f"${out_rate:.2f} / M output. Actual $/MTok depends on the chosen "
            "OpenRouter model; see https://openrouter.ai/models."
        )
    usd = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return CostStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        input_usd_per_mtok=in_rate,
        output_usd_per_mtok=out_rate,
        notes=notes,
    )


def _score_model(
    *,
    name: str,
    tickets: Sequence[Ticket],
    results: Sequence[ClassificationResult],
    high_confidence_threshold: float,
    pricing_model: str,
) -> ModelReport:
    paired = [
        (ticket, result)
        for ticket, result in zip(tickets, results)
        if result.error is None and result.label
    ]
    n_errors = len(tickets) - len(paired)
    gold = [ticket.label for ticket, _ in paired]
    pred = [result.label for _, result in paired]
    matrix = confusion_matrix(gold, pred, DEPARTMENTS)
    correct = sum(g == p for g, p in zip(gold, pred))
    accuracy = correct / len(paired) if paired else 0.0
    latencies = [result.latency_ms for _, result in paired]
    latency = LatencyStats(
        mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        n=len(latencies),
    )
    input_tokens = sum(result.input_tokens for _, result in paired)
    output_tokens = sum(result.output_tokens for _, result in paired)
    cost = estimate_cost(
        name=name,
        model=pricing_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    confidences = [
        result.confidence for _, result in paired if result.confidence is not None
    ]
    confidence: ConfidenceStats | None = None
    if confidences:
        high = [
            (ticket, result)
            for ticket, result in paired
            if result.confidence is not None
            and result.confidence >= high_confidence_threshold
        ]
        high_acc = (
            sum(ticket.label == result.label for ticket, result in high) / len(high)
            if high
            else None
        )
        confidence = ConfidenceStats(
            mean_confidence=sum(confidences) / len(confidences),
            high_confidence_threshold=high_confidence_threshold,
            high_confidence_count=len(high),
            high_confidence_accuracy=high_acc,
        )
    mistakes = [
        (ticket, result) for ticket, result in paired if ticket.label != result.label
    ]
    return ModelReport(
        name=name,
        n=len(paired),
        n_errors=n_errors,
        accuracy=accuracy,
        per_class=per_class_metrics(matrix, DEPARTMENTS),
        confusion=matrix,
        latency=latency,
        cost=cost,
        confidence=confidence,
        predictions=list(results),
        mistakes=mistakes,
    )


def run_classifier(
    tickets: Sequence[Ticket],
    classify: Callable[[str], ClassificationResult],
    *,
    on_ticket: Callable[[Ticket, ClassificationResult, int], None] | None = None,
) -> list[ClassificationResult]:
    results: list[ClassificationResult] = []
    for index, ticket in enumerate(tickets, start=1):
        try:
            result = classify(ticket.text)
        except Exception as exc:  # noqa: BLE001 — surface per-ticket failures in the report
            result = ClassificationResult(
                label="",
                latency_ms=0.0,
                error=str(exc),
            )
        results.append(result)
        if on_ticket is not None:
            on_ticket(ticket, result, index)
    return results


def evaluate(
    tickets: Sequence[Ticket],
    named_results: Sequence[tuple[str, Sequence[ClassificationResult], str]],
    *,
    high_confidence_threshold: float = HIGH_CONFIDENCE_DEFAULT,
) -> BenchmarkReport:
    models = [
        _score_model(
            name=name,
            tickets=tickets,
            results=results,
            high_confidence_threshold=high_confidence_threshold,
            pricing_model=pricing_model,
        )
        for name, results, pricing_model in named_results
    ]
    return BenchmarkReport(models=models, n_tickets=len(tickets))


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _fmt_ms(value: float) -> str:
    return f"{value:7.1f} ms"


def _fmt_usd(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.0001:
        return f"${value:.6f}"
    return f"${value:.4f}"


def format_confusion(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    header = ["gold\\pred", *labels]
    widths = [max(len(header[0]), max(len(label) for label in labels))]
    widths.extend(
        max(len(label), max(len(str(matrix[r][c])) for r in range(len(labels))))
        for c, label in enumerate(labels)
    )
    lines = [
        "  ".join(title.rjust(widths[i]) for i, title in enumerate(header))
    ]
    for row_i, label in enumerate(labels):
        cells = [label.rjust(widths[0])]
        for col_i in range(len(labels)):
            cells.append(str(matrix[row_i][col_i]).rjust(widths[col_i + 1]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def format_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("# Ticket routing benchmark")
    lines.append("")
    lines.append(f"Tickets evaluated: {report.n_tickets}")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append(
        "| Model | Accuracy | Mean latency | p50 | p95 | Est. cost | Input tok | Output tok |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in report.models:
        lines.append(
            "| {name} | {acc} | {mean} | {p50} | {p95} | {cost} | {inp} | {out} |".format(
                name=model.name,
                acc=_pct(model.accuracy).strip(),
                mean=_fmt_ms(model.latency.mean_ms).strip(),
                p50=_fmt_ms(model.latency.p50_ms).strip(),
                p95=_fmt_ms(model.latency.p95_ms).strip(),
                cost=_fmt_usd(model.cost.usd),
                inp=f"{model.cost.input_tokens:,}",
                out=f"{model.cost.output_tokens:,}",
            )
        )
    lines.append("")
    for model in report.models:
        lines.append(f"## {model.name}")
        lines.append("")
        lines.append(
            f"Accuracy: **{_pct(model.accuracy).strip()}** "
            f"({model.n - len(model.mistakes)}/{model.n})"
        )
        if model.n_errors:
            lines.append(f"Failed calls: {model.n_errors}")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in model.per_class:
            lines.append(
                f"| {row.label} | {_pct(row.precision).strip()} | "
                f"{_pct(row.recall).strip()} | {_pct(row.f1).strip()} | {row.support} |"
            )
        lines.append("")
        lines.append("Confusion matrix (rows = gold, columns = predicted):")
        lines.append("")
        lines.append("```")
        lines.append(format_confusion(model.confusion, DEPARTMENTS))
        lines.append("```")
        lines.append("")
        lines.append(
            f"Latency: mean {_fmt_ms(model.latency.mean_ms).strip()}, "
            f"p50 {_fmt_ms(model.latency.p50_ms).strip()}, "
            f"p95 {_fmt_ms(model.latency.p95_ms).strip()}"
        )
        lines.append(
            f"Tokens: {model.cost.input_tokens:,} input, "
            f"{model.cost.output_tokens:,} output. {model.cost.notes}"
        )
        lines.append(f"Estimated cost: {_fmt_usd(model.cost.usd)}")
        if model.confidence is not None:
            mean_c = model.confidence.mean_confidence or 0.0
            lines.append(f"Average Choice confidence: {mean_c:.3f}")
            high_acc = model.confidence.high_confidence_accuracy
            high_acc_s = _pct(high_acc).strip() if high_acc is not None else "n/a"
            lines.append(
                "High-confidence accuracy "
                f"(confidence ≥ {model.confidence.high_confidence_threshold:.2f}): "
                f"{high_acc_s} "
                f"on {model.confidence.high_confidence_count}/{model.n} tickets"
            )
        if model.mistakes:
            lines.append("")
            lines.append("Example mistakes:")
            for ticket, result in model.mistakes[:8]:
                snippet = " ".join(ticket.text.split())
                if len(snippet) > 140:
                    snippet = snippet[:137] + "..."
                extra = ""
                if result.confidence is not None:
                    extra = f", confidence={result.confidence:.2f}"
                lines.append(
                    f"- `{ticket.id}` gold={ticket.label} pred={result.label}{extra}: {snippet}"
                )
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- Accuracy is exact-match of the predicted department against the gold label."
    )
    lines.append(
        "- Jev also returns Noul(urgency) and Score(frustration) in the same call; "
        "those extras are not scored here."
    )
    lines.append(
        "- Cost is estimated from reported usage tokens. Jev uses the published "
        "$0.042 / M input list price (output free). The LLM line is a rough "
        "OpenRouter estimate; actual $/MTok depends on the chosen model."
    )
    lines.append(
        "- High-confidence accuracy is only reported when the model returns a "
        "Choice confidence (Jev, and the dry-run heuristic)."
    )
    return "\n".join(lines).rstrip() + "\n"
