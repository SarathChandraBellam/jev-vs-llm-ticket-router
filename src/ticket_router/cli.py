"""CLI: python scripts/run_benchmark.py [--dry-run] [--limit N] ..."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optional for --dry-run if python-dotenv is not installed
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

from ticket_router.dataset import label_counts, load_tickets
from ticket_router.evaluate import HIGH_CONFIDENCE_DEFAULT, evaluate, format_report, run_classifier
from ticket_router.heuristic import classify_heuristic, frustration_level, looks_urgent
from ticket_router.types import ClassificationResult, Ticket

ROOT = Path(__file__).resolve().parents[2]


def _progress(ticket: Ticket, result: ClassificationResult, index: int, total: int) -> None:
    status = result.label or "ERROR"
    mark = "ok" if result.error is None and result.label == ticket.label else "miss"
    if result.error:
        mark = "err"
    conf = f" conf={result.confidence:.2f}" if result.confidence is not None else ""
    print(
        f"[{index}/{total}] {ticket.id:4} gold={ticket.label:10} "
        f"pred={status:10} {result.latency_ms:6.1f}ms {mark}{conf}",
        flush=True,
    )


def _print_samples(tickets: Sequence[Ticket], results: Sequence[ClassificationResult], n: int = 8) -> None:
    print("\nSample predictions")
    print("------------------")
    for ticket, result in list(zip(tickets, results))[:n]:
        snippet = " ".join(ticket.text.split())
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        extras = []
        if result.confidence is not None:
            extras.append(f"conf={result.confidence:.2f}")
        urgent = result.extras.get("urgent")
        if isinstance(urgent, (int, float)):
            extras.append(f"urgent={urgent:.2f}")
        frustration = result.extras.get("frustration")
        if isinstance(frustration, (int, float)):
            extras.append(f"frustration={frustration:.2f}")
        extra_s = ("  " + " ".join(extras)) if extras else ""
        print(f"{ticket.id}  gold={ticket.label:10} pred={result.label:10}{extra_s}")
        print(f"     {snippet}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TypeSafe Jev vs an OpenRouter LLM (OpenAI-compatible "
            "structured outputs) on support-ticket department routing."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "tickets.jsonl"),
        help="JSONL with id, text, label fields (default: data/tickets.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tickets")
    parser.add_argument("--jev-only", action="store_true", help="Skip the LLM baseline")
    parser.add_argument("--llm-only", action="store_true", help="Skip Jev")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls. Score a keyword heuristic so the repo is demoable without keys.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the markdown report to this path (in addition to stdout)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=HIGH_CONFIDENCE_DEFAULT,
        help="Jev high-confidence cutoff (default: 0.75)",
    )
    parser.add_argument(
        "--no-fan-out",
        action="store_true",
        help="Ask only the Choice question (skip Noul urgency and Score frustration)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args(argv)
    if args.jev_only and args.llm_only:
        print("Choose at most one of --jev-only / --llm-only.", file=sys.stderr)
        return 2

    tickets = load_tickets(args.dataset, limit=args.limit)
    counts = label_counts(tickets)
    print(f"Loaded {len(tickets)} tickets from {args.dataset}")
    print("Class mix: " + ", ".join(f"{label}={n}" for label, n in counts.items()))

    named_results: list[tuple[str, list[ClassificationResult], str]] = []

    if args.dry_run:
        print("\nDry-run: keyword heuristic (no API keys required)\n")

        def classify(text: str) -> ClassificationResult:
            result = classify_heuristic(text)
            result.extras["urgent"] = looks_urgent(text)
            result.extras["frustration"] = frustration_level(text)
            return result

        results = run_classifier(
            tickets,
            classify,
            on_ticket=lambda t, r, i: _progress(t, r, i, len(tickets)),
        )
        named_results.append(("heuristic (dry-run)", results, "heuristic"))
        _print_samples(tickets, results)
    else:
        run_jev = not args.llm_only
        run_llm = not args.jev_only
        missing: list[str] = []
        if run_jev and not (os.environ.get("TYPESAFE_API_KEY") or "").strip():
            missing.append("TYPESAFE_API_KEY")
        from ticket_router.llm_classifier import resolve_llm_api_key

        if run_llm and not resolve_llm_api_key():
            missing.append("OPENROUTER_API_KEY")
        if missing:
            hint = "Set them in .env or re-run with --dry-run."
            if "OPENROUTER_API_KEY" in missing:
                hint += " OPENAI_API_KEY is accepted if OPENROUTER_API_KEY is unset."
            print("Missing " + ", ".join(missing) + ". " + hint, file=sys.stderr)
            return 2

        if run_jev:
            from ticket_router.jev_classifier import JevClassifier

            print("\nJev (TypeSafe System One)\n")
            with JevClassifier(fan_out=not args.no_fan_out) as jev:
                results = run_classifier(
                    tickets,
                    jev.classify,
                    on_ticket=lambda t, r, i: _progress(t, r, i, len(tickets)),
                )
            model_name = os.environ.get("TYPESAFE_MODEL") or "jev-latest"
            named_results.append((f"Jev ({model_name})", results, model_name))
            _print_samples(tickets, results)

        if run_llm:
            from ticket_router.llm_classifier import LLMClassifier

            print("\nLLM structured-output baseline (OpenRouter)\n")
            with LLMClassifier() as llm:
                results = run_classifier(
                    tickets,
                    llm.classify,
                    on_ticket=lambda t, r, i: _progress(t, r, i, len(tickets)),
                )
                model_name = llm.model
            named_results.append((f"LLM via OpenRouter ({model_name})", results, model_name))
            _print_samples(tickets, results)

    report = evaluate(
        tickets,
        named_results,
        high_confidence_threshold=args.confidence_threshold,
    )
    rendered = format_report(report)
    print("\n" + rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
