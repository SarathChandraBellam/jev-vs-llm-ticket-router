from __future__ import annotations

import json
from pathlib import Path

from ticket_router.categories import DEPARTMENTS
from ticket_router.types import Ticket

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "tickets.jsonl"


def load_tickets(path: str | Path | None = None, limit: int | None = None) -> list[Ticket]:
    dataset_path = Path(path) if path else DEFAULT_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tickets: list[Ticket] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path}:{line_no}: invalid JSON") from exc
            for key in ("id", "text", "label"):
                if key not in row:
                    raise ValueError(f"{dataset_path}:{line_no}: missing field {key!r}")
            label = str(row["label"]).strip().lower()
            if label not in DEPARTMENTS:
                raise ValueError(
                    f"{dataset_path}:{line_no}: unknown label {label!r}; "
                    f"expected one of {list(DEPARTMENTS)}"
                )
            tickets.append(
                Ticket(id=str(row["id"]), text=str(row["text"]).strip(), label=label)
            )
            if limit is not None and len(tickets) >= limit:
                break
    if not tickets:
        raise ValueError(f"No tickets loaded from {dataset_path}")
    return tickets


def label_counts(tickets: list[Ticket]) -> dict[str, int]:
    counts = {label: 0 for label in DEPARTMENTS}
    for ticket in tickets:
        counts[ticket.label] += 1
    return counts
