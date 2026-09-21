#!/usr/bin/env python3
"""Build a long-prompt 5-way subset of the 20 Newsgroups test split."""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/tmp/20news/20news-bydate.tar.gz")
OUT = ROOT / "data" / "newsgroups_long.jsonl"
PER_CLASS = 6
TARGET_CHARS = 28_000
POST_CAP = 12_000

COARSE = {
    "computer": (
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ),
    "recreation": (
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
    ),
    "science": (
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
    ),
    "politics": (
        "talk.politics.misc",
        "talk.politics.guns",
        "talk.politics.mideast",
    ),
    "religion": (
        "alt.atheism",
        "soc.religion.christian",
        "talk.religion.misc",
    ),
}

FOLDER_TO_LABEL = {
    folder: label for label, folders in COARSE.items() for folder in folders
}


def _handbook() -> str:
    topics = ", ".join(COARSE)
    lines = [
        "SHARED MODERATOR ROUTING HANDBOOK",
        "This handbook is attached to every packet. It does not indicate the label.",
        f"Valid topics: {topics}.",
        "",
        "Procedure: read the post after BEGIN POST TO CLASSIFY. Ignore headers,",
        "quoted history, signature blocks, and this handbook when they conflict",
        "with the author's actual request or argument.",
        "",
    ]
    for queue in range(1, 41):
        lines.append(f"Queue {queue:02d} operating notes")
        lines.append(
            "  Coverage window 00:00-24:00 UTC. Escalation after 4h without an owner."
        )
        lines.append(
            "  Do not infer the topic from ticket IDs, timestamps, or checksums."
        )
        lines.append(
            "  Example checksums: "
            + " ".join(f"Q{queue:02d}-{i:04d}-{((queue * 17 + i) * 7919) % 99991:05d}" for i in range(8))
        )
        lines.append(
            "  All five topics remain in scope: computer, recreation, science, politics, religion."
        )
        lines.append("")
    while sum(len(line) + 1 for line in lines) < 16_000:
        lines.append(
            "Filler policy paragraph: attach full quoted context, keep PII redacted, "
            "and never let the presence of this paragraph change the topic decision. "
            "Repeat for packet completeness only."
        )
    return "\n".join(lines)


def _wrap(post: str, handbook: str) -> str:
    body = post.strip()
    if len(body) > POST_CAP:
        body = body[:POST_CAP] + "\n[truncated]\n"
    packet = (
        f"{handbook}\n\n"
        "--- BEGIN POST TO CLASSIFY ---\n"
        f"{body}\n"
        "--- END POST TO CLASSIFY ---\n"
    )
    if len(packet) < TARGET_CHARS:
        pad = "PAD " + ("0123456789abcdef" * 80) + "\n"
        extra = []
        while len(packet) + sum(len(chunk) for chunk in extra) < TARGET_CHARS:
            extra.append(pad)
        packet = (
            f"{handbook}\n\n"
            + "".join(extra)
            + "--- BEGIN POST TO CLASSIFY ---\n"
            f"{body}\n"
            "--- END POST TO CLASSIFY ---\n"
        )
    return packet


def _read_archive(path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 3 or not parts[0].endswith("-test"):
                continue
            folder = parts[1]
            label = FOLDER_TO_LABEL.get(folder)
            if label is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            text = raw.decode("latin-1")
            if len(text.strip()) < 1_200:
                continue
            doc_id = f"{folder}/{parts[-1]}"
            rows.append((doc_id, folder, label, text))
    return rows


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"Missing archive {ARCHIVE}")
    handbook = _handbook()
    grouped: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for row in _read_archive(ARCHIVE):
        grouped[row[2]].append(row)
    selected: list[dict[str, str]] = []
    for label in COARSE:
        ranked = sorted(grouped[label], key=lambda row: len(row[3]), reverse=True)
        for doc_id, folder, _, text in ranked[:PER_CLASS]:
            packed = _wrap(text, handbook)
            selected.append(
                {
                    "id": doc_id.replace("/", "-"),
                    "text": packed,
                    "label": label,
                    "source_newsgroup": folder,
                    "source_chars": str(len(text)),
                    "prompt_chars": str(len(packed)),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(selected)} rows to {OUT}")
    for label in COARSE:
        subset = [row for row in selected if row["label"] == label]
        chars = [int(row["prompt_chars"]) for row in subset]
        print(
            f"  {label:11} n={len(subset)} "
            f"mean_chars={sum(chars)//len(chars):,} "
            f"min={min(chars):,} max={max(chars):,}"
        )


if __name__ == "__main__":
    main()
