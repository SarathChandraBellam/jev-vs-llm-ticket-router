#!/usr/bin/env python3
"""Run the Jev vs LLM ticket-routing benchmark.

Examples:
    python scripts/run_benchmark.py --dry-run
    python scripts/run_benchmark.py --limit 20
    python scripts/run_benchmark.py --jev-only
    python scripts/run_benchmark.py --llm-only --limit 10
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ticket_router.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
