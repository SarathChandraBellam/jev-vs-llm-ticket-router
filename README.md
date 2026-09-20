# jev-vs-llm-ticket-router

A small, runnable Python benchmark that routes support tickets to a department and compares:

1. **TypeSafe Jev** (System One) — one `Choice` question, plus optional `Noul` / `Score` in the same call
2. **A traditional LLM** — OpenAI-compatible **structured output** forced to the same five labels

The goal is a **fair classification comparison** (same tickets, same label set, parallel instructions) plus latency and estimated cost.

## What Jev is

Jev is TypeSafe's flagship model and the first **System One** model. You send **state** plus typed **questions**; you get structured answers (`choice` + probabilities + confidence, a `score`, or a yes/no `noul`) instead of generated text you then parse.

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`
- Model: `jev-latest` (or pin `jev-1.13.0`)
- Auth: `Authorization: Bearer $TYPESAFE_API_KEY`
- Python: `pip install typesafe-sdk` → `TypeSafeClient`, `Choice`, `Noul`, `Score`

See [docs/how-jev-works.md](docs/how-jev-works.md) for Choice / Score / Noul, speculative fan-out, and a copy-paste `curl`. Official docs: [docs.typesafe.ai](https://docs.typesafe.ai).

## Classification problem

Gold-labeled **support ticket department routing**. Five classes:

| Label | Criteria (shared by Jev Choice and the LLM prompt) |
| --- | --- |
| `billing` | Payments, charges, refunds, invoices, subscriptions |
| `technical` | Bugs, outages, errors, integrations, API issues |
| `shipping` | Delivery, tracking, lost packages, address changes |
| `account` | Login, password, permissions, profile, 2FA |
| `sales` | Pricing, upgrades, demos, new accounts, plans |

Dataset: [`data/tickets.jsonl`](data/tickets.jsonl) (~110 synthetic but realistic tickets: short, long, ambiguous, multi-intent). Fields: `id`, `text`, `label`. Accuracy is **exact match on `label`**. Jev's extra urgency / frustration answers are recorded, not scored.

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Keys

| Variable | Required for | Where to get it |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | Jev | [console.typesafe.ai](https://console.typesafe.ai) |
| `OPENAI_API_KEY` | LLM baseline | [platform.openai.com](https://platform.openai.com) |

Optional:

| Variable | Default | Notes |
| --- | --- | --- |
| `TYPESAFE_MODEL` | `jev-latest` | Pin with e.g. `jev-1.13.0` |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI-compatible chat model that supports JSON schema |
| `OPENAI_BASE_URL` | OpenAI default | Compatible proxies / gateways |

**Do not commit `.env`.** Only `.env.example` is in git.

## Run

No keys (keyword heuristic, so the repo is demoable):

```bash
python scripts/run_benchmark.py --dry-run
python scripts/run_benchmark.py --dry-run --limit 10
```

Full comparison (needs both keys):

```bash
python scripts/run_benchmark.py
python scripts/run_benchmark.py --limit 20 --output results/latest_report.md
```

One side only:

```bash
python scripts/run_benchmark.py --jev-only
python scripts/run_benchmark.py --llm-only --limit 15
```

After `pip install -e .` you can also run `run-benchmark --dry-run`.

### CLI flags

| Flag | Meaning |
| --- | --- |
| `--limit N` | First N tickets only |
| `--jev-only` / `--llm-only` | Skip the other model |
| `--dry-run` | No API calls; keyword baseline |
| `--output PATH` | Write the markdown report |
| `--confidence-threshold 0.75` | Jev high-confidence cutoff |
| `--no-fan-out` | Choice only (skip urgency + frustration) |
| `--dataset PATH` | Alternate JSONL |

## How to interpret results

The report prints:

- **Overall accuracy** — predicted department vs gold
- **Per-class precision / recall / F1** and a **confusion matrix**
- **Latency** — mean, p50, p95 of wall time per ticket (sequential calls, one shared client)
- **Estimated cost** — usage tokens × documented list prices (not an invoice)
  - Jev: **$0.042 / M input tokens**, output free
  - Default LLM (`gpt-4o-mini`): **$0.15 / M input**, **$0.60 / M output** (standard API, not batch)
- **Jev confidence** — mean Choice confidence, and accuracy on tickets with confidence ≥ threshold

A filled-in example of the tables is in [`results/sample_report.md`](results/sample_report.md) (illustrative numbers, not a live run).

Jev is built for fast structured decisions; the interesting comparison is usually **accuracy at much lower latency and cost**, plus whether high-confidence Choice answers are safe to auto-route. Multi-intent tickets (refund + login, shipping + upgrade) are in the set on purpose — both models are instructed to pick the **primary owner**.

## Project layout

```
data/tickets.jsonl              labeled dataset
src/ticket_router/
  jev_classifier.py             TypeSafeClient Choice (+ Noul/Score fan-out)
  llm_classifier.py             OpenAI structured outputs
  evaluate.py                   accuracy, F1, latency, cost
  heuristic.py                  dry-run keyword baseline
scripts/run_benchmark.py        CLI
docs/how-jev-works.md           primitives + curl
results/sample_report.md        comparison table shape
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests cover dataset loading, metrics, and the dry-run heuristic. They do not call paid APIs.
