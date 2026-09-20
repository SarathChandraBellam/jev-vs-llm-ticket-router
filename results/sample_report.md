# Sample report (illustrative)

This file shows the **shape** of `python scripts/run_benchmark.py` output. Numbers are **not** from a live API run. They are a realistic filled-in example so you can see the tables before you spend tokens.

Tickets evaluated: 110

## Comparison

| Model | Accuracy | Mean latency | p50 | p95 | Est. cost | Input tok | Output tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev (jev-latest) | 90.9% | 184.2 ms | 161.0 ms | 312.4 ms | $0.0016 | 38,940 | 5,280 |
| LLM (gpt-4o-mini) | 89.1% | 641.8 ms | 598.0 ms | 1104.2 ms | $0.0079 | 42,150 | 2,640 |

## Jev (jev-latest)

Accuracy: **90.9%** (100/110)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| billing | 91.3% | 95.5% | 93.3% | 22 |
| technical | 90.5% | 86.4% | 88.4% | 22 |
| shipping | 95.2% | 90.9% | 93.0% | 22 |
| account | 87.0% | 90.9% | 88.9% | 22 |
| sales | 90.9% | 90.9% | 90.9% | 22 |

Confusion matrix (rows = gold, columns = predicted):

```
gold\pred  billing  technical  shipping  account  sales
  billing       21          0         0        1      0
technical        1         19         0        1      1
 shipping        0          1        20        0      1
  account        1          1         0       20      0
    sales        0          0         1        1     20
```

Latency: mean 184.2 ms, p50 161.0 ms, p95 312.4 ms
Tokens: 38,940 input, 5,280 output. Jev list price: $0.042 / M input tokens; output free.
Estimated cost: $0.0016
Average Choice confidence: 0.812
High-confidence accuracy (confidence ≥ 0.75): 96.2% on 79/110 tickets

Example mistakes:
- `t014` gold=billing pred=technical, confidence=0.41: I updated my card and now I can't use the product until billing retries...
- `t039` gold=technical pred=billing, confidence=0.55: I also want a refund later maybe, but first please fix the 500 error...
- `t080` gold=account pred=shipping, confidence=0.48: Also our shipment is late, but first I cannot log in...

## LLM (gpt-4o-mini)

Accuracy: **89.1%** (98/110)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| billing | 87.5% | 95.5% | 91.3% | 22 |
| technical | 90.0% | 81.8% | 85.7% | 22 |
| shipping | 91.3% | 95.5% | 93.3% | 22 |
| account | 90.5% | 86.4% | 88.4% | 22 |
| sales | 86.4% | 86.4% | 86.4% | 22 |

Confusion matrix (rows = gold, columns = predicted):

```
gold\pred  billing  technical  shipping  account  sales
  billing       21          0         0        0      1
technical        2         18         0        1      1
 shipping        0          0        21        0      1
  account        1          2         0       19      0
    sales        0          0         2        1     19
```

Latency: mean 641.8 ms, p50 598.0 ms, p95 1104.2 ms
Tokens: 42,150 input, 2,640 output. Assumed gpt-4o-mini list price: $0.15 / M input, $0.60 / M output (standard API, not batch).
Estimated cost: $0.0079

Example mistakes:
- `t076` gold=account pred=billing: Permissions on the billing page are read-only for me even though I'm listed as admin...
- `t018` gold=billing pred=account: I asked support last week about a missing invoice and also mentioned I forgot my password...
- `t059` gold=shipping pred=sales: I know I also asked about upgrading seats, but the urgent thing is the pallet...

## How to read this

- Accuracy is exact-match of the predicted department against the gold label.
- Jev also returns Noul(urgency) and Score(frustration) in the same call; those extras are not scored here.
- Cost is estimated from reported usage tokens and the documented list prices. It is not a bill.
- High-confidence accuracy is only reported when the model returns a Choice confidence (Jev, and the dry-run heuristic).

Replace this file by running:

```bash
python scripts/run_benchmark.py --output results/latest_report.md
```
