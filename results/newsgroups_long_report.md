# Ticket routing benchmark

Tickets evaluated: 30

## Comparison

| Model | Accuracy | Mean latency | p50 | p95 | Est. cost | Input tok | Output tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev (jev-latest) | 96.7% | 416.6 ms | 384.1 ms | 427.5 ms | $0.0150 | 357,174 | 2,734 |
| LLM via OpenRouter (openai/gpt-4o-mini) | 96.7% | 2350.8 ms | 2082.2 ms | 3228.1 ms | $0.0405 | 269,144 | 167 |

## Jev (jev-latest)

Accuracy: **96.7%** (29/30)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| computer | 100.0% | 100.0% | 100.0% | 6 |
| recreation | 100.0% | 100.0% | 100.0% | 6 |
| science | 85.7% | 100.0% | 92.3% | 6 |
| politics | 100.0% | 100.0% | 100.0% | 6 |
| religion | 100.0% | 83.3% | 90.9% | 6 |

Confusion matrix (rows = gold, columns = predicted):

```
 gold\pred  computer  recreation  science  politics  religion
  computer         6           0        0         0         0
recreation         0           6        0         0         0
   science         0           0        6         0         0
  politics         0           0        0         6         0
  religion         0           0        1         0         5
```

Latency: mean 416.6 ms, p50 384.1 ms, p95 427.5 ms
Tokens: 357,174 input, 2,734 output. Jev list price: $0.042 / M input tokens; output free.
Estimated cost: $0.0150
Average Choice confidence: 0.999
High-confidence accuracy (confidence ≥ 0.75): 96.7% on 30/30 tickets

Example mistakes:
- `talk.religion.misc-84449` gold=religion pred=science, confidence=0.97: SHARED MODERATOR ROUTING HANDBOOK This handbook is attached to every packet. It does not indicate the label. Valid topics: computer, recr...

## LLM via OpenRouter (openai/gpt-4o-mini)

Accuracy: **96.7%** (29/30)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| computer | 100.0% | 100.0% | 100.0% | 6 |
| recreation | 100.0% | 100.0% | 100.0% | 6 |
| science | 85.7% | 100.0% | 92.3% | 6 |
| politics | 100.0% | 100.0% | 100.0% | 6 |
| religion | 100.0% | 83.3% | 90.9% | 6 |

Confusion matrix (rows = gold, columns = predicted):

```
 gold\pred  computer  recreation  science  politics  religion
  computer         6           0        0         0         0
recreation         0           6        0         0         0
   science         0           0        6         0         0
  politics         0           0        0         6         0
  religion         0           0        1         0         5
```

Latency: mean 2350.8 ms, p50 2082.2 ms, p95 3228.1 ms
Tokens: 269,144 input, 167 output. Estimated OpenRouter cost for openai/gpt-4o-mini at $0.15 / M input, $0.60 / M output. Actual $/MTok depends on the chosen OpenRouter model; see https://openrouter.ai/models.
Estimated cost: $0.0405

Example mistakes:
- `talk.religion.misc-84449` gold=religion pred=science: SHARED MODERATOR ROUTING HANDBOOK This handbook is attached to every packet. It does not indicate the label. Valid topics: computer, recr...

## How to read this

- Accuracy is exact-match of the predicted department against the gold label.
- Jev also returns Noul(urgency) and Score(frustration) in the same call; those extras are not scored here.
- Cost is estimated from reported usage tokens. Jev uses the published $0.042 / M input list price (output free). The LLM line is a rough OpenRouter estimate; actual $/MTok depends on the chosen model.
- High-confidence accuracy is only reported when the model returns a Choice confidence (Jev, and the dry-run heuristic).
