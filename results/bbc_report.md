# Ticket routing benchmark

Tickets evaluated: 50

## Comparison

| Model | Accuracy | Macro F1 | Input tok | Input $ | Output tok | Output $ | Est. cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev (jev-latest) | 92.0% | 91.6% | 85,473 | $0.0036 | 4,505 | $0 | $0.0036 |
| LLM via OpenRouter (openai/gpt-4o-mini) | 94.0% | 93.9% | 65,643 | $0.0098 | 286 | $0.0002 | $0.0100 |

## Jev (jev-latest)

Accuracy: **92.0%** (46/50)
Macro F1: **91.6%**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| business | 100.0% | 70.0% | 82.4% | 10 |
| entertainment | 90.9% | 100.0% | 95.2% | 10 |
| politics | 90.9% | 100.0% | 95.2% | 10 |
| sport | 90.9% | 100.0% | 95.2% | 10 |
| tech | 90.0% | 90.0% | 90.0% | 10 |

Confusion matrix (rows = gold, columns = predicted):

```
    gold\pred  business  entertainment  politics  sport  tech
     business         7              0         1      1     1
entertainment         0             10         0      0     0
     politics         0              0        10      0     0
        sport         0              0         0     10     0
         tech         0              1         0      0     9
```

Latency: mean 376.2 ms, p50 358.6 ms, p95 464.7 ms
Tokens: 85,473 input, 4,505 output. Jev list price: $0.042 / M input tokens; output free.
Estimated cost: $0.0036
Average Choice confidence: 0.916
High-confidence accuracy (confidence ≥ 0.75): 97.6% on 41/50 tickets

Example mistakes:
- `bbc-business-06` gold=business pred=politics, confidence=0.70: world leaders gather to face uncertainty more than 2 000 business and political leaders from around the globe are arriving in the swiss m...
- `bbc-business-07` gold=business pred=tech, confidence=0.65: bt offers equal access to rivals bt has moved to pre-empt a possible break-up of its business by offering to cut wholesale broadband pric...
- `bbc-business-08` gold=business pred=sport, confidence=0.51: q&a: malcolm glazer and man utd the battle for control of manchester united has taken another turn after the club confirmed it had receiv...
- `bbc-tech-10` gold=tech pred=entertainment, confidence=0.75: gta sequel is criminally good the grand theft auto series of games have set themselves the very highest of standards in recent years but ...

## LLM via OpenRouter (openai/gpt-4o-mini)

Accuracy: **94.0%** (47/50)
Macro F1: **93.9%**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| business | 100.0% | 100.0% | 100.0% | 10 |
| entertainment | 76.9% | 100.0% | 87.0% | 10 |
| politics | 100.0% | 100.0% | 100.0% | 10 |
| sport | 100.0% | 100.0% | 100.0% | 10 |
| tech | 100.0% | 70.0% | 82.4% | 10 |

Confusion matrix (rows = gold, columns = predicted):

```
    gold\pred  business  entertainment  politics  sport  tech
     business        10              0         0      0     0
entertainment         0             10         0      0     0
     politics         0              0        10      0     0
        sport         0              0         0     10     0
         tech         0              3         0      0     7
```

Latency: mean 1464.9 ms, p50 1074.1 ms, p95 2750.0 ms
Tokens: 65,643 input, 286 output. Estimated OpenRouter cost for openai/gpt-4o-mini at $0.15 / M input, $0.60 / M output. Actual $/MTok depends on the chosen OpenRouter model; see https://openrouter.ai/models.
Estimated cost: $0.0100

Example mistakes:
- `bbc-tech-01` gold=tech pred=entertainment: losing yourself in online gaming online role playing games are time-consuming but enthralling flights from reality. but are some people t...
- `bbc-tech-08` gold=tech pred=entertainment: what high-definition will do to dvds first it was the humble home video then it was the dvd and now hollywood is preparing for the next r...
- `bbc-tech-10` gold=tech pred=entertainment: gta sequel is criminally good the grand theft auto series of games have set themselves the very highest of standards in recent years but ...

## How to read this

- Accuracy is exact-match of the predicted department against the gold label.
- Jev also returns Noul(urgency) and Score(frustration) in the same call; those extras are not scored here.
- Cost is estimated from reported usage tokens. Jev uses the published $0.042 / M input list price (output free). The LLM line is a rough OpenRouter estimate; actual $/MTok depends on the chosen model.
- High-confidence accuracy is only reported when the model returns a Choice confidence (Jev, and the dry-run heuristic).
