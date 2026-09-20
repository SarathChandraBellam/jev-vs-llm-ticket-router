# How Jev works

Jev is TypeSafe's flagship **System One** model. A System One model does not generate prose for a human to read. You send **state** (the thing to judge) plus one or more typed **questions**. Jev evaluates every question against that state in parallel and returns structured answers your code can branch on.

Official docs: [docs.typesafe.ai](https://docs.typesafe.ai). Python SDK: [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/).

## Request shape

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
```

Body:

```json
{
  "model": "jev-latest",
  "state": { "ticket_text": "I was charged twice. Please refund the duplicate." },
  "questions": { }
}
```

- **model**: `jev-latest`, or pin a version such as `jev-1.13.0`.
- **state**: a string or JSON object. This benchmark uses `{"ticket_text": "..."}` so instructions can point at `` `ticket_text` ``.
- **questions**: a map of IDs you choose → Choice, Score, or Noul.

Question IDs are for your code. They are not sent to the model. Write the full question in `instructions`.

## The three primitives

| Type | Ask | Get back |
| --- | --- | --- |
| **Choice** | Which of these options? | `choice`, `probabilities`, `confidence` |
| **Score** | Where on this ordered scale? | `score`, `legend`, `probabilities`, `confidence` |
| **Noul** | Is this true? | `noul` in `[0, 1]` (probability of yes) |

Choice and Score constrain the answer to the options or levels you supplied. Noul has no separate confidence field; values near `0.5` are uncertain.

This benchmark's **accuracy metric uses only Choice** (department). The same request also asks Noul(urgency) and Score(frustration) to show **speculative fan-out**: extra questions are evaluated in parallel, add almost no latency, and cost only their question tokens. The harness records those extras but does not score them against gold labels.

## Minimal curl

```bash
curl -sS -X POST https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": { "ticket_text": "Help! My payouts have been failing for 3 days." },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle `ticket_text`? Pick the single primary owner.",
        "criteria": {
          "billing": "Payments, charges, refunds, invoices, subscriptions",
          "technical": "Bugs, outages, errors, integrations, API issues",
          "shipping": "Delivery, tracking, lost packages, address changes",
          "account": "Login, password, permissions, profile, 2FA",
          "sales": "Pricing, upgrades, demos, new accounts, plans"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does `ticket_text` convey urgency that needs same-day attention?"
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer appear in `ticket_text`?",
        "criteria": [
          "Calm, just stating facts.",
          "Frustrated but civil.",
          "Very angry or using strong language."
        ]
      }
    }
  }'
```

A successful Choice answer looks like:

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.91,
        "technical": 0.04,
        "shipping": 0.01,
        "account": 0.02,
        "sales": 0.02
      },
      "confidence": 0.88
    }
  },
  "usage": { "input_tokens": 312, "output_tokens": 48 }
}
```

## Python SDK

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={"ticket_text": "I was charged twice. Please refund the duplicate."},
        questions={
            "department": Choice(
                instructions="Which department should handle `ticket_text`?",
                criteria={
                    "billing": "Payments, charges, refunds, invoices, subscriptions",
                    "technical": "Bugs, outages, errors, integrations, API issues",
                    "shipping": "Delivery, tracking, lost packages, address changes",
                    "account": "Login, password, permissions, profile, 2FA",
                    "sales": "Pricing, upgrades, demos, new accounts, plans",
                },
            ),
            "urgent": Noul(
                instructions="Does `ticket_text` convey urgency that needs same-day attention?",
            ),
            "frustration": Score(
                instructions="How frustrated does the customer appear in `ticket_text`?",
                criteria=[
                    "Calm, just stating facts.",
                    "Frustrated but civil.",
                    "Very angry or using strong language.",
                ],
            ),
        },
    )

print(response.choices["department"].choice)
print(response.choices["department"].confidence)
print(response.nouls["urgent"].noul)
print(response.scores["frustration"].score)
print(response.usage.input_tokens)
```

`TypeSafeClient` reads `TYPESAFE_API_KEY`. Optional `TYPESAFE_MODEL` / `TYPESAFE_BASE_URL` override the default model and API root. This repo wraps that call in `src/ticket_router/jev_classifier.py`. If the SDK cannot be imported, the same JSON body is posted with `httpx`.

## Pricing used in the harness

Jev: **$0.042 per million input tokens**, output free. The report estimates cost from `usage.input_tokens`. See [jevapi.dev](https://jevapi.dev/) for current list prices.
