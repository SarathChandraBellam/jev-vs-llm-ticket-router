"""Shared department labels and routing criteria.

Both Jev Choice criteria and the LLM structured-output instructions
use this map so the comparison is on the same decision problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Department = Literal["billing", "technical", "shipping", "account", "sales"]

DEPARTMENTS: tuple[Department, ...] = (
    "billing",
    "technical",
    "shipping",
    "account",
    "sales",
)

DEPARTMENT_CRITERIA: dict[Department, str] = {
    "billing": (
        "Payments, charges, refunds, invoices, failed cards, taxes, "
        "promo codes, and subscription billing or auto-renewals."
    ),
    "technical": (
        "Bugs, outages, errors, crashes, integrations, webhooks, "
        "API issues, and product-not-working reports."
    ),
    "shipping": (
        "Delivery, tracking, lost or delayed packages, damaged shipments, "
        "and shipping address changes."
    ),
    "account": (
        "Login, password, permissions, profile details, two-factor "
        "authentication, and account access or lockouts."
    ),
    "sales": (
        "Pricing, upgrades, plan comparisons, demos, new accounts, "
        "and questions about buying or changing a plan."
    ),
}

ROUTING_INSTRUCTIONS = (
    "Which department should handle `ticket_text`? "
    "Pick the single primary owner. If the ticket mentions more than one "
    "issue, choose the department that should take first action."
)

URGENCY_INSTRUCTIONS = (
    "Does `ticket_text` convey urgency that needs same-day attention "
    "(outage, blocked work, lost package in transit today, locked out now)?"
)

FRUSTRATION_CRITERIA = (
    "Calm, just stating facts.",
    "Frustrated but civil.",
    "Very angry or using strong language.",
)

FRUSTRATION_INSTRUCTIONS = (
    "How frustrated does the customer appear in `ticket_text`?"
)

LLM_SYSTEM_PROMPT = """You route customer support tickets to exactly one department.

Pick the department that should own the ticket. If the ticket spans multiple
issues, choose the department that should take first action.

Departments:
- billing: Payments, charges, refunds, invoices, failed cards, taxes, promo codes, and subscription billing or auto-renewals.
- technical: Bugs, outages, errors, crashes, integrations, webhooks, API issues, and product-not-working reports.
- shipping: Delivery, tracking, lost or delayed packages, damaged shipments, and shipping address changes.
- account: Login, password, permissions, profile details, two-factor authentication, and account access or lockouts.
- sales: Pricing, upgrades, plan comparisons, demos, new accounts, and questions about buying or changing a plan.

Return only the structured department label."""


def department_schema() -> dict:
    """JSON Schema used by the LLM structured-output baseline."""
    return classification_schema(DEFAULT_TAXONOMY)


@dataclass(frozen=True)
class Taxonomy:
    """Label set + instructions shared by Jev Choice and the LLM baseline."""

    labels: tuple[str, ...]
    criteria: dict[str, str]
    instructions: str
    llm_system_prompt: str
    choice_key: str = "department"
    schema_name: str = "department_route"


def classification_schema(taxonomy: Taxonomy) -> dict:
    return {
        "type": "object",
        "properties": {
            taxonomy.choice_key: {
                "type": "string",
                "enum": list(taxonomy.labels),
                "description": taxonomy.instructions,
            }
        },
        "required": [taxonomy.choice_key],
        "additionalProperties": False,
    }


DEFAULT_TAXONOMY = Taxonomy(
    labels=DEPARTMENTS,
    criteria=dict(DEPARTMENT_CRITERIA),
    instructions=ROUTING_INSTRUCTIONS,
    llm_system_prompt=LLM_SYSTEM_PROMPT,
    choice_key="department",
    schema_name="department_route",
)


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    if path is None:
        return DEFAULT_TAXONOMY
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    criteria = dict(payload["criteria"])
    labels = tuple(payload.get("labels") or criteria.keys())
    if set(labels) != set(criteria):
        raise ValueError(f"{path}: labels and criteria keys must match")
    return Taxonomy(
        labels=labels,
        criteria=criteria,
        instructions=str(payload["instructions"]),
        llm_system_prompt=str(payload["llm_system_prompt"]),
        choice_key=str(payload.get("choice_key") or "department"),
        schema_name=str(payload.get("schema_name") or "department_route"),
    )
