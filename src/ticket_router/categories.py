"""Shared department labels and routing criteria.

Both Jev Choice criteria and the LLM structured-output instructions
use this map so the comparison is on the same decision problem.
"""

from __future__ import annotations

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
    return {
        "type": "object",
        "properties": {
            "department": {
                "type": "string",
                "enum": list(DEPARTMENTS),
                "description": ROUTING_INSTRUCTIONS,
            }
        },
        "required": ["department"],
        "additionalProperties": False,
    }
