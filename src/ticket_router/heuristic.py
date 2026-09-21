"""Keyword heuristic used for --dry-run when API keys are absent."""

from __future__ import annotations

import re
import time

from ticket_router.categories import DEPARTMENTS, Department
from ticket_router.types import ClassificationResult

# Weighted phrases. Longer / more specific phrases get higher weights so
# "shipping address" beats a generic "account" hit on the same ticket.
KEYWORD_WEIGHTS: dict[Department, tuple[tuple[str, float], ...]] = {
    "billing": (
        ("duplicate charge", 4.0),
        ("charged twice", 4.0),
        ("double billed", 4.0),
        ("refund", 3.5),
        ("invoice", 3.5),
        ("receipt", 2.0),
        ("declined", 2.5),
        ("card was declined", 3.5),
        ("subscription", 2.5),
        ("auto-renew", 3.0),
        ("autorenew", 3.0),
        ("charged", 2.5),
        ("payment failed", 3.5),
        ("declined card", 3.5),
        ("credit card", 2.0),
        ("promo code", 3.0),
        ("coupon", 2.5),
        ("tax", 2.0),
        ("vat", 2.0),
        ("billing", 3.5),
        ("overcharged", 3.5),
        ("prorat", 2.5),
        ("stripe", 2.0),
        ("paypal", 2.0),
        ("chargeback", 3.0),
        ("unsubscribed", 2.0),
        ("cancel my subscription", 3.0),
        ("still being billed", 4.0),
        ("wired the payment", 2.5),
    ),
    "technical": (
        ("500 error", 4.0),
        ("stack trace", 4.0),
        ("webhook", 3.5),
        ("api key", 2.5),
        ("api ", 2.5),
        ("integration", 3.0),
        ("outage", 3.5),
        ("downtime", 3.5),
        ("timeout", 3.0),
        ("crash", 3.0),
        ("bug", 3.0),
        ("error code", 3.0),
        ("http 4", 3.0),
        ("http 5", 3.0),
        ("sdk", 2.5),
        ("oauth callback", 3.0),
        ("rate limit", 3.0),
        ("null pointer", 3.5),
        ("not loading", 2.5),
        ("white screen", 3.0),
        ("console error", 3.0),
        ("cannot connect", 2.5),
        ("sync is stuck", 3.0),
        ("dashboard is blank", 3.0),
        ("feature flag", 2.0),
        ("graphql", 2.5),
        ("webhook signature", 3.5),
    ),
    "shipping": (
        ("tracking number", 4.0),
        ("tracking", 3.5),
        ("lost package", 4.0),
        ("package never arrived", 4.0),
        ("shipment never arrived", 4.0),
        ("delivery", 3.0),
        ("shipped", 2.5),
        ("shipment", 3.0),
        ("carrier", 2.5),
        ("fedex", 3.0),
        ("ups ", 3.0),
        ("usps", 3.0),
        ("dhl", 3.0),
        ("warehouse", 2.5),
        ("wrong address", 3.0),
        ("shipping address", 3.5),
        ("change the address", 3.0),
        ("damaged box", 3.5),
        ("damaged in transit", 4.0),
        ("out for delivery", 3.5),
        ("left at the", 2.5),
        ("porch", 2.0),
        ("eta", 2.0),
        ("fulfillment", 2.5),
        ("package", 2.5),
        ("parcel", 2.5),
        ("customs", 2.5),
        ("return label", 2.5),
    ),
    "account": (
        ("two-factor", 4.0),
        ("2fa", 4.0),
        ("password reset", 4.0),
        ("forgot my password", 3.5),
        ("reset link", 3.0),
        ("locked out", 3.5),
        ("can't log in", 3.5),
        ("cannot log in", 3.5),
        ("cant log in", 3.5),
        ("login", 3.0),
        ("log in", 3.0),
        ("sign in", 2.5),
        ("permissions", 3.0),
        ("role", 2.0),
        ("sso", 3.0),
        ("single sign-on", 3.0),
        ("profile", 2.0),
        ("display name", 2.5),
        ("email change", 3.0),
        ("change my email", 3.0),
        ("account recovery", 3.5),
        ("verification code", 3.0),
        ("authenticator", 3.0),
        ("deactivated", 2.5),
        ("invite teammate", 2.5),
        ("remove a user", 2.5),
        ("admin access", 2.5),
        ("session expired", 2.5),
    ),
    "sales": (
        ("enterprise plan", 4.0),
        ("upgrade", 3.0),
        ("downgrade", 2.5),
        ("pricing", 3.5),
        ("price list", 3.5),
        ("how much", 2.0),
        ("quote", 3.0),
        ("demo", 3.5),
        ("sales call", 3.5),
        ("new account", 3.0),
        ("annual plan", 2.5),
        ("seat", 2.5),
        ("volume discount", 3.5),
        ("nonprofit discount", 3.5),
        ("switch to the", 2.0),
        ("compare plans", 3.5),
        ("which plan", 3.0),
        ("trial", 2.0),
        ("procurement", 3.0),
        ("contract", 2.5),
        ("reseller", 3.0),
        ("onboarding package", 2.5),
        ("talk to sales", 4.0),
        ("buy extra", 3.0),
        ("add-on", 2.5),
    ),
}


def _scores(text: str) -> dict[str, float]:
    lowered = f" {text.lower()} "
    scores = {label: 0.0 for label in DEPARTMENTS}
    for label, weighted in KEYWORD_WEIGHTS.items():
        for phrase, weight in weighted:
            if phrase in lowered:
                scores[label] += weight
    return scores


def classify_heuristic(text: str) -> ClassificationResult:
    """Return a department guess from keyword overlap. No API calls."""
    started = time.perf_counter()
    scores = _scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1]
    total = sum(scores.values())
    if total <= 0:
        probabilities = {label: 1.0 / len(DEPARTMENTS) for label in DEPARTMENTS}
        confidence = 0.20
        label = "technical"
    else:
        probabilities = {label: score / total for label, score in scores.items()}
        margin = (top_score - second_score) / top_score if top_score else 0.0
        confidence = max(0.25, min(0.95, 0.45 + 0.5 * margin))
        label = top_label
    latency_ms = (time.perf_counter() - started) * 1000
    approx_tokens = max(1, len(text.split()))
    return ClassificationResult(
        label=label,
        latency_ms=latency_ms,
        confidence=confidence,
        probabilities=probabilities,
        input_tokens=approx_tokens,
        output_tokens=0,
        extras={"backend": "heuristic"},
    )


def looks_urgent(text: str) -> float:
    needles = (
        "asap",
        "urgent",
        "immediately",
        "right now",
        "blocked",
        "down",
        "can't work",
        "cannot work",
        "lost",
        "locked out",
        "today",
        "emergency",
    )
    lowered = text.lower()
    hits = sum(1 for needle in needles if needle in lowered)
    return min(0.95, 0.15 + 0.2 * hits)


def frustration_level(text: str) -> float:
    lowered = text.lower()
    angry = len(re.findall(r"\b(furious|unacceptable|ridiculous|lawsuit|angry)\b", lowered))
    angry += lowered.count("!")
    if angry >= 3:
        return 2.0
    if angry >= 1:
        return 1.0
    return 0.0
