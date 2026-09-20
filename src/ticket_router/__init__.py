"""Support-ticket routing benchmark: TypeSafe Jev vs a structured-output LLM."""

from ticket_router.categories import DEPARTMENT_CRITERIA, DEPARTMENTS
from ticket_router.evaluate import BenchmarkReport, evaluate, format_report
from ticket_router.types import ClassificationResult, Ticket

__all__ = [
    "DEPARTMENT_CRITERIA",
    "DEPARTMENTS",
    "BenchmarkReport",
    "ClassificationResult",
    "Ticket",
    "evaluate",
    "format_report",
]
