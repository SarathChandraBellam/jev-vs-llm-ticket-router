from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ticket_router.categories import DEPARTMENTS
from ticket_router.dataset import label_counts, load_tickets
from ticket_router.evaluate import (  # noqa: E402
    confusion_matrix,
    estimate_cost,
    evaluate,
    format_report,
    percentile,
    per_class_metrics,
)
from ticket_router.heuristic import classify_heuristic
from ticket_router.llm_classifier import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    LLMClassifier,
    openrouter_headers,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)
from ticket_router.types import ClassificationResult, Ticket


class PercentileTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)

    def test_single(self) -> None:
        self.assertEqual(percentile([10.0], 95), 10.0)

    def test_median_odd(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)

    def test_p95_interpolation(self) -> None:
        values = [float(i) for i in range(1, 21)]
        self.assertAlmostEqual(percentile(values, 95), 19.05, places=2)


class MetricsTests(unittest.TestCase):
    def test_perfect_confusion(self) -> None:
        labels = ("a", "b")
        matrix = confusion_matrix(["a", "b", "a"], ["a", "b", "a"], labels)
        self.assertEqual(matrix, [[2, 0], [0, 1]])
        rows = per_class_metrics(matrix, labels)
        self.assertEqual(rows[0].f1, 1.0)
        self.assertEqual(rows[1].support, 1)

    def test_off_diagonal(self) -> None:
        labels = ("billing", "technical")
        matrix = confusion_matrix(
            ["billing", "billing", "technical"],
            ["technical", "billing", "technical"],
            labels,
        )
        rows = {row.label: row for row in per_class_metrics(matrix, labels)}
        self.assertAlmostEqual(rows["billing"].precision, 1.0)
        self.assertAlmostEqual(rows["billing"].recall, 0.5)


class EvaluateReportTests(unittest.TestCase):
    def test_format_includes_table(self) -> None:
        tickets = [
            Ticket(id="1", text="refund please", label="billing"),
            Ticket(id="2", text="app crash", label="technical"),
        ]
        results = [
            ClassificationResult(label="billing", latency_ms=10, confidence=0.9, input_tokens=12),
            ClassificationResult(label="shipping", latency_ms=20, confidence=0.4, input_tokens=8),
        ]
        report = evaluate(tickets, [("Jev (jev-latest)", results, "jev-latest")])
        text = format_report(report)
        self.assertIn("| Model | Accuracy |", text)
        self.assertIn("Confusion matrix", text)
        self.assertIn("Average Choice confidence", text)
        self.assertIn("50.0%", text)
        llm_report = evaluate(
            tickets,
            [("LLM via OpenRouter (openai/gpt-4o-mini)", results, "openai/gpt-4o-mini")],
        )
        llm_text = format_report(llm_report)
        self.assertIn("OpenRouter", llm_text)
        self.assertIn("openrouter.ai/models", llm_text)


class OpenRouterConfigTests(unittest.TestCase):
    def test_prefers_openrouter_key_and_model(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "or-key",
            "OPENAI_API_KEY": "oa-key",
            "OPENROUTER_MODEL": "anthropic/claude-3.5-haiku",
            "OPENAI_MODEL": "gpt-4o-mini",
        }
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(resolve_llm_api_key(), "or-key")
            self.assertEqual(resolve_llm_model(), "anthropic/claude-3.5-haiku")

    def test_falls_back_to_openai_key_and_defaults_to_openrouter(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "oa-key",
            "OPENROUTER_MODEL": "",
            "OPENAI_MODEL": "",
            "OPENAI_BASE_URL": "",
        }
        with patch.dict("os.environ", env, clear=False):
            # empty strings should be treated as missing by resolve helpers
            self.assertEqual(resolve_llm_api_key(), "oa-key")
            self.assertEqual(resolve_llm_model(), DEFAULT_LLM_MODEL)
            self.assertEqual(resolve_llm_base_url(), DEFAULT_OPENROUTER_BASE_URL)
            self.assertTrue(DEFAULT_LLM_MODEL.startswith("openai/"))
            self.assertEqual(DEFAULT_OPENROUTER_BASE_URL, "https://openrouter.ai/api/v1")

    def test_attribution_headers(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_HTTP_REFERER": "https://example.test",
                "OPENROUTER_X_TITLE": "custom-title",
            },
            clear=False,
        ):
            headers = openrouter_headers()
            self.assertEqual(headers["HTTP-Referer"], "https://example.test")
            self.assertEqual(headers["X-Title"], "custom-title")
        with patch.dict(
            "os.environ",
            {"OPENROUTER_HTTP_REFERER": "", "OPENROUTER_X_TITLE": ""},
            clear=False,
        ):
            headers = openrouter_headers()
            self.assertNotIn("HTTP-Referer", headers)
            self.assertEqual(headers["X-Title"], "jev-vs-llm-ticket-router")

    def test_classifier_defaults_without_constructing_client(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "or-key",
                "OPENAI_API_KEY": "",
                "OPENROUTER_MODEL": "",
                "OPENAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENROUTER_HTTP_REFERER": "",
                "OPENROUTER_X_TITLE": "",
            },
            clear=False,
        ):
            clf = LLMClassifier()
            self.assertEqual(clf.base_url, DEFAULT_OPENROUTER_BASE_URL)
            self.assertEqual(clf.model, DEFAULT_LLM_MODEL)
            self.assertEqual(clf.api_key, "or-key")
            self.assertEqual(clf.default_headers["X-Title"], "jev-vs-llm-ticket-router")
            self.assertNotIn("HTTP-Referer", clf.default_headers)

    def test_openrouter_cost_note(self) -> None:
        cost = estimate_cost(
            name="LLM via OpenRouter (openai/gpt-4o-mini)",
            model="openai/gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        self.assertAlmostEqual(cost.usd, 0.15)
        self.assertIn("OpenRouter", cost.notes)
        self.assertIn("openrouter.ai/models", cost.notes)


class HeuristicTests(unittest.TestCase):
    def test_refund_routes_billing(self) -> None:
        result = classify_heuristic("Please refund the duplicate charge on invoice INV-1.")
        self.assertEqual(result.label, "billing")
        self.assertGreater(result.confidence or 0, 0.4)

    def test_tracking_routes_shipping(self) -> None:
        result = classify_heuristic("Where is my package? Tracking number 1Z999 went dark.")
        self.assertEqual(result.label, "shipping")

    def test_login_routes_account(self) -> None:
        result = classify_heuristic("Can't log in. Password reset email never arrives.")
        self.assertEqual(result.label, "account")


class DatasetTests(unittest.TestCase):
    def test_real_dataset_loads_and_is_balanced(self) -> None:
        tickets = load_tickets()
        self.assertGreaterEqual(len(tickets), 80)
        self.assertLessEqual(len(tickets), 120)
        counts = label_counts(tickets)
        self.assertEqual(set(counts), {"billing", "technical", "shipping", "account", "sales"})
        for label, count in counts.items():
            self.assertGreaterEqual(count, 15, msg=label)
        ids = [ticket.id for ticket in tickets]
        self.assertEqual(len(ids), len(set(ids)))
        prefix = load_tickets(limit=10)
        self.assertEqual(set(t.label for t in prefix), set(DEPARTMENTS))

    def test_limit_and_bad_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text(
                json.dumps({"id": "a", "text": "hello", "label": "billing"})
                + "\n"
                + json.dumps({"id": "b", "text": "x", "label": "nope"})
                + "\n",
                encoding="utf-8",
            )
            loaded = load_tickets(path, limit=1)
            self.assertEqual(len(loaded), 1)
            with self.assertRaises(ValueError):
                load_tickets(path)


if __name__ == "__main__":
    unittest.main()
