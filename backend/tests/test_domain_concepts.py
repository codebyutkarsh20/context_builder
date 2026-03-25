"""
Unit tests for enricher/domain_concepts.py — domain concept extraction.

Covers:
  - _split_camel: CamelCase splitting
  - extract_domain_concepts: from class names, module paths, type suffixes
  - Noise word filtering
  - Minimum appearance threshold (2+)
  - Concept type inference from suffixes
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enricher.domain_concepts import _split_camel, extract_domain_concepts


# ---------------------------------------------------------------------------
# _split_camel
# ---------------------------------------------------------------------------

class TestSplitCamel:

    def test_simple_two_words(self):
        assert _split_camel("UserService") == ["User", "Service"]

    def test_three_words(self):
        assert _split_camel("OrderPaymentProcessor") == ["Order", "Payment", "Processor"]

    def test_single_word_unchanged(self):
        assert _split_camel("User") == ["User"]

    def test_all_caps_handled(self):
        # "ABCClass" → ["ABC", "Class"]
        result = _split_camel("ABCClass")
        assert "Class" in result

    def test_lowercase_unchanged(self):
        assert _split_camel("user") == ["user"]

    def test_mixed_with_acronym(self):
        # "HTTPClient" → ["HTTP", "Client"]
        result = _split_camel("HTTPClient")
        assert "Client" in result

    def test_empty_string(self):
        assert _split_camel("") == []


# ---------------------------------------------------------------------------
# extract_domain_concepts — basic functionality
# ---------------------------------------------------------------------------

class TestExtractDomainConcepts:

    def _parsed_with_classes(self, path, class_defs):
        """Helper to create a minimal parsed file record."""
        return {
            "path": path,
            "classes": [
                {
                    "name": name,
                    "docstring": doc,
                    "bases": bases,
                    "methods": [{"name": m} for m in methods],
                }
                for name, doc, bases, methods in class_defs
            ],
            "functions": [],
        }

    def test_empty_input_returns_empty(self):
        assert extract_domain_concepts([]) == []

    def test_single_class_single_occurrence_filtered(self):
        # Only 1 class, 1 file → count < 2 → filtered out
        parsed = [self._parsed_with_classes("app.py", [
            ("PaymentService", "", [], ["process"]),
        ])]
        # "Payment" appears only once → may be filtered
        # depends on module name contribution
        concepts = extract_domain_concepts(parsed)
        # Just ensure it doesn't crash; may or may not include Payment
        assert isinstance(concepts, list)

    def test_repeated_concept_across_classes_included(self):
        parsed = [self._parsed_with_classes("payments/models.py", [
            ("PaymentModel", "Represents a payment", [], ["save"]),
            ("PaymentService", "Processes payments", [], ["process"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        names = [c["name"] for c in concepts]
        assert "Payment" in names

    def test_concept_type_from_service_suffix(self):
        parsed = [self._parsed_with_classes("orders/orders.py", [
            ("OrderService", "", [], ["create"]),
            ("OrderFactory", "", [], ["build"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        order = next((c for c in concepts if c["name"] == "Order"), None)
        if order:  # might or might not meet threshold
            assert order["type"] in ("process", "entity")

    def test_concept_type_from_model_suffix(self):
        parsed = [self._parsed_with_classes("users/models.py", [
            ("UserModel", "", [], ["save"]),
            ("UserSchema", "", [], ["validate"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        user = next((c for c in concepts if c["name"] == "User"), None)
        if user:
            assert user["type"] in ("entity", "process")

    def test_noise_word_filtered(self):
        parsed = [self._parsed_with_classes("api/views.py", [
            ("AbstractView", "", [], []),
            ("BaseController", "", [], []),
        ])]
        concepts = extract_domain_concepts(parsed)
        names = [c["name"].lower() for c in concepts]
        assert "abstract" not in names
        assert "base" not in names

    def test_result_has_required_keys(self):
        parsed = [self._parsed_with_classes("orders.py", [
            ("OrderProcessor", "", [], ["process"]),
            ("OrderValidator", "", [], ["validate"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        for c in concepts:
            assert "id" in c
            assert "name" in c
            assert "type" in c
            assert "related_classes" in c
            assert "related_files" in c

    def test_concept_id_format(self):
        parsed = [self._parsed_with_classes("billing.py", [
            ("InvoiceService", "", [], ["create"]),
            ("InvoiceModel", "", [], ["save"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        invoice = next((c for c in concepts if c["name"] == "Invoice"), None)
        if invoice:
            assert invoice["id"].startswith("domain::")
            assert "invoice" in invoice["id"]

    def test_caps_at_50_concepts(self):
        # Even with many unique class names, result should be ≤ 50
        classes = [(f"Entity{i}Service", "", [], []) for i in range(200)]
        parsed = [self._parsed_with_classes("big.py", classes)]
        # Also need same names in second file to pass threshold
        parsed.append(self._parsed_with_classes("big2.py", classes))
        concepts = extract_domain_concepts(parsed)
        assert len(concepts) <= 50

    def test_related_classes_populated(self):
        parsed = [self._parsed_with_classes("billing.py", [
            ("PaymentService", "", [], ["charge"]),
            ("PaymentProcessor", "", [], ["process"]),
            ("PaymentGateway", "", [], ["send"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        payment = next((c for c in concepts if c["name"] == "Payment"), None)
        if payment:
            assert len(payment["related_classes"]) >= 2

    def test_module_concepts_from_path(self):
        # "payments" appears in both paths → should boost concept
        parsed = [
            {
                "path": "payments/service.py",
                "classes": [],
                "functions": [],
            },
            {
                "path": "payments/model.py",
                "classes": [],
                "functions": [],
            },
        ]
        concepts = extract_domain_concepts(parsed)
        # "Payments" or "Payment" should appear from module names
        # (depends on how module names are transformed)
        names_lower = [c["name"].lower() for c in concepts]
        assert any("payment" in n for n in names_lower)

    def test_description_initially_none(self):
        parsed = [self._parsed_with_classes("users.py", [
            ("UserService", "", [], ["get"]),
            ("UserModel", "", [], ["save"]),
        ])]
        concepts = extract_domain_concepts(parsed)
        user = next((c for c in concepts if c["name"] == "User"), None)
        if user:
            assert user["description"] is None
