"""Tests for the FastAPI HTTP layer.

These tests verify the WIRING, not the chains:
  - Endpoints exist at the right paths
  - Pydantic validation rejects bad input (returns 422)
  - Response shapes match the documented schemas
  - Mocked happy-paths return the right structure and status

What these tests DON'T do: call real LLMs or real ChromaDB. Chain
quality and retrieval correctness live in test_chains.py and
test_retriever.py respectively. Each test file has one concern.

Running:
    pytest tests/test_api.py -v
"""

from unittest.mock import patch

from models.schemas import (
    CoverLetter,
    FitReport,
    IntentClassification,
)


# ─── Health ────────────────────────────────────────────────────────
class TestHealth:
    """Liveness check — must always be cheap and never depend on LLMs."""

    def test_health_returns_200(self, api_client):
        resp = api_client.get("/api/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, api_client):
        body = api_client.get("/api/health").json()
        assert body == {"status": "ok", "service": "JobFit API"}


# ─── Frontend ──────────────────────────────────────────────────────
class TestFrontend:
    """The HTML page is served by Jinja2 — verify it actually renders."""

    def test_root_returns_html(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_root_html_contains_form(self, api_client):
        # Smoke-test: the page actually renders the form structure
        # we expect, not just any HTML. If the template path or
        # static mount breaks, this catches it.
        body = api_client.get("/").text.lower()
        assert "<form" in body
        assert "jd_text" in body
        assert "user_request" in body


# ─── Validation ────────────────────────────────────────────────────
class TestProcessValidation:
    """Pydantic catches bad input BEFORE the route function runs.

    Status code is 422 (Unprocessable Entity) — that's the FastAPI /
    Pydantic convention for schema-validation failures, distinct from
    400 (Bad Request) which we'd use for semantically-rejected input.
    """

    def test_rejects_short_jd(self, api_client):
        resp = api_client.post(
            "/api/process",
            json={"jd_text": "too short", "user_request": "Should I apply?"},
        )
        assert resp.status_code == 422

    def test_rejects_short_user_request(self, api_client, sample_jd):
        resp = api_client.post(
            "/api/process",
            json={"jd_text": sample_jd, "user_request": "x"},
        )
        assert resp.status_code == 422

    def test_rejects_missing_user_request(self, api_client, sample_jd):
        resp = api_client.post("/api/process", json={"jd_text": sample_jd})
        assert resp.status_code == 422


# ─── Mocked happy-path ─────────────────────────────────────────────
class TestProcessHappyPath:
    """Verify response shape with a stubbed process_request.

    We mock at the IMPORT SITE in routes.process — not at the source
    in assistant.core — because Python's `from x import y` rebinds the
    name. Patching the source after import wouldn't reach the alias
    that routes.process actually calls.

    This is the single most-common pytest-mock gotcha; worth the
    teaching note in your presentation.
    """

    @staticmethod
    def _fake_classification(intent: str = "generate_cover_letter"):
        return IntentClassification(
            intent=intent,
            confidence=0.95,
            reasoning="Test fixture — bypassed router.",
        )

    @staticmethod
    def _fake_cover_letter():
        return CoverLetter(
            opening_paragraph="Dear hiring team,",
            body_paragraphs=["I have built RAG systems and agentic workflows."],
            closing_paragraph="Thank you for your consideration.",
            word_count=15,
            tone="enthusiastic",
        )

    def test_process_returns_classification_and_result(
        self, api_client, process_request_payload
    ):
        with patch("routes.process.process_request") as mock_proc:
            mock_proc.return_value = (
                self._fake_classification(),
                self._fake_cover_letter(),
            )
            resp = api_client.post("/api/process", json=process_request_payload)

        assert resp.status_code == 200
        body = resp.json()
        # Top-level shape
        assert "classification" in body
        assert "result" in body
        # Classification fields preserved end-to-end
        assert body["classification"]["intent"] == "generate_cover_letter"
        assert body["classification"]["confidence"] == 0.95
        # Result fields preserved end-to-end
        assert body["result"]["tone"] == "enthusiastic"
        assert body["result"]["word_count"] == 15

    def test_process_500_on_chain_error(self, api_client, process_request_payload):
        with patch("routes.process.process_request") as mock_proc:
            mock_proc.side_effect = RuntimeError("LLM provider down")
            resp = api_client.post("/api/process", json=process_request_payload)

        assert resp.status_code == 500
        assert "Processing failed" in resp.json()["detail"]


# ─── Dedicated handlers ────────────────────────────────────────────
class TestDedicatedHandlers:
    """The three /api/{analyze-fit,cover-letter,interview-prep} endpoints
    bypass the router and call their chain directly.

    Two contracts to verify per endpoint:
      1. Returns 200 with the bare schema shape
      2. Does NOT wrap in ProcessResponse — that's /api/process's job

    The 'no classification key' assertion is the architectural test
    for the V4-step-1 design decision: dedicated endpoints return bare
    schemas, only /api/process wraps in ProcessResponse.
    """

    def test_analyze_fit_returns_bare_fit_report(self, api_client, fit_request_payload):
        fake = FitReport(
            overall_score=80,
            matched_skills=["Python", "RAG"],
            gap_skills=["Kubernetes"],
            strengths=["Strong Python background."],
            concerns=["No K8s experience."],
            recommendation="apply",
            reasoning="Solid match on core stack.",
        )
        with patch(
            "routes.handlers.analyze_fit.get_relevant_context",
            return_value="mock context",
        ), patch(
            "routes.handlers.analyze_fit.analyze_fit",
            return_value=fake,
        ):
            resp = api_client.post("/api/analyze-fit", json=fit_request_payload)

        assert resp.status_code == 200
        body = resp.json()
        # Bare schema, not wrapped:
        assert "classification" not in body
        assert body["overall_score"] == 80
        assert body["recommendation"] == "apply"

    def test_cover_letter_returns_bare_cover_letter(
        self, api_client, fit_request_payload
    ):
        fake = CoverLetter(
            opening_paragraph="Dear team,",
            body_paragraphs=["..."],
            closing_paragraph="Thanks.",
            word_count=5,
            tone="formal",
        )
        with patch(
            "routes.handlers.cover_letter.get_relevant_context",
            return_value="mock context",
        ), patch(
            "routes.handlers.cover_letter.generate_cover_letter",
            return_value=fake,
        ):
            resp = api_client.post("/api/cover-letter", json=fit_request_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "classification" not in body
        assert body["tone"] == "formal"
