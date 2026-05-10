"""Shared pytest fixtures for the JobFit test suite.

Anything used by multiple test files lives here. Three categories:

1. The FastAPI TestClient (api_client) — so test_api.py can hit
   endpoints without spinning up a real server.

2. Sample data (sample_jd, *_payload) — realistic inputs reused
   across many tests. Defined once here, imported via fixture.

3. (Future) shared mock factories for real-LLM tests so test_router.py
   and test_chains.py can opt in/out of real OpenAI calls.

Why pytest fixtures vs. unittest setUp/tearDown?
  - Fixtures are explicit dependencies (declared as function params)
  - Fixtures support scopes (function/module/session) so expensive
    resources like the TestClient are built once per session
  - Fixtures compose: one fixture can depend on another (see
    process_request_payload depending on sample_jd)
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """A FastAPI TestClient bound to the JobFit app.

    Session-scoped: creating a TestClient triggers the FastAPI lifespan
    (ChromaDB existence check, logging configure). We only want to pay
    that cost once per test run.

    The `with` block ensures startup runs and shutdown is called
    cleanly even if a test raises.
    """
    from api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_jd() -> str:
    """A realistic JD that satisfies min_length=20.

    Mirrors a real Elad-Systems-style posting — short, structured, and
    naming concrete tech so the router has signal to classify against.
    """
    return (
        "AI Developer at Elad Systems (Tel Aviv).\n"
        "We are looking for an AI Developer to join our growing AI division.\n"
        "Responsibilities: design and develop production-grade RAG systems,\n"
        "agentic workflows with LangChain or LangGraph, and integrate AI\n"
        "services with enterprise systems.\n"
        "Required: Python, LLMs, vector databases, FastAPI.\n"
        "Nice to have: AWS, Docker, observability (LangSmith / Prometheus)."
    )


@pytest.fixture
def fit_request_payload(sample_jd) -> dict:
    """Payload for any endpoint that accepts JDPlusContextRequest
    (i.e. the three dedicated handlers)."""
    return {"jd_text": sample_jd}


@pytest.fixture
def process_request_payload(sample_jd) -> dict:
    """Payload for /api/process (jd_text + user_request)."""
    return {
        "jd_text": sample_jd,
        "user_request": "Should I apply for this role?",
    }
