"""Test suite for JobFit.

Layout mirrors the source: one test_*.py per concern.
  - test_api.py        -> FastAPI HTTP layer (mocked chains)
  - test_router.py     -> V2 IntentClassification (real LLM, labeled set)
  - test_chains.py     -> V2 chains (real LLM, schema-shape checks)
  - test_retriever.py  -> V3 portfolio retriever (no LLM, hit-rate)

Shared fixtures live in conftest.py.
Run all: `pytest`
Run one file: `pytest tests/test_api.py -v`
Run one test: `pytest tests/test_api.py::TestHealth::test_health_returns_200 -v`
"""
