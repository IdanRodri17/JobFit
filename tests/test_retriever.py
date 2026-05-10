"""Tests for the V3 portfolio retriever (retrieval/portfolio_retriever.py).

These tests are FREE — they exercise vector similarity in ChromaDB,
not LLM calls. No opt-in env var needed; they run on every push that
has ChromaDB built.

Auto-skipped on fresh clones or in CI where data/chroma_db doesn't
exist yet, so the suite stays green even before ingestion has run.

Two test classes:
  TestRetrieverContract — basic shape/non-empty/no-crash guarantees
  TestRetrieverHitRate  — for known-relevant queries, expected keywords
                          appear in the retrieved context

Hit-rate is the V5 evaluation metric in miniature: "for query X, did
the right chunk show up?" These tests pin down a few high-confidence
expectations so a regression in chunking, embedding, or ChromaDB
config gets caught immediately.

If a hit-rate test fails, suspect in order:
  1. Portfolio content drifted (file removed/renamed in data/portfolio)
  2. Chunking changed (chunk_size or overlap tweaked)
  3. Embedding model changed
  4. ChromaDB needs re-ingestion
"""
import pytest

from config.settings import settings
from retrieval.portfolio_retriever import get_relevant_context


# ─── Auto-skip if ChromaDB isn't built ─────────────────────
# Graceful on fresh clones and CI without ingestion.
requires_chroma = pytest.mark.skipif(
    not settings.chroma_persist_dir.exists(),
    reason=(
        f"ChromaDB not found at {settings.chroma_persist_dir}. "
        "Run `python -m ingestion.portfolio_ingest` first."
    ),
)


# ─── Contract checks ───────────────────────────────────────
@requires_chroma
class TestRetrieverContract:
    """The retriever must always return a non-empty string for any
    plausible JD, and never crash on edge inputs.
    """

    def test_returns_string(self):
        result = get_relevant_context("Python developer with ML experience")
        assert isinstance(result, str)

    def test_returns_non_empty_for_realistic_jd(self, sample_jd):
        result = get_relevant_context(sample_jd)
        assert len(result) > 0

    def test_handles_short_query(self):
        # Even minimal queries shouldn't crash — they may return less
        # relevant results, but the contract holds.
        result = get_relevant_context("Python")
        assert isinstance(result, str)


# ─── Hit-rate ──────────────────────────────────────────────
@requires_chroma
class TestRetrieverHitRate:
    """For queries that are well-supported by the portfolio, verify
    the right keywords surface in the returned context.

    NOTE: keywords below are picked to match Idan's portfolio
    (Multi-Source RAG Hub mentions RAG + LangGraph; the CV mentions
    Python). If your portfolio content changes, update these accordingly.
    """

    @pytest.mark.parametrize(
        ("query", "expected_keyword"),
        [
            ("RAG retrieval-augmented generation pipelines", "RAG"),
            ("LangGraph agentic workflow orchestration",     "LangGraph"),
            ("Python developer experience",                   "Python"),
        ],
    )
    def test_query_retrieves_expected_keyword(self, query, expected_keyword):
        context = get_relevant_context(query)
        assert expected_keyword.lower() in context.lower(), (
            f"Query {query!r} should retrieve a chunk containing "
            f"{expected_keyword!r}; got first 200 chars: {context[:200]!r}"
        )
