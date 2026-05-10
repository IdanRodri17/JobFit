"""Tests for the V3 portfolio retriever (retrieval/portfolio_retriever.py).

These tests need real OpenAI access (for embedding calls -- not LLM
completions) and a built ChromaDB. Both are true locally for anyone
with v3-complete tagged + a real .env; neither is true in CI.

Auto-skipped via @requires_real_openai whenever OPENAI_API_KEY is
missing or fake. Local cost is ~$0.00001 per embedding call -- free
for practical purposes.

WHY ENV VAR INSTEAD OF FILESYSTEM CHECK:
We originally gated on `settings.chroma_persist_dir.exists()`. That
gate fired AFTER the module-level `from retrieval.portfolio_retriever
import ...` line -- by which point ChromaDB's PersistentClient had
already auto-created the directory as a side effect of import. The
gate saw a freshly-created (empty) directory and let tests run, which
then tried real OpenAI calls and 401'd in CI on the fake key.

Lesson: gate on something an import can't accidentally create. Env
vars can't be conjured by a Python import; filesystem state can.
"""
import os

import pytest

from retrieval.portfolio_retriever import get_relevant_context


# --- Gate: real OpenAI access -----------------------------
def _has_real_openai_key() -> bool:
    """True if OPENAI_API_KEY looks like a real OpenAI key.

    Catches the two CI/fresh-clone scenarios:
      - missing/empty -> not real
      - 'fake' substring -> CI placeholder
    A genuinely-wrong real-shaped key would 401 at call time, which
    is informative (test fails loudly) rather than silently skipping.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    return key.startswith("sk-") and "fake" not in key.lower()


requires_real_openai = pytest.mark.skipif(
    not _has_real_openai_key(),
    reason=(
        "OPENAI_API_KEY is missing or fake -- retriever needs real "
        "embeddings. Set a real key in .env to run these locally."
    ),
)


# --- Contract checks --------------------------------------
@requires_real_openai
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
        # Even minimal queries shouldn't crash -- they may return less
        # relevant results, but the contract holds.
        result = get_relevant_context("Python")
        assert isinstance(result, str)


# --- Hit-rate ---------------------------------------------
@requires_real_openai
class TestRetrieverHitRate:
    """For queries that are well-supported by the portfolio, verify
    the right keywords surface in the returned context.
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
