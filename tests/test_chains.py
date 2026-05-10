"""Tests for the V2 chains (assistant/chains/*.py).

Like test_router.py these are REAL LLM tests, but they're EXPENSIVE
because each test does retrieval + a chain LLM call. Total cost when
fully opted in: ~$0.008 across 3 chains.

Gated by TWO decorators stacked:
  - @real_llm        — opt-in via REAL_LLM=1
  - @requires_chroma — auto-skip if ChromaDB isn't built

To run:
    REAL_LLM=1 pytest tests/test_chains.py -v

What's verified per chain:
  - The chain returned the right Pydantic schema
  - Required fields are populated (non-empty strings, list lengths >= 1)
  - Literal values are in the documented sets

What's NOT verified: specific content. LLM jitter makes assertions
like "the cover letter should mention LangGraph" flaky. Content
quality is a V5 evaluation question with LLM-as-judge scoring.
"""
import os

import pytest

from assistant.chains.cover_letter import generate_cover_letter
from assistant.chains.fit_analyzer import analyze_fit
from assistant.chains.interview_prep import prepare_interview
from config.settings import settings
from models.schemas import CoverLetter, FitReport, InterviewPrep
from retrieval.portfolio_retriever import get_relevant_context


# ─── Gates ─────────────────────────────────────────────────
real_llm = pytest.mark.skipif(
    os.getenv("REAL_LLM") != "1",
    reason="Real LLM tests are opt-in. Set REAL_LLM=1 to run them.",
)

requires_chroma = pytest.mark.skipif(
    not settings.chroma_persist_dir.exists(),
    reason=(
        f"ChromaDB not found at {settings.chroma_persist_dir}. "
        "Run `python -m ingestion.portfolio_ingest` first."
    ),
)


# ─── FitAnalyzer ───────────────────────────────────────────
@real_llm
@requires_chroma
class TestFitAnalyzerChain:
    def test_returns_valid_fit_report(self, sample_jd):
        context = get_relevant_context(sample_jd)
        result = analyze_fit(sample_jd, context)

        assert isinstance(result, FitReport)
        assert 0 <= result.overall_score <= 100
        assert result.recommendation in {
            "strong_apply", "apply", "stretch", "skip"
        }
        assert result.reasoning, "reasoning should be non-empty"
        # matched_skills / gap_skills lists may be empty for low-fit
        # JDs — we don't pin those down.


# ─── CoverLetter ───────────────────────────────────────────
@real_llm
@requires_chroma
class TestCoverLetterChain:
    def test_returns_valid_cover_letter(self, sample_jd):
        context = get_relevant_context(sample_jd)
        result = generate_cover_letter(sample_jd, context)

        assert isinstance(result, CoverLetter)
        assert result.opening_paragraph, "opening_paragraph should be non-empty"
        assert result.closing_paragraph, "closing_paragraph should be non-empty"
        assert len(result.body_paragraphs) >= 1, "expected >=1 body paragraph"
        assert all(p.strip() for p in result.body_paragraphs), (
            "every body paragraph should be non-empty"
        )
        assert result.word_count > 0
        assert result.tone in {"formal", "conversational", "enthusiastic"}


# ─── InterviewPrep ─────────────────────────────────────────
@real_llm
@requires_chroma
class TestInterviewPrepChain:
    def test_returns_valid_interview_prep(self, sample_jd):
        context = get_relevant_context(sample_jd)
        result = prepare_interview(sample_jd, context)

        assert isinstance(result, InterviewPrep)
        assert len(result.technical_questions) >= 1, "expected >=1 technical Q"
        assert len(result.behavioral_questions) >= 1, "expected >=1 behavioral Q"
        assert len(result.questions_to_ask_them) >= 1, "expected >=1 question to ask"

        # Verify the QA pairs have substance (not empty strings).
        for qa in result.technical_questions + result.behavioral_questions:
            assert qa.question, "QA.question should be non-empty"
            assert qa.suggested_answer, "QA.suggested_answer should be non-empty"
            assert qa.relevant_experience, "QA.relevant_experience should be non-empty"
