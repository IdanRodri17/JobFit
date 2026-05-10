"""Tests for the V2 router (assistant/router.py:classify_intent).

These are REAL LLM tests — each test makes an OpenAI call and produces
a LangSmith trace. Gated by the @real_llm decorator so they're skipped
by default (and on CI), preventing accidental cost.

To run them:
    REAL_LLM=1 pytest tests/test_router.py -v

What's verified:
  - Clear-intent inputs produce the correct classification
  - Confidence on clear inputs is at least 0.7 (router-is-confident floor)
  - Ambiguous inputs return a *valid* IntentClassification rather than
    crashing — we don't pin down WHICH intent they pick (that's a V5
    evaluation question with a labeled dataset, not a unit-test concern)

Why test classify_intent directly and not via process_request?
The router is one LLM call. process_request runs the router AND the
dispatched chain — two calls plus retrieval. To measure router quality
we isolate the variable.

Test phrasings deliberately differ from the smoke-test cases in
assistant/router.py — we want to test generalization, not memorization.
"""
import os

import pytest

from assistant.router import classify_intent
from models.schemas import IntentClassification


# ─── Opt-in gate ───────────────────────────────────────────
# Skip by default. Tests run only when REAL_LLM=1 is set in the env.
real_llm = pytest.mark.skipif(
    os.getenv("REAL_LLM") != "1",
    reason="Real LLM tests are opt-in. Set REAL_LLM=1 to run them.",
)


# ─── Labeled test set ──────────────────────────────────────
# (user_request, expected_intent) — one per intent, plus an ambiguous block.
CLEAR_CASES = [
    ("Am I a good match for this role?",            "analyze_fit"),
    ("Draft a cover letter for me.",                "generate_cover_letter"),
    ("What interview questions should I expect?",   "interview_prep"),
    ("Rewrite my resume bullets for this posting.", "tailor_resume"),
    ("What do you know about Elad Systems?",        "company_research"),
]

AMBIGUOUS_INPUTS = [
    "Help me with this.",
    "I'm not sure what to do.",
    "Make this work for me.",
]

CONFIDENCE_FLOOR = 0.7  # Below this on a clear case = router is shaky.


# ─── Clear cases ───────────────────────────────────────────
@real_llm
class TestRouterClearCases:
    """Parametrized so each case shows as its own line in pytest output —
    easier to spot a single regression than scanning a monolithic test.
    """

    @pytest.mark.parametrize(("user_request", "expected_intent"), CLEAR_CASES)
    def test_classifies_clear_input(self, user_request, expected_intent):
        result = classify_intent(user_request)

        assert result.intent == expected_intent, (
            f"Misclassified {user_request!r}: expected {expected_intent}, "
            f"got {result.intent} (reasoning={result.reasoning!r})"
        )
        assert result.confidence >= CONFIDENCE_FLOOR, (
            f"Low confidence on clear input {user_request!r}: "
            f"got {result.confidence:.2f}, want >= {CONFIDENCE_FLOOR}"
        )


# ─── Ambiguous cases ───────────────────────────────────────
@real_llm
class TestRouterAmbiguous:
    """The router must still return a valid IntentClassification on
    ambiguous input — never raise, never return None. We verify the
    contract, not the specific intent picked.
    """

    @pytest.mark.parametrize("user_request", AMBIGUOUS_INPUTS)
    def test_returns_valid_classification(self, user_request):
        result = classify_intent(user_request)

        assert isinstance(result, IntentClassification)
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning, "Reasoning should be non-empty"
