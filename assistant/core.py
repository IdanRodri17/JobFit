"""V2/V3/V5 orchestration layer: router + rewriter + filtered retrieval + dispatch.

This module is the "glue" that turns the independent chains into a
coherent system. It does exactly four things:

  1. Classifies the user's intent via the V2 router.
  2. Rewrites (jd + request) into a focused retrieval query (V5).
  3. Retrieves candidate context with intent-aware category filtering
     (V5) — different chains see different portfolio slices.
  4. Maps each intent to its handler via dict dispatch.

Smoke test:
    python -m assistant.core
    (pre-requisite: run `python -m ingestion.portfolio_ingest` once first)
"""

from typing import Callable

from assistant.chains.cover_letter import generate_cover_letter
from assistant.chains.fit_analyzer import analyze_fit
from assistant.chains.interview_prep import prepare_interview
from assistant.chains.query_rewriter import rewrite_query
from assistant.router import classify_intent
from models.schemas import (
    CoverLetter,
    FitReport,
    IntentClassification,
    InterviewPrep,
)
from retrieval.portfolio_retriever import Category, get_relevant_context

# ─── Dispatch Table ────────────────────────────────────────────────
HandlerFunc = Callable[[str, str], FitReport | CoverLetter | InterviewPrep]

HANDLERS: dict[str, HandlerFunc] = {
    "analyze_fit": analyze_fit,
    "generate_cover_letter": generate_cover_letter,
    "interview_prep": prepare_interview,
}


# ─── V5: intent → category filter ──────────────────────────────────
# Each chain is best served by a different slice of the portfolio.
#   - analyze_fit needs comprehensive coverage (skills + languages
#     in cv.md + concrete project evidence). No filter.
#   - generate_cover_letter wants concrete project examples; CV
#     boilerplate (education, languages section) just dilutes top-k.
#   - interview_prep references real project work; same logic.
#
# Missing-key default via dict.get() is None (no filter), so adding
# a new handler later without an entry here gracefully gets every
# chunk — explicit opt-in to filtering.
INTENT_TO_CATEGORIES: dict[str, list[Category] | None] = {
    "analyze_fit": None,  # all signal welcome
    "generate_cover_letter": ["projects"],
    "interview_prep": ["projects"],
}


# ─── Public API ────────────────────────────────────────────────────
def process_request(
    jd_text: str,
    user_request: str,
) -> tuple[IntentClassification, FitReport | CoverLetter | InterviewPrep | str]:
    """Route a user request to the correct specialized handler.

    Args:
        jd_text: Raw text of the job posting.
        user_request: What the user wants to do.

    Returns:
        A tuple of (IntentClassification, handler_result).
    """
    # Step 1: classify the user's intent
    classification = classify_intent(user_request)

    # Step 2: dispatch to the matching handler (or explain absence)
    handler = HANDLERS.get(classification.intent)
    if handler is None:
        result = (
            f"The '{classification.intent}' handler is not implemented in V2. "
            "Available handlers: analyze_fit, generate_cover_letter, "
            "interview_prep. (tailor_resume and company_research are "
            "planned for a later version.)"
        )
        return classification, result

    # Step 3: rewrite the query (V5), then retrieve with intent-aware
    # category filtering (V5). The rewriter handles 'what to look for';
    # the category filter handles 'where to look'.
    retrieval_query = rewrite_query(jd_text, user_request)
    categories = INTENT_TO_CATEGORIES.get(classification.intent)
    candidate_context = get_relevant_context(
        retrieval_query.query,
        categories=categories,
    )

    # Step 4: run the handler with the JD and retrieved context
    result = handler(jd_text, candidate_context)
    return classification, result


# ─── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_jd = """\
AI Developer — Elad Systems (Tel Aviv, Hybrid)

Required:
- 3+ years of Python development
- Experience with LangChain and RAG systems
- PostgreSQL and pgvector
- FastAPI and Docker

Nice to have:
- LangGraph experience
- Hebrew language skills
"""

    test_requests = [
        "Should I apply to this role?",
        "Write me a cover letter for this position.",
        "What might they ask me in the interview?",
        "Tailor my resume for this JD.",
    ]

    for user_request in test_requests:
        print("═" * 70)
        print(f"USER REQUEST: {user_request!r}")
        print("═" * 70)

        classification, result = process_request(sample_jd, user_request)

        print(f"\n→ Routed to: {classification.intent}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Reasoning:  {classification.reasoning}\n")

        if isinstance(result, str):
            print(f"⚠️  {result}\n")
        else:
            schema_name = type(result).__name__
            preview = result.model_dump_json(indent=2)
            if len(preview) > 600:
                preview = preview[:600] + "\n  ...[truncated]"
            print(f"✓ Result ({schema_name}):")
            print(preview)
            print()
