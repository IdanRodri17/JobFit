"""V2/V3 orchestration layer: router + dispatch + retrieval.

This module is the "glue" that turns the independent router and handler
chains into a coherent system. It does exactly three things:

  1. Retrieves candidate context per-request via the V3 portfolio
     retriever (ChromaDB-backed). In V2 this was a hardcoded constant
     — now it's dynamic and query-aware.

  2. Maps each intent to its handler chain via dict dispatch — a
     declarative routing table. Adding new handlers later is a
     one-line change.

  3. Exposes process_request() — the single public entry point. Given
     a JD and a user request, it routes to the right handler with
     freshly-retrieved context and returns the structured result.

Smoke test:
    python -m assistant.core
    (pre-requisite: run `python -m ingestion.portfolio_ingest` once first)
"""
from typing import Callable

from assistant.chains.cover_letter import generate_cover_letter
from assistant.chains.fit_analyzer import analyze_fit
from assistant.chains.interview_prep import prepare_interview
from assistant.router import classify_intent
from models.schemas import (
    CoverLetter,
    FitReport,
    IntentClassification,
    InterviewPrep,
)
from retrieval.portfolio_retriever import get_relevant_context


# ─── Dispatch Table ────────────────────────────────────────────────
# Maps each Intent to the function that handles it.
# Each handler has the signature: (jd_text: str, candidate_context: str) -> SomeSchema.
HandlerFunc = Callable[[str, str], FitReport | CoverLetter | InterviewPrep]

HANDLERS: dict[str, HandlerFunc] = {
    "analyze_fit": analyze_fit,
    "generate_cover_letter": generate_cover_letter,
    "interview_prep": prepare_interview,
}


# ─── Retrieval helper ──────────────────────────────────────────────
def _build_retrieval_query(jd_text: str, user_request: str) -> str:
    """Build the query string used to retrieve relevant portfolio chunks.

    V3 baseline: concatenate JD text and user request. The JD provides
    semantic grounding (skills, tech, role context), and the user
    request adds intent specificity. V5 will replace this with a
    dedicated query_rewriter chain.
    """
    return f"{user_request}\n\n{jd_text}"


# ─── Public API ────────────────────────────────────────────────────
def process_request(
    jd_text: str,
    user_request: str,
) -> tuple[IntentClassification, FitReport | CoverLetter | InterviewPrep | str]:
    """Route a user request to the correct specialized handler.

    Args:
        jd_text: Raw text of the job posting.
        user_request: What the user wants to do (e.g.
            "Should I apply?", "Write me a cover letter").

    Returns:
        A tuple of (IntentClassification, handler_result). The
        IntentClassification lets callers see the routing decision
        (intent + confidence + reasoning) alongside the actual output.
        For unimplemented intents, handler_result is a friendly
        explanation string.
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

    # Step 3: retrieve relevant candidate context from the portfolio
    # via ChromaDB. This is the V3 swap — V2 used a hardcoded constant.
    retrieval_query = _build_retrieval_query(jd_text, user_request)
    candidate_context = get_relevant_context(retrieval_query)

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
        "Tailor my resume for this JD.",  # not implemented — fallthrough
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
