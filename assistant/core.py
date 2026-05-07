"""V2 orchestration layer: router + dispatch + candidate context.

This module is the "glue" that turns the independent router and handler
chains into a coherent system. It does exactly three things:

  1. Defines CANDIDATE_CONTEXT — the hardcoded text describing the
     candidate. In V3 this will be replaced by RAG retrieval over a
     real portfolio in ChromaDB. The rest of this file will not change.

  2. Maps each intent to its handler chain via dict dispatch — a
     declarative routing table. Adding new handlers later is a one-line
     change.

  3. Exposes process_request() — the single public entry point. Given
     a JD and a user request, it routes to the right handler and
     returns the structured result.

Smoke test:
    python -m assistant.core
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


# ─── Candidate Context (V2 — hardcoded; V3 — RAG-retrieved) ────────
# This is the "candidate side" of every (JD, candidate) pair. In V2 it
# is a single string passed to every handler. In V3 we will replace
# this with a retriever that fetches the most relevant portfolio
# chunks per query — but the handler chains will not change, because
# they consume `candidate_context` as a string either way.
CANDIDATE_CONTEXT = """\
Idan — AI Developer

Background:
- B.Sc. in Computer Science with Machine Learning specialization
  (Holon Institute of Technology, Israel)
- Currently completing the CyberPro AI Developer Bootcamp at ELAD Software
- Native Hebrew speaker; fluent English
- ~1 year of relevant project experience in production AI development

Recent flagship projects:

1. Multi-Source RAG Knowledge Hub
   Production-grade Retrieval-Augmented Generation system. Built across
   four phases: (1) FastAPI + PostgreSQL/pgvector + Redis backend;
   (2) LangGraph agentic orchestration with router/retriever/grader/
   generator nodes and a retry loop; (3) LLM provider abstraction layer
   supporting OpenAI and Ollama via factory pattern; (4) full test suite,
   GitHub Actions CI/CD pipeline, and Prometheus/Grafana monitoring stack.
   Stack: FastAPI, PostgreSQL, pgvector, Redis, LangGraph, GitHub Actions,
   Prometheus, Grafana, Docker.

2. ShelfGuard (ELAD Software 24-hour hackathon, April 2026)
   AI shelf gap detection system using GPT-4o to compare baseline vs.
   current shelf photos. pgvector stores product embeddings for
   similarity-based substitute suggestions (e.g., milk gap → dairy
   substitutes). Idan led AI/Backend on a 4-person team.
   Stack: FastAPI, React, PostgreSQL/pgvector, Redis, Docker, GPT-4o.

3. PokerScan / PokerVision
   YOLOv8 card detection app deployed to Hugging Face Spaces and Netlify.
   Stack: YOLOv8, Python, React, FastAPI.

4. DocTor (דוקתור)
   Hospital management system in active development. Stack: PostgreSQL,
   SQLAlchemy ORM, CLI, FastAPI, React frontend.

Tech stack summary:
- Languages: Python (advanced), Kotlin
- Backend: FastAPI, SQLAlchemy
- Databases: PostgreSQL, pgvector, Redis, MongoDB
- AI/ML: LangChain, LangGraph, OpenAI, Ollama, YOLOv8, ResNet,
  transfer learning
- Infra: Docker, GitHub Actions, Prometheus, Grafana, Render
- Other: React (basics), n8n automation
"""


# ─── Dispatch Table ────────────────────────────────────────────────
# Maps each Intent to the function that handles it.
# Each handler has the signature: (jd_text: str, candidate_context: str) -> SomeSchema.
#
# Intents not yet implemented (tailor_resume, company_research) are
# absent from this dict; process_request handles them with a friendly
# "not implemented" message.
HandlerFunc = Callable[[str, str], FitReport | CoverLetter | InterviewPrep]

HANDLERS: dict[str, HandlerFunc] = {
    "analyze_fit": analyze_fit,
    "generate_cover_letter": generate_cover_letter,
    "interview_prep": prepare_interview,
}


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

    # Step 3: run the handler with the JD and candidate context
    result = handler(jd_text, CANDIDATE_CONTEXT)
    return classification, result


# ─── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # End-to-end test: same JD, three different user requests.
    # Each call: 1 router call + 1 handler call ≈ 2 OpenAI requests.
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
        "Tailor my resume for this JD.",  # not implemented — should fall through
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
            # Unimplemented intent — friendly message
            print(f"⚠️  {result}\n")
        else:
            # Structured result — show its type and a short preview
            schema_name = type(result).__name__
            preview = result.model_dump_json(indent=2)
            if len(preview) > 600:
                preview = preview[:600] + "\n  ...[truncated]"
            print(f"✓ Result ({schema_name}):")
            print(preview)
            print()
