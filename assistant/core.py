"""V2/V3/V5/V6 orchestration layer.

Two top-level entry points:

  process_request(jd, request)         -- V5 retrieval-only flow,
                                          kept for HTTP API compat.
  process_request_v6(jd, request)      -- V6 full action-selector
                                          dispatch (this is the new one).

V6 architecture:

    USER REQUEST
        │
        ▼
    select_action(request)  ← V6 action selector
        │
        ├── direct_answer → answer_directly()         → string
        ├── retrieval     → V5 retrieval flow         → Pydantic
        └── tool_use      → TOOLS[name](input)
                              → synthesize_tool_response → string

Smoke test:
    python -m assistant.core
    (pre-requisite: python -m ingestion.portfolio_ingest)
"""

from typing import Callable

from assistant.action_selector import select_action
from assistant.chains.cover_letter import generate_cover_letter
from assistant.chains.direct_answer import answer_directly
from assistant.chains.fit_analyzer import analyze_fit
from assistant.chains.interview_prep import prepare_interview
from assistant.chains.query_rewriter import rewrite_query
from assistant.router import classify_intent
from assistant.synthesizer import synthesize_tool_response
from assistant.tools import TOOLS
from models.schemas import (
    ActionDecision,
    CoverLetter,
    FitReport,
    IntentClassification,
    InterviewPrep,
)
from retrieval.portfolio_retriever import Category, get_relevant_context

# ─── V2 dispatch table ─────────────────────────────────────
HandlerFunc = Callable[[str, str], FitReport | CoverLetter | InterviewPrep]

HANDLERS: dict[str, HandlerFunc] = {
    "analyze_fit": analyze_fit,
    "generate_cover_letter": generate_cover_letter,
    "interview_prep": prepare_interview,
}

# ─── V5: intent → category filter ──────────────────────────
INTENT_TO_CATEGORIES: dict[str, list[Category] | None] = {
    "analyze_fit": None,
    "generate_cover_letter": ["projects"],
    "interview_prep": ["projects"],
}


# Type alias for V6's possible result shapes
V6Result = str | FitReport | CoverLetter | InterviewPrep


# ─── V5 entry point (HTTP API uses this) ───────────────────
def process_request(
    jd_text: str,
    user_request: str,
) -> tuple[IntentClassification, FitReport | CoverLetter | InterviewPrep | str]:
    """V5 retrieval-only flow. Preserved for HTTP API backward compat.

    Same behavior as before V6: classify intent → rewrite → retrieve →
    dispatch to handler. Returns (IntentClassification, result).
    """
    classification = classify_intent(user_request)
    handler = HANDLERS.get(classification.intent)
    if handler is None:
        result = (
            f"The '{classification.intent}' handler is not implemented in V2. "
            "Available handlers: analyze_fit, generate_cover_letter, "
            "interview_prep."
        )
        return classification, result

    retrieval_query = rewrite_query(jd_text, user_request)
    categories = INTENT_TO_CATEGORIES.get(classification.intent)
    candidate_context = get_relevant_context(
        retrieval_query.query,
        categories=categories,
    )
    result = handler(jd_text, candidate_context)
    return classification, result


# ─── V6 entry point (new) ──────────────────────────────────
def process_request_v6(
    jd_text: str,
    user_request: str,
) -> tuple[ActionDecision, IntentClassification | None, V6Result]:
    """V6 top-level dispatch: action selector → path-specific handler.

    Args:
        jd_text: Job description text. Used only by the retrieval path.
        user_request: What the user wants.

    Returns:
        A 3-tuple (action, classification, result):
          - action: ActionDecision from select_action()
          - classification: IntentClassification when action='retrieval',
                            else None
          - result: string (direct_answer / tool_use) or Pydantic
                    (retrieval path)
    """
    action = select_action(user_request)

    if action.action == "direct_answer":
        return action, None, answer_directly(user_request)

    if action.action == "tool_use":
        return action, None, _run_tool(action, user_request)

    # action.action == "retrieval" — fall through to V5 flow
    classification, result = process_request(jd_text, user_request)
    return action, classification, result


# ─── Tool-use helper ───────────────────────────────────────
def _run_tool(action: ActionDecision, user_request: str) -> str:
    """Dispatch action.tool_name with action.tool_input, then synthesize.

    Errors in tool dispatch (unknown tool, missing input) produce a
    user-facing error string rather than raising — the synthesizer
    contract is 'always return a string the user can read'.
    """
    if action.tool_name is None or action.tool_input is None:
        return (
            "The action selector chose tool_use but didn't provide a "
            "tool name or input. Try rephrasing your question."
        )

    tool_fn = TOOLS.get(action.tool_name)
    if tool_fn is None:
        return (
            f"Tool {action.tool_name!r} is registered in the schema but not in TOOLS."
        )

    tool_result = tool_fn(action.tool_input)
    return synthesize_tool_response(
        user_request=user_request,
        tool_name=action.tool_name,
        tool_result=tool_result,
    )


# ─── Smoke test — all three V6 paths end-to-end ────────────
if __name__ == "__main__":
    sample_jd = """\
AI Developer — Elad Systems (Tel Aviv, Hybrid)

Required:
- 3+ years Python development experience
- LangChain and RAG systems
- PostgreSQL and pgvector
- FastAPI and Docker

Nice to have:
- LangGraph for agentic workflows
- Hebrew language skills
"""

    test_cases = [
        # direct_answer path
        "How long should a cover letter be?",
        # retrieval path
        "Should I apply to this role?",
        # tool_use paths
        "How many years of Python experience do I have?",
        "What salary should I ask for as a junior AI developer?",
        "What's the latest news about Anthropic Claude?",
    ]

    for user_request in test_cases:
        print("═" * 70)
        print(f"REQUEST: {user_request!r}")
        print("═" * 70)

        action, classification, result = process_request_v6(sample_jd, user_request)

        print(f"\n→ Action: {action.action}", end="")
        if action.tool_name:
            print(f"  |  tool: {action.tool_name}  |  input: {action.tool_input!r}")
        else:
            print()
        print(f"  Reasoning: {action.reasoning}")

        if classification:
            print(
                f"\n  V2 intent: {classification.intent} "
                f"(confidence {classification.confidence:.2f})"
            )

        print(f"\n  RESULT ({type(result).__name__}):")
        if isinstance(result, str):
            print(f"  {result}")
        else:
            preview = result.model_dump_json(indent=2)
            if len(preview) > 600:
                preview = preview[:600] + "\n  ...[truncated]"
            print(preview)
        print()

    print("→ Open https://smith.langchain.com (project: JobFit) for the full traces")
