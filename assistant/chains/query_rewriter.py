"""V5 chain: rewrite a user request + JD into a retrieval-optimized query.

This is a PRE-RETRIEVAL chain. Same LCEL composition shape as the
V1/V2 chains, but it produces an *intermediate* artifact (a search
query) rather than a final user-facing artifact.

Composition:
    QUERY_REWRITER_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

Why this exists (V3 findings recap):
    V3 used `f"{user_request}\\n\\n{jd_text}"` as the retrieval query
    — a naive concatenation. With k=4 retrieval and 12+ skill topics
    in a typical Elad JD, important candidate chunks were missing
    from the top-k. Skills the candidate HAD were flagged as gaps
    because the documenting chunks never landed in retrieval.

    The fix isn't bigger k (more noise, diluted relevance). The fix
    is a denser, more focused query. That's this chain.

Smoke test:
    python -m assistant.chains.query_rewriter
    (note: this calls OpenAI — cost is ~$0.0005 per case with gpt-4o-mini)
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import RetrievalQuery
from prompts.templates import QUERY_REWRITER_PROMPT

# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=RetrievalQuery)


# ─── Model ─────────────────────────────────────────────────
# Temperature=0.0 — same input must produce the same query, every
# time. Without reproducibility we can't measure regressions in
# run_eval.py later in V5.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain ─────────────────────────────────────────────
query_rewriter_chain = (
    QUERY_REWRITER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def rewrite_query(jd_text: str, user_request: str) -> RetrievalQuery:
    """Rewrite (JD + user_request) into a retrieval-optimized query.

    Args:
        jd_text: Raw text of a job posting.
        user_request: Natural-language description of what to do.

    Returns:
        A validated RetrievalQuery with the optimized query string
        and the reasoning behind the term choices.
    """
    return query_rewriter_chain.invoke(
        {
            "jd_text": jd_text,
            "user_request": user_request,
        }
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    sample_jd = """\
AI Developer — Elad Systems (Tel Aviv, Hybrid)

Required:
- 3+ years of Python development
- Experience with LangChain and RAG systems
- PostgreSQL and pgvector
- FastAPI and Docker

Nice to have:
- LangGraph experience for agentic workflows
- Hebrew language skills
- Production observability (LangSmith, Prometheus)
"""

    # Three intents → three different optimal queries.
    # Watch how the rewriter biases its output for each.
    test_cases = [
        "Should I apply for this role?",
        "Write me a cover letter for this position.",
        "What technical questions might they ask in the interview?",
    ]

    print("⏳ Running query rewriter on 3 intents...\n")
    for user_request in test_cases:
        print("═" * 70)
        print(f"USER REQUEST: {user_request!r}")
        print("═" * 70)
        result = rewrite_query(sample_jd, user_request)
        print(f"\nOptimized query:")
        print(f"  {result.query}")
        print(f"\nReasoning:")
        print(f"  {result.reasoning}")
        print()

    print("→ Open https://smith.langchain.com (project: JobFit) for the traces")
