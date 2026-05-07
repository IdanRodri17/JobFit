"""V2 chain: produce a structured interview prep guide for a (JD, candidate) pair.

Same LCEL composition pattern as fit_analyzer and cover_letter:
    INTERVIEW_PREP_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

Inputs:
    - jd_text: the raw job description
    - candidate_context: text describing the candidate

Output:
    InterviewPrep — technical_questions, behavioral_questions,
    questions_to_ask_them. Each question is a QuestionAnswer object
    pairing the question with a suggested answer grounded in the
    candidate's real background.

Smoke test:
    python -m assistant.chains.interview_prep
    (note: this calls OpenAI — cost is ~$0.005 with gpt-4o-mini;
    interview prep generates more output than other chains)
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import InterviewPrep
from prompts.templates import INTERVIEW_PREP_PROMPT


# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=InterviewPrep)


# ─── Model ─────────────────────────────────────────────────
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain ─────────────────────────────────────────────
interview_prep_chain = (
    INTERVIEW_PREP_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def prepare_interview(jd_text: str, candidate_context: str) -> InterviewPrep:
    """Generate interview prep questions and suggested answers.

    Args:
        jd_text: Raw text of a job posting.
        candidate_context: Text describing the candidate's background.

    Returns:
        A validated InterviewPrep instance with technical questions,
        behavioral questions, and questions to ask the interviewer.
    """
    return interview_prep_chain.invoke({
        "jd_text": jd_text,
        "candidate_context": candidate_context,
    })


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    sample_jd = """\
AI Developer — Elad Systems (Tel Aviv, Hybrid)

Required:
- 3+ years of Python development
- Experience with LangChain and RAG systems
- Strong PostgreSQL skills (pgvector a plus)
- FastAPI for production APIs
- Docker

Responsibilities:
- Design and develop production RAG systems
- Build agentic workflows
- Integrate AI services with enterprise systems
"""

    sample_candidate = """\
Idan — AI Developer (Junior, ~1 year of relevant project experience)

Recent projects:
- Multi-Source RAG Knowledge Hub: production-grade RAG with FastAPI,
  PostgreSQL/pgvector, Redis, LangGraph for agentic orchestration,
  full Prometheus/Grafana monitoring stack with GitHub Actions CI/CD.
- ShelfGuard (ELAD hackathon, Apr 2026): GPT-4o-based shelf gap
  detection using pgvector for product embeddings; led AI/Backend
  on a 4-person team.
- DocTor: hospital management system with PostgreSQL + SQLAlchemy.

Tech stack: Python, FastAPI, PostgreSQL, pgvector, Redis, Docker,
LangChain, LangGraph, GitHub Actions, Prometheus/Grafana.
"""

    print("⏳ Generating interview prep for Elad Systems AI Developer role...\n")
    prep = prepare_interview(sample_jd, sample_candidate)

    print(f"✓ Interview prep generated\n")
    print("═" * 70)
    print("TECHNICAL QUESTIONS")
    print("═" * 70)
    for i, qa in enumerate(prep.technical_questions, 1):
        print(f"\nQ{i}. {qa.question}")
        print(f"\n    Suggested answer:")
        print(f"    {qa.suggested_answer}")
        print(f"\n    Reference: {qa.relevant_experience}")

    print("\n" + "═" * 70)
    print("BEHAVIORAL QUESTIONS")
    print("═" * 70)
    for i, qa in enumerate(prep.behavioral_questions, 1):
        print(f"\nQ{i}. {qa.question}")
        print(f"\n    Suggested answer:")
        print(f"    {qa.suggested_answer}")
        print(f"\n    Reference: {qa.relevant_experience}")

    print("\n" + "═" * 70)
    print("QUESTIONS TO ASK THE INTERVIEWER")
    print("═" * 70)
    for i, q in enumerate(prep.questions_to_ask_them, 1):
        print(f"  {i}. {q}")

    print()
    print("→ Open https://smith.langchain.com (project: JobFit) for the trace")
