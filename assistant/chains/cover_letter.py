"""V2 chain: generate a structured cover letter for a (JD, candidate) pair.

Same LCEL composition pattern as fit_analyzer:
    COVER_LETTER_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

Inputs:
    - jd_text: the raw job description
    - candidate_context: text describing the candidate (hardcoded in V2,
      RAG-retrieved in V3)

Output:
    CoverLetter — opening_paragraph, body_paragraphs, closing_paragraph,
    word_count, tone.

Smoke test:
    python -m assistant.chains.cover_letter
    (note: this calls OpenAI — cost is ~$0.003 with gpt-4o-mini)
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import CoverLetter
from prompts.templates import COVER_LETTER_PROMPT


# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=CoverLetter)


# ─── Model ─────────────────────────────────────────────────
# Cover letters benefit from a slightly higher temperature for natural
# prose variation, but we keep it at 0.0 here because the schema
# (with explicit calibration of tone/length) does the heavy lifting.
# Tunable later if outputs feel robotic.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain ─────────────────────────────────────────────
cover_letter_chain = (
    COVER_LETTER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def generate_cover_letter(jd_text: str, candidate_context: str) -> CoverLetter:
    """Generate a structured cover letter for a candidate applying to a JD.

    Args:
        jd_text: Raw text of a job posting.
        candidate_context: Text describing the candidate's background.

    Returns:
        A validated CoverLetter instance.
    """
    return cover_letter_chain.invoke({
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

Nice to have:
- LangGraph experience
- Hebrew language skills
"""

    sample_candidate = """\
Idan — AI Developer (Junior, ~1 year of relevant project experience)

Background:
- CS B.Sc. with ML specialization (Holon Institute of Technology)
- Currently completing the CyberPro AI Developer Bootcamp at ELAD Software
- Native Hebrew speaker

Recent projects:
- Multi-Source RAG Knowledge Hub: production-grade RAG with FastAPI,
  PostgreSQL/pgvector, Redis, LangGraph for agentic orchestration,
  full Prometheus/Grafana monitoring stack with GitHub Actions CI/CD.
- ShelfGuard (ELAD hackathon, Apr 2026): GPT-4o-based shelf gap
  detection using pgvector for product embeddings; led AI/Backend
  on a 4-person team.

Tech stack: Python, FastAPI, PostgreSQL, pgvector, Redis, Docker,
LangChain, LangGraph, GitHub Actions, Prometheus/Grafana.
"""

    print("⏳ Generating cover letter for Elad Systems...\n")
    letter = generate_cover_letter(sample_jd, sample_candidate)

    print(f"✓ Cover letter generated ({letter.word_count} words, tone: {letter.tone})\n")
    print("─" * 70)
    print(letter.opening_paragraph)
    print()
    for paragraph in letter.body_paragraphs:
        print(paragraph)
        print()
    print(letter.closing_paragraph)
    print("─" * 70)
    print()
    print("→ Open https://smith.langchain.com (project: JobFit) for the trace")
