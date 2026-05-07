"""V2 chain: produce a structured fit report for a (JD, candidate) pair.

Composes three Runnables using LCEL — same shape as jd_parser and router:
    FIT_ANALYZER_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

Inputs:
    - jd_text: the raw job description (or stringified JobDescription)
    - candidate_context: text describing the candidate's background.
      In V2 this is hardcoded in assistant/core.py.
      In V3 this will be filled by retrieved chunks from ChromaDB via RAG.

Output:
    FitReport — overall_score, matched_skills, gap_skills, strengths,
    concerns, recommendation, and reasoning.

Smoke test:
    python -m assistant.chains.fit_analyzer
    (note: this calls OpenAI — cost is ~$0.002 with gpt-4o-mini)
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import FitReport
from prompts.templates import FIT_ANALYZER_PROMPT


# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=FitReport)


# ─── Model ─────────────────────────────────────────────────
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain: prompt → model → parser ────────────────────
fit_analyzer_chain = (
    FIT_ANALYZER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def analyze_fit(jd_text: str, candidate_context: str) -> FitReport:
    """Produce a structured fit report for a candidate against a JD.

    Args:
        jd_text: Raw text of a job posting.
        candidate_context: Text describing the candidate's background,
            projects, and skills.

    Returns:
        A validated FitReport instance.
    """
    return fit_analyzer_chain.invoke({
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
- Production ML deployment experience
"""

    # Hardcoded candidate context for V2 testing.
    # In V3, this will be replaced by retrieved chunks from your portfolio.
    sample_candidate = """\
Idan — AI Developer (Junior, ~1 year of relevant project experience)

Background:
- CS B.Sc. with ML specialization (Holon Institute of Technology)
- Currently completing the CyberPro AI Developer Bootcamp at ELAD Software
- Native Hebrew speaker; fluent English

Recent projects:
- Multi-Source RAG Knowledge Hub: production-grade RAG system built with
  FastAPI, PostgreSQL/pgvector, Redis, LangGraph for agentic orchestration,
  and a full Prometheus/Grafana monitoring stack with GitHub Actions CI/CD.
- ShelfGuard (ELAD hackathon): GPT-4o-based shelf gap detection using
  pgvector for product embeddings; led AI/Backend on a 4-person team.
- DocTor: hospital management system with PostgreSQL + SQLAlchemy ORM,
  CLI, FastAPI, and React frontend.
- PokerScan/PokerVision: YOLOv8 card detection deployed to Hugging Face/Netlify.

Tech stack:
- Python (advanced), FastAPI, PostgreSQL, pgvector, Redis, Docker,
  LangChain, LangGraph, GitHub Actions, Prometheus/Grafana
- ML/CV: YOLOv8, ResNet, transfer learning
"""

    print("⏳ Running fit analyzer on Elad Systems JD...\n")
    report = analyze_fit(sample_jd, sample_candidate)

    print(f"✓ Fit report generated\n")
    print(f"  Overall score:    {report.overall_score}/100")
    print(f"  Recommendation:   {report.recommendation}")
    print(f"\n  Matched skills:   {', '.join(report.matched_skills) or '(none)'}")
    print(f"  Gap skills:       {', '.join(report.gap_skills) or '(none)'}")
    print(f"\n  Strengths:")
    for s in report.strengths:
        print(f"    • {s}")
    print(f"\n  Concerns:")
    for c in report.concerns:
        print(f"    • {c}")
    print(f"\n  Reasoning: {report.reasoning}")
    print()
    print("→ Open https://smith.langchain.com (project: JobFit) for the trace")
