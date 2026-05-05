"""Pydantic schemas for structured outputs from LangChain chains.

Each chain in JobFit returns one of these typed schemas instead of
free-form text. This is what makes JobFit a *tool* (producing artifacts)
rather than a *chatbot* (producing prose).

How these schemas reach the LLM:
    LangChain's PydanticOutputParser converts these classes into
    JSON Schema descriptions, which are injected into the prompt as
    format instructions. The Field(description=...) text becomes part
    of that prompt — clearer descriptions yield better extraction.

Smoke test:
    python -m models.schemas
"""
from typing import Literal

from pydantic import BaseModel, Field


# ─── V1: Job Description Parser ────────────────────────────
class JobDescription(BaseModel):
    """Structured representation of a parsed job description.

    Produced by the jd_parser chain in V1. All downstream chains
    (fit analyzer, cover letter generator, interview prep) consume
    this schema as their starting input.
    """

    title: str = Field(
        ...,
        description=(
            "The job title exactly as written in the posting "
            "(e.g. 'Senior AI Developer', 'Junior Backend Engineer')."
        ),
    )

    company_name: str = Field(
        ...,
        description=(
            "The hiring company's name. If the posting does not state it "
            "(e.g. anonymous recruiter listings), use 'Unknown'."
        ),
    )

    seniority_level: Literal["junior", "mid", "senior", "lead", "unknown"] = Field(
        ...,
        description=(
            "Seniority level inferred from the title and required years of "
            "experience. Use 'unknown' only if the posting gives no signal."
        ),
    )

    years_of_experience_required: int | None = Field(
        default=None,
        description=(
            "Minimum years of experience required, as an integer. "
            "Null if not specified."
        ),
    )

    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Hard requirements — must-have skills, technologies, or "
            "qualifications. Be specific: prefer concrete tools like "
            "'PyTorch', 'LangChain', 'PostgreSQL' over vague terms like "
            "'programming' or 'databases'."
        ),
    )

    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Preferred but not required skills, typically listed under "
            "'nice to have', 'bonus', 'plus', or 'preferred qualifications'."
        ),
    )

    key_responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "Main duties and responsibilities of the role, as 3-7 short "
            "bullet points. Each entry should be one responsibility."
        ),
    )

    location: str | None = Field(
        default=None,
        description=(
            "Geographic location (city and/or country, e.g. 'Tel Aviv, Israel'). "
            "Null if the posting is fully remote or location-agnostic."
        ),
    )

    work_arrangement: Literal["remote", "hybrid", "onsite", "unknown"] = Field(
        ...,
        description=(
            "Work arrangement: 'remote' (fully remote), 'hybrid' (mixed "
            "office/home), 'onsite' (must work from office), or 'unknown' "
            "if not specified."
        ),
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # Run with: python -m models.schemas
    import json

    # 1. Verify we can construct a valid instance
    example = JobDescription(
        title="Senior AI Developer",
        company_name="Elad Systems",
        seniority_level="senior",
        years_of_experience_required=5,
        required_skills=["Python", "LangChain", "RAG", "FastAPI", "PostgreSQL"],
        nice_to_have_skills=["LangGraph", "Hebrew", "Docker"],
        key_responsibilities=[
            "Design and build production RAG systems",
            "Lead AI architecture decisions",
            "Mentor junior developers on LLM best practices",
        ],
        location="Tel Aviv, Israel",
        work_arrangement="hybrid",
    )

    print("✓ JobDescription schema validated successfully\n")
    print("─── Serialized as JSON (what your chain returns) ───")
    print(example.model_dump_json(indent=2))

    print("\n─── JSON Schema (what gets injected into LLM prompts) ───")
    print(json.dumps(example.model_json_schema(), indent=2))
