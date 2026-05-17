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


# ─── V2: Intent Router ─────────────────────────────────────
# The five intents JobFit can handle. Defined as a Literal alias so
# we use the same set of strings everywhere (router output, dispatcher
# keys, presentation slides). Single source of truth.
Intent = Literal[
    "analyze_fit",
    "tailor_resume",
    "generate_cover_letter",
    "interview_prep",
    "company_research",
]


class IntentClassification(BaseModel):
    """Result of classifying a user's request against a job description.

    Produced by the router chain. The `intent` field is then used by
    the dispatcher to invoke the correct specialized handler chain.
    """

    intent: Intent = Field(
        ...,
        description=(
            "Which specialized handler to invoke:\n"
            "- 'analyze_fit': user wants a fit assessment (should I apply?)\n"
            "- 'tailor_resume': user wants resume bullets tailored to this JD\n"
            "- 'generate_cover_letter': user wants a cover letter for this role\n"
            "- 'interview_prep': user wants likely interview questions\n"
            "- 'company_research': user wants a brief on the company"
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the classification, between 0.0 and 1.0. "
            "Below 0.6 signals an ambiguous request that may need clarification."
        ),
    )

    reasoning: str = Field(
        ...,
        description="One short sentence explaining why this intent was chosen.",
    )


# ─── V2: Fit Analyzer ──────────────────────────────────────
class FitReport(BaseModel):
    """Structured assessment of how well a candidate matches a JD."""

    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Overall fit score from 0 (terrible match) to 100 (perfect match). "
            "Calibration: 80+ = strong apply, 60-79 = apply, "
            "40-59 = stretch, below 40 = skip."
        ),
    )

    matched_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills from the JD's requirements that the candidate clearly has, "
            "based on the candidate context provided."
        ),
    )

    gap_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills from the JD's requirements that the candidate does NOT "
            "have or has only weakly, based on the candidate context provided."
        ),
    )

    strengths: list[str] = Field(
        default_factory=list,
        description=(
            "Specific strengths beyond skill matching that make this candidate "
            "a strong fit (e.g. 'Has shipped production RAG systems', "
            "'Direct experience with the company's tech stack')."
        ),
    )

    concerns: list[str] = Field(
        default_factory=list,
        description=(
            "Honest concerns or red flags about the fit "
            "(e.g. 'Years of experience below requirement', "
            "'No prior enterprise experience')."
        ),
    )

    recommendation: Literal["strong_apply", "apply", "stretch", "skip"] = Field(
        ...,
        description=(
            "Final recommendation, derived from overall_score:\n"
            "- 'strong_apply' (80+): apply with high confidence\n"
            "- 'apply' (60-79): apply, prepare for some gaps\n"
            "- 'stretch' (40-59): consider only if highly motivated\n"
            "- 'skip' (<40): not a good fit, look elsewhere"
        ),
    )

    reasoning: str = Field(
        ...,
        description="2-3 sentence summary justifying the recommendation.",
    )


# ─── V2: Cover Letter Generator ────────────────────────────
class CoverLetter(BaseModel):
    """A structured cover letter, broken into reusable parts."""

    opening_paragraph: str = Field(
        ...,
        description=(
            "The opening paragraph: hook, the role being applied for, "
            "and one concrete reason this candidate is a strong match. "
            "2-4 sentences."
        ),
    )

    body_paragraphs: list[str] = Field(
        ...,
        description=(
            "1-3 body paragraphs. Each paragraph should highlight one "
            "concrete project, skill, or experience that maps directly to "
            "the JD's requirements. Be specific — name actual technologies "
            "and outcomes from the candidate context."
        ),
    )

    closing_paragraph: str = Field(
        ...,
        description=(
            "Closing paragraph: enthusiasm for the role, willingness to "
            "discuss further, polite sign-off. 2-3 sentences."
        ),
    )

    word_count: int = Field(
        ...,
        description="Approximate total word count across all paragraphs.",
    )

    tone: Literal["formal", "conversational", "enthusiastic"] = Field(
        ...,
        description=(
            "The tone of the letter. Match it to the company's likely culture: "
            "'formal' for traditional enterprise, 'conversational' for modern "
            "tech companies, 'enthusiastic' for startups."
        ),
    )


# ─── V2: Interview Preparation ─────────────────────────────
class QuestionAnswer(BaseModel):
    """A single interview question paired with a suggested answer."""

    question: str = Field(..., description="The interview question.")
    suggested_answer: str = Field(
        ...,
        description=(
            "A 2-4 sentence suggested answer drawing on the candidate's "
            "actual background. Be concrete about projects and outcomes."
        ),
    )
    relevant_experience: str = Field(
        ...,
        description=(
            "Which specific project, skill, or experience from the candidate's "
            "background to reference when answering this question."
        ),
    )


class InterviewPrep(BaseModel):
    """A structured interview preparation guide for a specific JD."""

    technical_questions: list[QuestionAnswer] = Field(
        ...,
        description=(
            "3-5 likely technical questions specific to this role's stack. "
            "Each with a suggested answer drawing on real candidate experience."
        ),
    )

    behavioral_questions: list[QuestionAnswer] = Field(
        ...,
        description=(
            "2-3 likely behavioral questions appropriate to the role's seniority. "
            "Each with a suggested answer using the STAR pattern where relevant."
        ),
    )

    questions_to_ask_them: list[str] = Field(
        ...,
        description=(
            "3-5 thoughtful questions the candidate should ask the interviewer. "
            "Should reflect genuine interest in the role and company."
        ),
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # Run with: python -m models.schemas
    print("✓ All V1 + V2 schemas imported and validated successfully\n")
    print("Available schemas:")
    for cls in [
        JobDescription,
        IntentClassification,
        FitReport,
        CoverLetter,
        InterviewPrep,
    ]:
        print(f"  • {cls.__name__:25s} ({len(cls.model_fields)} fields)")

    print("\n─── Sample IntentClassification ───")
    sample_intent = IntentClassification(
        intent="generate_cover_letter",
        confidence=0.95,
        reasoning="The user explicitly asked for a cover letter.",
    )
    print(sample_intent.model_dump_json(indent=2))


# ─── V5: Query Rewriter ────────────────────────────────────
class RetrievalQuery(BaseModel):
    """A retrieval-optimized search query produced by the query rewriter.

    Replaces V3's naive `f"{user_request}\\n\\n{jd_text}"` concatenation
    with a focused, keyword-dense string designed for vector search.
    The reasoning field is what makes this debuggable in LangSmith —
    when retrieval misses, you can see exactly what query the rewriter
    chose and why.
    """

    query: str = Field(
        ...,
        description=(
            "A concise search query optimized for vector retrieval. "
            "Should consist of concrete skills, technologies, project "
            "types, and domain terms — not conversational phrasing or "
            "filler words. Aim for 5-15 high-signal keyword tokens. "
            "Use the JD's exact capitalization (PyTorch, not pytorch)."
        ),
    )

    reasoning: str = Field(
        ...,
        description=(
            "One short sentence explaining why these terms were chosen, "
            "referring to the JD requirements and the user's intent."
        ),
    )


# ─── V6: Action Selector ───────────────────────────────────
# The set of deterministic tools the action selector can dispatch to.
# Adding a tool means: (a) implement assistant/tools/<name>.py,
# (b) register it in assistant/tools/__init__.py, (c) add it here.
# The Literal makes the binding type-safe end-to-end.
ToolName = Literal[
    "web_search",
    "experience_calculator",
    "mock_salary_lookup",
]


class ActionDecision(BaseModel):
    """High-level action path chosen by the V6 action selector.

    Sits ABOVE the V2 intent router. The action selector decides
    whether the request needs portfolio retrieval at all — and if
    so, hands off to the V2 router. If not, the request goes to a
    direct-answer chain (no RAG) or a deterministic tool.
    """

    action: Literal["direct_answer", "retrieval", "tool_use"] = Field(
        ...,
        description=(
            "Which high-level path to invoke:\n"
            "- 'direct_answer': general advice the LLM can answer from "
            "training data; no portfolio access, no tools.\n"
            "- 'retrieval': portfolio-grounded request; downstream V2 "
            "router will dispatch to the right specialized handler.\n"
            "- 'tool_use': deterministic computation or external data "
            "lookup; downstream tool executor will invoke 'tool_name' "
            "with 'tool_input' as its argument."
        ),
    )

    tool_name: ToolName | None = Field(
        default=None,
        description=(
            "When action='tool_use', the specific tool to call. "
            "MUST be one of the registered tool names. "
            "When action != 'tool_use', this MUST be null."
        ),
    )

    tool_input: str | None = Field(
        default=None,
        description=(
            "When action='tool_use', the single string argument passed "
            "to the tool function. Each tool takes one string:\n"
            "- experience_calculator: a skill/technology name "
            "(e.g. 'Python', 'FastAPI')\n"
            "- mock_salary_lookup: a seniority level — must be "
            "'junior', 'mid', 'senior', or 'lead'\n"
            "- web_search: a focused search query\n"
            "When action != 'tool_use', this MUST be null."
        ),
    )

    reasoning: str = Field(
        ...,
        description=(
            "One short sentence explaining why this action (and tool, "
            "if applicable) was chosen. Surfaces in LangSmith traces "
            "for debugging."
        ),
    )
