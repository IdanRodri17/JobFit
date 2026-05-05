"""V1 chain: parse a raw job description into a structured JobDescription.

Composes three Runnables using LCEL's pipe operator:
    JD_PARSER_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

When invoked, this chain:
    1. Renders the prompt with jd_text and pre-filled format_instructions
    2. Sends the rendered messages to OpenAI
    3. Parses + validates the response into a JobDescription instance
    4. (Automatically) traces every step in LangSmith

Smoke test:
    python -m assistant.chains.jd_parser
    (note: this calls OpenAI — cost is ~$0.001 with gpt-4o-mini)
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import JobDescription
from prompts.templates import JD_PARSER_PROMPT


# ─── Output parser ─────────────────────────────────────────
# Two responsibilities:
#   1. get_format_instructions() — returns the JSON Schema string
#      we inject into the prompt so the LLM knows what to return.
#   2. invoke(ai_message) — parses + validates the LLM's response
#      and returns a JobDescription instance (raises on failure).
parser = PydanticOutputParser(pydantic_object=JobDescription)


# ─── Model ─────────────────────────────────────────────────
# Temperature 0.0 — deterministic output. Critical for structured
# extraction: with temperature > 0, the model would occasionally
# return "Senior" instead of "senior", breaking our Literal enum.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain: prompt → model → parser ────────────────────
# .partial() pre-fills format_instructions once at chain construction,
# so callers only provide jd_text at invoke time.
jd_parser_chain = (
    JD_PARSER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def parse_job_description(jd_text: str) -> JobDescription:
    """Parse a raw job description into a validated JobDescription.

    Args:
        jd_text: Raw text of a job posting (paste from LinkedIn, etc.)

    Returns:
        A validated JobDescription instance.

    Raises:
        OutputParserException: if the LLM response cannot be parsed as JSON.
        ValidationError: if the parsed JSON fails schema validation.
    """
    return jd_parser_chain.invoke({"jd_text": jd_text})


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE: This makes a real OpenAI API call. Cost ~$0.001 with gpt-4o-mini.
    sample_jd = """\
Senior AI Developer — Elad Systems (Tel Aviv, Hybrid)

We're hiring a Senior AI Developer with 5+ years of experience to join
our growing AI team. You'll lead the design of production RAG systems
for our enterprise clients.

Required:
- 5+ years of Python development
- Hands-on experience with LangChain and RAG architectures
- Strong PostgreSQL skills (pgvector a plus)
- FastAPI for production APIs
- Comfortable with a hybrid setup (3 days in office)

Nice to have:
- LangGraph experience for agentic workflows
- Hebrew language skills
- Prior production ML deployment experience

Responsibilities:
- Build production RAG systems for enterprise clients
- Lead architecture decisions for new AI products
- Mentor junior team members on LLM best practices
"""

    print("⏳ Calling OpenAI to parse the JD...\n")
    result = parse_job_description(sample_jd)

    print("✓ JD parsed successfully!\n")
    print(result.model_dump_json(indent=2))
    print()
    print("→ Open https://smith.langchain.com and look in the 'JobFit' project")
    print("  for a new trace named 'RunnableSequence' (or similar).")
