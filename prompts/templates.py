"""Prompt templates for all chains in JobFit.

Centralizing prompts here keeps business logic clean and makes prompt
iteration trivial: to tune extraction quality, edit this file alone —
no chain code changes required.

Each template uses ChatPromptTemplate with two roles:
- system: defines the model's behavior, constraints, and format expectations
- human: contains the user's input (changes per invocation)

The {format_instructions} placeholder is filled at chain construction time
by LangChain's PydanticOutputParser with the auto-generated JSON Schema
(the same schema you saw via `python -m models.schemas`).

Smoke test:
    python -m prompts.templates
"""
from langchain_core.prompts import ChatPromptTemplate


# ─── V1: Job Description Parser ────────────────────────────
JD_PARSER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert recruitment assistant specializing in parsing "
        "job descriptions for AI and software engineering roles.\n\n"
        "Your task is to extract structured information from raw job "
        "posting text. Be precise — do not infer information that is not "
        "explicitly present in the posting. For ambiguous cases, prefer "
        "the 'unknown' enum value or null over guessing.\n\n"
        "When extracting skills, distinguish carefully between:\n"
        "- HARD requirements: must-have skills, marked with 'required', "
        "'essential', or stated years of experience.\n"
        "- NICE-TO-HAVE skills: optional preferences, marked with "
        "'preferred', 'bonus', 'plus', or 'would be an advantage'.\n\n"
        "Extract concrete technologies (e.g. 'PyTorch', 'PostgreSQL', "
        "'LangChain') rather than abstract categories (e.g. 'ML frameworks', "
        "'databases') whenever the posting names them specifically.\n\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Parse the following job posting and return the structured JSON:\n\n"
        "--- JOB POSTING ---\n"
        "{jd_text}\n"
        "--- END POSTING ---"
    ),
])


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # Render the template with sample data to verify variable substitution
    sample_jd = (
        "We are hiring a Senior AI Developer at Elad Systems.\n"
        "Required: 5+ years of Python, experience with LangChain and RAG.\n"
        "Nice to have: Hebrew, LangGraph experience.\n"
        "Hybrid role based in Tel Aviv."
    )
    sample_format_instructions = (
        "[At runtime, PydanticOutputParser injects the JSON Schema here]"
    )

    rendered_messages = JD_PARSER_PROMPT.format_messages(
        jd_text=sample_jd,
        format_instructions=sample_format_instructions,
    )

    print("✓ Prompt template loaded and rendered successfully\n")
    print(f"  Required input variables: {JD_PARSER_PROMPT.input_variables}")
    print(f"  Number of messages:       {len(rendered_messages)}\n")

    for i, msg in enumerate(rendered_messages, 1):
        print(f"─── Message {i}: role = {msg.type.upper()} ───")
        print(msg.content)
        print()
