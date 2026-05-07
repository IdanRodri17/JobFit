"""V2 chain: classify a user request into one of five intents.

Composes three Runnables using LCEL — same shape as V1's jd_parser:
    ROUTER_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

The output is an IntentClassification with three fields:
    - intent: one of the five Literal values (analyze_fit, etc.)
    - confidence: 0.0-1.0 (below 0.6 signals ambiguity)
    - reasoning: short explanation of the choice

The dispatcher in assistant/core.py uses `intent` to pick which
specialized handler chain to invoke.

Smoke test:
    python -m assistant.router
    (note: this calls OpenAI — cost is ~$0.0005 with gpt-4o-mini)
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import IntentClassification
from prompts.templates import ROUTER_PROMPT


# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=IntentClassification)


# ─── Model ─────────────────────────────────────────────────
# Same temperature=0.0 reasoning as V1: deterministic classification.
# A router that returns different intents for the same input on
# different calls is a router that cannot be debugged.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain: prompt → model → parser ────────────────────
router_chain = (
    ROUTER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def classify_intent(user_request: str) -> IntentClassification:
    """Classify a user request into one of the five JobFit intents.

    Args:
        user_request: The raw text of what the user wants to do
                      (e.g. "Write me a cover letter for this role").

    Returns:
        An IntentClassification instance with the intent, confidence,
        and reasoning.
    """
    return router_chain.invoke({"user_request": user_request})


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # Test the router on a variety of inputs to verify routing quality.
    # Each call is a real OpenAI request — total cost ~$0.003 for all six.
    test_cases = [
        # Clear single-intent cases
        ("Should I apply to this Senior AI Developer role?", "analyze_fit"),
        ("Write me a cover letter for this position.", "generate_cover_letter"),
        ("What technical questions might they ask me?", "interview_prep"),
        ("Tell me about Elad Systems as a company.", "company_research"),
        ("Tailor my resume bullets for this JD.", "tailor_resume"),
        # Ambiguous case — confidence should be lower
        ("Help me with this job.", "analyze_fit"),  # disambiguation rule
    ]

    print("⏳ Running router smoke test on 6 cases...\n")
    correct = 0
    for user_request, expected_intent in test_cases:
        result = classify_intent(user_request)
        is_correct = result.intent == expected_intent
        correct += int(is_correct)

        marker = "✓" if is_correct else "✗"
        print(f"{marker}  Input:      {user_request!r}")
        print(f"   Expected:   {expected_intent}")
        print(f"   Got:        {result.intent}  (confidence: {result.confidence:.2f})")
        print(f"   Reasoning:  {result.reasoning}")
        print()

    print(f"─── {correct}/{len(test_cases)} correct ───")
    print("→ Open https://smith.langchain.com (project: JobFit) for the traces")
