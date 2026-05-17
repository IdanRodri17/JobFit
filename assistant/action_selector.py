"""V6 chain: classify a user request into one of three action paths.

This is the V6 architectural pivot. It sits ABOVE the V2 intent
router and decides whether the request needs to touch the
portfolio at all:

    USER REQUEST
        │
        ▼
    ACTION SELECTOR  (this chain)
        │
        ├── 'direct_answer' → direct LLM answer, no RAG, no tools
        ├── 'retrieval'     → V2 router → handler → structured output
        └── 'tool_use'      → tool executor → synthesize → answer

Composition is the standard JobFit shape:
    ACTION_SELECTOR_PROMPT  →  ChatOpenAI  →  PydanticOutputParser

Why a separate chain instead of merging into the V2 router?
    Single responsibility. The V2 router classifies INTENT for
    portfolio-grounded requests (fit / cover letter / interview prep).
    The action selector classifies PATH (direct / retrieval / tool).
    Merging them creates a five-way classifier with worse calibration
    than two two-way-ish classifiers in sequence — and harder to
    debug in LangSmith.

Smoke test:
    python -m assistant.action_selector
    (note: real OpenAI calls — cost is ~$0.0005 per case with gpt-4o-mini)
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import ActionDecision
from prompts.templates import ACTION_SELECTOR_PROMPT

# ─── Output parser ─────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=ActionDecision)


# ─── Model ─────────────────────────────────────────────────
# Temperature=0.0 for reproducibility. Same input MUST yield the
# same action — otherwise tool dispatch becomes non-deterministic.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)


# ─── The chain ─────────────────────────────────────────────
action_selector_chain = (
    ACTION_SELECTOR_PROMPT.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)


def select_action(user_request: str) -> ActionDecision:
    """Classify a user request into action, tool_name, reasoning.

    Args:
        user_request: The raw text of what the user wants.

    Returns:
        A validated ActionDecision with action ∈ {direct_answer,
        retrieval, tool_use} and, when applicable, tool_name.
    """
    return action_selector_chain.invoke({"user_request": user_request})


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # direct_answer cases
        ("How long should a cover letter be?", "direct_answer", None),
        ("What's the standard tone for a thank-you email?", "direct_answer", None),
        # retrieval cases
        ("Should I apply for this AI Developer role?", "retrieval", None),
        ("Write me a cover letter for this position.", "retrieval", None),
        # tool_use cases
        (
            "How many years of Python experience do I have?",
            "tool_use",
            "experience_calculator",
        ),
        (
            "What salary should I ask for as a junior AI dev?",
            "tool_use",
            "mock_salary_lookup",
        ),
        ("What's the latest news about Elad Systems?", "tool_use", "web_search"),
    ]

    print("⏳ Running action selector on 7 cases...\n")
    correct_action = 0
    correct_tool = 0
    for user_request, expected_action, expected_tool in test_cases:
        result = select_action(user_request)
        action_ok = result.action == expected_action
        tool_ok = result.tool_name == expected_tool
        correct_action += int(action_ok)
        correct_tool += int(tool_ok)

        marker = "✓" if (action_ok and tool_ok) else "✗"
        print(f"{marker}  Input:     {user_request!r}")
        print(f"   Expected:  action={expected_action}, tool={expected_tool}")
        print(f"   Got:       action={result.action}, tool={result.tool_name}")
        print(f"   Reasoning: {result.reasoning}")
        print()

    print(
        f"─── {correct_action}/{len(test_cases)} actions correct, "
        f"{correct_tool}/{len(test_cases)} tools correct ───"
    )
    print("→ Open https://smith.langchain.com (project: JobFit) for the traces")
