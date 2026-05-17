"""V6 chain: answer general career questions without portfolio or tools.

Used for the 'direct_answer' action path. The user is asking something
the LLM can answer from training knowledge — career advice, formatting
questions, generic tips. Routing this through retrieval would just
dilute the answer with irrelevant portfolio context.

Composition is the simplest LCEL shape in the project — no parser:
    DIRECT_ANSWER_PROMPT  →  ChatOpenAI  →  StrOutputParser

Smoke test:
    python -m assistant.chains.direct_answer
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.settings import settings
from prompts.templates import DIRECT_ANSWER_PROMPT

parser = StrOutputParser()

# Slight temperature — career advice benefits from natural phrasing,
# unlike the structured-extraction chains that demand temperature=0.0.
model = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.3,
    api_key=settings.openai_api_key,
)


direct_answer_chain = DIRECT_ANSWER_PROMPT | model | parser


def answer_directly(user_request: str) -> str:
    """Answer a general career / application question.

    No portfolio access, no tool calls. The LLM responds from
    training knowledge alone.
    """
    return direct_answer_chain.invoke({"user_request": user_request})


if __name__ == "__main__":
    test_qs = [
        "How long should a cover letter be?",
        "What's the standard tone for a thank-you email after an interview?",
        "Should I include a photo on my resume in Israel?",
    ]
    for q in test_qs:
        print("═" * 70)
        print(f"Q: {q}")
        print("─" * 70)
        print(answer_directly(q))
        print()
