"""V6 helper: synthesize a tool result into a natural-language answer.

After the V6 action selector picks a tool and the dispatcher runs it,
the result is a typed Pydantic model. The user wants prose. This
function bridges that gap.

Why an LLM call instead of a per-tool template? The result schemas
vary (ExperienceCalculation, SalaryRange, WebSearchResponse) and each
has nuances — is_known=False signals 'no data', SalaryRange.data_source
'placeholder' must be flagged honestly, web_search results may need
URL citations. An LLM seeing the schema + the user_request can phrase
the answer better than any per-tool template would.

The synthesizer prompt embeds grounding rules so the LLM doesn't
fabricate from training data when the tool said 'no data'.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config.settings import settings
from prompts.templates import TOOL_SYNTHESIS_PROMPT

_parser = StrOutputParser()

_model = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.2,
    api_key=settings.openai_api_key,
)


_chain = TOOL_SYNTHESIS_PROMPT | _model | _parser


def synthesize_tool_response(
    user_request: str,
    tool_name: str,
    tool_result: BaseModel,
) -> str:
    """Turn a tool's structured output into a natural-language answer.

    Args:
        user_request: What the user originally asked.
        tool_name: Which tool produced the result (for context in the prompt).
        tool_result: The Pydantic instance returned by the tool.

    Returns:
        A natural-language string answering the user's question,
        grounded in the tool's structured output.
    """
    result_json = tool_result.model_dump_json(indent=2)
    return _chain.invoke(
        {
            "user_request": user_request,
            "tool_name": tool_name,
            "tool_result_json": result_json,
        }
    )
