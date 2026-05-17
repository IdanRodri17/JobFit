"""Tool registry for V6 dispatch.

The V6 action selector outputs ActionDecision.tool_name as a string;
TOOLS maps that name to the actual callable. Adding a new tool is
a four-step change:
  1. Implement the tool module (one file in assistant/tools/).
  2. Register it here.
  3. Add its name to ToolName Literal in models/schemas.py.
  4. Add its description to ACTION_SELECTOR_PROMPT in prompts/templates.py.

Each tool MUST accept exactly one string positional argument and
return a Pydantic BaseModel (so the synthesizer can introspect the
fields). That contract is what makes the dispatcher in core.py a
one-liner.
"""

from typing import Callable

from pydantic import BaseModel

from assistant.tools.experience_calculator import calculate_experience
from assistant.tools.mock_salary_lookup import lookup_salary
from assistant.tools.web_search import search_web

# Tool callable signature: (str) -> BaseModel
ToolCallable = Callable[[str], BaseModel]


TOOLS: dict[str, ToolCallable] = {
    "experience_calculator": calculate_experience,
    "mock_salary_lookup": lookup_salary,
    "web_search": search_web,
}
