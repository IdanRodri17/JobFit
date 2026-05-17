# V6 Architecture — Controlled Action Selection

## The problem V6 solves

After V5, JobFit was a strong RAG system: it analyzed fit, drafted cover letters, and prepared interview questions, all grounded in retrieved portfolio context. But every request went through the same flow:

```
classify_intent → rewrite_query → retrieve → handler
```

This forced retrieval onto questions that didn't need it ("how long should a cover letter be?"), and gave the system no way to answer questions that *required* information outside the portfolio (current company news, deterministic date math).

V6 adds a routing layer **above** the V2 intent router: an action selector that decides whether the request needs retrieval at all, can be answered directly, or requires a tool call.

## The three paths

```
process_request_v6(jd, request)
        │
        ▼
   select_action(request)  →  ActionDecision(action, tool_name, tool_input, reasoning)
        │
        ├── direct_answer  → answer_directly(request)               → string
        ├── retrieval      → V5 flow (router → retrieve → handler)  → Pydantic
        └── tool_use       → TOOLS[tool_name](tool_input)
                                → synthesize_tool_response          → string
```

**direct_answer** — general advice the LLM can give from training knowledge alone. No portfolio access, no external calls. 2–4 sentences of concrete guidance. Examples: *"How long should a cover letter be?"*, *"What's the standard tone for a thank-you email?"*

**retrieval** — the existing V5 flow, unchanged. Used when the request is about *this* candidate or *this* job. Examples: *"Write me a cover letter for this position"*, *"What gaps should I address?"*

**tool_use** — deterministic computation or external data lookup. Three tools registered in `assistant/tools/__init__.py`:

- `experience_calculator(skill)` — date math against `data/portfolio/skills.json`
- `mock_salary_lookup(seniority)` — JSON lookup of ILS salary ranges (illustrative, not live)
- `web_search(query)` — Tavily integration for recent news

## Design principle: LLMs for language, Python for facts

V6's central architectural decision: **LLMs handle language understanding and routing; deterministic Python handles math, lookups, and API calls.**

- The LLM is good at understanding *"how many years of Python do I have?"* means *"call experience_calculator with input='Python'"*.
- The LLM is **bad** at actually computing year differences between dates.
- The deterministic tool is **good** at date math.
- The LLM is good at phrasing the result as *"You have 4.6 years of Python experience."*

Same pattern for salary and web search: LLM picks the tool and extracts the input string; Python (or Tavily) does the actual work; LLM phrases the answer.

This separation is the opposite of fully-agentic frameworks where the LLM is in a tool-calling loop and may re-invoke tools, second-guess outputs, or hallucinate facts. V6 is **single-pass and bounded**: one action selection, one tool call, one synthesis. Cheaper, more observable, and (for this scope) sufficient. When stateful multi-step flows are needed, LangGraph is the next step up.

## The synthesizer's grounding contract

When a tool returns a structured result, the synthesizer's prompt enforces specific rules:

1. **Base every fact on the tool's output.** No training-data padding.
2. **If `is_known: false`, say so.** Never fabricate a number to fill the gap.
3. **If `data_source: 'placeholder'`, flag it.** The salary tool's output explicitly carries this marker, and the synthesizer surfaces it: *"Keep in mind that these figures are illustrative and not based on live market data."*
4. **For web search results, cite URLs only when they materially support a claim.**

These rules are why V6 doesn't lie when a tool can't help. They're testable: ask *"What salary should I ask for as a junior AI dev?"* and the synthesized answer should always include the placeholder caveat.

## Adding a new tool — the 4-step recipe

1. Implement the tool as a single function in `assistant/tools/<name>.py`: `def my_tool(input: str) -> BaseModel`.
2. Register it in `assistant/tools/__init__.py` by adding an entry to the `TOOLS` dict.
3. Add the tool name to the `ToolName` Literal in `models/schemas.py`.
4. Add the tool's purpose, examples, and `tool_input` format to `ACTION_SELECTOR_PROMPT` in `prompts/templates.py`.

The contract is: one string in, one Pydantic model out. The Pydantic model should include an `is_known: bool` field if the tool can fail to find data — that's the signal the synthesizer reads to avoid fabrication.

## Known limitations and V6.1 candidates

- **The action selector is JD-blind.** It sees only the user request, not the JD text. When a user asks *"Should I apply to this role?"*, the action selector has no JD context to recognize this as a retrieval question; for short ambiguous phrasings it sometimes defaults to `direct_answer`. Passing the JD text into the action selector prompt would resolve this.
- **The HTTP API still uses the V5 flow.** `routes/process.py` calls `process_request()` (the V5 2-tuple signature), not `process_request_v6()`. Migrating the route to V6 means extending `ProcessResponse` with `ActionDecision` and updating the frontend renderer.
- **Tool input is a single string.** Tools that need multiple structured parameters require workarounds. For richer tool calling, switching to LangChain's tool-binding pattern would be cleaner.
- **No memory or state.** Each request is independent. A V7 with LangGraph would enable stateful flows like *"draft a cover letter, then refine based on feedback"*.

## Why this design vs. fully agentic

A fully agentic framework (LangChain `AgentExecutor`, OpenAI function-calling loop, LangGraph ReAct agent) would let the LLM decide on every turn which tool to call next, retry on failure, and chain multiple tool calls. That's more flexible — but for this project, more flexible was the wrong trade-off:

- **Observability**: single-pass dispatch produces one clean LangSmith trace per request. Agentic loops produce nested traces that are harder to reason about.
- **Cost**: bounded calls. An agent that mis-routes can spend 5–10× more tokens.
- **Debuggability**: when something goes wrong, you can point at one of three handlers and reproduce the failure. With a free-running agent, you have to replay the whole episode.
- **Sufficient for scope**: every request type JobFit handles fits in one tool call.

V6 is the **pragmatic agentic minimum**: enough action selection to handle three distinct paths, none of the looping complexity. When the requirements outgrow single-pass dispatch (multi-turn refinement, conditional branching on tool output), LangGraph is the next architecture.
