# JobFit

**AI job-application assistant with controlled action selection over a personal portfolio.**

JobFit analyzes how well a candidate's portfolio (CV + project documents) fits a job description, drafts grounded cover letters, prepares interview questions, and answers general application questions or live web/computation queries — all routed through a deliberate dispatcher that decides up front whether to retrieve, answer directly, or call a tool.

Built as the LangChain/LangSmith capstone for the CyberPro AI Developer Bootcamp.

## What it does

Given a job description and a user request, JobFit:

1. **Classifies the request's high-level action** — direct answer, portfolio retrieval, or tool use.
2. For **portfolio retrieval**, rewrites the query into keyword-dense form, filters by document category, retrieves the top-k matching chunks from a Chroma vector store, and synthesizes a structured response (fit report, cover letter, or interview prep).
3. For **direct answer**, sends the question to a lightweight prompted chain (no RAG, no tools).
4. For **tool use**, dispatches to one of three deterministic tools (experience calculator, mock salary lookup, web search), then synthesizes the structured result into prose.

Every chain, retrieval, and tool call is observable in **LangSmith**.

## Version arc

| Version | Headline | What it added |
|---|---|---|
| V1 | LCEL pipeline | JD parsing chain (`prompt \| model \| parser`), Pydantic schemas, basic error handling. |
| V2 | Intent router | Multi-handler dispatcher: `analyze_fit`, `generate_cover_letter`, `interview_prep`. Per-handler prompts. |
| V3 | RAG over portfolio | Chroma vector store, embedding ingestion, top-k retrieval into handlers. Diagnosed "phantom gap" failure mode (LLM hallucinated missing skills because retrieval surfaced cv.md chunks only). |
| V4 | Architecture refactor | Modular chains, central settings, FastAPI HTTP layer, web frontend. |
| V5 | **Measurement** | LangSmith evaluation pipeline. Query rewriter, metadata-filtered retrieval, 12-case labeled eval dataset, 8 custom evaluators, 4 experiments. **`sources_in_context` improved 0.10 → 1.00 at k=6.** |
| V6 | **Controlled action selection** | Top-level action selector chooses between three paths. Three deterministic tools (date math, mock salary lookup, Tavily web search). Synthesizer turns structured tool results into prose with explicit grounding rules (`is_known=false` → honest, `data_source='placeholder'` → flag it). |

## Architecture

```
USER REQUEST
    │
    ▼
ACTION SELECTOR (V6)
    │
    ├── direct_answer  → answer_directly()           → string
    ├── retrieval      → V2 router → V5 retrieval
    │                    → handler                   → Pydantic
    └── tool_use       → TOOLS[name](input)
                          → synthesize_tool_response → string
```

LLMs handle language understanding (action selection, intent classification, query rewriting, synthesis). Deterministic Python handles math, file lookups, and external API calls. Each layer is independently observable in LangSmith. Full design rationale in [`docs/v6_architecture.md`](docs/v6_architecture.md).

## Eval results (V5)

12-case labeled dataset, four experiments compared at retrieval `k ∈ {2, 4, 6}` vs. V3 baseline:

| Evaluator | V3 baseline | V5 k=4 | V5 k=6 |
|---|---|---|---|
| intent_correct | 1.00 | 1.00 | 1.00 |
| fit_score_in_range | 1.00 | 1.00 | 1.00 |
| **sources_in_context** | **0.10** | 0.80 | **1.00** |
| gap_skills_present | 1.00 | 1.00 | 1.00 |
| total cost (12 cases) | $0.0059 | $0.0079 | $0.0078 |

V5 with k=6 is the recommended production config: perfect retrieval, equivalent synthesis quality, comparable cost. Full discussion in [`docs/eval_results.md`](docs/eval_results.md).

## Tech stack

- **LLM**: OpenAI `gpt-4o-mini` (orchestration), `gpt-4o` (eval judge)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector store**: ChromaDB (local persistence)
- **Frameworks**: LangChain (LCEL), LangSmith (tracing + eval), Pydantic v2
- **API**: FastAPI
- **Web search**: Tavily
- **Language**: Python 3.12

## Quick start

```bash
# 1. Setup
git clone https://github.com/IdanRodri17/JobFit.git
cd JobFit
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add OPENAI_API_KEY, TAVILY_API_KEY (optional), LANGSMITH_API_KEY (optional)

# 3. Ingest the portfolio into the vector store
python -m ingestion.portfolio_ingest

# 4a. CLI demo (V6 dispatcher — all three paths)
python -m assistant.core

# 4b. Or run the HTTP API (V5 retrieval flow)
uvicorn api:app --reload
# Open http://localhost:8000
```

## Project layout

```
JobFit/
├── assistant/
│   ├── core.py                 # V5 + V6 entry points
│   ├── action_selector.py      # V6: direct / retrieval / tool routing
│   ├── router.py               # V2: intent classification
│   ├── synthesizer.py          # V6: tool result → prose
│   ├── chains/
│   │   ├── parser.py           # V1: JD parsing
│   │   ├── fit_analyzer.py     # V2: analyze_fit handler
│   │   ├── cover_letter.py     # V2: generate_cover_letter handler
│   │   ├── interview_prep.py   # V2: interview_prep handler
│   │   ├── query_rewriter.py   # V5: keyword-dense rewrite
│   │   └── direct_answer.py    # V6: general advice chain
│   └── tools/
│       ├── experience_calculator.py
│       ├── mock_salary_lookup.py
│       └── web_search.py       # Tavily integration
├── retrieval/
│   └── portfolio_retriever.py  # Chroma + category filter
├── evaluation/
│   ├── eval_dataset.py         # 12 labeled cases
│   └── run_eval.py             # LangSmith experiment runner
├── ingestion/
│   └── portfolio_ingest.py     # Chunk + embed portfolio docs
├── prompts/templates.py
├── models/schemas.py
├── data/
│   ├── portfolio/              # CV + project markdown
│   └── market/salaries.json
├── routes/                     # FastAPI routes
├── tests/
└── docs/
    ├── PROJECT_SPECIFICATION.md
    ├── eval_results.md
    └── v6_architecture.md
```

## Author

**Idan Rodriguez** — B.Sc. Computer Science (HIT, ML specialization). Currently in the CyberPro AI Developer Bootcamp.

Portfolio: [idanportfolio.netlify.app](https://idanportfolio.netlify.app) · GitHub: [@IdanRodri17](https://github.com/IdanRodri17)