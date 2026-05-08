# V3 Findings — RAG Implementation

This document captures what V3 revealed about JobFit's behavior and
sets up the work for V5 (RAG Quality Optimization).

## Summary

V3 added real Retrieval-Augmented Generation: 5 portfolio markdown
files (CV + 4 projects), chunked and embedded into ChromaDB, retrieved
per-request from `assistant/core.py`, and consumed by the three V2
handler chains with new grounding language.

The architecture works. The candidate context is now dynamic and
query-aware, not a hardcoded constant.

## What's working

- **Ingestion:** 5 markdown files → 37 chunks → embedded with
  `text-embedding-3-small` and persisted to ChromaDB. End-to-end
  cost is negligible (~$0.0001 per ingestion run).
- **Retrieval:** Semantic similarity surfaces relevant chunks across
  the portfolio. Test queries for "LangChain and RAG", "computer
  vision and YOLOv8", "production deployment", and "Hebrew language"
  each returned topic-appropriate chunks without keyword overlap.
- **Grounding language:** The shared `GROUNDING_RULES` block reduced
  hallucination compared to V2. The cover letter now references real
  project sentences from the RAG Hub README rather than paraphrased
  content from a hardcoded blurb.
- **Architectural payoff:** V3's biggest capability change was a
  ~5-line diff in `assistant/core.py`. Handlers were never modified —
  they consumed `candidate_context` as a string in V2 and still do
  in V3. The retrieval source changed; the contract didn't.

## What V3 revealed — the "phantom gaps" pattern

Running `python main.py examples/elad_jd.txt --request "Should I apply?"`
against the full-length Elad Systems JD produced a `FitReport` with
gap_skills entries that the candidate actually has:

- "Hands-on experience with LangChain..."
- "Solid understanding of vector databases..."
- "Comfortable with FastAPI for production APIs"
- "Experience with PostgreSQL and SQL"
- "Familiarity with Docker and Git workflows"

All of these are present in the candidate's portfolio. Why did the
model flag them as gaps?

**Root cause: retrieval coverage vs JD breadth.**

The retrieval pulled `k=4` chunks. The Elad JD covers 12+ distinct
skill topics. With only 4 chunks, the retrieval cannot fully cover a
JD this broad. Skills that exist in the portfolio but don't land in
the top-4 chunks become invisible to the handler.

The grounding rule (*"treat anything not in retrieved context as a
gap"*) is then doing its job correctly — the model has no way to
distinguish "candidate lacks X" from "X is in portfolio but wasn't
retrieved."

This is not a bug. It's V3 exposing the *retrieval quality* dimension
that V5 was designed to address.

## Other V3 observations

- **Score sensitivity to retrieval scope.** The same candidate
  scored 90/strong_apply against a short, focused JD (5 explicit
  requirements) and 75/apply against the full Elad JD (12+ topics).
  Score quality is bounded by retrieval coverage of the JD.
- **One small "current role" slip in the cover letter.** The model
  introduced the phrase *"In my current role, I have hands-on
  experience..."* — the candidate is in a bootcamp, not a current
  professional role. Worth a small prompt rule in V5: *"Do not
  assume professional employment context unless the retrieved
  context explicitly states one."*
- **Markdown header noise in chunks.** A few chunks end with orphan
  headers like `## Tech Stack` (the next section's header). Minor
  retrieval-quality issue. Solvable with a markdown-aware splitter
  (`MarkdownTextSplitter`) but adds complexity. Defer to V5+.

## V5 planning hooks

What V3 has set up for V5:

1. **Query rewriting** (assistant/chains/query_rewriter.py) — turn
   a user request + JD into a concise, retrieval-optimal search
   query. Replaces V3's `_build_retrieval_query()` concatenation.
2. **Tune `k` based on JD breadth.** A short JD might need `k=2`;
   a long Elad-style JD might need `k=8`. Eval set will quantify
   the trade-off.
3. **Metadata filtering by category.** When checking comprehensive
   skill coverage, prefer cv.md chunks (full skill list) over
   project chunks (specific topic depth). When generating cover
   letters, prefer projects/ chunks for concrete examples.
4. **Eval dataset.** ~10-15 (JD, expected-fit-score, expected-skills)
   tuples covering easy / medium / tricky / out-of-scope cases.
   Fixes the phantom-gap problem only if we can measure it.
5. **Cost tracking.** Per-query token + dollar accounting via
   `get_openai_callback()`, surfaced into LangSmith.

## Demo material for the bootcamp presentation

Three slide-worthy moments from V3:

1. **Side-by-side `core.py` V2 vs V3** — same dispatch logic, the
   only real change is `CANDIDATE_CONTEXT` (constant) → `get_relevant_
   context(query)` (dynamic). Architectural payoff.
2. **Semantic retrieval beats keyword search** — the *"Hebrew language
   and Israeli tech market"* query surfaces ELAD Software, Holon
   Institute, and "Looking For" CV sections without containing the
   word "Israeli". Show the chunks.
3. **The phantom gaps slide** — display the gap_skills list, point
   out the candidate has every one of those, explain `k=4` vs JD
   breadth, then promise V5 fixes it with measurement, not vibes.
