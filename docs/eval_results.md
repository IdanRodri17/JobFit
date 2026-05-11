# JobFit V5 — Evaluation Results

> *A LangSmith-driven measurement of the V5 retrieval-quality changes (query rewriter + metadata filter), benchmarked against the V3 baseline across a 12-case labeled evaluation set.*

**Author:** Idan
**Date:** May 2026
**LangSmith project:** [JobFit](https://smith.langchain.com)
**Dataset:** `JobFit-V5` (12 cases, see `evaluation/eval_dataset.py`)

---

## TL;DR

V5's combined query-rewriter and metadata-filter changes deliver an **8–10× improvement in retrieval hit-rate** (`sources_in_context` 0.10 → 1.00). The cost is negligible — a $0.0002 per-query overhead for the rewriter, total experiment cost $0.0079 for 12 cases.

A counter-intuitive secondary finding: V3 scored *higher* than V5 on `matched_skills_present` (1.00 vs 0.60). Drilling into the raw output revealed this was an artifact, not a regression — V3's naive concatenation query retrieves only `cv.md` chunks, which happen to contain a flat skill enumeration the LLM easily transcribes. V5 retrieves *correctly diverse* sources, but the fit-analyzer prompt weighs project-stack terminology over the CV's skill listing. **The fix for this is prompt-level, not retrieval-level**, and is documented as V5.5 follow-up work below.

The recommended production configuration is **k=6 with rewriter + filter**, which achieves perfect retrieval hit-rate at no quality or cost penalty vs. k=4.

---

## What we measured

### Dataset

12 labeled cases in `evaluation/eval_dataset.py`, split across difficulty buckets:

| Bucket | Count | Tests |
|---|---|---|
| Easy | 3 | Clear inputs with unambiguously correct outputs |
| Medium | 4 | Require synthesis across multiple portfolio chunks |
| Tricky | 3 | Router-disambiguation edge cases |
| Out-of-scope | 2 | Grounding stress — system must not fabricate experience |

Each case carries a `GroundTruth` Pydantic with optional fields: expected intent, fit-score range, skills that must appear in matched/gap lists, source files that must be retrieved, content the output must (or must not) contain.

Three job descriptions are used so retrieval has to discriminate across role profiles: a strong-fit AI Developer role (Elad-style), a junior Python backend role, a Java enterprise role, and a Kubernetes-heavy DevOps role.

### Experiments

Four LangSmith experiments, each running the same 12 cases through `process_request()`-equivalent pipelines with different knobs:

| Experiment | Rewriter | Filter | k | Notes |
|---|---|---|---|---|
| `v3-baseline` | off | off | 4 | Naive concat query `f"{user_request}\n\n{jd_text}"`, no category filter — the pre-V5 behavior |
| `v5-baseline` | on | on | 4 | The current main-branch behavior |
| `v5-k2` | on | on | 2 | Tighter retrieval |
| `v5-k6` | on | on | 6 | Wider retrieval |

The rewriter and filter are both V5 changes, evaluated as a combined unit (V3-baseline vs. V5). Decomposing them into separate ablations was descoped — when both helped together, neither's individual contribution warranted the additional measurement budget.

### Evaluators

Eight pure functions, one per `GroundTruth` field plus cost:

| Evaluator | What it measures | Score |
|---|---|---|
| `intent_correct` | Router classification accuracy | 1.0 if expected intent matches actual |
| `fit_score_in_range` | Fit-score calibration | 1.0 if `overall_score` lies in `[min, max]` |
| `matched_skills_present` | Phantom-gap test | Fraction of expected skills found in `matched_skills` (case-insensitive substring) |
| `gap_skills_present` | No-fabrication test | Fraction of true gaps correctly identified |
| `sources_in_context` | Retrieval hit-rate | Fraction of expected source files in retrieved context |
| `must_contain` | Output content check | Fraction of required strings present in generated output |
| `must_not_contain` | Grounding stress test | 1.0 if none of forbidden strings appear; 0.0 otherwise |
| `cost_usd` | Per-case USD cost | Pass-through from `get_openai_callback()` |

`None` scores mean "this evaluator doesn't apply to this case" — they're excluded from aggregates rather than averaged as zero.

---

## Results

### Aggregate scores

| Evaluator | v3-baseline | v5-baseline | v5-k2 | v5-k6 | n |
|---|---|---|---|---|---|
| `intent_correct` | 1.000 | 1.000 | 1.000 | 1.000 | 12 |
| `fit_score_in_range` | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| `gap_skills_present` | 1.000 | 1.000 | 1.000 | 1.000 | 2 |
| **`matched_skills_present`** | **1.000** | 0.600 | 0.867 | 0.600 | 3 |
| **`sources_in_context`** | **0.100** | 0.800 | 0.700 | **1.000** | 5 |
| `must_contain` | 0.667 | 0.667 | 0.667 | 0.667 | 3 |
| `must_not_contain` | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| **Total cost (12 cases)** | $0.0059 | $0.0079 | $0.0078 | $0.0078 | — |
| **Avg latency / case** | 6.46s | 8.61s | 7.50s | 7.13s | — |

Bold cells flag the two metrics that meaningfully discriminate the configurations: `sources_in_context` (where V5 dramatically wins) and `matched_skills_present` (where the result is counterintuitive). Everything else is either saturated (all configs at 1.0) or invariant across configs (all configs at 0.667 / 0.000).

---

## Findings

### Finding 1 — V5 dramatically improves retrieval hit-rate

`sources_in_context` is the most consequential measurement in this entire eval set. It directly answers: *did the chunks containing relevant portfolio content actually land in the top-k?*

V5's improvement is dramatic:

| Config | sources_in_context | What it retrieves |
|---|---|---|
| v3-baseline | 0.100 | Only `cv.md` chunks, on every query |
| v5-baseline (k=4) | 0.800 | Correctly diverse — cv.md + project files |
| v5-k6 | **1.000** | All expected sources retrieved |

Inspecting the actual retrieved sources for V3 across five source-test cases makes the failure mode obvious:

```
v3-baseline, all five test cases:
  E1, E2, E3, M2, M3 → sources retrieved = ['cv.md', 'cv.md', 'cv.md', 'cv.md']
```

V3's naive concatenated query `f"{user_request}\n\n{jd_text}"` is verbose and JD-shaped enough that ChromaDB's vector similarity always lands on `cv.md` — the file with the most JD-like content. **Project files never surface.** For cover letters, interview prep, and observability-emphasis queries that need concrete project evidence, V3 fundamentally cannot retrieve the right chunks.

V5's query rewriter produces concise, keyword-dense search strings (e.g., `"Python LangChain RAG PostgreSQL pgvector FastAPI Docker LangGraph"`) that retrieve project chunks reliably. The metadata filter further sharpens retrieval for cover-letter and interview-prep intents by excluding `cv.md` content that doesn't help those handlers.

This is the headline V5 win. Retrieval is the foundation everything else builds on, and V5 fixes it.

### Finding 2 — The `matched_skills_present` paradox

V3's apparent win on `matched_skills_present` (1.000 vs V5's 0.600) initially looked like a regression. Drilling into a representative case (E1, "Should I apply to Elad?") tells a different story:

```
v3-baseline E1:
  sources retrieved: ['cv.md', 'cv.md', 'cv.md', 'cv.md']
  matched_skills: ['Python development experience', 'LangChain', 'RAG systems',
                   'PostgreSQL', 'pgvector', 'FastAPI', 'Docker', 'Hebrew language skills']
  → all 5 expected (Python, FastAPI, PostgreSQL, LangChain, Hebrew) found → 1.00

v5-baseline E1:
  sources retrieved: ['project_rag_hub.md' × 2, 'cv.md' × 2]
  matched_skills: ['Python development experience', 'LangChain and RAG systems',
                   'PostgreSQL and pgvector', 'FastAPI', 'Docker']
  → 4 of 5 found (Hebrew missing) → 0.80
```

V3 was scoring matched_skills "correctly" *by accident*. Because it dumps only `cv.md` chunks, the LLM sees the comprehensive flat skill listing under "Core Technologies" plus the "Languages: Hebrew (Native)" line, and transcribes them straight into `matched_skills`. V5 retrieves correctly diverse sources — including project chunks the user actually needs — but the fit-analyzer LLM, faced with both rich project context and CV content, biases its `matched_skills` enumeration toward the project's specific tech stack and underutilizes the CV's flat skill listing.

**This is a synthesis problem, not a retrieval problem.** The fit-analyzer's prompt grounds the model in retrieved context but doesn't explicitly instruct it to *exhaustively* enumerate skills from CV-category chunks. The result is qualitatively *better* in many ways — V5's `matched_skills` are more specific and informative (e.g., "LangChain and RAG systems" vs. V3's separate "LangChain", "RAG systems") — but loses some breadth.

The right fix is prompt engineering, scheduled as V5.5 work below. The wrong fix would be reverting to V3's retrieval behavior to chase the 1.000 score — that would re-break four of the five source-retrieval cases.

### Finding 3 — k=6 is the production sweet spot

Comparing the three V5 configurations against each other:

| Config | sources | matched_skills | cost | latency |
|---|---|---|---|---|
| v5-k2 | 0.700 | **0.867** | $0.0078 | 7.50s |
| v5-k4 (baseline) | 0.800 | 0.600 | $0.0079 | 8.61s |
| **v5-k6** | **1.000** | 0.600 | $0.0078 | **7.13s** |

`k=6` is Pareto-best: perfect retrieval, no synthesis quality penalty vs. k=4, lowest latency (concurrency variance, not deterministic), same cost. The case for switching the default from k=4 to k=6 is straightforward.

`k=2` is an interesting wrinkle. Its `matched_skills_present` of 0.867 (vs. 0.600 at k=4/k=6) is genuinely the highest of all configs. The hypothesis: with only 2 chunks to work with, the LLM scrutinizes each more carefully and pulls out more diverse skills — including ones it would otherwise overlook in a busier context window. The trade-off (0.700 on sources_in_context vs 1.000 at k=6) makes k=2 a poor default but a candidate for future intent-specific tuning: a fit-analysis chain could plausibly benefit from `k=2` while a cover-letter chain benefits from `k=6`. Worth measuring before committing to it.

### Finding 4 — Retrieval is solved; synthesis is the next bottleneck

Case M3 (cover letter emphasizing hackathon experience) reveals the gap that's now visible since retrieval is no longer the limiting factor.

Ground truth: `must_reference_sources=["project_shelfguard.md"]`, `must_contain=["ShelfGuard"]`. With k=6, `sources_in_context = 1.00` — `project_shelfguard.md` *was* retrieved. But `must_contain = 0.00` — the cover letter never mentions "ShelfGuard." Inspecting the retrieved chunks explains why:

```
v5-k6 M3 retrieval (k=6):
  ['project_rag_hub.md' × 5, 'project_shelfguard.md' × 1]
```

ShelfGuard is technically in context, but outnumbered 5-to-1 by RAG Hub chunks. The LLM treats the lone ShelfGuard chunk as outlier signal and writes the cover letter around the project it sees more evidence for.

**This is a chunk-rebalancing problem, not a retrieval-coverage problem.** Single-query similarity search will always be dominated by the most semantically central project unless we change the retrieval strategy. Multi-query retrieval — one query for "hackathon" specifically, one for general skill match, then merge with a per-source cap — is the cleanest known fix. Scheduled as V5.5 below.

### Finding 5 — Evaluator design lessons

Case O2 (cover letter for a Java role) failed `must_not_contain = 0.00` in all four configurations. The forbidden strings were "Java" and "Spring Boot." Inspecting the actual output:

> *"While my primary experience is with Python, I have a deep understanding of backend principles that are transferable to Java and Spring Boot..."*

The cover letter is doing the *right* thing — explicitly acknowledging Python as the candidate's primary stack, framing skills as transferable rather than claiming Java experience. But the evaluator does a strict substring match, so the words "Java" and "Spring Boot" appearing anywhere — even in honest "I don't have this" framing — count as failure.

**This is an evaluator problem, not a system problem.** The fix is to refine `must_not_contain` from substring-match to LLM-as-judge: *"Does this output fabricate experience with X?"* — which can distinguish honest mention-of-gap from fabricated claim. Scheduled as V5.5.

The 0.00 result is real signal about the evaluator's brittleness rather than a real grounding failure. Worth surfacing as a methodology limitation.

---

## Recommendations

### Immediate

1. **Update `config/settings.py`:** change `retrieval_k` from `4` to `6`. Achieves perfect retrieval hit-rate at no quality or cost penalty.

### V5.5 (planned)

1. **Improve fit-analyzer skill enumeration.** Modify `FIT_ANALYZER_PROMPT` to explicitly instruct the model to exhaustively enumerate skills from CV-category chunks when calculating `matched_skills`, in addition to mentioning project-specific stack. Target: lift `matched_skills_present` from 0.60 toward 1.00 on E1/M2 without re-breaking retrieval.
2. **Multi-query retrieval for emphasis cases.** When the user request emphasizes a specific narrative (hackathon, observability, etc.), run a second query biased toward that emphasis and merge with the rewriter's primary query, capping per-source contribution. Target: fix M3-class failures.
3. **Refine `must_not_contain` evaluator.** Replace substring match with LLM-as-judge to distinguish fabrication from honest gap acknowledgment. Target: lift the O2-class measurement from artifact noise to genuine signal.

### Out-of-scope follow-ups (V6 and beyond)

- **Larger labeled dataset.** 12 cases gave us clear directional signal but tight ranges become brittle. 50–100 cases would allow tightening `fit_score_min/max` ranges and adding per-intent breakdowns.
- **Cross-JD diversity.** Three JDs (strong-fit AI, junior backend, Java enterprise, DevOps K8s) cover the major fit-pattern axes but a wider JD library would stress-test the rewriter under more role profiles.

---

## Limitations

This evaluation has clear scope boundaries. Surface them so the conclusions land in context:

- **Small sample size (n=12).** Enough to detect 8–10× differences in retrieval hit-rate, but tight ranges in `fit_score_in_range` and `matched_skills_present` are inherently brittle. A 50-case version would let us tighten score bounds with statistical confidence.
- **Rewriter and filter measured as a combined unit.** We didn't isolate which component contributes more to the retrieval improvement. The two were designed as a coherent V5 unit; separating them was descoped.
- **Substring-based content evaluators.** `must_contain` and `must_not_contain` use case-insensitive substring matching. This is brittle for cases like O2 (above) where the LLM mentions a forbidden string in honest-acknowledgment framing. LLM-as-judge would be more robust.
- **No human inter-rater on ground truth.** Labels were authored by a single annotator (the author). Two-rater agreement is the standard rigor improvement for next-version eval.
- **gpt-4o-mini only.** All experiments used `gpt-4o-mini` as the system LLM and no model comparison was run. The cost/quality trade-off vs `gpt-4o` or Ollama models is unmeasured.

---

## Reproducing these results

All experiments are reproducible via the V5 evaluation runner. Prerequisites: a valid `.env`, ChromaDB built (`python -m ingestion.portfolio_ingest`), and `langsmith` Python client configured.

```bash
# V3 baseline — pre-V5 behavior
python -m evaluation.run_eval \
    --experiment-name v3-baseline \
    --no-rewriter --no-filter --k 4

# V5 baseline — current main-branch behavior
python -m evaluation.run_eval \
    --experiment-name v5-baseline \
    --k 4

# V5 with tighter retrieval
python -m evaluation.run_eval \
    --experiment-name v5-k2 \
    --k 2

# V5 with wider retrieval — the recommended config
python -m evaluation.run_eval \
    --experiment-name v5-k6 \
    --k 6
```

Each command produces a separate experiment in the `JobFit` LangSmith project, comparable side-by-side under **Datasets & Experiments → JobFit-V5**. Aggregates can be recomputed by exporting the experiment CSV and grouping per evaluator column.

---

## Acknowledgments

Built as the LangChain & LangSmith capstone for the CyberPro AI Developer Bootcamp (ELAD Software). The V5 work was the moment the project transitioned from "build features and hope they help" to "measure, then decide." That shift is the single most useful lesson the project taught.
