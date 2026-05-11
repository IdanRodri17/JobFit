"""V5 evaluation runner: measure JobFit pipeline quality via LangSmith experiments.

This is the file that turns 'we have a labeled dataset' into 'we have
a command that runs the full pipeline against it and gives us a
scorecard.' Four moving parts:

  1. Dataset upload — pushes the local DATASET to LangSmith as a hosted
     Dataset. Runs once, then reuses on every subsequent eval run.
     Force re-upload with --rebuild-dataset.

  2. Configurable pipeline (make_target) — wraps the JobFit pipeline
     so the rewriter, the metadata filter, and the retrieval k are all
     toggleable. That's how we compare V3-baseline vs V5-current vs
     k=2 vs k=6 in separate experiments.

  3. Evaluators — eight pure functions, one per ground-truth field.
     Each consumes (run, example) and returns a dict {key, score,
     comment}. Score None signals 'no expectation for this case'.

  4. Cost tracking — get_openai_callback() wraps the per-example
     pipeline call. Token + dollar counts ride out in the run's
     outputs and surface as the 'cost_usd' evaluator column.

Recommended experiments (matches PROJECT_SPECIFICATION.md §9):
    python -m evaluation.run_eval --experiment-name v3-baseline    --no-rewriter --no-filter --k 4
    python -m evaluation.run_eval --experiment-name v5-rewriter    --no-filter --k 4
    python -m evaluation.run_eval --experiment-name v5-full        --k 4
    python -m evaluation.run_eval --experiment-name v5-k2          --k 2
    python -m evaluation.run_eval --experiment-name v5-k6          --k 6

Pre-requisites:
    - LANGCHAIN_API_KEY in .env
    - OPENAI_API_KEY in .env
    - ChromaDB built (python -m ingestion.portfolio_ingest)
"""

import argparse
import json

from langchain_community.callbacks import get_openai_callback
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.utils import LangSmithNotFoundError

from assistant.chains.query_rewriter import rewrite_query
from assistant.core import HANDLERS, INTENT_TO_CATEGORIES
from assistant.router import classify_intent
from config.logging import configure_logging, get_logger
from config.settings import settings
from evaluation.eval_dataset import DATASET
from retrieval.portfolio_retriever import get_relevant_context

configure_logging()
logger = get_logger("jobfit.eval")

DATASET_NAME = "JobFit-V5"


# ──────────────────────────────────────────────────────────
# Dataset upload — runs once per dataset name.
# ──────────────────────────────────────────────────────────
def ensure_dataset(client: Client, force_rebuild: bool = False):
    """Get the LangSmith dataset, creating it from local DATASET if needed."""
    if force_rebuild:
        try:
            existing = client.read_dataset(dataset_name=DATASET_NAME)
            client.delete_dataset(dataset_id=existing.id)
            logger.info("Deleted existing dataset %r for rebuild", DATASET_NAME)
        except LangSmithNotFoundError:
            pass

    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info("Using existing LangSmith dataset %r", DATASET_NAME)
        return dataset
    except LangSmithNotFoundError:
        pass

    logger.info(
        "Creating LangSmith dataset %r with %d examples",
        DATASET_NAME,
        len(DATASET),
    )
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "JobFit V5 labeled evaluation set: 12 cases across "
            "easy/medium/tricky/out_of_scope buckets. See "
            "evaluation/eval_dataset.py for ground-truth schema."
        ),
    )
    # Parallel lists — most portable across langsmith versions.
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {"jd_text": c.jd_text, "user_request": c.user_request} for c in DATASET
        ],
        outputs=[c.ground_truth.model_dump() for c in DATASET],
        metadata=[
            {
                "case_id": c.case_id,
                "category": c.category,
                "rationale": c.rationale,
            }
            for c in DATASET
        ],
    )
    return dataset


# ──────────────────────────────────────────────────────────
# Configurable pipeline runner — closure pattern.
# ──────────────────────────────────────────────────────────
def make_target(k: int, use_rewriter: bool, use_filter: bool):
    """Build the target function passed to evaluate().

    The closure bakes in the experiment configuration. evaluate() calls
    target(inputs) once per example; the return becomes run.outputs in
    the evaluators below.

    Why duplicate process_request() here instead of parameterizing it?
    Keeping the knobs in the eval runner means core.py stays a clean
    'do the right thing' default. Experimental variations live in the
    place that experiments belong — the evaluation module.
    """

    def target(inputs: dict) -> dict:
        jd_text = inputs["jd_text"]
        user_request = inputs["user_request"]

        with get_openai_callback() as cb:
            # 1. Classify intent
            classification = classify_intent(user_request)

            # 2. Build retrieval query
            if use_rewriter:
                rq = rewrite_query(jd_text, user_request)
                query = rq.query
            else:
                # V3 baseline behavior — naive concatenation
                query = f"{user_request}\n\n{jd_text}"

            # 3. Pick category filter (or skip)
            categories = (
                INTENT_TO_CATEGORIES.get(classification.intent) if use_filter else None
            )

            # 4. Retrieve
            context = get_relevant_context(query, k=k, categories=categories)

            # 5. Dispatch to handler
            handler = HANDLERS.get(classification.intent)
            handler_result = handler(jd_text, context) if handler else None

        result_dict = (
            handler_result.model_dump() if handler_result is not None else None
        )

        return {
            "intent": classification.intent,
            "intent_confidence": classification.confidence,
            "retrieved_context": context,
            "result": result_dict,
            "_meta": {
                "total_tokens": cb.total_tokens,
                "total_cost_usd": cb.total_cost,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
            },
        }

    return target


# ──────────────────────────────────────────────────────────
# Evaluators — one per GroundTruth field, plus cost.
# Signature: (run, example) -> dict {key, score, comment}
# Returning score=None means "this evaluator doesn't apply to this case."
# ──────────────────────────────────────────────────────────
def _flatten_result(result: dict | None) -> str:
    """Flatten a Pydantic-dumped chain result into one searchable string.

    Used by content evaluators so they can substring-match any field
    (opening_paragraph, body_paragraphs, suggested_answer, reasoning…)
    uniformly without per-schema logic.
    """
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False)


def _all_present(needles: list[str], haystack: str) -> tuple[int, int]:
    """Count case-insensitive substring hits."""
    h = haystack.lower()
    hits = sum(1 for n in needles if n.lower() in h)
    return hits, len(needles)


def eval_intent(run, example) -> dict:
    """Score 1.0 if router picked the expected intent."""
    expected = example.outputs.get("expected_intent")
    if expected is None:
        return {"key": "intent_correct", "score": None}
    actual = run.outputs.get("intent")
    return {
        "key": "intent_correct",
        "score": 1.0 if actual == expected else 0.0,
        "comment": f"expected={expected}, got={actual}",
    }


def eval_fit_score_in_range(run, example) -> dict:
    """Score 1.0 if FitReport.overall_score sits inside [min, max]."""
    lo = example.outputs.get("fit_score_min")
    hi = example.outputs.get("fit_score_max")
    if lo is None or hi is None:
        return {"key": "fit_score_in_range", "score": None}
    result = run.outputs.get("result")
    if not result or "overall_score" not in result:
        return {
            "key": "fit_score_in_range",
            "score": 0.0,
            "comment": "no FitReport in output",
        }
    score = result["overall_score"]
    return {
        "key": "fit_score_in_range",
        "score": 1.0 if lo <= score <= hi else 0.0,
        "comment": f"expected [{lo},{hi}], got {score}",
    }


def eval_matched_skills_present(run, example) -> dict:
    """Fraction of must_appear_in_matched that show up in matched_skills.

    The phantom-gap test. A low score here means retrieval missed
    chunks documenting skills the candidate actually has.
    """
    expected = example.outputs.get("must_appear_in_matched", [])
    if not expected:
        return {"key": "matched_skills_present", "score": None}
    result = run.outputs.get("result")
    if not result or "matched_skills" not in result:
        return {"key": "matched_skills_present", "score": 0.0}
    matched_blob = " ".join(result.get("matched_skills", []))
    hits, total = _all_present(expected, matched_blob)
    return {
        "key": "matched_skills_present",
        "score": hits / total,
        "comment": f"{hits}/{total} expected skills found in matched_skills",
    }


def eval_gap_skills_present(run, example) -> dict:
    """Fraction of must_appear_in_gap that show up in gap_skills.

    Honesty test. A low score means the system fabricated matches for
    skills the candidate doesn't actually have.
    """
    expected = example.outputs.get("must_appear_in_gap", [])
    if not expected:
        return {"key": "gap_skills_present", "score": None}
    result = run.outputs.get("result")
    if not result or "gap_skills" not in result:
        return {"key": "gap_skills_present", "score": 0.0}
    gap_blob = " ".join(result.get("gap_skills", []))
    hits, total = _all_present(expected, gap_blob)
    return {
        "key": "gap_skills_present",
        "score": hits / total,
        "comment": f"{hits}/{total} expected gaps found in gap_skills",
    }


def eval_sources_in_context(run, example) -> dict:
    """Fraction of expected source filenames that surfaced in retrieval.

    Retrieval hit-rate — directly measures whether the chunks we need
    actually appear in the top-k. Chunks are prefixed with
    [Source: filename.md] so a substring match is sufficient.
    """
    expected = example.outputs.get("must_reference_sources", [])
    if not expected:
        return {"key": "sources_in_context", "score": None}
    context = run.outputs.get("retrieved_context", "")
    hits, total = _all_present(expected, context)
    return {
        "key": "sources_in_context",
        "score": hits / total,
        "comment": f"{hits}/{total} expected sources retrieved",
    }


def eval_must_contain(run, example) -> dict:
    """Fraction of must_contain strings present in the generated output."""
    expected = example.outputs.get("must_contain", [])
    if not expected:
        return {"key": "must_contain", "score": None}
    flat = _flatten_result(run.outputs.get("result"))
    hits, total = _all_present(expected, flat)
    return {
        "key": "must_contain",
        "score": hits / total,
        "comment": f"{hits}/{total} required strings present",
    }


def eval_must_not_contain(run, example) -> dict:
    """1.0 if NONE of must_not_contain appear; 0.0 if any do.

    Grounding stress test — cover letters must not fabricate experience
    the candidate doesn't have.
    """
    forbidden = example.outputs.get("must_not_contain", [])
    if not forbidden:
        return {"key": "must_not_contain", "score": None}
    flat = _flatten_result(run.outputs.get("result")).lower()
    hits = [f for f in forbidden if f.lower() in flat]
    if hits:
        return {
            "key": "must_not_contain",
            "score": 0.0,
            "comment": f"fabricated content detected: {hits}",
        }
    return {"key": "must_not_contain", "score": 1.0}


def eval_cost_usd(run, example) -> dict:
    """Pass-through: surface per-example cost as an experiment column."""
    meta = (run.outputs or {}).get("_meta") or {}
    return {
        "key": "cost_usd",
        "score": meta.get("total_cost_usd", 0.0),
        "comment": f"{meta.get('total_tokens', 0)} tokens",
    }


EVALUATORS = [
    eval_intent,
    eval_fit_score_in_range,
    eval_matched_skills_present,
    eval_gap_skills_present,
    eval_sources_in_context,
    eval_must_contain,
    eval_must_not_contain,
    eval_cost_usd,
]


# ──────────────────────────────────────────────────────────
# CLI runner.
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a JobFit V5 evaluation experiment against LangSmith."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=settings.retrieval_k,
        help=f"Retrieval top-k (default: {settings.retrieval_k}).",
    )
    parser.add_argument(
        "--no-rewriter",
        action="store_true",
        help="Disable the V5 query rewriter (use V3 concat baseline).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable the V5 metadata category filter.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="jobfit-v5",
        help="Prefix for the LangSmith experiment name.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Delete and re-upload the LangSmith dataset before running.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Concurrent target calls (default: 4).",
    )
    args = parser.parse_args()

    use_rewriter = not args.no_rewriter
    use_filter = not args.no_filter

    config_label = (
        f"k={args.k}, "
        f"rewriter={'on' if use_rewriter else 'off'}, "
        f"filter={'on' if use_filter else 'off'}"
    )
    full_experiment_name = (
        f"{args.experiment_name}-k{args.k}" f"-rw{int(use_rewriter)}-f{int(use_filter)}"
    )

    logger.info("Experiment: %r — %s", full_experiment_name, config_label)

    client = Client()
    dataset = ensure_dataset(client, force_rebuild=args.rebuild_dataset)

    target = make_target(
        k=args.k,
        use_rewriter=use_rewriter,
        use_filter=use_filter,
    )

    evaluate(
        target,
        data=dataset.name,
        evaluators=EVALUATORS,
        experiment_prefix=full_experiment_name,
        max_concurrency=args.max_concurrency,
        metadata={
            "k": args.k,
            "use_rewriter": use_rewriter,
            "use_filter": use_filter,
        },
        client=client,
    )

    print(f"\n✓ Experiment {full_experiment_name!r} complete.")
    print(f"  Config: {config_label}")
    print(f"  Open: https://smith.langchain.com\n")


if __name__ == "__main__":
    main()
