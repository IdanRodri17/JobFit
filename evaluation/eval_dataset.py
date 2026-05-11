"""V5 evaluation dataset: labeled examples for measuring JobFit pipeline quality.

This module is the single source of truth for what 'correct' looks
like across the JobFit pipeline. Every experiment in run_eval.py
measures system outputs against these expected values.

Composition (12 cases):
    - 3 easy: clear inputs with clear correct behavior
    - 4 medium: require synthesis across multiple portfolio chunks
    - 3 tricky: edge cases for router and retrieval
    - 2 out-of-scope: grounding test — system must not fabricate

Why Pydantic and not raw JSON/YAML?
    - Validation at import: a typo in 'expected_intent' fails at
      module load, not at eval time.
    - IDE autocomplete on every field.
    - Easy mapping to LangSmith's example dict format in run_eval.py
      via .model_dump().

Smoke test:
    python -m evaluation.eval_dataset
"""

from typing import Literal

from pydantic import BaseModel, Field

from models.schemas import Intent

# ─── JD constants ──────────────────────────────────────────
# Inlined here so the dataset is self-contained — no file I/O,
# no path resolution surprises during eval runs. Kept short:
# eval JDs don't need to read like LinkedIn postings, they just
# need enough signal to discriminate fit profiles.

ELAD_JD = """\
AI Developer — Elad Systems (Tel Aviv, Hybrid)

Required:
- 3+ years Python development experience
- Hands-on with LangChain and RAG systems
- PostgreSQL and pgvector
- FastAPI and Docker

Nice to have:
- LangGraph for agentic workflows
- Hebrew language skills
- Production observability (LangSmith, Prometheus, Grafana)
"""

JUNIOR_BACKEND_JD = """\
Junior Backend Developer — TechStart (Tel Aviv)

Required:
- 1+ years Python experience
- FastAPI or Flask
- PostgreSQL basics
- Git workflow

Nice to have:
- Docker
- REST API design
"""

JAVA_ENTERPRISE_JD = """\
Senior Java Backend Engineer — FinCorp Israel

Required:
- 5+ years Java/Spring Boot
- JPA / Hibernate
- Oracle or MSSQL
- Microservices architecture
- Kafka

Nice to have:
- Kubernetes
- AWS
"""

DEVOPS_K8S_JD = """\
DevOps Engineer — CloudOps (Remote)

Required:
- 3+ years DevOps experience
- Kubernetes in production
- Terraform / Pulumi
- AWS or GCP
- CI/CD pipelines
- Linux administration

Nice to have:
- Helm charts
- Service mesh (Istio)
"""


# ─── Schemas ───────────────────────────────────────────────
CaseCategory = Literal["easy", "medium", "tricky", "out_of_scope"]


class GroundTruth(BaseModel):
    """Expected system behavior for a labeled example.

    Every field is optional — different cases test different things.
    An empty list means 'no expectation', not 'must be empty'.
    """

    # Routing — used to evaluate the V2 router
    expected_intent: Intent | None = None

    # Fit-score band — used to evaluate analyze_fit calibration
    fit_score_min: int | None = None
    fit_score_max: int | None = None

    # Skill-grounding checks (analyze_fit):
    #   must_appear_in_matched: candidate HAS these → must be in
    #     FitReport.matched_skills (phantom-gap test).
    #   must_appear_in_gap: candidate truly LACKS these → must be
    #     in FitReport.gap_skills (no-fabrication test).
    must_appear_in_matched: list[str] = Field(default_factory=list)
    must_appear_in_gap: list[str] = Field(default_factory=list)

    # Retrieval hit-rate — source filenames expected to appear in the
    # retrieved context (e.g. 'project_rag_hub.md', 'cv.md').
    must_reference_sources: list[str] = Field(default_factory=list)

    # Free-text content checks — strings the generated output MUST
    # or MUST NOT contain. Used for cover letter / interview prep
    # cases where structured fields don't tell the whole story.
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)


class LabeledExample(BaseModel):
    """A single labeled case in the eval dataset."""

    case_id: str
    category: CaseCategory
    jd_text: str
    user_request: str
    ground_truth: GroundTruth
    rationale: str


# ─── The dataset ───────────────────────────────────────────
DATASET: list[LabeledExample] = [
    # ════════════════════════ EASY ════════════════════════
    LabeledExample(
        case_id="E1_fit_strong_match",
        category="easy",
        jd_text=ELAD_JD,
        user_request="Should I apply to this role?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
            fit_score_min=75,
            fit_score_max=95,
            # Candidate evidence for each: cv.md Core Technologies +
            # project_rag_hub.md tech stack + cv.md Languages section.
            # This case is the V3-phantom-gap regression test rolled in:
            # if LangChain or Hebrew appears in gap_skills instead of
            # matched_skills, retrieval is missing the cv.md chunks.
            must_appear_in_matched=[
                "Python",
                "FastAPI",
                "PostgreSQL",
                "LangChain",
                "Hebrew",
            ],
            must_reference_sources=["project_rag_hub.md", "cv.md"],
        ),
        rationale=(
            "Baseline strong-fit case. Tests router accuracy, score "
            "calibration, AND the V3 phantom-gap pattern (LangChain + "
            "Hebrew are documented in cv.md but were flagged as gaps "
            "by V3-baseline retrieval)."
        ),
    ),
    LabeledExample(
        case_id="E2_cover_letter_strong_match",
        category="easy",
        jd_text=ELAD_JD,
        user_request="Write me a cover letter for this position.",
        ground_truth=GroundTruth(
            expected_intent="generate_cover_letter",
            must_reference_sources=["project_rag_hub.md"],
            must_contain=["RAG"],
        ),
        rationale=(
            "Baseline cover letter. Should retrieve and name the RAG "
            "Hub project — the most directly relevant in the portfolio."
        ),
    ),
    LabeledExample(
        case_id="E3_interview_prep_strong_match",
        category="easy",
        jd_text=ELAD_JD,
        user_request="What technical questions might they ask me?",
        ground_truth=GroundTruth(
            expected_intent="interview_prep",
            must_reference_sources=["project_rag_hub.md"],
            must_contain=["RAG"],
        ),
        rationale=(
            "Baseline interview prep. Technical questions should pull "
            "from real project work, not abstract concepts."
        ),
    ),
    # ════════════════════════ MEDIUM ════════════════════════
    LabeledExample(
        case_id="M1_fit_junior_role",
        category="medium",
        jd_text=JUNIOR_BACKEND_JD,
        user_request="Am I a good fit for this junior role?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
            fit_score_min=70,
            fit_score_max=95,
            must_appear_in_matched=["Python", "FastAPI", "PostgreSQL"],
        ),
        rationale=(
            "Mismatch in seniority direction — candidate is over-qualified "
            "for a junior role but matches all required skills. Tests "
            "that the score reflects skill match, not penalizes over-fit."
        ),
    ),
    LabeledExample(
        case_id="M2_fit_observability_emphasis",
        category="medium",
        jd_text=ELAD_JD,
        user_request="The role mentions observability. Do I have relevant experience for that part?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
            must_appear_in_matched=["Prometheus", "Grafana"],
            must_reference_sources=["project_rag_hub.md"],
        ),
        rationale=(
            "Tests whether the rewriter pulls observability-specific "
            "terms when the user emphasizes them. Prometheus/Grafana "
            "evidence is in project_rag_hub.md Phase 4."
        ),
    ),
    LabeledExample(
        case_id="M3_cover_letter_hackathon_emphasis",
        category="medium",
        jd_text=ELAD_JD,
        user_request="Write a cover letter that emphasizes my hackathon experience.",
        ground_truth=GroundTruth(
            expected_intent="generate_cover_letter",
            must_reference_sources=["project_shelfguard.md"],
            must_contain=["ShelfGuard"],
        ),
        rationale=(
            "Cover letter with a specific narrative requirement. Should "
            "retrieve project_shelfguard.md (the hackathon project) and "
            "name it explicitly."
        ),
    ),
    LabeledExample(
        case_id="M4_fit_java_role_low_match",
        category="medium",
        jd_text=JAVA_ENTERPRISE_JD,
        user_request="Should I apply for this Java role?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
            fit_score_min=10,
            fit_score_max=45,
            must_appear_in_gap=["Java", "Spring"],
        ),
        rationale=(
            "Low-fit case. Candidate has no Java/Spring experience. "
            "Tests that the system correctly identifies skill absence "
            "instead of fabricating matches."
        ),
    ),
    # ════════════════════════ TRICKY ════════════════════════
    LabeledExample(
        case_id="T1_ambiguous_request",
        category="tricky",
        jd_text=ELAD_JD,
        user_request="Help me with this job posting.",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
        ),
        rationale=(
            "Per the V2 router disambiguation rules, 'help me with this' "
            "with no other signal routes to analyze_fit (the natural "
            "starting point). Low-confidence is acceptable here — we "
            "test the routing decision, not the score."
        ),
    ),
    LabeledExample(
        case_id="T2_multi_intent_request",
        category="tricky",
        jd_text=ELAD_JD,
        user_request="Analyze my fit and write me a cover letter.",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
        ),
        rationale=(
            "Two intents in one request. Per router rules, pick the "
            "FIRST/PRIMARY intent (analyze_fit) with confidence ~0.7. "
            "Tests router conflict resolution."
        ),
    ),
    LabeledExample(
        case_id="T3_emotional_phrasing",
        category="tricky",
        jd_text=ELAD_JD,
        user_request="I'm nervous about applying. Am I even qualified for this?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
        ),
        rationale=(
            "Non-canonical phrasing wrapped in emotional context. The "
            "router must look past the framing to the underlying intent "
            "(fit assessment)."
        ),
    ),
    # ═══════════════════════ OUT-OF-SCOPE ═══════════════════════
    LabeledExample(
        case_id="O1_devops_k8s_no_match",
        category="out_of_scope",
        jd_text=DEVOPS_K8S_JD,
        user_request="Should I apply for this DevOps role?",
        ground_truth=GroundTruth(
            expected_intent="analyze_fit",
            fit_score_min=10,
            fit_score_max=40,
            must_appear_in_gap=["Kubernetes", "Terraform"],
        ),
        rationale=(
            "Hard skill mismatch with no portfolio evidence. System must "
            "correctly identify the gap rather than fabricating K8s/"
            "Terraform experience. Score should reflect the mismatch."
        ),
    ),
    LabeledExample(
        case_id="O2_cover_letter_java_no_match",
        category="out_of_scope",
        jd_text=JAVA_ENTERPRISE_JD,
        user_request="Write me a cover letter for this Java role.",
        ground_truth=GroundTruth(
            expected_intent="generate_cover_letter",
            # The output MUST NOT claim Java/Spring experience because
            # none exists in the portfolio. Grounding stress test.
            must_not_contain=["Java", "Spring Boot"],
        ),
        rationale=(
            "Grounding stress test. Candidate has no Java experience, "
            "so the cover letter must not fabricate any. Honest output "
            "based on transferable skills only — or a refusal."
        ),
    ),
]


# ─── Convenience helpers ───────────────────────────────────
def get_by_id(case_id: str) -> LabeledExample:
    """Lookup a single example by ID."""
    for case in DATASET:
        if case.case_id == case_id:
            return case
    raise KeyError(f"No labeled example with case_id={case_id!r}")


def get_by_category(category: CaseCategory) -> list[LabeledExample]:
    """Filter examples by difficulty category."""
    return [c for c in DATASET if c.category == category]


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    print(f"✓ Loaded {len(DATASET)} labeled examples\n")

    categories: list[CaseCategory] = ["easy", "medium", "tricky", "out_of_scope"]
    print("Breakdown:")
    for cat in categories:
        cases = get_by_category(cat)
        print(f"  {cat:15s} {len(cases)}")
        for c in cases:
            print(f"      • {c.case_id}")

    print(f"\n─── Sample case (full JSON) ───")
    print(DATASET[0].model_dump_json(indent=2))
