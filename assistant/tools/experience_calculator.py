"""V6 tool: deterministic years-of-experience calculation.

Why this tool exists: the LLM cannot do date math reliably. ...
[keep the original docstring]

Skill data lives in data/portfolio/skills.json — co-located with the
portfolio markdown so a single content owner maintains one place.
The tool loads it at module import and caches in SKILL_START_DATES.

To add a skill: edit skills.json. No code change required.
"""

import json
from datetime import date

from pydantic import BaseModel, Field

from config.settings import settings

# ─── Skill registry — loaded from JSON at module import ─────
_SKILLS_PATH = settings.portfolio_dir / "skills.json"


def _load_skill_dates() -> dict[str, date]:
    """Load and normalize the skill → start-date map from disk.

    Keys lowercased for case-insensitive lookup. Values parsed from
    ISO date strings into datetime.date so the rest of the module
    can use them directly without type conversion at call time.

    Returns an empty dict if the file is missing — the tool then
    treats every skill as 'unknown', surfaced via is_known=False.

    Multi-user note: for a hosted deployment, this loader would take
    a user_id and resolve to user-specific paths
    (e.g. data/users/<user_id>/portfolio/skills.json). The tool's
    public signature would gain a user_id parameter threaded through
    from the request context. That refactor is out of scope for V6 —
    JobFit is single-user by design — but the data-file layout above
    is the right foundation for it.
    """
    if not _SKILLS_PATH.exists():
        return {}
    raw = json.loads(_SKILLS_PATH.read_text(encoding="utf-8"))
    return {
        key.strip().lower(): date.fromisoformat(value) for key, value in raw.items()
    }


SKILL_START_DATES: dict[str, date] = _load_skill_dates()


# ─── Output schema ─────────────────────────────────────────
class ExperienceCalculation(BaseModel):
    """Structured result of a years-of-experience lookup.

    Returned in BOTH the success path (is_known=True) and the
    unknown-skill path (is_known=False). The boolean lets the LLM
    synthesizer downstream distinguish "0 years because I don't have
    data" from "0 years because the skill is brand new" — without
    that flag, the LLM would fabricate the difference.
    """

    skill: str = Field(
        ...,
        description="Normalized (lowercased, trimmed) skill name.",
    )
    is_known: bool = Field(
        ...,
        description=(
            "True if the skill is in the registry. False signals "
            "'no data' — synthesizer must NOT report a number."
        ),
    )
    years: float = Field(
        ...,
        description="Years of experience, rounded to one decimal place. 0.0 if not known.",
    )
    months: int = Field(
        ...,
        description="Total months of experience, rounded to integer. 0 if not known.",
    )
    start_date: date | None = Field(
        default=None,
        description="When experience with this skill began. None if not known.",
    )
    end_date: date = Field(
        ...,
        description="Reference 'now' date used in the calculation.",
    )


# ─── Public API ────────────────────────────────────────────
def calculate_experience(
    skill: str,
    today: date | None = None,
) -> ExperienceCalculation:
    """Compute years and months of experience with a given skill.

    Args:
        skill: Skill or technology name (case-insensitive). Whitespace
            and case are normalized before lookup.
        today: Override the reference 'now' date. Defaults to
            date.today(). Used in tests to keep results reproducible.

    Returns:
        ExperienceCalculation. Always returns — when the skill is not
        in the registry, returns with is_known=False rather than
        raising. The caller decides how to handle 'unknown'.
    """
    reference = today or date.today()
    key = skill.strip().lower()
    start = SKILL_START_DATES.get(key)

    if start is None:
        return ExperienceCalculation(
            skill=key,
            is_known=False,
            years=0.0,
            months=0,
            start_date=None,
            end_date=reference,
        )

    delta_days = (reference - start).days
    # 365.25 accounts for leap years over a multi-year span.
    years = round(delta_days / 365.25, 1)
    # 30.4375 = 365.25 / 12, average days per month.
    months = int(round(delta_days / 30.4375))

    return ExperienceCalculation(
        skill=key,
        is_known=True,
        years=years,
        months=months,
        start_date=start,
        end_date=reference,
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    test_skills = [
        "Python",  # known
        "FastAPI",  # known
        "LangChain",  # known
        "  pgvector  ",  # known + tests whitespace/case normalization
        "Java",  # unknown — tests the is_known=False branch
        "Rust",  # unknown
    ]

    print(f"Reference 'now': {date.today()}\n")
    print(f"Registry size: {len(SKILL_START_DATES)} skills\n")

    for skill in test_skills:
        result = calculate_experience(skill)
        marker = "✓" if result.is_known else "⚠"
        if result.is_known:
            print(
                f"{marker} {skill:15s}  →  {result.years} years "
                f"({result.months} months)  start: {result.start_date}"
            )
        else:
            print(f"{marker} {skill:15s}  →  unknown (not in registry)")
    print()
    print("→ Note: no LangSmith trace for this run — pure Python, no LLM call.")
