"""V6 tool: deterministic Israeli AI Developer salary lookup.

Why this tool exists: the LLM might know rough salary ranges from
training data, but those numbers are STALE (training cutoff predates
the application date) and likely WRONG (model averages across
markets, currencies, and role types). A real salary lookup uses
current market data from a structured source.

This is a MOCK tool — data is illustrative placeholder values in
data/market/salaries.json, NOT live market data. In production, the
loader would be replaced by a call to a real salary API (LinkedIn
Talent Insights, Glassdoor, an Israel-specific service like
SalaryUp). The tool's structured-output contract stays the same;
only _load_salary_data() changes.

Data ownership: salaries.json lives under data/market/ rather than
data/portfolio/ because market data is global (not per-user).
Portfolio data is per-user. The directory layout makes that
ownership distinction visible at a glance.

Smoke test:
    python -m assistant.tools.mock_salary_lookup
"""

import json

from pydantic import BaseModel, Field

from config.settings import settings

# ─── Data loader — placeholder ranges from JSON ────────────
_SALARY_DATA_PATH = settings.project_root / "data" / "market" / "salaries.json"


def _load_salary_data() -> dict:
    """Load the illustrative salary data from disk.

    In production, this is the function you'd replace with a real
    API call. The rest of the module would not need to change.
    """
    if not _SALARY_DATA_PATH.exists():
        return {}
    return json.loads(_SALARY_DATA_PATH.read_text(encoding="utf-8"))


_SALARY_DATA: dict = _load_salary_data()


# ─── Output schema ─────────────────────────────────────────
class SalaryRange(BaseModel):
    """Structured salary range for a seniority level.

    Same is_known pattern as ExperienceCalculation: when the
    requested seniority isn't in the registry, the tool returns a
    valid SalaryRange with is_known=False rather than raising. The
    downstream synthesizer can then correctly report 'no data'
    instead of fabricating a number.
    """

    seniority: str = Field(
        ...,
        description="Seniority level requested (normalized lowercase).",
    )
    is_known: bool = Field(
        ...,
        description=(
            "True if the seniority is registered. False signals "
            "'no data' — the synthesizer must NOT report a number."
        ),
    )
    currency: str = Field(
        ...,
        description="ISO currency code, e.g. 'ILS' for Israeli new shekel.",
    )
    min_monthly: int = Field(
        ...,
        description="Lower bound of gross monthly salary. 0 if unknown.",
    )
    max_monthly: int = Field(
        ...,
        description="Upper bound of gross monthly salary. 0 if unknown.",
    )
    median_monthly: int = Field(
        ...,
        description="Midpoint of the range, for quick reference. 0 if unknown.",
    )
    annual_estimate: str = Field(
        ...,
        description=(
            "Approximate annual gross (12 × midpoint), human-readable. "
            "Useful for cross-currency context in cover letters."
        ),
    )
    data_source: str = Field(
        ...,
        description=(
            "Where this data came from. 'placeholder' marks "
            "illustrative non-live data that the synthesizer "
            "should caveat when reporting."
        ),
    )


# ─── Public API ────────────────────────────────────────────
def lookup_salary(seniority: str) -> SalaryRange:
    """Look up a salary range for a given seniority level.

    Args:
        seniority: One of 'junior', 'mid', 'senior', 'lead'
            (case-insensitive). Unknown values return is_known=False.

    Returns:
        SalaryRange. is_known=False means 'no data for this level',
        not 'zero salary'. The caller decides how to surface that.
    """
    key = seniority.strip().lower()
    currency = _SALARY_DATA.get("currency", "ILS")
    source = _SALARY_DATA.get("source", "placeholder")
    ranges = _SALARY_DATA.get("ranges", {})

    band = ranges.get(key)
    if band is None:
        return SalaryRange(
            seniority=key,
            is_known=False,
            currency=currency,
            min_monthly=0,
            max_monthly=0,
            median_monthly=0,
            annual_estimate="unknown",
            data_source=source,
        )

    median = (band["min_monthly"] + band["max_monthly"]) // 2
    annual = median * 12
    return SalaryRange(
        seniority=key,
        is_known=True,
        currency=currency,
        min_monthly=band["min_monthly"],
        max_monthly=band["max_monthly"],
        median_monthly=median,
        annual_estimate=f"~{annual:,} {currency} / year",
        data_source=source,
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Salary data source:  {_SALARY_DATA.get('source', 'unknown')!r}")
    print(f"Role context:        {_SALARY_DATA.get('role', 'unknown')!r}")
    print(f"Currency:            {_SALARY_DATA.get('currency', 'unknown')}")
    print(f"Last updated:        {_SALARY_DATA.get('last_updated', 'unknown')}")
    print()

    test_levels = ["Junior", "MID", "senior", "  lead  ", "principal"]
    for level in test_levels:
        result = lookup_salary(level)
        if result.is_known:
            print(
                f"✓ {level:12s}  →  "
                f"{result.min_monthly:>6,}–{result.max_monthly:>6,} "
                f"{result.currency}/month  "
                f"(median {result.median_monthly:,}, {result.annual_estimate})"
            )
        else:
            print(f"⚠ {level:12s}  →  unknown seniority level")
    print()
    print("→ No LangSmith trace — pure Python, no LLM call.")
