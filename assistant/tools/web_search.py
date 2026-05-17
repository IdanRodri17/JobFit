"""V6 tool: live web search for recent company / news lookups.

Why this tool exists: the LLM's training-data knowledge of companies,
hiring news, and recent events is stale by definition. Anything
'recent' must come from a live source. This tool wraps Tavily's
search API — chosen for its generous free tier and built-in
result summarization.

Unlike experience_calculator and mock_salary_lookup, this tool
needs an external API key. The is_known pattern from the other
tools applies: when TAVILY_API_KEY is missing or the call fails,
the tool returns a structured 'no' (is_known=False, with an
error message) rather than raising. The downstream synthesizer
sees the error and can degrade gracefully — 'I can't fetch
current news right now' beats a 500.

Smoke test:
    python -m assistant.tools.web_search
"""

import logging

from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Output schemas ────────────────────────────────────────
class WebSearchResult(BaseModel):
    """A single hit from a web search."""

    title: str = Field(..., description="Page title as returned by the search engine.")
    url: str = Field(..., description="Canonical URL of the result.")
    snippet: str = Field(
        ...,
        description=(
            "Short content excerpt. Tavily's 'content' field — usually "
            "the most relevant 1-3 paragraphs of the page."
        ),
    )
    published_date: str | None = Field(
        default=None,
        description="Publication date if the search engine provides one (often missing for non-news pages).",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Tavily's relevance score, 0.0–1.0. Higher = more relevant to query.",
    )


class WebSearchResponse(BaseModel):
    """Full structured response from a web search call.

    is_known pattern matches the other V6 tools: when the search
    couldn't run (no API key, network failure, no results), the
    response carries is_known=False and an `error` message rather
    than raising.
    """

    query: str = Field(..., description="The query that was searched (echoed back).")
    is_known: bool = Field(
        ...,
        description=(
            "True if the search ran successfully and returned at least "
            "one result. False signals 'no data' — synthesizer must "
            "NOT fabricate news, dates, or facts."
        ),
    )
    answer: str | None = Field(
        default=None,
        description=(
            "Tavily's pre-synthesized one-line answer, when available. "
            "Often the cleanest input for the downstream LLM synthesizer."
        ),
    )
    results: list[WebSearchResult] = Field(
        default_factory=list,
        description="Individual search hits, ordered by relevance.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when is_known=False.",
    )


# ─── Public API ────────────────────────────────────────────
def search_web(
    query: str,
    max_results: int = 5,
    days: int = 30,
    topic: str = "news",
) -> WebSearchResponse:
    """Search the live web via Tavily's API.

    Args:
        query: Natural-language search query. Tavily handles this
            without needing keyword-engineering on our side.
        max_results: How many hits to return. 3-5 is a good default;
            more = noisier, fewer = might miss relevant results.
        days: Recency window for 'news' topic (ignored for 'general').
        topic: 'news' (default — biases toward recent journalism)
            or 'general' (broader web search). Use 'news' for company
            updates and current events; 'general' for evergreen content.

    Returns:
        WebSearchResponse. Always returns a valid object — failure
        modes are encoded via is_known=False + error message.
    """
    # Fail-fast on missing key, but return structured response so
    # the dispatcher can keep working.
    if not settings.tavily_api_key:
        return WebSearchResponse(
            query=query,
            is_known=False,
            error="TAVILY_API_KEY not set in .env",
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        return WebSearchResponse(
            query=query,
            is_known=False,
            error="tavily-python package not installed (pip install tavily-python)",
        )

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        # include_answer=True gets us Tavily's pre-synthesized
        # one-liner — saves an LLM call downstream when the LLM
        # would otherwise summarize raw snippets itself.
        raw = client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            days=days,
            include_answer=True,
        )
    except Exception as e:
        # Network errors, rate limits, invalid keys etc. all funnel
        # here. Log for debugging, return structured failure.
        logger.warning("Tavily search failed for query %r: %s", query, e)
        return WebSearchResponse(
            query=query,
            is_known=False,
            error=f"Tavily search failed: {e}",
        )

    results = [
        WebSearchResult(
            title=item.get("title", "")[:200],
            url=item.get("url", ""),
            snippet=item.get("content", "")[:500],
            published_date=item.get("published_date"),
            relevance_score=float(item.get("score", 0.0)),
        )
        for item in raw.get("results", [])
    ]

    if not results:
        return WebSearchResponse(
            query=query,
            is_known=False,
            error="No results returned by Tavily for this query.",
        )

    return WebSearchResponse(
        query=query,
        is_known=True,
        answer=raw.get("answer"),
        results=results,
    )


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "Elad Systems Israel hiring news",
        "Anthropic Claude latest release",
    ]

    for q in queries:
        print("═" * 70)
        print(f"QUERY: {q!r}")
        print("═" * 70)
        response = search_web(q, max_results=3)

        if not response.is_known:
            print(f"\n⚠  No results: {response.error}\n")
            continue

        if response.answer:
            print(f"\nTavily answer:")
            print(f"  {response.answer}\n")

        print(f"Top {len(response.results)} results:")
        for i, r in enumerate(response.results, 1):
            print(f"\n  [{i}]  {r.title}")
            print(f"       {r.url}")
            print(f"       score: {r.relevance_score:.2f}", end="")
            if r.published_date:
                print(f"  |  published: {r.published_date}")
            else:
                print()
            print(f"       {r.snippet[:200]}...")
        print()

    print("→ This tool DID call an external service. Cost: ~$0 (Tavily free tier).")
