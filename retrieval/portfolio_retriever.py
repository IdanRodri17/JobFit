"""V3/V5 retrieval module: query the persisted ChromaDB for relevant portfolio chunks.

This module is the clean interface the rest of the codebase uses to
fetch context from the portfolio. It hides ChromaDB, embeddings,
collection names, and persistence paths behind two public functions:

    get_retriever() -> VectorStoreRetriever
        Returns the underlying LangChain retriever for advanced use
        (e.g. composing into LCEL chains).

    get_relevant_context(query, k=None, categories=None) -> str
        High-level helper: takes a query and (optionally) a category
        filter and returns the top-k chunks formatted as one string
        ready for a handler prompt.

V5 update: both functions now accept an optional category filter.
A cover-letter chain wants 'projects' chunks, not the languages
section of the CV. Filtering by metadata sharpens retrieval at the
same k.

Pre-requisite: ingestion/portfolio_ingest.py must have been run at
least once to populate data/chroma_db/ with category-tagged chunks.

Smoke test:
    python -m retrieval.portfolio_retriever
"""

from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config.settings import settings

# ─── Category type ─────────────────────────────────────────
# Single source of truth for the metadata categories assigned during
# ingestion. Using a Literal makes typos a type-check error rather
# than a silent empty-result-set at runtime.
Category = Literal["cv", "projects", "other"]


# ─── Module-level singletons ───────────────────────────────
_embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)

_vectorstore = Chroma(
    persist_directory=str(settings.chroma_persist_dir),
    embedding_function=_embeddings,
    collection_name="portfolio",
)


# ─── Internal helpers ──────────────────────────────────────
def _build_chroma_filter(
    categories: Category | list[Category] | None,
) -> dict | None:
    """Convert one or more categories into a ChromaDB filter dict.

    Hides Chroma's filter quirks (`{"key": value}` vs `{"key": {"$in": [...]}}`)
    behind a single Python signature. Returns None when no filter is
    requested so callers can omit the `filter` key entirely.
    """
    if categories is None:
        return None
    cats = [categories] if isinstance(categories, str) else list(categories)
    if len(cats) == 1:
        return {"category": cats[0]}
    return {"category": {"$in": cats}}


# ─── Public API ────────────────────────────────────────────
def get_retriever(
    k: int | None = None,
    categories: Category | list[Category] | None = None,
) -> VectorStoreRetriever:
    """Return the LangChain retriever, optionally filtered by category.

    Args:
        k: Number of chunks per query (defaults to settings.retrieval_k).
        categories: If provided, retrieval is restricted to chunks whose
            'category' metadata matches. Pass a single Category for a
            simple equality filter, a list for multi-value, or None
            (default) to disable filtering entirely.

    Returns:
        A LangChain VectorStoreRetriever — itself a Runnable, so it
        can be piped with `|` in a future LCEL chain.
    """
    search_kwargs: dict = {"k": k or settings.retrieval_k}
    chroma_filter = _build_chroma_filter(categories)
    if chroma_filter is not None:
        search_kwargs["filter"] = chroma_filter
    return _vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_relevant_context(
    query: str,
    k: int | None = None,
    categories: Category | list[Category] | None = None,
) -> str:
    """Fetch top-k portfolio chunks for a query, formatted as a single string.

    The high-level helper used by core.py and the handler chains.
    Each chunk is prefixed with its source filename so the LLM can
    reason about provenance.

    Args:
        query: Natural-language description of what context is needed.
        k: Number of chunks to retrieve (default from settings).
        categories: Optional category filter (see get_retriever).

    Returns:
        A single string with all retrieved chunks concatenated.
    """
    retriever = get_retriever(k=k, categories=categories)
    docs: list[Document] = retriever.invoke(query)
    return _format_chunks(docs)


def _format_chunks(docs: list[Document]) -> str:
    """Format a list of retrieved chunks as a single context string.

    Each chunk is prefixed with [Source: <filename>] so the LLM can
    distinguish provenance and the trace stays human-readable.
    """
    sections = []
    for doc in docs:
        source = doc.metadata.get("filename", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(sections)


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    print("✓ Retriever loaded from persisted ChromaDB\n")

    query = "Python developer with FastAPI and RAG experience"
    print("═" * 70)
    print(f"QUERY: {query!r}")
    print("═" * 70)

    # Same query, three filter modes — easy way to see the effect.
    cases: list[tuple[str, Category | list[Category] | None]] = [
        ("no filter (V3 baseline behavior)", None),
        ("filter: ['projects']", ["projects"]),
        ("filter: 'cv'", "cv"),
    ]

    for label, categories in cases:
        print(f"\n--- {label} ---")
        ctx = get_relevant_context(query, k=3, categories=categories)
        for chunk in ctx.split("\n\n"):
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(preview)
        print()
