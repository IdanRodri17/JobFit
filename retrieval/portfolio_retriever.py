"""V3 retrieval module: query the persisted ChromaDB for relevant portfolio chunks.

This module is the clean interface the rest of the codebase uses to
fetch context from the portfolio. It hides ChromaDB, embeddings,
collection names, and persistence paths behind two public functions:

    get_retriever() -> VectorStoreRetriever
        Returns the underlying LangChain retriever for advanced use
        (e.g. composing into LCEL chains in V5/V6).

    get_relevant_context(query: str, k: int | None = None) -> str
        The high-level convenience function: takes a natural-language
        query and returns the top-k chunks formatted as a single string
        ready to drop into a handler prompt.

Pre-requisite: ingestion/portfolio_ingest.py must have been run at
least once to populate data/chroma_db/.

Smoke test:
    python -m retrieval.portfolio_retriever
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


# ─── Module-level singletons ───────────────────────────────
# Instantiated once at import. Re-using them across queries avoids
# re-opening ChromaDB and re-loading the embedding client on every call.
_embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)

_vectorstore = Chroma(
    persist_directory=str(settings.chroma_persist_dir),
    embedding_function=_embeddings,
    collection_name="portfolio",
)


# ─── Public API ────────────────────────────────────────────
def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return the underlying LangChain retriever.

    Use this when you want the full Document objects (with metadata)
    or when you want to compose retrieval into an LCEL chain.

    Args:
        k: Number of chunks to retrieve per query. Defaults to
           settings.retrieval_k (configured in config/settings.py).

    Returns:
        A LangChain VectorStoreRetriever — itself a Runnable that
        can be piped with `|` in V5/V6.
    """
    return _vectorstore.as_retriever(
        search_kwargs={"k": k or settings.retrieval_k}
    )


def get_relevant_context(query: str, k: int | None = None) -> str:
    """Fetch top-k portfolio chunks for a query, formatted as a single string.

    This is the high-level helper used by core.py and the handler chains.
    Each chunk is prefixed with its source filename so the LLM can reason
    about provenance.

    Args:
        query: Natural-language description of what context is needed,
               e.g. "experience with LangChain and RAG systems".
        k: Number of chunks to retrieve (default from settings).

    Returns:
        A single string with all retrieved chunks concatenated.
    """
    retriever = get_retriever(k=k)
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
    test_queries = [
        "experience with LangChain and RAG systems",
        "production deployment and CI/CD",
        "computer vision and YOLOv8",
        "Hebrew language and Israeli tech market",
    ]

    print("✓ Retriever loaded from persisted ChromaDB\n")

    for query in test_queries:
        print("═" * 70)
        print(f"QUERY: {query!r}")
        print("═" * 70)
        context = get_relevant_context(query, k=3)
        # Truncate each chunk for readable output
        truncated_lines = []
        for chunk in context.split("\n\n"):
            if len(chunk) > 250:
                chunk = chunk[:250] + "..."
            truncated_lines.append(chunk)
        print("\n".join(truncated_lines))
        print()
