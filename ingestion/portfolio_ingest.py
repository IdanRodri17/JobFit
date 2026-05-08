"""V3 ingestion script: turn portfolio markdown files into embedded chunks.

This is a one-time setup step (re-run whenever portfolio files change).
The script:
    1. Loads every .md file in data/portfolio/ via DirectoryLoader
    2. Splits each document into ~500-token chunks with 50-token overlap
       using RecursiveCharacterTextSplitter
    3. Attaches metadata to each chunk (source filename + category)
    4. Embeds chunks via OpenAI's text-embedding-3-small
    5. Persists everything to ChromaDB at data/chroma_db/

Usage:
    python -m ingestion.portfolio_ingest

The persisted store is then consumed by retrieval/portfolio_retriever.py
at query time.
"""
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


# ─── Categorize each source file ───────────────────────────
def categorize(filename: str) -> str:
    """Map a portfolio filename to a metadata category.

    Used at retrieval time (V5) to filter by category, e.g.
    "only fetch project chunks for cover letters".
    """
    if filename == "cv.md":
        return "cv"
    if filename.startswith("project_"):
        return "projects"
    return "other"


# ─── The ingestion pipeline ────────────────────────────────
def ingest_portfolio() -> Chroma:
    """Build a fresh ChromaDB from the portfolio markdown files.

    Returns:
        The persisted Chroma vector store (also saved to disk so the
        retriever module can re-open it without re-embedding).
    """
    # 1. Verify the source directory exists and has content
    portfolio_dir = settings.portfolio_dir
    if not portfolio_dir.exists():
        sys.exit(f"❌ Portfolio directory not found: {portfolio_dir}")

    md_files = list(portfolio_dir.glob("*.md"))
    if not md_files:
        sys.exit(f"❌ No .md files found in {portfolio_dir}")

    print(f"📂 Found {len(md_files)} markdown files in {portfolio_dir}:")
    for f in md_files:
        print(f"   • {f.name}")

    # 2. Wipe any existing ChromaDB so this is a clean rebuild
    if settings.chroma_persist_dir.exists():
        print(f"\n🧹 Wiping existing ChromaDB at {settings.chroma_persist_dir}")
        shutil.rmtree(settings.chroma_persist_dir)

    # 3. Load all .md files into LangChain Document objects.
    #    DirectoryLoader uses TextLoader under the hood for plain text.
    print("\n📥 Loading documents...")
    loader = DirectoryLoader(
        path=str(portfolio_dir),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    raw_docs: list[Document] = loader.load()
    print(f"   Loaded {len(raw_docs)} documents.")

    # 4. Attach the category metadata to each document BEFORE splitting,
    #    so every resulting chunk inherits it automatically.
    for doc in raw_docs:
        source_path = Path(doc.metadata["source"])
        doc.metadata["filename"] = source_path.name
        doc.metadata["category"] = categorize(source_path.name)

    # 5. Split into chunks. Recursive splitter prefers semantic boundaries
    #    (paragraphs > sentences > words) over arbitrary character cuts.
    print(
        f"\n✂️  Splitting into chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})..."
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = splitter.split_documents(raw_docs)
    print(f"   Produced {len(chunks)} chunks.")

    # 6. Embed and persist. Chroma.from_documents handles the embedding
    #    call internally — one round-trip per batch to OpenAI.
    print(
        f"\n🧠 Embedding chunks with {settings.embedding_model} "
        "and persisting to ChromaDB..."
    )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
        collection_name="portfolio",
    )

    print(f"\n✅ Ingestion complete.")
    print(f"   Chunks embedded:  {len(chunks)}")
    print(f"   Persisted to:     {settings.chroma_persist_dir}")
    return vectorstore


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    vectorstore = ingest_portfolio()

    # Sanity check — query the freshly-built store with a sample question
    print("\n─── Sanity check: querying the store ───")
    sample_query = "What experience do I have with LangChain and RAG?"
    print(f"Query: {sample_query!r}\n")

    results = vectorstore.similarity_search(sample_query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"Chunk {i} (source: {doc.metadata.get('filename')}, "
              f"category: {doc.metadata.get('category')})")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"   {preview}\n")
