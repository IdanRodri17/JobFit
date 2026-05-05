"""Application configuration loaded from environment variables.

Uses Pydantic Settings to validate types, provide defaults, and
auto-load values from the .env file at startup.

Usage:
    from config.settings import settings

    print(settings.openai_api_key)   # str (validated)
    print(settings.llm_model)        # str (default: "gpt-4o-mini")
    print(settings.chunk_size)       # int (default: 500)

Smoke test:
    python -m config.settings
"""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Project Root ──────────────────────────────────────────
# Computed from this file's location, so it works regardless
# of where the script is launched from.
# config/settings.py → config/ → JobFit/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralized, type-safe configuration for JobFit.

    All values come from environment variables, with .env as the
    fallback source. Required fields (marked Field(...)) cause the
    app to fail at startup if missing — fail-fast is better than
    runtime surprises.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars instead of erroring
    )

    # ─── Project Paths ─────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    portfolio_dir: Path = PROJECT_ROOT / "data" / "portfolio"

    # ─── OpenAI ────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key from platform.openai.com")
    llm_model: str = "gpt-4o-mini"               # default model for chains
    judge_model: str = "gpt-4o"                  # used for LLM-as-judge in V5
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0                     # deterministic by default

    # ─── LangSmith ─────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_api_key: str = Field(..., description="LangSmith API key")
    langchain_project: str = "JobFit"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ─── Tavily (used in V6 — Web Search Tool) ─────────────
    tavily_api_key: str | None = None            # optional until V6

    # ─── Retrieval Defaults (used from V3) ─────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4

    # ─── Application Behavior ──────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Singleton instance — import this everywhere instead of recreating Settings()
settings = Settings()


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    # Run with: python -m config.settings
    print("✓ Settings loaded and validated successfully\n")
    print(f"  Project root:        {settings.project_root}")
    print(f"  Chroma persist dir:  {settings.chroma_persist_dir}")
    print(f"  LLM model:           {settings.llm_model}")
    print(f"  Embedding model:     {settings.embedding_model}")
    print(f"  Temperature:         {settings.temperature}")
    print(f"  LangSmith project:   {settings.langchain_project}")
    print(f"  Tracing enabled:     {settings.langchain_tracing_v2}")
    print(f"  Chunk size / k:      {settings.chunk_size} / {settings.retrieval_k}")
    print()
    print(f"  OpenAI key prefix:    {settings.openai_api_key[:7]}...")
    print(f"  LangSmith key prefix: {settings.langchain_api_key[:7]}...")
