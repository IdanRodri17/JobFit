"""V4 API entry point: app setup, middleware, lifespan, router mounts.

Endpoint definitions live in routes/*.py. This file is intentionally
minimal -- its only responsibilities are:

  1. Configure logging + verify ChromaDB exists at startup (lifespan).
  2. Configure the FastAPI app, CORS middleware, static files,
     templates.
  3. Log every HTTP request via middleware.
  4. Mount the route modules from routes/.

Run locally:
    uvicorn api:app --reload --port 8000

Then visit:
    http://localhost:8000          (HTML form)
    http://localhost:8000/docs     (interactive Swagger UI)
    http://localhost:8000/redoc    (ReDoc-style API docs)
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.logging import configure_logging, get_logger
from config.settings import settings

# Infrastructure + meta routes live at routes/ root.
from routes import frontend, health, jd, process

# Chain-backed handlers live in routes/handlers/ — one module per V2 intent.
from routes.handlers import analyze_fit, cover_letter, interview_prep

logger = get_logger("jobfit.api")


# Lifespan: startup / shutdown lifecycle.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging FIRST so the rest of the startup uses our format.
    configure_logging()

    if not settings.chroma_persist_dir.exists():
        raise RuntimeError(
            f"ChromaDB not found at {settings.chroma_persist_dir}. "
            "Run `python -m ingestion.portfolio_ingest` first to "
            "build the portfolio vector store."
        )
    logger.info("JobFit API ready -- ChromaDB at %s", settings.chroma_persist_dir)
    yield
    logger.info("JobFit API shutting down")


# App initialization.
app = FastAPI(
    title="JobFit API",
    description=(
        "AI job application assistant. Parses job descriptions, "
        "analyzes candidate fit, drafts cover letters, and prepares "
        "interview questions -- all grounded in a portfolio retrieved "
        "from ChromaDB via RAG."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# CORS -- permissive for now so a future React frontend on a different
# port can call this API without backend changes. Tighten to specific
# origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTTP request logging middleware.
# Logs every request with method, path, status code, and wall-clock
# duration. Runs for every endpoint -- one centralized observability
# layer instead of repeating logging in every route handler.
@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Static files (CSS, JS) for the HTML frontend.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount all routers. Order doesn't matter for correctness, but grouping
# them visibly here serves as a table-of-contents for the API surface.

# Infrastructure + meta endpoints
app.include_router(health.router)
app.include_router(frontend.router)
app.include_router(jd.router)
app.include_router(process.router)

# Chain-backed handlers — one router per V2 intent
app.include_router(analyze_fit.router)
app.include_router(cover_letter.router)
app.include_router(interview_prep.router)
# V6 will add: tailor_resume.router, company_research.router
