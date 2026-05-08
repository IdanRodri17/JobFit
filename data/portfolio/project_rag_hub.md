# Multi-Source RAG Knowledge Hub

## Overview

The Multi-Source RAG Knowledge Hub is a production-grade Retrieval-Augmented Generation system built as a portfolio project to demonstrate end-to-end ownership of an AI engineering project, from architecture through CI/CD to monitoring.

The system answers questions over multiple knowledge sources by routing queries through an agentic LangGraph workflow that retrieves relevant context, grades retrieval quality, generates grounded responses, and retries when retrieval fails. It is designed to be a reference implementation of how a senior AI Developer should approach a real RAG product.

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic v2
- **Vector store:** PostgreSQL with the pgvector extension
- **Cache:** Redis
- **Agentic orchestration:** LangGraph (router, retriever, grader, generator nodes with retry loop)
- **LLM providers:** OpenAI and Ollama, swappable via a factory-pattern provider abstraction
- **Frontend:** React (separate folder)
- **CI/CD:** GitHub Actions (test, lint, build)
- **Monitoring:** Prometheus + Grafana stack with custom metrics
- **Containerization:** Docker + docker-compose

## Architecture (built across four phases)

**Phase 1 — Backend foundation.** Set up FastAPI with PostgreSQL/pgvector for vector storage and Redis for caching. Implemented document ingestion with `RecursiveCharacterTextSplitter` and OpenAI embeddings. Exposed `/query` and `/ingest` endpoints with Pydantic-validated request/response models.

**Phase 2 — Agentic orchestration with LangGraph.** Replaced the simple linear chain with a LangGraph state machine: a router node classifies query intent, a retriever node fetches top-k chunks from pgvector, a grader node scores retrieval quality with the LLM, and a generator node produces the final grounded answer. A retry loop re-runs retrieval with rephrased queries when grading fails.

**Phase 3 — LLM provider abstraction.** Added a factory-pattern provider layer supporting both OpenAI (gpt-4o-mini for production, gpt-4o for evaluation) and Ollama (llama3.1:8b for cost-free local development). The provider is selected at runtime via configuration. The rest of the codebase consumes the abstract interface, never the concrete provider.

**Phase 4 — Production polish.** Added a full pytest test suite, a GitHub Actions CI/CD pipeline that runs tests on every PR, and a Prometheus/Grafana monitoring stack tracking request latency, token usage, retrieval hit rate, and cost per query.

## Key Engineering Decisions

- **pgvector over Pinecone/Weaviate.** Chosen to keep infrastructure self-contained and avoid third-party lock-in. PostgreSQL is already the system of record, so co-locating the vectors avoided a separate service.
- **LangGraph over plain LangChain agents.** LangChain's `AgentExecutor` is essentially deprecated; LangGraph's explicit state-machine model gives full control over the retrieval-grading-retry loop and produces much cleaner LangSmith traces.
- **Factory pattern for LLM providers.** Allowed switching between OpenAI and Ollama with one config change — important during development for cost reasons, and useful in production for failover.

## Outcomes

- Green CI on every commit since Phase 4 completion.
- Full LangSmith tracing across all four nodes; every query is observable end-to-end.
- Custom Grafana dashboards showing real-time retrieval hit rate, p95 latency, and per-query cost.
- Polish work remaining: streaming SSE responses, SSRF protection, CORS lockdown, README expansion.
