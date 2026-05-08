# ShelfGuard — AI Shelf Gap Detection

## Overview

ShelfGuard is an AI-powered retail shelf gap detection system built during the ELAD Software 24-hour internal hackathon (April 2026). The system uses GPT-4o vision to compare a baseline "full shelf" photo against a current shelf photo, identifies gaps where products are missing, and suggests substitute products via vector similarity search over a product embedding catalog.

I served as **AI/Backend lead** on a four-person team, owning the GPT-4o integration, the pgvector substitute-recommendation engine, and the FastAPI service layer.

## Real-World Problem

Retail shelf gaps cost stores significant revenue per missing-product-hour. Manual shelf audits are slow and infrequent. A camera-driven AI system can detect gaps in near-real-time and suggest the most semantically appropriate substitute (a milk gap should be filled by another dairy product, not by bread on the next shelf).

## Tech Stack

- **AI/Vision:** GPT-4o (vision modality) for shelf image comparison
- **Backend:** FastAPI, Pydantic
- **Vector store:** PostgreSQL + pgvector for product embeddings
- **Cache:** Redis
- **Frontend:** React
- **Containerization:** Docker + docker-compose

## How it works

1. **Baseline capture.** A clean, full shelf photo is taken at the start of the day and stored as the reference image.
2. **Periodic comparison.** Subsequent shelf photos are sent to GPT-4o along with the baseline; the model identifies which product slots are now empty.
3. **Substitute lookup.** For each detected gap, the system fetches the missing product's embedding and queries pgvector for the top-k most similar products in stock. This produces semantically meaningful suggestions (dairy → dairy, snacks → snacks).
4. **Recommendation surface.** The frontend shows store staff a prioritized list of "missing items" plus suggested substitutes for restocking.

## What I owned

- **GPT-4o vision integration** — designed the prompt that asks the model to identify gaps from a pair of images, with structured JSON output.
- **pgvector substitute engine** — embedded the product catalog, built the similarity search query, and tuned `k` for relevance vs noise.
- **FastAPI service layer** — exposed the comparison and recommendation endpoints with Pydantic validation.

## Outcomes

- Working end-to-end demo within the 24-hour window.
- Substitute suggestions visibly outperformed naive nearest-shelf suggestions in the live demo (dairy gaps → dairy products, not adjacent biscuits).
- Team won [hackathon outcome — fill in if applicable, otherwise delete this line].
