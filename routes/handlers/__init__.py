"""Chain-backed dedicated endpoints — one router module per V2 intent.

Each module here exposes `router: APIRouter` and is included by api.py.
Mirrors the assistant/chains/ layout: one file per intent, one chain
behind each endpoint. V6 will add tailor_resume.py and company_research.py.
"""
