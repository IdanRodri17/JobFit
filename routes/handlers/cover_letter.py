"""POST /api/cover-letter — direct invocation of the V2 cover_letter chain (V2 + V3 RAG).

Same architectural pattern as routes/handlers/analyze_fit.py — see that
file's docstring for the design rationale (bypass router, return bare
schema, contrast with /api/process).
"""

from fastapi import APIRouter, HTTPException

from assistant.chains.cover_letter import generate_cover_letter
from models.api_schemas import JDPlusContextRequest
from models.schemas import CoverLetter
from retrieval.portfolio_retriever import get_relevant_context

router = APIRouter(prefix="/api", tags=["handlers"])


@router.post("/cover-letter", response_model=CoverLetter)
def api_cover_letter(req: JDPlusContextRequest) -> CoverLetter:
    """Generate a cover letter for a JD (V2 + V3 RAG)."""
    try:
        candidate_context = get_relevant_context(req.jd_text)
        return generate_cover_letter(req.jd_text, candidate_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cover letter failed: {e}")
