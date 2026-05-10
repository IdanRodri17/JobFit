"""POST /api/analyze-fit — direct invocation of the V2 fit_analyzer chain (V2 + V3 RAG).

Bypasses the V2 router. Useful when the caller already knows what they
want — e.g. a React "Analyze fit" button, or any client that doesn't
need intent classification. Returns a bare FitReport, not a wrapped
ProcessResponse.

For natural-language / mixed-intent input where intent must be inferred,
use POST /api/process instead.
"""

from fastapi import APIRouter, HTTPException

from assistant.chains.fit_analyzer import analyze_fit
from models.api_schemas import JDPlusContextRequest
from models.schemas import FitReport
from retrieval.portfolio_retriever import get_relevant_context

router = APIRouter(prefix="/api", tags=["handlers"])


@router.post("/analyze-fit", response_model=FitReport)
def api_analyze_fit(req: JDPlusContextRequest) -> FitReport:
    """Analyze candidate fit against a JD (V2 + V3 RAG)."""
    try:
        candidate_context = get_relevant_context(req.jd_text)
        return analyze_fit(req.jd_text, candidate_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fit analysis failed: {e}")
