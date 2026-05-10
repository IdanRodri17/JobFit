"""POST /api/interview-prep — direct invocation of the V2 interview_prep chain (V2 + V3 RAG).

Same architectural pattern as routes/handlers/analyze_fit.py — see that
file's docstring for the design rationale.
"""

from fastapi import APIRouter, HTTPException

from assistant.chains.interview_prep import prepare_interview
from models.api_schemas import JDPlusContextRequest
from models.schemas import InterviewPrep
from retrieval.portfolio_retriever import get_relevant_context

router = APIRouter(prefix="/api", tags=["handlers"])


@router.post("/interview-prep", response_model=InterviewPrep)
def api_interview_prep(req: JDPlusContextRequest) -> InterviewPrep:
    """Generate interview prep for a JD (V2 + V3 RAG)."""
    try:
        candidate_context = get_relevant_context(req.jd_text)
        return prepare_interview(req.jd_text, candidate_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview prep failed: {e}")
