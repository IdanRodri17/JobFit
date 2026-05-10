"""Process endpoint -- the full V2/V3 router + dispatch flow.

This is the most general endpoint. Accepts any natural-language user
request alongside a JD, classifies intent, dispatches to the matching
specialized handler, and returns both the routing decision and the
resulting structured artifact.

The HTML frontend uses this endpoint exclusively, since the user's
input is free-text and intent must be inferred.
"""
from fastapi import APIRouter, HTTPException

from assistant.core import process_request
from models.api_schemas import ProcessRequestBody, ProcessResponse

router = APIRouter(prefix="/api", tags=["process"])


@router.post("/process", response_model=ProcessResponse)
def api_process(req: ProcessRequestBody) -> ProcessResponse:
    """Classify intent and dispatch to the matching handler."""
    try:
        classification, result = process_request(req.jd_text, req.user_request)
        return ProcessResponse(classification=classification, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
