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
from config.logging import get_logger
from models.api_schemas import ProcessRequestBody, ProcessResponse

router = APIRouter(prefix="/api", tags=["process"])
logger = get_logger("jobfit.process")


@router.post("/process", response_model=ProcessResponse)
def api_process(req: ProcessRequestBody) -> ProcessResponse:
    """Classify intent and dispatch to the matching handler."""
    # Truncate the user_request preview to avoid leaking long inputs into
    # the terminal; jd_len is enough to confirm the JD reached us.
    logger.info(
        "Processing request (jd_len=%d, user_request=%r)",
        len(req.jd_text),
        req.user_request[:80],
    )
    try:
        classification, result = process_request(req.jd_text, req.user_request)
        logger.info(
            "Routed to '%s' (confidence=%.2f) -> %s",
            classification.intent,
            classification.confidence,
            type(result).__name__,
        )
        return ProcessResponse(classification=classification, result=result)
    except Exception as e:
        # logger.exception() includes the full traceback; logger.error() wouldn't.
        # This is the right call for unexpected exceptions in production code.
        logger.exception("Processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
