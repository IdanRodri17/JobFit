"""JD parsing endpoint -- exposes the V1 jd_parser chain over HTTP."""
from fastapi import APIRouter, HTTPException

from assistant.chains.jd_parser import parse_job_description
from models.api_schemas import ParseJDRequest
from models.schemas import JobDescription

router = APIRouter(prefix="/api", tags=["jd"])


@router.post("/parse-jd", response_model=JobDescription)
def api_parse_jd(req: ParseJDRequest) -> JobDescription:
    """Parse a raw job description into a structured JobDescription (V1)."""
    try:
        return parse_job_description(req.jd_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JD parse failed: {e}")
