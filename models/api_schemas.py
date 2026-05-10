"""API-layer Pydantic schemas: request bodies and response wrappers.

Separated from models/schemas.py to keep two concerns distinct:

  models/schemas.py     -> what the LangChain chains PRODUCE
                            (FitReport, CoverLetter, InterviewPrep, ...)

  models/api_schemas.py -> what the API endpoints ACCEPT and return
                            as composite responses (this file).

Chain output schemas (FitReport et al.) ARE used as response_model for
endpoints that return a single artifact -- those are imported directly
from models/schemas.py at the route level. This file holds only the
schemas unique to the HTTP boundary.
"""
from pydantic import BaseModel, Field

from models.schemas import (
    CoverLetter,
    FitReport,
    IntentClassification,
    InterviewPrep,
)


# Request bodies.
class ParseJDRequest(BaseModel):
    jd_text: str = Field(
        ...,
        min_length=20,
        description="Raw text of a job posting.",
    )


class JDPlusContextRequest(BaseModel):
    """Used by /analyze-fit, /cover-letter, /interview-prep.

    Currently identical to ParseJDRequest, but defined separately so
    the two requests can diverge in the future (e.g. V6 may add tool
    parameters here without touching ParseJDRequest).
    """
    jd_text: str = Field(
        ...,
        min_length=20,
        description="Raw text of a job posting.",
    )


class ProcessRequestBody(BaseModel):
    jd_text: str = Field(
        ...,
        min_length=20,
        description="Raw text of a job posting.",
    )
    user_request: str = Field(
        ...,
        min_length=3,
        description=(
            "Natural-language description of what to do "
            "(e.g. 'Should I apply?', 'Write me a cover letter')."
        ),
    )


# Composite response wrappers.
class ProcessResponse(BaseModel):
    """Tuple-style response from the /process endpoint.

    Wraps both the routing decision and the resulting structured
    artifact, so the caller can show users the routing reasoning
    alongside the output.
    """
    classification: IntentClassification
    result: FitReport | CoverLetter | InterviewPrep | str
