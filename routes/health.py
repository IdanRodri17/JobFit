"""Health check endpoint for liveness monitoring."""
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check -- used by CI and uptime monitoring."""
    return {"status": "ok", "service": "JobFit API"}
