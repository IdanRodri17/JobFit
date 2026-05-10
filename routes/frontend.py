"""HTML frontend route -- serves the minimal Jinja2 form at /."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["frontend"])

# Jinja2 templates loader. Templates live at JobFit/templates/.
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the minimal HTML form."""
    # Keyword args — explicit and version-stable. Starlette ≥0.29 made
    # `request` the first positional arg of TemplateResponse, which broke
    # the older `(name, context_dict)` style. Naming everything sidesteps
    # the churn entirely.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )
