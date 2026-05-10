"""HTML frontend route -- serves the minimal Jinja2 form at /."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["frontend"])

# Jinja2 templates loader. Templates live at JobFit/templates/.
# Created in V4 step 2.
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the minimal HTML form."""
    return templates.TemplateResponse("index.html", {"request": request})
