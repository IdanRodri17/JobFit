"""HTTP route modules for JobFit's FastAPI app.

Each module defines an APIRouter that's mounted by api.py. This
keeps endpoint definitions cleanly separated by concern:

  - health.py    -> liveness check
  - frontend.py  -> HTML frontend
  - jd.py        -> JD parsing
  - handlers.py  -> direct chain invocation (analyze-fit, cover-letter,
                    interview-prep)
  - process.py   -> full V2/V3 router + dispatch flow
"""
