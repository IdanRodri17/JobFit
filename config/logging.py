"""Centralized logging configuration for JobFit.

Called once at app startup (from api.py lifespan). Owns log format,
handler setup, and noisy-third-party silencing.

Why a separate module?
- config/settings.py owns env vars and paths
- config/logging.py owns log format and handlers
- Both are configuration, but at different layers; splitting keeps
  each module single-purpose. Same single-source-of-truth principle
  as everything else in config/.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Set up the root logger with a development-friendly format.

    Output looks like:
        2026-05-10 18:42:11 [INFO    ] jobfit.api      | JobFit API ready
        2026-05-10 18:42:25 [INFO    ] jobfit.process  | Processing request (...)
        2026-05-10 18:42:34 [INFO    ] jobfit.api      | POST /api/process -> 200 (8923 ms)
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Clear handlers uvicorn / chromadb may have added so we control format.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame chatty third-party loggers — we use LangSmith for chain detail,
    # not stdout. This keeps the terminal readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Convention: 'jobfit.<module>' (e.g. 'jobfit.api').

    Named loggers let you filter or change levels per subsystem later
    (e.g. silence 'jobfit.api' but keep 'jobfit.process' verbose).
    """
    return logging.getLogger(name)
