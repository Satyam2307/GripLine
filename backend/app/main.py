"""
GripLine FastAPI application entry point.

    uvicorn backend.app.main:app --reload --port 8000

No model loading, no network calls, no side effects at import time.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import config
from backend.app.routes.analysis import router as analysis_router
from backend.app.schemas import GripLineError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GripLine",
    description="AI co-pilot for racing track-condition analysis.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(analysis_router)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(GripLineError)
async def gripline_error_handler(
    request: Request, exc: GripLineError
) -> JSONResponse:
    """Return structured JSON for all GripLineError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler — never expose stack traces to API consumers.
    """
    logging.getLogger("gripline.error").error(
        "Unhandled exception: %s", exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred. Please try again.",
            }
        },
    )
