import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import setup_logging
from app.middleware.request_id import RequestIDMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical RAG API",
    version="1.0.0",
)

app.add_middleware(RequestIDMiddleware)

app.include_router(health_router)
app.include_router(chat_router)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
