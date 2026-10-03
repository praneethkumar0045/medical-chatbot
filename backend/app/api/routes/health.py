from fastapi import APIRouter

from app.core.config import (
    PINECONE_API_KEY,
    GOOGLE_API_KEY,
)

router = APIRouter(
    tags=["Health"],
)


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/ready")
def readiness():

    checks = {
        "pinecone": bool(PINECONE_API_KEY),
        "gemini": bool(GOOGLE_API_KEY),
    }

    ready = all(checks.values())

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
