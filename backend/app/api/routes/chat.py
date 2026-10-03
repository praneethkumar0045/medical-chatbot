import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import ask_question

router = APIRouter(prefix="/api/v1", tags=["Chat"])

logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest):

    request_id = request.state.request_id

    logger.info("Chat request started | request_id=%s", request_id)

    try:

        answer = ask_question(body.question)

        logger.info("Chat request completed | request_id=%s", request_id)

        return ChatResponse(answer=answer)

    except RuntimeError:

        logger.exception("Chat request failed | request_id=%s", request_id)

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        )
