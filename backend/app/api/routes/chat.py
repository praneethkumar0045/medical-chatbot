import logging
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import ask_question

router = APIRouter(prefix="/api/v1", tags=["Chat"])

logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    logger.info("Processing chat request")
    answer = ask_question(request.question)

    return ChatResponse(answer=answer)
