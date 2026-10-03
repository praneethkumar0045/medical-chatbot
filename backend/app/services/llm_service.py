from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import (
    LLM_TEMPERATURE,
    GEMINI_MODEL,
    MAX_OUTPUT_TOKENS,
    GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    google_api_key=GOOGLE_API_KEY,
)
