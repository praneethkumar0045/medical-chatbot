import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K"))
