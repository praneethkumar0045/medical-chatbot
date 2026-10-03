from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import PINECONE_INDEX_NAME, RETRIEVAL_K

# Create Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name=PINECONE_INDEX_NAME
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
