from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import PINECONE_INDEX_NAME

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_documents():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    return splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest():
    print("Loading PDF documents...")

    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    embeddings = create_embeddings()

    print("Uploading chunks to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()
