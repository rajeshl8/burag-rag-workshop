"""End-to-end helpers: build the index and load the retriever."""

from pathlib import Path

from burag.chunking import split_documents
from burag.config import PDF_DIR
from burag.embeddings import EmbeddingManager
from burag.data_ingestion import process_all_pdfs
from burag.retriever import RAGRetriever
from burag.vectorstore import VectorStore


def build_index(pdf_dir: str | Path = PDF_DIR) -> RAGRetriever:
    """Ingest PDFs, chunk, embed and store them (the store is rebuilt from scratch)."""
    documents = process_all_pdfs(pdf_dir)
    chunks = split_documents(documents)

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(reset=True)
    embeddings = embedding_manager.generate_embeddings([c.page_content for c in chunks])
    vector_store.add_documents(chunks, embeddings)

    return RAGRetriever(vector_store, embedding_manager)


def load_retriever() -> RAGRetriever:
    """Open the existing vector store without re-embedding anything."""
    return RAGRetriever(VectorStore(reset=False), EmbeddingManager())
