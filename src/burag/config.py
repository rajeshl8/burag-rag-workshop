"""Project-wide paths and default settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
LLM_MODEL = "gpt-4o-mini"
