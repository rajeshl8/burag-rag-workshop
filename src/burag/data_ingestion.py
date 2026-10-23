"""Data ingestion: read PDFs into LangChain Documents."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)


def process_all_pdfs(pdf_directory: str | Path) -> list[Document]:
    """Load every PDF found (recursively) in a directory, one Document per page."""
    all_documents: list[Document] = []
    pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            documents = PyPDFLoader(str(pdf_file)).load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
