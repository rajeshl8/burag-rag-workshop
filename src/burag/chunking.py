"""Data parsing: split documents into chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from burag.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks for better RAG performance."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs
