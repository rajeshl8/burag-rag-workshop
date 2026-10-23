"""ChromaDB vector store."""

import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from burag.config import COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = VECTOR_STORE_DIR,
        reset: bool = False,
    ):
        """
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
            reset: Delete any existing collection first (avoids duplicates on re-runs)
        """
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.reset = reset
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            if self.reset:
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass  # collection did not exist yet

            # cosine so that similarity = 1 - distance
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG",
                    "hnsw:space": "cosine",
                },
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: list[Any], embeddings: np.ndarray) -> None:
        """Add LangChain documents and their embeddings to the collection."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, texts, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(
                {
                    **doc.metadata,
                    "doc_index": i,
                    "content_length": len(doc.page_content),
                }
            )
            texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts,
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
