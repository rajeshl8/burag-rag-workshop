"""Query-based retrieval from the vector store."""

from typing import Any

from burag.embeddings import EmbeddingManager
from burag.vectorstore import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return the top_k chunks whose similarity score is >= score_threshold."""
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs: list[dict[str, Any]] = []

            if results["documents"] and results["documents"][0]:
                rows = zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
                for i, (doc_id, document, metadata, distance) in enumerate(rows):
                    similarity_score = 1 - distance  # collection uses cosine distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
