"""Generation: build the prompt from retrieved context and call the LLM."""

from typing import Any

from burag.retriever import RAGRetriever

PROMPT_TEMPLATE = """Use the following context to answer the question concisely.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _sources(results: list[dict[str, Any]], preview_chars: int = 300) -> list[dict[str, Any]]:
    return [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:preview_chars] + "...",
        }
        for doc in results
    ]


def rag_simple(query: str, retriever: RAGRetriever, llm, top_k: int = 3) -> str:
    """Retrieve context and generate an answer."""
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return "No relevant context found to answer the question."

    context = "\n\n".join(doc["content"] for doc in results)
    return llm.invoke(build_prompt(context, query)).content


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> dict[str, Any]:
    """RAG with sources and a confidence score (the best similarity score)."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    context = "\n\n".join(doc["content"] for doc in results)
    output: dict[str, Any] = {
        "answer": llm.invoke(build_prompt(context, query)).content,
        "sources": _sources(results),
        "confidence": max(doc["similarity_score"] for doc in results),
    }
    if return_context:
        output["context"] = context
    return output


class AdvancedRAGPipeline:
    """RAG with streaming, citations, query history and optional summarization."""

    def __init__(self, retriever: RAGRetriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history: list[dict[str, Any]] = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        stream: bool = False,
        summarize: bool = False,
    ) -> dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            answer, sources = "No relevant context found.", []
        else:
            context = "\n\n".join(doc["content"] for doc in results)
            sources = _sources(results, preview_chars=120)
            prompt = build_prompt(context, question)

            if stream:
                chunks = []
                for part in self.llm.stream(prompt):
                    print(part.content, end="", flush=True)
                    chunks.append(part.content)
                print()
                answer = "".join(chunks)
            else:
                answer = self.llm.invoke(prompt).content

        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer
        )

        summary = None
        if summarize and sources:
            summary = self.llm.invoke(
                f"Summarize the following answer in 2 sentences:\n{answer}"
            ).content

        self.history.append(
            {"question": question, "answer": answer, "sources": sources, "summary": summary}
        )

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }
