"""Command line entry point: `burag ingest` and `burag ask "<question>"`."""

import argparse

from burag.llm import get_llm
from burag.pipeline import build_index, load_retriever
from burag.rag import AdvancedRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="burag", description="PDF RAG demo")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Rebuild the vector store from data/pdfs")

    ask = sub.add_parser("ask", help="Ask a question about the ingested PDFs")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--min-score", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "ingest":
        build_index()
    else:
        pipeline = AdvancedRAGPipeline(load_retriever(), get_llm())
        result = pipeline.query(args.question, top_k=args.top_k, min_score=args.min_score, stream=True)
        # the answer was already streamed to the terminal; only add the citations
        citations = result["answer"].split("\n\nCitations:\n", 1)
        if len(citations) == 2:
            print("\nCitations:\n" + citations[1])
