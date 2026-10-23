# BURAG: PDF RAG Workshop

A small Retrieval-Augmented Generation (RAG) pipeline for the BU workshop. It ingests PDFs, chunks and embeds them into a local Chroma vector store, retrieves the most relevant chunks for a question, and asks an OpenAI model to answer using only that context.

```
PDFs → load → chunk → embed → Chroma ──┐
                                        ├→ retrieve top-k → prompt (context + question) → LLM → answer + citations
question → embed ───────────────────────┘
```

## Project layout

```
src/burag/
  config.py          paths and defaults (chunk size, models)
  data_ingestion.py  read PDFs into LangChain Documents
  chunking.py        split documents into chunks
  embeddings.py      SentenceTransformer embeddings (all-MiniLM-L6-v2)
  vectorstore.py     ChromaDB store (cosine similarity)
  retriever.py       query → top-k chunks with similarity scores
  llm.py             OpenAI chat model
  rag.py             prompt + rag_simple / rag_advanced / AdvancedRAGPipeline
  pipeline.py        build_index() and load_retriever()
  cli.py             `burag` command
notebook/
  document.ipynb           LangChain Document basics
  pdf_data_pipeline.ipynb  the full pipeline, step by step
data/pdfs/                 PDFs to ingest
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set your OpenAI key, either in your shell or in a `.env` file in the project root (already git-ignored):

```bash
OPENAI_API_KEY=sk-...
```

## Run

Put PDFs in `data/pdfs/`, then:

```bash
uv run burag ingest                                          # build the vector store (rebuilds from scratch)
uv run burag ask "Which ship is nicknamed Old Ironsides?"    # streamed answer + citations
uv run burag ask "What is the Freedom Trail?" --top-k 5 --min-score 0.2
```

Run `ingest` again whenever you add or change PDFs.

## Use from Python

```python
from burag.pipeline import load_retriever
from burag.llm import get_llm
from burag.rag import rag_simple

print(rag_simple("What is the Freedom Trail?", load_retriever(), get_llm()))
```

## Sample questions (Boston PDFs)

- What is the biggest unsolved art theft in history, and where did it happen?
- What is the Freedom Trail and how many historic sites does it connect?
- Which ship is nicknamed Old Ironsides?
- Which museum has a butterfly garden and a planetarium?

Ask something the PDFs can't answer (for example "What is the attention mechanism in transformers?") to see low similarity scores and the "I don't know" behaviour.

## Notes

- `data/vector_store/` is generated locally and not committed.
- Image-only or scanned PDFs yield little text; they need OCR to be useful.
