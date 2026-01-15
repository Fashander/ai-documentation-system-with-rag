# Docs RAG Demo (OpenAI embeddings + LLM)

This repo is a tiny, deployable reference implementation for the tutorial sections:

- **Ingestion**: Markdown header-aware splitting + overlap
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector store**: Chroma (persisted locally)
- **Keyword search**: BM25
- **Hybrid retrieval**: Reciprocal Rank Fusion (RRF)
- **Generation**: OpenAI chat model (default `gpt-4o-mini`) grounded strictly on retrieved context, with citations

## Requirements

- Python 3.11+
- An OpenAI API key

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."

# build indexes
python cli.py --ingest

# ask questions
python cli.py --ask "How do I authenticate?"
python cli.py --ask "How do I reset my key?"
python cli.py --ask "What does ERR-AD-99 mean?"
```

### API server

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"query":"What does ERR-AD-99 mean?"}'
```

## Docker

```bash
docker build -t docs-rag-openai-demo .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" docs-rag-openai-demo
```

## Notes

- This demo uses **real** OpenAI models. If `OPENAI_API_KEY` is missing, commands will fail fast.
- The reranker in `app/rag.py` is a lightweight heuristic. In production you'd replace it with a true cross-encoder reranker (e.g., Cohere Rerank, bge-reranker, etc.).
- Chunks store `source`, `section_path`, and `chunk_id` metadata to support traceability.

## File map

- `docs/api_guide.md` — test doc
- `app/ingest.py` — load + split + embed + persist (Chroma + BM25)
- `app/rag.py` — hybrid retrieval (BM25 + vector) + RRF fusion + grounded answering
- `app/main.py` — FastAPI endpoints
- `cli.py` — CLI
