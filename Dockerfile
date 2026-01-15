FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docs ./docs
COPY cli.py ./cli.py
COPY .env.example ./.env.example

ENV DOCS_PATH=./docs \
    CHROMA_DIR=./chroma_db \
    BM25_PATH=./bm25_index.pkl \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-lc", "python cli.py --ingest && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
