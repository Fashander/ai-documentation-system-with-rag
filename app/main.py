from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ingest import run_ingest
from .rag import answer_question

app = FastAPI(title="Docs RAG Demo (OpenAI)")


class AskRequest(BaseModel):
    query: str


@app.post("/ingest")
def ingest():
    try:
        return run_ingest()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer_question(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
