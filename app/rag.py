from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from .config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    section_path: str
    score: float
    meta: dict


def load_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    return Chroma(
        collection_name="docs",
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )


def load_bm25_payload() -> dict:
    with open(settings.bm25_path, "rb") as f:
        return pickle.load(f)


def bm25_search(query: str, payload: dict, k: int) -> List[RetrievedChunk]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(payload["tokenized"])
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    # get top k indices
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    out: List[RetrievedChunk] = []
    for i in top_idx:
        meta = payload["metas"][i]
        out.append(
            RetrievedChunk(
                chunk_id=payload["ids"][i],
                text=payload["corpus"][i],
                source=str(meta.get("source", "")),
                section_path=str(meta.get("section_path", "")),
                score=float(scores[i]),
                meta=meta,
            )
        )
    return out


def vector_search(query: str, vectordb: Chroma, k: int) -> List[RetrievedChunk]:
    # returns (Document, distance) — distance is NOT used for ranking
    results = vectordb.similarity_search_with_score(query, k=k)

    out: List[RetrievedChunk] = []
    for rank, (doc, _distance) in enumerate(results, start=1):
        meta = doc.metadata or {}
        out.append(
            RetrievedChunk(
                chunk_id=str(meta.get("chunk_id", "")),
                text=doc.page_content,
                source=str(meta.get("source", "")),
                section_path=str(meta.get("section_path", "")),
                # score is rank-based because RRF consumes rank, not distance
                score=1.0 / rank,
                meta=meta,
            )
        )
    return out

def rrf_fuse(vector_results: List[RetrievedChunk], keyword_results: List[RetrievedChunk], k: int, rrf_k: int) -> List[RetrievedChunk]:
    # Build id -> chunk map, prefer vector text if duplicates
    doc_map: Dict[str, RetrievedChunk] = {}
    for c in vector_results + keyword_results:
        if c.chunk_id and c.chunk_id not in doc_map:
            doc_map[c.chunk_id] = c

    fusion_scores: Dict[str, float] = {}

    for rank, c in enumerate(vector_results, start=1):
        if not c.chunk_id:
            continue
        fusion_scores.setdefault(c.chunk_id, 0.0)
        fusion_scores[c.chunk_id] += 1.0 / (rrf_k + rank)

    for rank, c in enumerate(keyword_results, start=1):
        if not c.chunk_id:
            continue
        fusion_scores.setdefault(c.chunk_id, 0.0)
        fusion_scores[c.chunk_id] += 1.0 / (rrf_k + rank)

    sorted_ids = sorted(fusion_scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = [doc_map[doc_id] for doc_id, _ in sorted_ids[:k] if doc_id in doc_map]
    return fused


def simple_rerank(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Lightweight rerank without extra vendors.

    This is NOT a cross-encoder reranker; it's a pragmatic heuristic that boosts exact token matches
    for identifiers (error codes, endpoint paths) while keeping order mostly stable.
    """
    q = query.lower()
    def bonus(c: RetrievedChunk) -> float:
        b = 0.0
        if any(tok in q for tok in ["err-", "/api/"]):
            # boost exact substring matches in the chunk text
            if q.strip() in c.text.lower():
                b += 2.0
        return b

    return sorted(chunks, key=lambda c: (bonus(c), len(set(simple_tokenize(c.text)) & set(simple_tokenize(query)))), reverse=True)


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        source = c.source
        section = c.section_path or ""
        parts.append(
            f"Context Chunk {i}:\n"
            f"Content: {c.text.strip()}\n"
            f"Source: {source} | {section} | chunk_id={c.chunk_id}\n"
        )
    return "\n".join(parts)


SYSTEM_PROMPT = """You are a technical support assistant.
Use the following pieces of context to answer the user's question.
If the answer is not present in the context, do NOT make up an answer.
Instead, simply say: "I'm sorry, that information is not in the documentation."

Rules:
- Answer based strictly on the provided context.
- Every sentence that states a fact MUST end with a citation in this format: (Source: <source> | <section_path>).
"""


def answer_question(query: str) -> dict:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. This demo uses real OpenAI models.")

    vectordb = load_vectordb()
    bm25_payload = load_bm25_payload()

    v = vector_search(query, vectordb, settings.top_k_vector)
    k = bm25_search(query, bm25_payload, settings.top_k_keyword)

    fused = rrf_fuse(v, k, settings.top_k_fused, settings.rrf_k)
    fused = simple_rerank(query, fused)

    context = build_context(fused)

    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)

    user_prompt = f"""Answer the question based strictly on the context below.
Always cite the 'Source' provided for each chunk. When citing sources:
- Cite only the document name and section path.
- Do NOT include internal identifiers such as chunk IDs or hashes.
- Use this format exactly:

(Source: <document> → <section path>)


{context}

Question: {query}
"""

    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    answer = resp.content.strip() if hasattr(resp, "content") else str(resp)

    # Minimal citation sanity check (non-blocking): require at least one (Source: ...)
    has_citation = "(Source:" in answer

    return {
        "query": query,
        "answer": answer,
        "used_chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "section_path": c.section_path,
            }
            for c in fused
        ],
        "citation_present": has_citation,
        "models": {
            "chat": settings.openai_chat_model,
            "embedding": settings.openai_embedding_model,
        },
    }


def simple_tokenize(text: str):
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]
