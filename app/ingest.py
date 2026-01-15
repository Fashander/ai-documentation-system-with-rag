from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import settings


def stable_id(*parts: str) -> str:
    """Deterministic id from stable parts."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


def read_markdown_files(docs_path: str) -> List[Tuple[str, str]]:
    """Return list of (relative_path, text)."""
    root = Path(docs_path).resolve()
    files = sorted(root.rglob("*.md"))
    out: List[Tuple[str, str]] = []
    for fp in files:
        rel = str(fp.relative_to(root))
        out.append((rel, fp.read_text(encoding="utf-8")))
    return out


def split_markdown(rel_path: str, markdown_text: str):
    """Chained splitting: headers -> recursive size splitting.

    Notes:
      - Header splitter preserves section metadata.
      - Recursive splitter uses character counts.
    """
    headers_to_split_on = [("#", "Section"), ("##", "Subsection"), ("###", "Topic")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = header_splitter.split_text(markdown_text)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    final_chunks = size_splitter.split_documents(header_docs)

    # Attach stable metadata and ids
    for ordinal, d in enumerate(final_chunks):
        d.metadata.setdefault("source", rel_path)
        # Build a human-friendly section path
        section_path = " / ".join(
            [
                str(d.metadata.get("Section", "")).strip(),
                str(d.metadata.get("Subsection", "")).strip(),
                str(d.metadata.get("Topic", "")).strip(),
            ]
        ).replace("  ", " ").strip(" /")
        d.metadata["section_path"] = section_path
        d.metadata["ordinal"] = ordinal
        d.metadata["chunk_id"] = stable_id(rel_path, section_path, str(ordinal))

    return final_chunks


def build_bm25_index(chunks):
    """Build a BM25 index and return (bm25, corpus, metas).

    We store a minimal, pickle-able structure (tokens + metadata) so we can load it in the API.
    """
    from rank_bm25 import BM25Okapi

    corpus = [c.page_content for c in chunks]
    tokenized = [simple_tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized)

    metas = [c.metadata for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]

    payload = {
        "tokenized": tokenized,
        "corpus": corpus,
        "metas": metas,
        "ids": ids,
    }
    return bm25, payload


def simple_tokenize(text: str):
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def run_ingest() -> dict:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. This demo uses real OpenAI embeddings and a chat model."
        )

    md_files = read_markdown_files(settings.docs_path)
    all_chunks = []
    for rel_path, text in md_files:
        all_chunks.extend(split_markdown(rel_path, text))

    # Embed + persist to Chroma
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_dir,
        collection_name="docs",
    )

    # Build and persist BM25 side index
    _, bm25_payload = build_bm25_index(all_chunks)
    with open(settings.bm25_path, "wb") as f:
        pickle.dump(bm25_payload, f)

    return {
        "docs": len(md_files),
        "chunks": len(all_chunks),
        "chroma_dir": os.path.abspath(settings.chroma_dir),
        "bm25_path": os.path.abspath(settings.bm25_path),
        "embedding_model": settings.openai_embedding_model,
    }


if __name__ == "__main__":
    print(run_ingest())
