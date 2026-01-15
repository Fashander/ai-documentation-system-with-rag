from __future__ import annotations

import os

from app.ingest import run_ingest
from app.rag import answer_question


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run this smoke test.")

    ingest = run_ingest()
    print("ingest:", ingest)

    for q in [
        "How do I authenticate?",
        "How do I reset my key?",
        "What does ERR-AD-99 mean?",
        "What is the moon made of?",
    ]:
        out = answer_question(q)
        print("\nQ:", q)
        print(out["answer"])
        assert "answer" in out


if __name__ == "__main__":
    main()
