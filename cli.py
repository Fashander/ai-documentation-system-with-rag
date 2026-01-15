from __future__ import annotations

import argparse
import json

from app.ingest import run_ingest
from app.rag import answer_question


def main():
    p = argparse.ArgumentParser(description="Docs RAG demo (OpenAI)")
    p.add_argument("--ingest", action="store_true", help="Ingest docs into Chroma + BM25")
    p.add_argument("--ask", type=str, default=None, help="Ask a question")
    args = p.parse_args()

    if args.ingest:
        res = run_ingest()
        print(json.dumps(res, indent=2))
        return

    if args.ask:
        res = answer_question(args.ask)
        print(json.dumps(res, indent=2))
        return

    p.print_help()


if __name__ == "__main__":
    main()
