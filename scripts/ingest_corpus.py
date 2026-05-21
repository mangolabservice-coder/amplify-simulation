"""
CLI helper: ingest everything in ./corpus into Qdrant.

Usage:
    python -m scripts.ingest_corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import ingest_paths  # noqa: E402

CORPUS = ROOT / "corpus"


def main() -> None:
    files = [p for p in CORPUS.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"No files in {CORPUS}. Drop PDFs/markdown there and re-run.")
        return
    print(f"Ingesting {len(files)} file(s) from {CORPUS}…")
    for p in files:
        print(f"  - {p.name}")
    result = ingest_paths(files)
    print("Done:", result)


if __name__ == "__main__":
    main()
