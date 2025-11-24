#!/usr/bin/env python3
"""
Build a Chroma vector DB from a RAG JSONL corpus (e.g. Jigsaw toxic comments).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAG_DIR = DATA_DIR / "rag_corpus"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma DB from RAG JSONL.")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=str(RAG_DIR / "jigsaw_corpus.jsonl"),
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DATA_DIR / "chroma" / "jigsaw_db"),
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="jigsaw_toxic",
    )
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading entries from {in_path} ...")
    entries = list(load_jsonl(in_path))
    print(f"Loaded {len(entries)} entries.")

    # Set up Chroma (persistent DB)
    client = chromadb.PersistentClient(path=str(db_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=embed_fn,
    )

    ids = []
    docs = []
    metadatas = []

    for idx, e in enumerate(entries):
        # Use dataset id if present, else fallback to index
        eid = e.get("id") or f"row-{idx}"
        ids.append(str(eid))
        docs.append(e["text"])
        metadatas.append(
            {
                "source": e.get("source", ""),
                "persona_variant": e.get("persona_variant", ""),
                "labels": ",".join(e.get("labels", [])),
            }
        )

    print("Adding to Chroma collection ...")
    # Chroma can only ingest chunks at a time; we’ll batch
    batch_size = 512
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end} / {len(ids)}")

    print(f"Done. Chroma DB at: {db_path}")
    print(f"Collection name: {args.collection_name}")


if __name__ == "__main__":
    main()
