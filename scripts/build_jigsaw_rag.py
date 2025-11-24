#!/usr/bin/env python3
"""
Build RAG corpus JSONL from the Jigsaw toxic comment dataset (train.csv).

Input:
  data/jigsaw/train.csv

Output:
  data/rag_corpus/jigsaw_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JIGSAW_DIR = DATA_DIR / "jigsaw"
RAG_DIR = DATA_DIR / "rag_corpus"
RAG_DIR.mkdir(parents=True, exist_ok=True)


LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def row_to_tags(row: pd.Series) -> List[str]:
    """
    Convert the 0/1 label columns into a list of textual tags.
    If all labels are 0 → treat as 'benign'.
    """
    tags: List[str] = []

    for col in LABEL_COLS:
        if int(row[col]) == 1:
            tags.append(col)

    if not tags:
        tags.append("benign")

    # Optional: we can include a dataset tag too
    tags.append("dataset:jigsaw")
    return tags


def build_entries(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        text = str(row["comment_text"]).strip()
        if not text:
            continue

        tags = row_to_tags(row)

        entry = {
            "id": str(row["id"]),
            "persona_variant": "jigsaw_toxic",   # generic variant name
            "text": text,
            "source": "jigsaw_train",
            "tags": tags,                        # <-- used later to split by class
            "labels": [t for t in tags if not t.startswith("dataset:")],
        }
        entries.append(entry)
    return entries


def write_jsonl(entries, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG corpus from Jigsaw toxic comment dataset.")
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(JIGSAW_DIR / "train.csv"),
        help="Path to Jigsaw train.csv",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=str(RAG_DIR / "jigsaw_corpus.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    train_path = Path(args.train_path)
    out_path = Path(args.out_jsonl)

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    print(f"Loading {train_path} ...")
    df = pd.read_csv(train_path)

    print(f"Rows in train.csv: {len(df)}")
    entries = build_entries(df)
    print(f"Non-empty comments: {len(entries)}")

    print(f"Writing JSONL to {out_path} ...")
    write_jsonl(entries, out_path)

    print("Done.")
    print(f"Example labels counts (first 100 rows):")
    sample = df.head(100)
    print(sample[LABEL_COLS].sum())


if __name__ == "__main__":
    main()
