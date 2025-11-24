#!/usr/bin/env python3
"""
Split a RAG JSONL corpus into one file per label (class).

Input:
  data/rag_corpus/jigsaw_corpus.jsonl

Output:
  data/rag_corpus/by_class/toxic.jsonl
  data/rag_corpus/by_class/insult.jsonl
  ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAG_DIR = DATA_DIR / "rag_corpus"
OUT_DIR = RAG_DIR / "by_class"


# These are the class names we expect to see in `labels` or `tags`
CLASS_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "benign",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split RAG JSONL into per-class JSONL files.")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=str(RAG_DIR / "jigsaw_corpus.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
    )
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    buckets: Dict[str, List[dict]] = defaultdict(list)

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Prefer explicit 'labels' field, fallback to 'tags'
            labels = obj.get("labels")
            if labels is None:
                labels = obj.get("tags", [])
            labels = [str(l) for l in labels]

            for label in CLASS_LABELS:
                if label in labels:
                    buckets[label].append(obj)

    # Write each class file
    for label, entries in buckets.items():
        if not entries:
            continue
        out_path = out_dir / f"{label}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} entries to {out_path}")

    if not buckets:
        print("No entries found for any CLASS_LABELS – check your labels/tags fields.")


if __name__ == "__main__":
    main()
