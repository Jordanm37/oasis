#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAG_DIR = DATA_DIR / "rag_corpus"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAG corpus JSONL from a labelled dataset (e.g. Jigsaw toxic comments)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV (train.csv or processed file).",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="comment_text",
        help="Name of the text column.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Name of the class label column (already mapped to ontology labels).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(RAG_DIR / "dataset_corpus.jsonl"),
        help="Where to write the JSONL RAG corpus.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in CSV.")
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in CSV.")

    # Basic clean-up: drop missing text
    df = df.dropna(subset=[args.text_col])

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = str(row[args.text_col]).strip()
            label = str(row[args.label_col]).strip()

            if not text:
                continue

            entry = {
                "persona_variant": label,       # reuse field name from persona_corpus.jsonl
                "text": text,
                "source": "dataset_jigsaw",
                "tags": ["dataset_sample", label],
                "allowed_labels": label,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote dataset RAG corpus to: {out_path}")


if __name__ == "__main__":
    main()
