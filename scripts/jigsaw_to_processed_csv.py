#!/usr/bin/env python3
"""
Convert raw Jigsaw toxic comments train.csv into a processed CSV
with columns: id, text, label, dataset_name.

Label mapping:
  - If ANY of [toxic, severe_toxic, obscene, threat, insult, identity_hate] == 1
      -> label = "incel_misogyny"
  - Else
      -> label = "benign"

This processed file is then used by build_dataset_rag.py.
"""

import unicodedata
import re
from pathlib import Path

import pandas as pd

# Adjust these paths if your layout is different
RAW_PATH = Path("data/jigsaw/train.csv")
OUT_PATH = Path("data/processed/jigsaw_processed.csv")


def clean_text(text: str) -> str:
    """Lightly clean comment text while keeping emojis and non-English chars."""
    if not isinstance(text, str):
        text = str(text)

    # 1) Normalize Unicode (fixes weird encodings / ambiguous forms)
    text = unicodedata.normalize("NFC", text)

    # 2) Remove control characters except basic whitespace
    def _is_ok(ch: str) -> bool:
        # Keep normal whitespace (space, tab, newline)
        if ch in {"\n", "\t"}:
            return True
        return ch >= " "

    text = "".join(ch for ch in text if _is_ok(ch))

    # 3) Collapse multiple whitespace to a single space
    text = re.sub(r"\s+", " ", text)

    # 4) Strip leading/trailing spaces
    text = text.strip()

    return text


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find Jigsaw train.csv at {RAW_PATH}")

    print(f"Loading raw Jigsaw data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Check required columns
    required_cols = [
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in train.csv: {missing}")

    # Binary toxic flag: any of the toxicity columns > 0
    toxic_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    toxic_flag = (df[toxic_cols].sum(axis=1) > 0).astype(int)

    # Map to ontology-aligned labels
    # 1 -> harmful (mapped to incel_misogyny bucket in this ontology)
    # 0 -> benign
    label_map = {1: "incel_misogyny", 0: "benign"}
    labels = toxic_flag.map(label_map)

    # Build processed dataframe
    df_out = pd.DataFrame(
        {
            "id": df["id"],
            "text": df["comment_text"].astype(str).apply(clean_text),
            "label": labels,
            "dataset_name": "jigsaw_toxic",
        }
    )

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"Saved processed Jigsaw data to: {OUT_PATH}")
    print("Label distribution:")
    print(df_out["label"].value_counts())


if __name__ == "__main__":
    main()
