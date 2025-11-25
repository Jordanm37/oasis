#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "hate_speech" / "labeled_data.csv"   # adjust if your path is different
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "hate_speech_processed.csv"


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)

    # Normalise some common unicode quotes / dashes
    t = (
        t.replace("\u2019", "'")
         .replace("\u2018", "'")
         .replace("\u201c", '"')
         .replace("\u201d", '"')
         .replace("\u2014", "-")
         .replace("\u2013", "-")
    )

    # Collapse runs of exclamation marks to a single "!"
    t = re.sub(r"!{2,}", "!", t)

    # (Optional) collapse runs of question marks too:
    # t = re.sub(r"\?{2,}", "?", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)

    # Strip weird control chars
    t = "".join(ch for ch in t if ch.isprintable())

    return t.strip()



def map_class_to_label(cls: int) -> str:
    """
    Dataset 'class' meanings:
      0 = hate_speech
      1 = offensive_language
      2 = neither
    We map:
      0 or 1 -> 'incel_misogyny'   (harmful)
      2      -> 'benign'           (non-harmful)
    """
    if cls in (0, 1):
        return "incel_misogyny"
    return "benign"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Basic sanity checks
    required_cols = {"class", "tweet"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Build processed DataFrame
    out_rows = []
    for idx, row in df.iterrows():
        text_raw = row["tweet"]
        text_clean = clean_text(text_raw)
        if not text_clean:
            continue

        cls = int(row["class"])
        label = map_class_to_label(cls)

        # Use Unnamed: 0 as stable ID if present, else idx
        if "Unnamed: 0" in df.columns:
            row_id = int(row["Unnamed: 0"])
        else:
            row_id = int(idx)

        out_rows.append(
            {
                "id": row_id,
                "text": text_clean,
                "label": label,
                "dataset_name": "davidson_hate_offensive",
            }
        )

    out_df = pd.DataFrame(out_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote processed Davidson dataset to: {OUT_PATH}")
    print(f"   Rows: {len(out_df)}")
    print("   Label counts:")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
