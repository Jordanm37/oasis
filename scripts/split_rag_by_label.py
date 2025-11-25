#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, TextIO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a RAG JSONL corpus into one file per label."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL (e.g. data/rag_corpus/jigsaw_corpus.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write per-label JSONL files into.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: Dict[str, TextIO] = {}

    def get_labels(entry: dict) -> list[str]:
        # 1) Jigsaw-style: single 'label' field
        if "label" in entry and entry["label"]:
            return [str(entry["label"]).strip()]

        # 2) Persona-style: 'allowed_labels' as string or list
        allowed = entry.get("allowed_labels", "")
        if isinstance(allowed, str):
            labels = [lbl.strip() for lbl in allowed.split(";") if lbl.strip()]
            if labels:
                return labels
        elif isinstance(allowed, list):
            labels = [str(lbl).strip() for lbl in allowed if str(lbl).strip()]
            if labels:
                return labels

        # 3) Nothing usable
        return []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            labels = get_labels(entry)
            if not labels:
                continue

            for lbl in labels:
                safe_lbl = (
                    lbl.replace("/", "_")
                    .replace(" ", "_")
                    .replace("[", "")
                    .replace("]", "")
                )
                out_path = out_dir / f"{safe_lbl}.jsonl"
                if safe_lbl not in writers:
                    writers[safe_lbl] = out_path.open("w", encoding="utf-8")
                writers[safe_lbl].write(
                    json.dumps(entry, ensure_ascii=False) + "\n"
                )

    for fh in writers.values():
        fh.close()

    print(f"Done. Wrote {len(writers)} label files to {out_dir}")


if __name__ == "__main__":
    main()
