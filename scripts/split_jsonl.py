#!/usr/bin/env python3
"""Split a JSONL dataset into train/test/holdout by user, stratified by persona."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def split_jsonl(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    test_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Split JSONL file by user into train/test/holdout sets."""
    random.seed(seed)

    # Load all records
    records: List[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    # Group records by user_id
    user_to_records: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        user_id = rec.get("user_id", "unknown")
        user_to_records[user_id].append(rec)

    print(f"Found {len(user_to_records)} unique users")

    # Group users by persona for stratified splitting
    persona_to_users: Dict[str, List[str]] = defaultdict(list)
    for user_id, user_records in user_to_records.items():
        # Get persona from first record (they should all be the same for a user)
        persona = user_records[0].get("persona_id", "unknown")
        persona_to_users[persona].append(user_id)

    print(f"Persona distribution:")
    for persona, users in sorted(persona_to_users.items()):
        print(f"  {persona}: {len(users)} users")

    # Assign users to splits, stratified by persona
    user_to_split: Dict[str, str] = {}
    for persona, users in persona_to_users.items():
        random.shuffle(users)
        n = len(users)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)

        for i, user_id in enumerate(users):
            if i < n_train:
                user_to_split[user_id] = "train"
            elif i < n_train + n_test:
                user_to_split[user_id] = "test"
            else:
                user_to_split[user_id] = "holdout"

    # Count split distribution
    split_counts: Dict[str, int] = defaultdict(int)
    for split in user_to_split.values():
        split_counts[split] += 1
    print(f"\nUser split distribution:")
    print(f"  train: {split_counts['train']} users")
    print(f"  test: {split_counts['test']} users")
    print(f"  holdout: {split_counts['holdout']} users")

    # Assign records to splits and write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    split_records: Dict[str, List[dict]] = {"train": [], "test": [], "holdout": []}
    for rec in records:
        user_id = rec.get("user_id", "unknown")
        split = user_to_split.get(user_id, "train")
        rec["split"] = split
        split_records[split].append(rec)

    # Write split files
    for split_name, split_recs in split_records.items():
        out_path = output_dir / f"{stem}_{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in split_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(split_recs)} records to {out_path}")

    # Also write combined file with split field
    combined_path = output_dir / f"{stem}_split.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} records (with split field) to {combined_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")
    print(f"  train:   {len(split_records['train']):,} ({100*len(split_records['train'])/len(records):.1f}%)")
    print(f"  test:    {len(split_records['test']):,} ({100*len(split_records['test'])/len(records):.1f}%)")
    print(f"  holdout: {len(split_records['holdout']):,} ({100*len(split_records['holdout'])/len(records):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Split JSONL dataset by user")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    split_jsonl(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
