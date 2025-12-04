#!/usr/bin/env python3
"""
Single-label baseline for CLEAN dataset (id, content, label).
TF-IDF (word bigrams) + Logistic Regression (OvR).
Stratified train/val split (default 80/20).
Reports Accuracy and Macro-F1 on val.

Usage:
    python baseline_tfidf_single.py --data data/runs/FINAL_training_dataset_CLEAN.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("content") or ""
                lab = obj.get("label") or obj.get("labels") or ""
                text = str(text)
                lab = str(lab).strip()
                if not lab:
                    continue
                texts.append(text)
                labels.append(lab)
    else:  # CSV
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("content") or ""
                lab = row.get("label") or row.get("labels") or ""
                text = str(text)
                lab = str(lab).strip()
                if not lab:
                    continue
                texts.append(text)
                labels.append(lab)
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/runs/FINAL_training_dataset_CLEAN.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    texts, labels = load_data(Path(args.data))
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    clf = LogisticRegression(
        max_iter=200,
        C=4.0,
        class_weight="balanced",
        multi_class="ovr",
    )

    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xva)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"Samples: {len(texts)} | Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
