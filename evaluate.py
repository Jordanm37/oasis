#!/usr/bin/env python3
"""
Single-label evaluation for CLEAN dataset (id, content, label).
Inputs:
  --truth: JSONL/CSV with fields id, label (content unused for scoring)
  --pred:  CSV with fields id, label (submission)
Metrics:
  Accuracy, Macro-F1, per-class F1.
Usage:
  python evaluate.py --truth data/runs/FINAL_training_dataset_CLEAN.jsonl --pred submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_truth(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    labels: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = str(obj.get("id") or "").strip()
                lab = str(obj.get("label") or "").strip()
                if not rid or not lab:
                    continue
                ids.append(rid)
                labels.append(lab)
    else:  # CSV
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = str(row.get("id") or "").strip()
                lab = str(row.get("label") or "").strip()
                if not rid or not lab:
                    continue
                ids.append(rid)
                labels.append(lab)
    return ids, labels


def load_pred(path: Path) -> Dict[str, str]:
    pred: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id") or "").strip()
            lab = str(row.get("label") or "").strip()
            if not rid:
                continue
            pred[rid] = lab
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="Ground truth JSONL/CSV path (id,label)")
    ap.add_argument("--pred", required=True, help="Submission CSV path (id,label)")
    args = ap.parse_args()

    truth_ids, truth_labels = load_truth(Path(args.truth))
    pred_map = load_pred(Path(args.pred))

    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0
    for tid, tlab in zip(truth_ids, truth_labels):
        y_true.append(tlab)
        plab = pred_map.get(tid, "")
        if not plab:
            missing += 1
        y_pred.append(plab)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Samples: {len(y_true)}")
    print(f"Missing preds: {missing}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print("\nPer-class F1:")
    for lab, stats in report.items():
        if lab in {"accuracy", "macro avg", "weighted avg"}:
            continue
        print(f"  {lab:25s} f1={stats.get('f1-score',0):.4f} support={stats.get('support',0)}")

    print("\nLabel counts (truth):")
    cnt = Counter(y_true)
    for lab, v in cnt.most_common():
        print(f"  {lab:25s} {v}")


if __name__ == "__main__":
    main()
