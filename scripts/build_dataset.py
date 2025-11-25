#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from oasis.imputation.utils import (
    LABEL_TOKEN_PATTERN,
    StaticBank,
    extract_label_tokens,
)


DEFAULT_LABEL_MAPPING: Dict[str, List[str]] = {
    "LBL:INCEL_SLANG": ["incel"],
    "LBL:MISINFO_CLAIM": ["misinfo", "conspiracy"],
    "LBL:SUPPORTIVE": ["recovery", "benign"],
}


PERSONA_ALLOWED: Dict[str, List[str]] = {
    "incel": ["incel"],
    "misinfo": ["misinfo", "conspiracy"],
    "benign": ["benign"],
}


def extract_tokens(text: str) -> List[str]:
    return extract_label_tokens(text)


def impute_text(
    raw_text: str, static_bank: StaticBank, seed: int, post_id: int
) -> Tuple[str, List[str]]:
    tokens = extract_tokens(raw_text)
    occurrence: Dict[str, int] = {}

    def replace_match(m: re.Match[str]) -> str:
        full = f"LBL:{m.group(1)}"
        occurrence[full] = occurrence.get(full, 0) + 1
        return static_bank.deterministic_choice(full, seed, post_id, occurrence[full] - 1)

    new_text = LABEL_TOKEN_PATTERN.sub(replace_match, raw_text)
    return new_text, tokens


def token_to_categories(token: str, mapping: Dict[str, List[str]]) -> List[str]:
    return mapping.get(token, [])


def assign_labels(tokens: List[str], persona: Optional[str]) -> List[str]:
    cats: List[str] = []
    for t in tokens:
        cats.extend(token_to_categories(t, DEFAULT_LABEL_MAPPING))
    cats = sorted(set(cats))
    if persona:
        allowed = PERSONA_ALLOWED.get(persona, cats)
        cats = [c for c in cats if c in allowed]
        if not cats and allowed:
            cats = [allowed[0]]
    return cats


def infer_persona_from_username(username: str) -> Optional[str]:
    """Fallback persona inference from username prefix."""
    lowered = username.lower()
    # Check common prefixes from ontology
    prefixes = [
        "incel", "misinfo", "benign", "conspiracy", "recovery",
        "edrisk", "ed_risk", "trad", "gamergate", "pro_ana",
        "alpha", "extremist", "hate_speech", "bullying",
    ]
    for prefix in prefixes:
        if lowered.startswith(f"{prefix}_"):
            # Normalize ed_risk variants
            if prefix in ("edrisk", "ed_risk"):
                return "ed_risk"
            return prefix
    return None


def load_personas_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load personas CSV and return a map of username -> persona data.
    
    Returns:
        Dict mapping username to dict with keys: primary_label, secondary_label, etc.
    """
    if not csv_path or not csv_path.exists():
        return {}
    personas: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            username = row.get("username", "").strip()
            if username:
                personas[username] = {
                    "primary_label": row.get("primary_label", "").strip(),
                    "secondary_label": row.get("secondary_label", "").strip(),
                    "allowed_labels": row.get("allowed_labels", "").strip(),
                }
    return personas


def isoformat_timestamp(ts_val) -> str:
    # post.created_at may be integer (twitter clock) or datetime (reddit mode)
    # Map both to ISO-8601 string deterministically.
    if isinstance(ts_val, (int, float)):
        # Interpret as minutes since start; render as a pseudo-UTC time.
        base = datetime(2025, 1, 1)
        return (base.replace(microsecond=0) + (ts_val * 60) * 1e-6 * 0).isoformat() + "Z"
    try:
        return datetime.fromisoformat(str(ts_val)).isoformat() + "Z"
    except Exception:
        return str(ts_val)


def _load_sidecar(sidecar_path: Optional[Path]) -> tuple[Dict[int, dict], Dict[int, dict]]:
    if not sidecar_path or not sidecar_path.exists():
        return {}, {}
    post_map: Dict[int, dict] = {}
    comment_map: Dict[int, dict] = {}
    with sidecar_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            pid = rec.get("post_id")
            cid = rec.get("comment_id")
            if isinstance(pid, int):
                post_map[pid] = rec
            if isinstance(cid, int):
                comment_map[cid] = rec
    return post_map, comment_map


def build_dataset(
    db_path: Path,
    out_path: Path,
    bank_path: Path,
    seed: int,
    skip_imputation: bool = False,
    sidecar: Optional[Path] = None,
    personas_csv: Optional[Path] = None,
) -> None:
    static_bank = StaticBank.load_simple_yaml(bank_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Load sidecar if present for authoritative label decisions
    post_sidecar, comment_sidecar = _load_sidecar(sidecar)

    # Load personas CSV if provided for authoritative persona/label mapping
    personas_map = load_personas_csv(personas_csv) if personas_csv else {}

    # Fetch users for persona inference
    cur.execute("SELECT user_id, user_name, name FROM user")
    user_rows = cur.fetchall()
    uid_to_username: Dict[int, str] = {}
    uid_to_persona: Dict[int, Dict[str, str]] = {}
    for uid, uname, name in user_rows:
        username = uname or name or f"user_{uid}"
        uid_to_username[uid] = username
        # Try to find this username in the personas CSV
        if username in personas_map:
            uid_to_persona[uid] = personas_map[username]

    # Check if text_rag_imputed column exists in post table
    cur.execute("PRAGMA table_info(post)")
    post_columns = {row[1] for row in cur.fetchall()}
    has_post_rag_col = "text_rag_imputed" in post_columns

    # Check if text_rag_imputed column exists in comment table
    cur.execute("PRAGMA table_info(comment)")
    comment_columns = {row[1] for row in cur.fetchall()}
    has_comment_rag_col = "text_rag_imputed" in comment_columns

    # Fetch posts - handle missing text_rag_imputed column gracefully
    if has_post_rag_col:
        cur.execute(
            "SELECT post_id, user_id, original_post_id, content, quote_content, created_at, text_rag_imputed "
            "FROM post ORDER BY post_id"
        )
    else:
        cur.execute(
            "SELECT post_id, user_id, original_post_id, content, quote_content, created_at, NULL "
            "FROM post ORDER BY post_id"
        )
    post_rows = cur.fetchall()

    # Fetch comments - handle missing text_rag_imputed column gracefully
    if has_comment_rag_col:
        cur.execute(
            "SELECT comment_id, post_id, user_id, content, created_at, text_rag_imputed "
            "FROM comment ORDER BY comment_id"
        )
    else:
        cur.execute(
            "SELECT comment_id, post_id, user_id, content, created_at, NULL "
            "FROM comment ORDER BY comment_id"
        )
    comment_rows = cur.fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        # Write posts
        for (
            post_id,
            user_id,
            original_post_id,
            content,
            quote_content,
            created_at,
            text_rag_imputed_db,
        ) in post_rows:
            text_raw = content or ""
            username = uid_to_username.get(user_id, f"user_{user_id}")
            
            # Get persona from CSV (authoritative) or fall back to username inference
            persona_data = uid_to_persona.get(user_id, {})
            persona = persona_data.get("primary_label") or infer_persona_from_username(username)
            
            tokens = extract_tokens(text_raw)
            normalized_imputed = (text_rag_imputed_db or "").strip()
            imputer_source: str
            if normalized_imputed:
                imputed_text = normalized_imputed
                imputer_source = "rag-llm"
            elif skip_imputation:
                imputed_text = text_raw
                imputer_source = "skip"
            else:
                imputed_text, tokens = impute_text(text_raw, static_bank, seed, int(post_id))
                imputer_source = "v0-mvp"
            
            # Determine category labels:
            # 1. Use persona's primary_label as the authoritative category
            # 2. Fall back to token-based labels if no persona
            # 3. Default to "benign" if neither available
            if persona:
                labels = [persona]
            else:
                labels = assign_labels(tokens, persona)
                if not labels:
                    labels = ["benign"]

            # Override with sidecar decisions when available (sidecar is most authoritative)
            sc = post_sidecar.get(int(post_id))
            if sc:
                mode = sc.get("expected_mode")
                sc_labels = sc.get("category_labels") or []
                if mode == "none":
                    labels = ["benign"]
                elif isinstance(sc_labels, list) and sc_labels:
                    labels = list(dict.fromkeys(sc_labels))  # preserve order/unique

            rec = {
                "post_id": f"p_{post_id}",
                "thread_id": f"p_{original_post_id}" if original_post_id else f"p_{post_id}",
                "user_id": str(user_id),
                "parent_id": None,
                "timestamp": str(created_at),
                "text": imputed_text,
                "category_labels": labels,
                "split": "train",
                "provenance": f"gen:mvp persona:{persona or 'unknown'} | imputer:{imputer_source}",
                "generation_seed": int(seed),
                "persona_id": persona or "unknown",
                "needs_thread_context": False,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1

        # Write comments as child posts with parent_id
        for (
            comment_id,
            post_id,
            user_id,
            content,
            created_at,
            text_rag_imputed_db,
        ) in comment_rows:
            text_raw = content or ""
            username = uid_to_username.get(user_id, f"user_{user_id}")
            
            # Get persona from CSV (authoritative) or fall back to username inference
            persona_data = uid_to_persona.get(user_id, {})
            persona = persona_data.get("primary_label") or infer_persona_from_username(username)
            
            tokens = extract_tokens(text_raw)
            normalized_imputed = (text_rag_imputed_db or "").strip()
            imputer_source: str
            if normalized_imputed:
                imputed_text = normalized_imputed
                imputer_source = "rag-llm"
            elif skip_imputation:
                imputed_text = text_raw
                imputer_source = "skip"
            else:
                imputed_text, tokens = impute_text(
                    text_raw, static_bank, seed, int(comment_id) + 100000
                )
                imputer_source = "v0-mvp"
            
            # Determine category labels:
            # 1. Use persona's primary_label as the authoritative category
            # 2. Fall back to token-based labels if no persona
            # 3. Default to "benign" if neither available
            if persona:
                labels = [persona]
            else:
                labels = assign_labels(tokens, persona)
                if not labels:
                    labels = ["benign"]

            # Override with sidecar decisions when available (sidecar is most authoritative)
            sc = comment_sidecar.get(int(comment_id))
            if sc:
                mode = sc.get("expected_mode")
                sc_labels = sc.get("category_labels") or []
                if mode == "none":
                    labels = ["benign"]
                elif isinstance(sc_labels, list) and sc_labels:
                    labels = list(dict.fromkeys(sc_labels))

            rec = {
                "post_id": f"c_{comment_id}",
                "thread_id": f"p_{post_id}",
                "user_id": str(user_id),
                "parent_id": f"p_{post_id}",
                "timestamp": str(created_at),
                "text": imputed_text,
                "category_labels": labels,
                "split": "train",
                "provenance": f"gen:mvp persona:{persona or 'unknown'} | imputer:{imputer_source}",
                "generation_seed": int(seed),
                "persona_id": persona or "unknown",
                "needs_thread_context": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1

    conn.close()
    print(f"Wrote {num_written} items (posts + comments) to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP posts JSONL from OASIS SQLite DB")
    parser.add_argument("--db", type=str, required=True, help="Path to OASIS SQLite DB")
    parser.add_argument("--out", type=str, default="./data/mvp/posts_mvp.jsonl")
    parser.add_argument(
        "--static-bank", type=str, default="./data/label_tokens_static_bank.yaml"
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--skip-imputation", action="store_true")
    parser.add_argument("--sidecar", type=str, default="", help="Optional path to sidecar JSONL for label overrides.")
    parser.add_argument(
        "--personas-csv", type=str, default="",
        help="Path to personas CSV for authoritative persona/label mapping."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(os.path.abspath(args.db))
    out_path = Path(os.path.abspath(args.out))
    bank_path = Path(os.path.abspath(args.static_bank))
    sidecar_path = Path(os.path.abspath(args.sidecar)) if args.sidecar else None
    personas_csv_path = Path(os.path.abspath(args.personas_csv)) if args.personas_csv else None
    build_dataset(
        db_path,
        out_path,
        bank_path,
        args.seed,
        skip_imputation=args.skip_imputation,
        sidecar=sidecar_path,
        personas_csv=personas_csv_path,
    )


if __name__ == "__main__":
    main()


