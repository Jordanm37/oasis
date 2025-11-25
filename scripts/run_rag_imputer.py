#!/usr/bin/env python3
"""Standalone script to run LLM-based RAG imputation on an existing database."""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from configs.llm_settings import (
    IMPUTATION_MAX_TOKENS,
    IMPUTATION_MODEL,
    IMPUTATION_PROVIDER,
    IMPUTATION_TEMPERATURE,
    RAG_IMPUTER_BATCH_SIZE,
    RAG_IMPUTER_MAX_WORKERS,
    RAG_IMPUTER_STATIC_BANK,
)
from oasis.imputation.rag_llm_imputer import RagImputer, RagImputerConfig
from orchestrator.model_provider import LLMProviderSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-based RAG imputation on existing DB")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--workers", type=int, default=RAG_IMPUTER_MAX_WORKERS, help="Number of concurrent workers")
    parser.add_argument("--batch-size", type=int, default=RAG_IMPUTER_BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--temperature", type=float, default=IMPUTATION_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=IMPUTATION_MAX_TOKENS, help="Max tokens per LLM call")
    parser.add_argument("--static-bank", type=str, default=RAG_IMPUTER_STATIC_BANK, help="Path to static phrase bank YAML")
    parser.add_argument("--reset", action="store_true", help="Reset text_rag_imputed column before running")
    return parser.parse_args()


def count_tokens_in_db(db_path: Path) -> tuple[int, int]:
    """Count posts and comments with label tokens."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post WHERE content LIKE '%LBL:%' OR content LIKE '%<LBL:%'")
    post_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE content LIKE '%LBL:%' OR content LIKE '%<LBL:%'")
    comment_count = cur.fetchone()[0]
    conn.close()
    return post_count, comment_count


def reset_imputed_columns(db_path: Path) -> None:
    """Clear text_rag_imputed columns to force re-imputation."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE post SET text_rag_imputed = NULL")
    conn.execute("UPDATE comment SET text_rag_imputed = NULL")
    conn.commit()
    conn.close()
    print("[imputer] Reset text_rag_imputed columns")


async def run_imputation(args: argparse.Namespace) -> None:
    db_path = Path(os.path.abspath(args.db))
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    # Count tokens before starting
    post_count, comment_count = count_tokens_in_db(db_path)
    print(f"[imputer] Found {post_count} posts and {comment_count} comments with label tokens")
    
    if args.reset:
        reset_imputed_columns(db_path)
    
    print(f"[imputer] Database: {db_path}")
    print(f"[imputer] Provider: {IMPUTATION_PROVIDER}")
    print(f"[imputer] Model: {IMPUTATION_MODEL}")
    print(f"[imputer] Workers: {args.workers}")
    print(f"[imputer] Batch size: {args.batch_size}")
    print(f"[imputer] Temperature: {args.temperature}")
    
    # Build LLM settings
    llm_settings = LLMProviderSettings(
        provider=IMPUTATION_PROVIDER,
        model_name=IMPUTATION_MODEL,
    )
    
    # Create imputer config
    config = RagImputerConfig(
        db_path=db_path,
        llm_settings=llm_settings,
        mode="sync",  # Use sync mode for standalone script
        batch_size=args.batch_size,
        max_workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        static_bank_path=Path(args.static_bank),
        run_seed=314159,
    )
    
    # Create and run imputer
    imputer = RagImputer(config)
    
    print("[imputer] Starting imputer...")
    await imputer.start()
    
    # Reset internal tracking to scan from beginning (for standalone mode)
    imputer._last_post_id = 0
    imputer._last_comment_id = 0
    print(f"[imputer] Reset tracking: last_post_id={imputer._last_post_id}, last_comment_id={imputer._last_comment_id}")
    print(f"[imputer] Queue size before enqueue: {imputer._queue.qsize()}")
    
    print("[imputer] Enqueueing rows with label tokens...")
    await imputer.enqueue_new_rows()
    print(f"[imputer] Queue size after enqueue: {imputer._queue.qsize()}")
    
    print("[imputer] Flushing queue (processing all pending items)...")
    await imputer.flush()
    
    print("[imputer] Shutting down...")
    await imputer.shutdown()
    
    print("[imputer] Done!")


def main() -> None:
    args = parse_args()
    asyncio.run(run_imputation(args))


if __name__ == "__main__":
    main()
