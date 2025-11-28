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
    parser.add_argument("--personas-csv", type=str, default=None, help="Path to personas CSV for persona-aware imputation")
    parser.add_argument("--reset", action="store_true", help="Reset text_rag_imputed column before running")
    parser.add_argument("--retry-failed", action="store_true", help="Re-process items where imputed text still contains LBL: tokens")
    return parser.parse_args()


def count_tokens_in_db(db_path: Path) -> tuple[int, int, dict]:
    """Count posts and comments with label tokens, and breakdown by token type."""
    import re
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Count posts/comments with tokens
    cur.execute("SELECT COUNT(*) FROM post WHERE content LIKE '%LBL:%' OR content LIKE '%<LBL:%'")
    post_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE content LIKE '%LBL:%' OR content LIKE '%<LBL:%'")
    comment_count = cur.fetchone()[0]
    
    # Count already imputed
    cur.execute("SELECT COUNT(*) FROM post WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    posts_imputed = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    comments_imputed = cur.fetchone()[0]
    
    # Get token breakdown (sample first 1000 for speed)
    token_counts: dict = {}
    label_pattern = re.compile(r'<?(LBL:[A-Z_]+)>?')
    
    cur.execute("SELECT content FROM post WHERE content LIKE '%LBL:%' LIMIT 1000")
    for (content,) in cur.fetchall():
        for match in label_pattern.findall(content or ""):
            token_counts[match] = token_counts.get(match, 0) + 1
    
    cur.execute("SELECT content FROM comment WHERE content LIKE '%LBL:%' LIMIT 1000")
    for (content,) in cur.fetchall():
        for match in label_pattern.findall(content or ""):
            token_counts[match] = token_counts.get(match, 0) + 1
    
    conn.close()
    
    stats = {
        "posts_with_tokens": post_count,
        "comments_with_tokens": comment_count,
        "posts_imputed": posts_imputed,
        "comments_imputed": comments_imputed,
        "posts_remaining": post_count - posts_imputed,
        "comments_remaining": comment_count - comments_imputed,
        "token_breakdown": dict(sorted(token_counts.items(), key=lambda x: -x[1])[:20]),  # Top 20
    }
    return post_count, comment_count, stats


def reset_imputed_columns(db_path: Path) -> None:
    """Clear text_rag_imputed columns to force re-imputation."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE post SET text_rag_imputed = NULL")
    conn.execute("UPDATE comment SET text_rag_imputed = NULL")
    conn.commit()
    conn.close()
    print("[imputer] Reset text_rag_imputed columns")


def reset_failed_imputations(db_path: Path) -> tuple[int, int]:
    """Reset items where text_rag_imputed still contains LBL: tokens.
    
    Returns:
        Tuple of (posts_reset, comments_reset) counts.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Count and reset posts with failed imputation
    cur.execute("""
        SELECT COUNT(*) FROM post 
        WHERE text_rag_imputed IS NOT NULL 
        AND (text_rag_imputed LIKE '%LBL:%' OR text_rag_imputed LIKE '%<LBL:%')
    """)
    posts_failed = cur.fetchone()[0]
    
    cur.execute("""
        UPDATE post SET text_rag_imputed = NULL
        WHERE text_rag_imputed IS NOT NULL 
        AND (text_rag_imputed LIKE '%LBL:%' OR text_rag_imputed LIKE '%<LBL:%')
    """)
    
    # Count and reset comments with failed imputation
    cur.execute("""
        SELECT COUNT(*) FROM comment 
        WHERE text_rag_imputed IS NOT NULL 
        AND (text_rag_imputed LIKE '%LBL:%' OR text_rag_imputed LIKE '%<LBL:%')
    """)
    comments_failed = cur.fetchone()[0]
    
    cur.execute("""
        UPDATE comment SET text_rag_imputed = NULL
        WHERE text_rag_imputed IS NOT NULL 
        AND (text_rag_imputed LIKE '%LBL:%' OR text_rag_imputed LIKE '%<LBL:%')
    """)
    
    conn.commit()
    conn.close()
    
    print(f"[imputer] Reset {posts_failed} posts with failed imputation (still had LBL tokens)")
    print(f"[imputer] Reset {comments_failed} comments with failed imputation (still had LBL tokens)")
    print(f"[imputer] Total items to re-process: {posts_failed + comments_failed}")
    
    return posts_failed, comments_failed


async def run_imputation(args: argparse.Namespace) -> None:
    db_path = Path(os.path.abspath(args.db))
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    # Count tokens before starting
    post_count, comment_count, stats = count_tokens_in_db(db_path)
    print(f"\n{'='*60}")
    print(f"[imputer] IMPUTATION STATUS")
    print(f"{'='*60}")
    print(f"[imputer] Posts with LBL tokens: {post_count}")
    print(f"[imputer] Comments with LBL tokens: {comment_count}")
    print(f"[imputer] Posts already imputed: {stats['posts_imputed']}")
    print(f"[imputer] Comments already imputed: {stats['comments_imputed']}")
    print(f"[imputer] Posts remaining: {stats['posts_remaining']}")
    print(f"[imputer] Comments remaining: {stats['comments_remaining']}")
    print(f"\n[imputer] Top LBL tokens found (sample):")
    for token, count in list(stats['token_breakdown'].items())[:15]:
        print(f"  {token}: {count}")
    print(f"{'='*60}\n")
    
    if args.reset:
        reset_imputed_columns(db_path)
    elif args.retry_failed:
        posts_reset, comments_reset = reset_failed_imputations(db_path)
        if posts_reset == 0 and comments_reset == 0:
            print("[imputer] No failed imputations found. Nothing to retry.")
            return
    
    print(f"[imputer] Database: {db_path}")
    print(f"[imputer] Provider: {IMPUTATION_PROVIDER}")
    print(f"[imputer] Model: {IMPUTATION_MODEL}")
    print(f"[imputer] Workers: {args.workers}")
    print(f"[imputer] Batch size: {args.batch_size}")
    print(f"[imputer] Temperature: {args.temperature}")
    if args.personas_csv:
        print(f"[imputer] Personas CSV: {args.personas_csv}")
    
    # Build LLM settings
    llm_settings = LLMProviderSettings(
        provider=IMPUTATION_PROVIDER,
        model_name=IMPUTATION_MODEL,
    )
    
    # Create imputer config
    personas_path = Path(args.personas_csv) if args.personas_csv else None
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
        personas_csv_path=personas_path,
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
