#!/usr/bin/env python3
"""
Generate SQL commands to update Sharkey post timestamps.

This script generates SQL that can be run on the server to set realistic
timestamps for uploaded posts.

Usage:
    # Generate SQL to spread posts over 7 days
    poetry run python3 scripts/botsocial/generate_timestamp_sql.py \
        --dataset data/runs/prod_100p_100s_65_35_dataset.jsonl \
        --spread-days 7 \
        --output timestamp_updates.sql

    # Then copy to server and run:
    scp timestamp_updates.sql jordan@botsocial.mlai.au:~/
    ssh jordan@botsocial.mlai.au
    sudo -u postgres psql -d sharkey -f ~/timestamp_updates.sql

Author: eSafety Hackathon Team
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def load_post_mapping(mapping_file: Path) -> dict[str, str]:
    """Load dataset post_id -> sharkey note_id mapping."""
    mapping = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            for post_id, note_id in data.items():
                mapping[post_id] = note_id
    return mapping


def generate_spread_sql(
    spread_days: int,
    start_date: Optional[datetime] = None,
    hours_lookback: int = 24
) -> str:
    """Generate SQL to spread recent posts over N days."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=spread_days)
    
    sql = f"""
-- =============================================================================
-- Spread posts over {spread_days} days
-- Generated: {datetime.now().isoformat()}
-- =============================================================================

-- First, let's see what we're working with
SELECT 
    COUNT(*) as total_posts,
    MIN("createdAt") as earliest,
    MAX("createdAt") as latest
FROM note
WHERE "createdAt" > NOW() - INTERVAL '{hours_lookback} hours';

-- Spread posts evenly over {spread_days} days
-- Posts will be ordered by their original creation time
WITH numbered_notes AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY "createdAt" ASC) as row_num,
        COUNT(*) OVER () as total_count
    FROM note
    WHERE "createdAt" > NOW() - INTERVAL '{hours_lookback} hours'
)
UPDATE note n
SET "createdAt" = '{start_date.strftime("%Y-%m-%d %H:%M:%S")}'::timestamp + (
    (nn.row_num::float / GREATEST(nn.total_count, 1)) * INTERVAL '{spread_days} days'
)
FROM numbered_notes nn
WHERE n.id = nn.id;

-- Verify the change
SELECT 
    DATE("createdAt") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE("createdAt")
ORDER BY post_date DESC
LIMIT 14;
"""
    return sql


def generate_timestamp_mapping_sql(
    dataset_path: Path,
    mapping_file: Optional[Path],
    base_date: datetime,
    minutes_per_timestamp: int = 30
) -> str:
    """Generate SQL with specific timestamps based on dataset."""
    
    # Load mapping if available
    mapping = {}
    if mapping_file and mapping_file.exists():
        mapping = load_post_mapping(mapping_file)
    
    # Load dataset
    posts = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            posts.append({
                'post_id': data['post_id'],
                'timestamp': int(data.get('timestamp', 0)),
                'user_id': data['user_id']
            })
    
    # Sort by timestamp
    posts.sort(key=lambda p: (p['timestamp'], p['post_id']))
    
    # Generate SQL
    sql_lines = [
        "-- =============================================================================",
        f"-- Timestamp mapping from dataset: {dataset_path.name}",
        f"-- Generated: {datetime.now().isoformat()}",
        f"-- Base date: {base_date.isoformat()}",
        f"-- Minutes per timestamp unit: {minutes_per_timestamp}",
        "-- =============================================================================",
        "",
        "BEGIN;",
        ""
    ]
    
    # Group posts by timestamp
    timestamp_groups: dict[int, list] = {}
    for post in posts:
        ts = post['timestamp']
        if ts not in timestamp_groups:
            timestamp_groups[ts] = []
        timestamp_groups[ts].append(post)
    
    # Generate updates
    for ts, group in sorted(timestamp_groups.items()):
        target_time = base_date + timedelta(minutes=ts * minutes_per_timestamp)
        
        sql_lines.append(f"-- Timestamp {ts}: {len(group)} posts at {target_time}")
        
        for i, post in enumerate(group):
            # Add small offset within same timestamp group
            post_time = target_time + timedelta(seconds=i * 30)
            
            if post['post_id'] in mapping:
                note_id = mapping[post['post_id']]
                sql_lines.append(
                    f"UPDATE note SET \"createdAt\" = '{post_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp "
                    f"WHERE id = '{note_id}';"
                )
            else:
                sql_lines.append(
                    f"-- SKIP: {post['post_id']} (no mapping found)"
                )
        
        sql_lines.append("")
    
    sql_lines.extend([
        "COMMIT;",
        "",
        "-- Verify changes",
        "SELECT ",
        "    DATE(\"createdAt\") as post_date,",
        "    COUNT(*) as post_count",
        "FROM note",
        "GROUP BY DATE(\"createdAt\")",
        "ORDER BY post_date DESC",
        "LIMIT 14;",
    ])
    
    return "\n".join(sql_lines)


def generate_backdate_sql(days: int, hours_lookback: int = 24) -> str:
    """Generate SQL to backdate all recent posts by N days."""
    return f"""
-- =============================================================================
-- Backdate posts by {days} days
-- Generated: {datetime.now().isoformat()}
-- =============================================================================

-- Show current state
SELECT 
    COUNT(*) as total_posts,
    MIN("createdAt") as earliest,
    MAX("createdAt") as latest
FROM note
WHERE "createdAt" > NOW() - INTERVAL '{hours_lookback} hours';

-- Backdate all recent posts
UPDATE note
SET "createdAt" = "createdAt" - INTERVAL '{days} days'
WHERE "createdAt" > NOW() - INTERVAL '{hours_lookback} hours';

-- Verify
SELECT 
    DATE("createdAt") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE("createdAt")
ORDER BY post_date DESC
LIMIT 14;
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SQL for updating Sharkey post timestamps"
    )
    
    parser.add_argument("--dataset", type=Path, help="Dataset JSONL file")
    parser.add_argument("--mapping", type=Path, help="Post ID to Note ID mapping JSON")
    parser.add_argument("--output", type=Path, default=Path("timestamp_updates.sql"),
                       help="Output SQL file")
    
    # Timestamp strategies
    parser.add_argument("--spread-days", type=int, 
                       help="Spread posts over N days")
    parser.add_argument("--backdate", type=int,
                       help="Backdate posts by N days")
    parser.add_argument("--use-dataset-timestamps", action="store_true",
                       help="Use timestamp field from dataset")
    
    # Options
    parser.add_argument("--base-date", type=str,
                       help="Base date for timestamps (YYYY-MM-DD)")
    parser.add_argument("--minutes-per-unit", type=int, default=30,
                       help="Minutes per timestamp unit")
    parser.add_argument("--hours-lookback", type=int, default=24,
                       help="Hours to look back for recent posts")
    
    args = parser.parse_args()
    
    # Parse base date
    if args.base_date:
        base_date = datetime.strptime(args.base_date, "%Y-%m-%d")
    else:
        base_date = datetime.now() - timedelta(days=args.spread_days or 7)
    
    # Generate SQL based on strategy
    if args.spread_days:
        sql = generate_spread_sql(
            args.spread_days,
            base_date,
            args.hours_lookback
        )
    elif args.backdate:
        sql = generate_backdate_sql(args.backdate, args.hours_lookback)
    elif args.use_dataset_timestamps and args.dataset:
        sql = generate_timestamp_mapping_sql(
            args.dataset,
            args.mapping,
            base_date,
            args.minutes_per_unit
        )
    else:
        parser.error("Specify --spread-days, --backdate, or --use-dataset-timestamps")
    
    # Write output
    with open(args.output, 'w') as f:
        f.write(sql)
    
    print(f"Generated SQL: {args.output}")
    print(f"\nTo apply on server:")
    print(f"  scp {args.output} jordan@botsocial.mlai.au:~/")
    print(f"  ssh -i ~/.ssh/digitalocean jordan@botsocial.mlai.au")
    print(f"  sudo -u postgres psql -d sharkey -f ~/{args.output.name}")


if __name__ == "__main__":
    main()

