#!/usr/bin/env python3
"""
Clean bracket patterns from dataset using rule-based replacements and LLM imputation.

Categories:
1. Style metadata: Remove entirely
2. GIF references: Replace with emoji
3. Link placeholders: LLM impute fake URLs
4. Embedded commentary: Unwrap (remove brackets, keep text)
5. Slur placeholders: LLM impute contextual slurs
6. Other (sarcasm, rageface, etc.): Replace with emoji or remove
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Emoji mappings for common patterns
EMOJI_MAP = {
    # Reactions
    "sarcasm": "😏",
    "rageface": "😤",
    "smh": "🤦",
    "sigh": "😔",
    "laugh": "😂",
    "lol": "😂",
    "eye roll": "🙄",
    "mocking": "😏",
    "whisper": "🤫",
    "sob": "😢",
    "cry": "😢",
    # Actions
    "flexing": "💪",
    "middle finger": "🖕",
    "nope": "🙅",
    "clap": "👏",
    # Misc
    "redpill": "💊",
    "blackpill": "💊",
    "incelrage": "😤",
    "bloodredemoji": "🩸",
    "broken_heart": "💔",
}

# Patterns to remove entirely
REMOVE_PATTERNS = [
    r'\[Style[:\s][^\]]*\]',  # [Style: ...] or [Style:...]
    r'\[CTA[:\s][^\]]*\]',    # [CTA: Share this thread]
    r'\[Author[^\]]*\]',      # [Author], [Author's Name]
    r'\[Post ID[^\]]*\]',     # [Post ID: 3336]
    r'\[Time[:\s][^\]]*\]',   # [Time: 3:33 AM]
    r'\[Framework[:\s][^\]]*\]',  # [Framework: ...]
    r'\[Case Study[^\]]*\]',  # [Case Study: ...]
    r'\[Dashboard[^\]]*\]',   # [Dashboard: ...]
    r'\[chunk \d+/\d+[^\]]*\]',  # [chunk 10/15 of a long message]
    r'\[REDACTED\]',
    r'\[redacted\]',
]

# GIF patterns -> emoji
GIF_EMOJI_MAP = {
    "dismissive": "🙄",
    "eye-roll": "🙄",
    "eye roll": "🙄",
    "middle finger": "🖕",
    "middle_finger": "🖕",
    "laughing": "😂",
    "flexing": "💪",
    "muscles": "💪",
    "nope": "🙅",
    "aggressive": "😤",
    "triumphant": "🎉",
    "dance": "💃",
}


def pattern_to_emoji(match_text: str) -> str:
    """Convert a bracket pattern to emoji based on content."""
    lower = match_text.lower()
    
    # Check emoji map
    for key, emoji in EMOJI_MAP.items():
        if key in lower:
            return emoji
    
    # Check GIF patterns
    for key, emoji in GIF_EMOJI_MAP.items():
        if key in lower:
            return emoji
    
    return ""  # Remove if no match


def is_embedded_commentary(match_text: str) -> bool:
    """Check if this is embedded commentary (long narrative text)."""
    # Long text (>40 chars) that looks like narrative
    if len(match_text) < 40:
        return False
    
    # Should contain spaces (it's a sentence)
    if " " not in match_text:
        return False
    
    # Not a structured tag
    structured_prefixes = ["style", "gif", "link", "source", "cta", "framework", "time", "author"]
    lower = match_text.lower()
    for prefix in structured_prefixes:
        if lower.startswith(prefix):
            return False
    
    return True


def is_slur_placeholder(match_text: str) -> bool:
    """Check if this is a slur placeholder that needs LLM imputation."""
    lower = match_text.lower()
    slur_indicators = [
        "slur", "racial", "religious", "hate_slur", "dehumaniz",
        "insert slur", "animal comparison", "target group"
    ]
    return any(ind in lower for ind in slur_indicators)


def is_link_placeholder(match_text: str) -> bool:
    """Check if this is a link placeholder that needs LLM imputation."""
    lower = match_text.lower()
    link_indicators = [
        "link", "http", "source:", "see:", "pdf", "archive",
        "leaked", "memo", "document", "url"
    ]
    return any(ind in lower for ind in link_indicators)


def is_style_or_gif(match_text: str) -> bool:
    """Check if this is a style tag or GIF reference."""
    lower = match_text.lower()
    return lower.startswith("style") or lower.startswith("gif")


def rule_based_clean(text: str) -> Tuple[str, List[str]]:
    """
    Apply rule-based cleaning. Returns cleaned text and list of patterns needing LLM.
    
    Returns:
        (cleaned_text, llm_patterns) where llm_patterns are slurs/links needing imputation
    """
    llm_needed = []
    
    # First pass: remove known patterns
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Second pass: process remaining bracket patterns
    bracket_pattern = re.compile(r'\[([^\]]+)\]')
    
    def replace_bracket(m: re.Match) -> str:
        match_text = m.group(1)
        full_match = m.group(0)
        
        # Style tags - remove
        if is_style_or_gif(match_text):
            emoji = pattern_to_emoji(match_text)
            return emoji
        
        # Slur placeholders - mark for LLM
        if is_slur_placeholder(match_text):
            llm_needed.append(("slur", full_match))
            return full_match  # Keep for now, LLM will replace
        
        # Link placeholders - mark for LLM
        if is_link_placeholder(match_text):
            llm_needed.append(("link", full_match))
            return full_match  # Keep for now, LLM will replace
        
        # Embedded commentary - unwrap
        if is_embedded_commentary(match_text):
            return match_text  # Remove brackets, keep text
        
        # Try emoji conversion
        emoji = pattern_to_emoji(match_text)
        if emoji:
            return emoji
        
        # Short patterns - likely should be removed or are artifacts
        if len(match_text) < 20:
            return ""
        
        # Default: unwrap
        return match_text
    
    text = bracket_pattern.sub(replace_bracket, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, llm_needed


async def llm_impute_slur(text: str, pattern: str, labels: List[str]) -> str:
    """Use Grok to impute a contextual slur based on the text and labels."""
    import aiohttp
    
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        # Fallback: just remove the pattern
        return text.replace(pattern, "")
    
    prompt = f"""You are helping clean a hate speech detection dataset. 
The text contains a placeholder {pattern} that needs to be replaced with a realistic slur or dehumanizing term that fits the context.

Text: {text}
Labels: {labels}

Provide ONLY the replacement word/phrase (no explanation). The replacement should be:
- Contextually appropriate for the hate category
- A realistic slur or dehumanizing term
- Just the word/phrase, nothing else"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "grok-3-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.7,
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    replacement = data["choices"][0]["message"]["content"].strip()
                    return text.replace(pattern, replacement)
    except Exception as e:
        print(f"LLM error for slur: {e}")
    
    return text.replace(pattern, "")


async def llm_impute_link(text: str, pattern: str, labels: List[str]) -> str:
    """Use Grok to impute a fake but realistic-looking link."""
    import aiohttp
    
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return text.replace(pattern, "")
    
    prompt = f"""You are helping clean a misinformation detection dataset.
The text contains a placeholder {pattern} that needs to be replaced with a fake but realistic-looking URL.

Text: {text}
Labels: {labels}

Provide ONLY the replacement URL (no explanation). The URL should be:
- Fake but realistic looking (e.g., archive.is/xyz123, leaked-docs.org/file.pdf)
- Contextually appropriate for the content
- Just the URL, nothing else"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "grok-3-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.7,
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    replacement = data["choices"][0]["message"]["content"].strip()
                    return text.replace(pattern, replacement)
    except Exception as e:
        print(f"LLM error for link: {e}")
    
    return text.replace(pattern, "")


async def process_record(rec: dict, stats: Counter) -> dict:
    """Process a single record, cleaning bracket patterns."""
    text = rec.get("text", "")
    labels = rec.get("category_labels", [])
    
    # Rule-based cleaning first
    cleaned_text, llm_patterns = rule_based_clean(text)
    
    # LLM imputation for slurs and links
    for pattern_type, pattern in llm_patterns:
        if pattern_type == "slur":
            cleaned_text = await llm_impute_slur(cleaned_text, pattern, labels)
            stats["slurs_imputed"] += 1
        elif pattern_type == "link":
            cleaned_text = await llm_impute_link(cleaned_text, pattern, labels)
            stats["links_imputed"] += 1
    
    rec["text"] = cleaned_text
    return rec


async def process_batch(records: List[dict], stats: Counter, batch_size: int = 50, concurrency: int = 20) -> List[dict]:
    """Process records in batches with high concurrency."""
    import asyncio
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(rec):
        async with semaphore:
            return await process_record(rec, stats)
    
    # Process all records concurrently with semaphore limiting
    print(f"  Processing {len(records):,} records with concurrency={concurrency}...")
    tasks = [process_with_semaphore(rec) for rec in records]
    results = await asyncio.gather(*tasks)
    
    print(f"  Done processing {len(results):,} records")
    return list(results)


def main():
    parser = argparse.ArgumentParser(description="Clean bracket patterns from dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Just analyze, don't write")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for LLM calls")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent LLM calls")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    records = []
    with open(args.input, "r") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")
    
    if args.dry_run:
        # Just analyze patterns
        print("\n=== DRY RUN: Analyzing patterns ===")
        stats = Counter()
        sample_changes = []
        
        for rec in records[:1000]:
            text = rec.get("text", "")
            cleaned, llm_patterns = rule_based_clean(text)
            
            if text != cleaned or llm_patterns:
                stats["records_with_changes"] += 1
                for ptype, _ in llm_patterns:
                    stats[f"{ptype}_patterns"] += 1
                
                if len(sample_changes) < 5:
                    sample_changes.append({
                        "before": text[:200],
                        "after": cleaned[:200],
                        "llm_needed": llm_patterns
                    })
        
        print(f"\nStats (first 1000 records):")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        print(f"\nSample changes:")
        for i, s in enumerate(sample_changes):
            print(f"\n--- Sample {i+1} ---")
            print(f"BEFORE: {s['before']}...")
            print(f"AFTER:  {s['after']}...")
            print(f"LLM needed: {s['llm_needed']}")
        
        return
    
    # Full processing
    print("\n=== Processing records ===")
    stats = Counter()
    
    processed = asyncio.run(process_batch(records, stats, args.batch_size, args.concurrency))
    
    print(f"\n=== Writing to {args.output} ===")
    with open(args.output, "w") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\nDone! Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

