#!/usr/bin/env python3
"""
Fix Display Names Script

Updates display names for existing BotSocial users to use neutral names
that match their username pattern.

Usage:
    poetry run python3 scripts/botsocial/fix_display_names.py \
        --credentials data/runs/botsocial_tokens_v3.json
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE = "https://botsocial.mlai.au/api"

# Neutral name components
FIRST_NAMES = [
    "alex", "jordan", "taylor", "casey", "morgan", "riley", "quinn", "avery",
    "blake", "cameron", "drew", "emery", "finley", "harper", "jamie", "kai",
    "lee", "max", "nico", "parker", "reese", "sam", "skyler", "toni", "val",
    "wren", "zion", "ash", "bay", "charlie", "dakota", "eden", "frankie",
    "gray", "hayden", "indigo", "jess", "kerry", "logan", "marley", "noel"
]

LAST_PARTS = [
    "smith", "jones", "lee", "chen", "patel", "kim", "wong", "silva", "cruz",
    "ali", "khan", "nguyen", "garcia", "martinez", "lopez", "wilson", "moore",
    "taylor", "brown", "davis", "miller", "anderson", "thomas", "jackson",
    "white", "harris", "martin", "clark", "lewis", "walker", "hall", "young"
]


def _get_name_components(user_id: str, seed: str) -> tuple:
    """Get consistent first name, last name, and number for a user."""
    hash_input = f"{user_id}-{seed}-identity"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    first = FIRST_NAMES[hash_val % len(FIRST_NAMES)]
    last = LAST_PARTS[(hash_val >> 8) % len(LAST_PARTS)]
    num = (hash_val >> 16) % 100
    
    return first, last, num


def generate_neutral_display_name(user_id: str, seed: str) -> str:
    """Generate a neutral display name matching the username."""
    first, last, num = _get_name_components(user_id, seed)
    
    # Use secondary hash for pattern selection only
    pattern_hash = int(hashlib.md5(f"{user_id}-display-pattern".encode()).hexdigest(), 16)
    pattern = pattern_hash % 4
    
    if pattern == 0:
        return f"{first.title()} {last.title()[0]}."
    elif pattern == 1:
        return first.title()
    elif pattern == 2:
        return f"{first.title()} {last.title()}"
    else:
        return f"{first.title()}{num}"


def generate_neutral_username(user_id: str, seed: str, suffix: str = "v3") -> str:
    """Generate a neutral username that doesn't reveal persona type."""
    first, last, num = _get_name_components(user_id, seed)
    
    # Use secondary hash for pattern selection only
    pattern_hash = int(hashlib.md5(f"{user_id}-pattern".encode()).hexdigest(), 16)
    pattern = pattern_hash % 5
    
    if pattern == 0:
        base = f"{first}_{last}{num}"
    elif pattern == 1:
        base = f"{first}{num}"
    elif pattern == 2:
        base = f"{first}{last}"
    elif pattern == 3:
        base = f"{first}_{num}"
    else:
        base = f"{first}{num}{last[:3]}"
    
    return f"{base[:17]}_{suffix}"


def update_profile(token: str, name: str) -> bool:
    """Update a user's display name."""
    try:
        response = requests.post(
            f"{API_BASE}/i/update",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": name},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix display names to be neutral")
    parser.add_argument("--credentials", required=True, help="Path to credentials JSON")
    parser.add_argument("--seed", default="prod_100p_100s_65_35", help="Seed for name generation")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    args = parser.parse_args()
    
    # Load credentials
    creds_path = Path(args.credentials)
    if not creds_path.exists():
        logger.error(f"Credentials file not found: {creds_path}")
        return
    
    with open(creds_path) as f:
        credentials = json.load(f)
    
    logger.info(f"Loaded {len(credentials)} users from {creds_path}")
    
    # Preview changes
    print("\n" + "=" * 70)
    print("DISPLAY NAME UPDATES")
    print("=" * 70)
    print(f"{'User ID':<8} {'Old Name':<20} {'New Name':<20} {'New Username':<25}")
    print("-" * 70)
    
    updates = []
    for user_id, user_data in sorted(credentials.items(), key=lambda x: int(x[0])):
        old_name = user_data.get("display_name", "")
        new_name = generate_neutral_display_name(user_id, args.seed)
        new_username = generate_neutral_username(user_id, args.seed)
        
        # Check if name needs updating
        if old_name != new_name:
            updates.append({
                "user_id": user_id,
                "token": user_data["token"],
                "old_name": old_name,
                "new_name": new_name,
                "new_username": new_username,
                "old_username": user_data.get("username", "")
            })
            print(f"{user_id:<8} {old_name:<20} {new_name:<20} {new_username:<25}")
    
    print("-" * 70)
    print(f"Total updates needed: {len(updates)}")
    print()
    
    if args.dry_run:
        logger.info("DRY RUN - no changes applied")
        return
    
    if not updates:
        logger.info("No updates needed!")
        return
    
    # Confirm
    confirm = input(f"Apply {len(updates)} display name updates? [y/N]: ")
    if confirm.lower() != 'y':
        logger.info("Aborted")
        return
    
    # Apply updates
    success = 0
    failed = 0
    
    for i, update in enumerate(updates):
        logger.info(f"[{i+1}/{len(updates)}] Updating user {update['user_id']}: "
                   f"'{update['old_name']}' -> '{update['new_name']}'")
        
        if update_profile(update["token"], update["new_name"]):
            success += 1
            # Update credentials file
            credentials[update["user_id"]]["display_name"] = update["new_name"]
        else:
            failed += 1
            logger.error(f"Failed to update user {update['user_id']}")
        
        time.sleep(args.delay)
    
    # Save updated credentials
    with open(creds_path, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    logger.info(f"Complete! Success: {success}, Failed: {failed}")
    logger.info(f"Updated credentials saved to {creds_path}")


if __name__ == "__main__":
    main()

