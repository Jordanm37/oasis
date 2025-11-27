#!/usr/bin/env python3
"""
BotSocial Universal Dataset Uploader

A flexible, reusable uploader for OASIS datasets to BotSocial (Sharkey).

Features:
- Upload new users with posts/comments
- Upload new posts/comments for existing users
- Incremental uploads (skip already uploaded content)
- Configurable timing strategies
- Resume from failures

Usage Examples:
    # Full upload (new dataset)
    poetry run python3 scripts/botsocial/botsocial_uploader.py \
        --dataset data/runs/my_dataset.jsonl \
        --personas data/runs/my_personas.csv \
        --edges data/runs/my_edges.csv \
        --admin-token YOUR_ADMIN_TOKEN

    # Add posts for existing users only
    poetry run python3 scripts/botsocial/botsocial_uploader.py \
        --dataset data/runs/new_posts.jsonl \
        --credentials data/runs/existing_tokens.json \
        --skip-accounts --skip-profiles

    # Custom timing (simulate real-time posting)
    poetry run python3 scripts/botsocial/botsocial_uploader.py \
        --dataset data/runs/my_dataset.jsonl \
        --timing-mode realtime \
        --time-scale 60  # 1 minute = 1 hour simulated

Author: eSafety Hackathon Team
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

API_BASE = "https://botsocial.mlai.au/api"
DEFAULT_PASSWORD = "SecureSimUser2025!"

# Neutral username components
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

DECORATORS = ["_", "x", "the", "real", "its", "im", "just", "hey", "yo", "mr", "ms"]


class TimingMode(Enum):
    """Post timing strategies."""
    FAST = "fast"           # Minimal delay (0.6s) - fastest upload
    NATURAL = "natural"     # Random delays (1-5s) - looks more natural
    REALTIME = "realtime"   # Use timestamp field to simulate real timing
    BURST = "burst"         # Batch posts then pause


@dataclass
class TimingConfig:
    """Configuration for post timing."""
    mode: TimingMode = TimingMode.FAST
    min_delay: float = 0.6      # Minimum delay between posts
    max_delay: float = 5.0      # Maximum delay (for natural mode)
    time_scale: float = 1.0     # For realtime: 1 = real time, 60 = 1 min = 1 hour
    burst_size: int = 20        # Posts per burst
    burst_pause: float = 30.0   # Pause between bursts


# =============================================================================
# DATA CLASSES
# =============================================================================

def _get_name_components(user_id: str, seed: str) -> tuple[str, str, int]:
    """Get consistent first name, last name, and number for a user."""
    hash_input = f"{user_id}-{seed}-identity"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    first = FIRST_NAMES[hash_val % len(FIRST_NAMES)]
    last = LAST_PARTS[(hash_val >> 8) % len(LAST_PARTS)]
    num = (hash_val >> 16) % 100
    
    return first, last, num


def generate_neutral_username(user_id: str, seed: str, suffix: str = "v1") -> str:
    """Generate a neutral username that doesn't reveal persona type."""
    first, last, num = _get_name_components(user_id, seed)
    
    # Use a secondary hash just for username pattern selection
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


def generate_neutral_display_name(user_id: str, seed: str) -> str:
    """Generate a neutral display name matching the username."""
    first, last, num = _get_name_components(user_id, seed)
    
    # Use a secondary hash just for display pattern selection
    pattern_hash = int(hashlib.md5(f"{user_id}-display-pattern".encode()).hexdigest(), 16)
    pattern = pattern_hash % 4
    
    if pattern == 0:
        return f"{first.title()} {last.title()[0]}."
    elif pattern == 1:
        return first.title()
    elif pattern == 2:
        return f"{first.title()} {last.title()}"
    else:
        return f"{first.lower()}"


@dataclass
class User:
    """Represents a user/persona."""
    user_id: str
    username: str  # Original from dataset
    description: str = ""
    primary_label: str = ""
    neutral_username: str = ""
    display_name: str = ""
    sharkey_id: str = ""
    token: str = ""
    
    def __post_init__(self) -> None:
        if not self.neutral_username:
            self.neutral_username = generate_neutral_username(
                self.user_id, self.username
            )
        if not self.display_name:
            self.display_name = generate_neutral_display_name(
                self.user_id, self.username
            )


@dataclass
class Post:
    """Represents a post from the dataset."""
    post_id: str
    thread_id: str
    user_id: str
    parent_id: Optional[str]
    timestamp: str
    text: str
    persona_id: str = ""
    category_labels: list[str] = field(default_factory=list)
    sharkey_note_id: str = ""
    uploaded: bool = False


# =============================================================================
# API CLIENT
# =============================================================================

class BotSocialClient:
    """Rate-limited API client for BotSocial."""
    
    def __init__(self, min_interval: float = 0.6) -> None:
        self.min_interval = min_interval
        self.last_request = 0.0
    
    def _wait(self) -> None:
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
    
    def post(
        self,
        endpoint: str,
        data: dict[str, Any],
        skip_wait: bool = False
    ) -> requests.Response:
        """Make a POST request with rate limiting."""
        if not skip_wait:
            self._wait()
        
        return requests.post(
            f"{API_BASE}/{endpoint}",
            json=data,
            timeout=30
        )


# =============================================================================
# UPLOADER CLASS
# =============================================================================

class BotSocialUploader:
    """Main uploader class."""
    
    def __init__(
        self,
        admin_token: str,
        credentials_path: Path,
        timing_config: TimingConfig,
        username_suffix: str = "v1"
    ) -> None:
        self.admin_token = admin_token
        self.credentials_path = credentials_path
        self.timing = timing_config
        self.username_suffix = username_suffix
        
        self.client = BotSocialClient(min_interval=timing_config.min_delay)
        self.users: dict[str, User] = {}
        self.posts: list[Post] = []
        self.edges: list[tuple[str, str]] = []
        self.post_id_map: dict[str, str] = {}  # dataset_id -> sharkey_id
        
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load existing credentials."""
        if self.credentials_path.exists():
            try:
                with open(self.credentials_path, 'r') as f:
                    creds = json.load(f)
                logger.info(f"Loaded {len(creds)} existing credentials")
                
                for user_id, data in creds.items():
                    if user_id not in self.users:
                        self.users[user_id] = User(
                            user_id=user_id,
                            username=data.get('username', ''),
                            neutral_username=data.get('neutral_username', ''),
                            display_name=data.get('display_name', ''),
                            sharkey_id=data.get('sharkey_id', ''),
                            token=data.get('token', '')
                        )
                    else:
                        self.users[user_id].token = data.get('token', '')
                        self.users[user_id].sharkey_id = data.get('sharkey_id', '')
            except json.JSONDecodeError:
                logger.warning("Could not parse credentials file")
    
    def _save_credentials(self) -> None:
        """Save credentials to file."""
        creds = {}
        for user_id, user in self.users.items():
            if user.token:
                creds[user_id] = {
                    'username': user.username,
                    'neutral_username': user.neutral_username,
                    'display_name': user.display_name,
                    'sharkey_id': user.sharkey_id,
                    'token': user.token
                }
        
        with open(self.credentials_path, 'w') as f:
            json.dump(creds, f, indent=2)
        logger.info(f"Saved {len(creds)} credentials")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_personas(self, path: Path) -> None:
        """Load personas from CSV."""
        logger.info(f"Loading personas from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                user_id = str(idx)
                if user_id not in self.users:
                    self.users[user_id] = User(
                        user_id=user_id,
                        username=row['username'],
                        description=row.get('description', '')[:200],
                        primary_label=row.get('primary_label', '')
                    )
                else:
                    # Update existing user with persona info
                    self.users[user_id].description = row.get('description', '')[:200]
                    self.users[user_id].primary_label = row.get('primary_label', '')
        
        logger.info(f"Loaded {len(self.users)} personas")
    
    def load_posts(self, path: Path) -> None:
        """Load posts from JSONL."""
        logger.info(f"Loading posts from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Create user if not exists (for datasets without personas file)
                user_id = str(data['user_id'])
                if user_id not in self.users:
                    self.users[user_id] = User(
                        user_id=user_id,
                        username=f"user_{user_id}",
                        persona_id=data.get('persona_id', '')
                    )
                
                post = Post(
                    post_id=data['post_id'],
                    thread_id=data['thread_id'],
                    user_id=user_id,
                    parent_id=data.get('parent_id'),
                    timestamp=str(data.get('timestamp', '0')),
                    text=data['text'],
                    persona_id=data.get('persona_id', ''),
                    category_labels=data.get('category_labels', [])
                )
                self.posts.append(post)
        
        # Sort by timestamp, then root posts before replies
        self.posts.sort(key=lambda p: (int(p.timestamp), p.parent_id is not None))
        
        root = sum(1 for p in self.posts if p.parent_id is None)
        logger.info(f"Loaded {len(self.posts)} posts ({root} root, {len(self.posts)-root} replies)")
    
    def load_edges(self, path: Path) -> None:
        """Load follow relationships."""
        logger.info(f"Loading edges from {path}")
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.edges.append((str(row['follower_id']), str(row['followee_id'])))
        
        logger.info(f"Loaded {len(self.edges)} follow relationships")
    
    # =========================================================================
    # ACCOUNT MANAGEMENT
    # =========================================================================
    
    def create_account(self, user: User) -> bool:
        """Create a user account."""
        if user.token:
            logger.debug(f"User {user.user_id} already has token, skipping")
            return True
        
        try:
            username = f"{user.neutral_username[:17]}_{self.username_suffix}"
            
            response = self.client.post(
                "admin/accounts/create",
                {
                    "i": self.admin_token,
                    "username": username,
                    "password": DEFAULT_PASSWORD
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                user.sharkey_id = result.get('id', '')
                user.token = result.get('token', '')
                
                if user.token:
                    logger.info(f"✓ Created @{username}")
                    return True
                else:
                    logger.warning(f"Created @{username} but no token")
                    return False
            
            elif response.status_code == 500:
                # Check if USED_USERNAME error
                error_info = response.json().get('error', {}).get('info', {})
                if 'USED_USERNAME' in str(error_info) or 'DUPLICATED' in str(error_info):
                    logger.warning(f"Username {username} already used - try different suffix")
                    return False
            
            logger.warning(f"Account creation failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error creating account: {e}")
            return False
    
    def create_all_accounts(self) -> int:
        """Create accounts for all users without tokens."""
        logger.info("Creating accounts...")
        created = 0
        
        for user_id, user in self.users.items():
            if not user.token:
                if self.create_account(user):
                    created += 1
                    if created % 10 == 0:
                        logger.info(f"Progress: {created} accounts created")
        
        self._save_credentials()
        logger.info(f"Created {created} new accounts")
        return created
    
    # =========================================================================
    # PROFILE SETUP
    # =========================================================================
    
    def upload_image_from_url(self, token: str, url: str) -> Optional[str]:
        """Upload image from URL (async endpoint)."""
        try:
            # Get current most recent file
            before_resp = requests.post(
                f"{API_BASE}/drive/files",
                json={"i": token, "limit": 1},
                timeout=30
            )
            before_id = None
            if before_resp.status_code == 200:
                files = before_resp.json()
                if files:
                    before_id = files[0].get('id')
            
            # Trigger upload
            requests.post(
                f"{API_BASE}/drive/files/upload-from-url",
                json={"i": token, "url": url},
                timeout=30
            )
            
            # Wait and check for new file
            for _ in range(6):
                time.sleep(2)
                after_resp = requests.post(
                    f"{API_BASE}/drive/files",
                    json={"i": token, "limit": 1},
                    timeout=30
                )
                if after_resp.status_code == 200:
                    files = after_resp.json()
                    if files:
                        after_id = files[0].get('id')
                        if after_id and after_id != before_id:
                            return after_id
            
            return None
        except Exception:
            return None
    
    def setup_profile(self, user: User) -> bool:
        """Set up user profile with avatar and banner."""
        if not user.token:
            return False
        
        try:
            # Update name and description
            self.client.post("i/update", {
                "i": user.token,
                "name": user.display_name,
                "description": user.description
            })
            
            # Upload avatar
            avatar_seed = f"avatar-{user.neutral_username}-{user.user_id}"
            avatar_url = f"https://picsum.photos/seed/{avatar_seed}/400/400"
            avatar_id = self.upload_image_from_url(user.token, avatar_url)
            
            if avatar_id:
                self.client.post("i/update", {"i": user.token, "avatarId": avatar_id})
            
            # Upload banner
            time.sleep(2)
            banner_seed = f"banner-{user.neutral_username}-{user.user_id}"
            banner_url = f"https://picsum.photos/seed/{banner_seed}/1500/500"
            banner_id = self.upload_image_from_url(user.token, banner_url)
            
            if banner_id:
                self.client.post("i/update", {"i": user.token, "bannerId": banner_id})
            
            return True
        except Exception as e:
            logger.error(f"Profile setup error: {e}")
            return False
    
    def setup_all_profiles(self) -> int:
        """Set up profiles for all users."""
        logger.info("Setting up profiles...")
        setup = 0
        
        for user_id, user in self.users.items():
            if user.token and self.setup_profile(user):
                setup += 1
                if setup % 10 == 0:
                    logger.info(f"Progress: {setup} profiles setup")
        
        logger.info(f"Setup {setup} profiles")
        return setup
    
    # =========================================================================
    # POST UPLOAD
    # =========================================================================
    
    def _get_delay(self, prev_timestamp: str, curr_timestamp: str) -> float:
        """Calculate delay based on timing mode."""
        if self.timing.mode == TimingMode.FAST:
            return self.timing.min_delay
        
        elif self.timing.mode == TimingMode.NATURAL:
            return random.uniform(self.timing.min_delay, self.timing.max_delay)
        
        elif self.timing.mode == TimingMode.REALTIME:
            # Use timestamp difference scaled by time_scale
            try:
                diff = int(curr_timestamp) - int(prev_timestamp)
                delay = diff / self.timing.time_scale
                return max(self.timing.min_delay, min(delay, 300))  # Cap at 5 min
            except ValueError:
                return self.timing.min_delay
        
        else:  # BURST mode handled separately
            return self.timing.min_delay
    
    def create_note(self, post: Post) -> Optional[str]:
        """Create a note/post."""
        user = self.users.get(post.user_id)
        if not user or not user.token:
            logger.debug(f"No token for user {post.user_id}")
            return None
        
        try:
            data = {
                "i": user.token,
                "text": post.text,
                "visibility": "public"
            }
            
            # Add reply reference if this is a reply
            if post.parent_id and post.parent_id in self.post_id_map:
                data["replyId"] = self.post_id_map[post.parent_id]
            
            response = self.client.post("notes/create", data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('createdNote', {}).get('id')
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                return self.create_note(post)
            
            return None
        except Exception as e:
            logger.error(f"Note creation error: {e}")
            return None
    
    def upload_all_posts(self) -> int:
        """Upload all posts."""
        logger.info(f"Uploading {len(self.posts)} posts...")
        uploaded = 0
        prev_timestamp = "0"
        burst_count = 0
        
        for i, post in enumerate(self.posts):
            # Skip already uploaded
            if post.uploaded or post.post_id in self.post_id_map:
                continue
            
            # Handle timing
            if self.timing.mode == TimingMode.BURST:
                burst_count += 1
                if burst_count >= self.timing.burst_size:
                    logger.info(f"Burst pause ({self.timing.burst_pause}s)...")
                    time.sleep(self.timing.burst_pause)
                    burst_count = 0
            else:
                delay = self._get_delay(prev_timestamp, post.timestamp)
                if delay > self.timing.min_delay:
                    time.sleep(delay - self.timing.min_delay)
            
            # Create note
            note_id = self.create_note(post)
            
            if note_id:
                self.post_id_map[post.post_id] = note_id
                post.sharkey_note_id = note_id
                post.uploaded = True
                uploaded += 1
                
                if uploaded % 20 == 0:
                    logger.info(f"Progress: {uploaded}/{len(self.posts)} posts")
            
            prev_timestamp = post.timestamp
        
        logger.info(f"Uploaded {uploaded} posts")
        return uploaded
    
    # =========================================================================
    # SOCIAL INTERACTIONS
    # =========================================================================
    
    def create_follow(self, follower: User, followee: User) -> bool:
        """Create a follow relationship."""
        if not follower.token or not followee.sharkey_id:
            return False
        
        try:
            response = self.client.post("following/create", {
                "i": follower.token,
                "userId": followee.sharkey_id
            })
            return response.status_code in [200, 204]
        except Exception:
            return False
    
    def create_all_follows(self) -> int:
        """Create all follow relationships."""
        logger.info(f"Creating {len(self.edges)} follows...")
        created = 0
        
        for follower_id, followee_id in self.edges:
            follower = self.users.get(follower_id)
            followee = self.users.get(followee_id)
            
            if follower and followee and self.create_follow(follower, followee):
                created += 1
                if created % 50 == 0:
                    logger.info(f"Progress: {created}/{len(self.edges)} follows")
        
        logger.info(f"Created {created} follows")
        return created
    
    def add_reactions(self, count: int = 200) -> int:
        """Add random reactions to posts."""
        logger.info(f"Adding ~{count} reactions...")
        
        posted = [p for p in self.posts if p.sharkey_note_id]
        if not posted:
            return 0
        
        reactions = ['❤️', '👍', '🎉', '😊', '🔥', '💯']
        tokens = [u.token for u in self.users.values() if u.token]
        created = 0
        
        for _ in range(count):
            post = random.choice(posted)
            token = random.choice(tokens)
            reaction = random.choice(reactions)
            
            try:
                response = self.client.post("notes/reactions/create", {
                    "i": token,
                    "noteId": post.sharkey_note_id,
                    "reaction": reaction
                })
                if response.status_code in [200, 204]:
                    created += 1
            except Exception:
                pass
        
        logger.info(f"Added {created} reactions")
        return created
    
    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================
    
    def run(
        self,
        dataset_path: Optional[Path] = None,
        personas_path: Optional[Path] = None,
        edges_path: Optional[Path] = None,
        skip_accounts: bool = False,
        skip_profiles: bool = False,
        skip_posts: bool = False,
        skip_follows: bool = False,
        skip_reactions: bool = False,
        reaction_count: int = 200
    ) -> dict[str, int]:
        """Run the upload workflow."""
        stats = {
            'accounts': 0,
            'profiles': 0,
            'posts': 0,
            'follows': 0,
            'reactions': 0
        }
        
        # Load data
        if personas_path and personas_path.exists():
            self.load_personas(personas_path)
        
        if dataset_path and dataset_path.exists():
            self.load_posts(dataset_path)
        
        if edges_path and edges_path.exists():
            self.load_edges(edges_path)
        
        # Phase 1: Accounts
        if not skip_accounts:
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: Account Creation")
            logger.info("="*60)
            stats['accounts'] = self.create_all_accounts()
        
        # Phase 2: Profiles
        if not skip_profiles:
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: Profile Setup")
            logger.info("="*60)
            stats['profiles'] = self.setup_all_profiles()
        
        # Phase 3: Posts
        if not skip_posts:
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: Post Upload")
            logger.info("="*60)
            stats['posts'] = self.upload_all_posts()
        
        # Phase 4: Follows
        if not skip_follows and self.edges:
            logger.info("\n" + "="*60)
            logger.info("PHASE 4: Follow Relationships")
            logger.info("="*60)
            stats['follows'] = self.create_all_follows()
        
        # Phase 5: Reactions
        if not skip_reactions:
            logger.info("\n" + "="*60)
            logger.info("PHASE 5: Reactions")
            logger.info("="*60)
            stats['reactions'] = self.add_reactions(reaction_count)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("UPLOAD COMPLETE")
        logger.info("="*60)
        for key, value in stats.items():
            logger.info(f"  {key.title()}: {value}")
        logger.info(f"\nView at: https://botsocial.mlai.au")
        
        return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload OASIS dataset to BotSocial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full upload
  %(prog)s --dataset data.jsonl --personas personas.csv --admin-token TOKEN

  # Add posts to existing users
  %(prog)s --dataset new_posts.jsonl --credentials tokens.json --skip-accounts

  # Simulate real-time posting
  %(prog)s --dataset data.jsonl --timing-mode realtime --time-scale 60
        """
    )
    
    # Data files
    parser.add_argument("--dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("--personas", type=Path, help="Personas CSV file")
    parser.add_argument("--edges", type=Path, help="Edges CSV file")
    
    # Authentication
    parser.add_argument("--admin-token", type=str, help="Admin API token")
    parser.add_argument("--credentials", type=Path, 
                       default=Path("botsocial_tokens.json"),
                       help="Credentials file path")
    
    # Skip phases
    parser.add_argument("--skip-accounts", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-posts", action="store_true")
    parser.add_argument("--skip-follows", action="store_true")
    parser.add_argument("--skip-reactions", action="store_true")
    
    # Timing
    parser.add_argument("--timing-mode", type=str, default="fast",
                       choices=["fast", "natural", "realtime", "burst"],
                       help="Post timing strategy")
    parser.add_argument("--time-scale", type=float, default=1.0,
                       help="Time scale for realtime mode")
    parser.add_argument("--min-delay", type=float, default=0.6,
                       help="Minimum delay between posts")
    
    # Other
    parser.add_argument("--username-suffix", type=str, default="v1",
                       help="Suffix for usernames (to avoid collisions)")
    parser.add_argument("--reaction-count", type=int, default=200,
                       help="Number of reactions to add")
    
    args = parser.parse_args()
    
    # Validate
    if not args.skip_accounts and not args.admin_token:
        parser.error("--admin-token required unless --skip-accounts")
    
    # Create timing config
    timing = TimingConfig(
        mode=TimingMode(args.timing_mode),
        min_delay=args.min_delay,
        time_scale=args.time_scale
    )
    
    # Create uploader and run
    uploader = BotSocialUploader(
        admin_token=args.admin_token or "",
        credentials_path=args.credentials,
        timing_config=timing,
        username_suffix=args.username_suffix
    )
    
    uploader.run(
        dataset_path=args.dataset,
        personas_path=args.personas,
        edges_path=args.edges,
        skip_accounts=args.skip_accounts,
        skip_profiles=args.skip_profiles,
        skip_posts=args.skip_posts,
        skip_follows=args.skip_follows,
        skip_reactions=args.skip_reactions,
        reaction_count=args.reaction_count
    )


if __name__ == "__main__":
    main()

