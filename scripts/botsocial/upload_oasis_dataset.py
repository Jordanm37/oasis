#!/usr/bin/env python3
"""
OASIS to BotSocial Dataset Uploader

Uploads OASIS simulation datasets to BotSocial (Sharkey) platform.
Handles: accounts, profiles, posts, replies, follows, and reactions.

Usage:
    poetry run python3 scripts/botsocial/upload_oasis_dataset.py \
        --dataset data/runs/prod_100p_100s_65_35_dataset.jsonl \
        --personas data/runs/personas_prod_100p_100s_65_35.csv \
        --edges data/runs/edges_prod_100p_100s_65_35.csv \
        --config config/botsocial_credentials.yaml

Author: eSafety Hackathon Team
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_BASE = "https://botsocial.mlai.au/api"
DEFAULT_PASSWORD = "SecureSimUser2025!"
RATE_LIMIT_DELAY = 0.6  # Tested: 114 req/min works, using 0.6s for safety (100 req/min)
USERNAME_SUFFIX = "v3"  # Suffix to avoid USED_USERNAME errors from deleted accounts

# Neutral username components (no persona labels exposed)
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


def _get_name_components(user_id: str, seed: str) -> tuple:
    """Get consistent first name, last name, and number for a user.
    
    Args:
        user_id: Numeric user ID for determinism.
        seed: Additional seed string for variety.
    
    Returns:
        Tuple of (first_name, last_name, number).
    """
    import hashlib
    
    hash_input = f"{user_id}-{seed}-identity"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    first = FIRST_NAMES[hash_val % len(FIRST_NAMES)]
    last = LAST_PARTS[(hash_val >> 8) % len(LAST_PARTS)]
    num = (hash_val >> 16) % 100
    
    return first, last, num


def generate_neutral_username(user_id: str, seed: str) -> str:
    """Generate a neutral username that doesn't reveal persona type.
    
    Args:
        user_id: Numeric user ID for determinism.
        seed: Additional seed string for variety.
    
    Returns:
        A neutral username like 'alex_smith42' or 'alex42'.
    """
    import hashlib
    
    first, last, num = _get_name_components(user_id, seed)
    
    # Use secondary hash for pattern selection only
    pattern_hash = int(hashlib.md5(f"{user_id}-pattern".encode()).hexdigest(), 16)
    pattern = pattern_hash % 5
    
    if pattern == 0:
        return f"{first}_{last}{num}"
    elif pattern == 1:
        return f"{first}{num}"
    elif pattern == 2:
        return f"{first}{last}"
    elif pattern == 3:
        return f"{first}_{num}"
    else:
        return f"{first}{num}{last[:3]}"


def generate_neutral_display_name(user_id: str, seed: str) -> str:
    """Generate a neutral display name matching the username.
    
    Args:
        user_id: Numeric user ID for determinism.
        seed: Additional seed string for variety.
    
    Returns:
        A display name like 'Alex S.' or 'Alex Smith'.
    """
    import hashlib
    
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


@dataclass
class Persona:
    """Represents a user persona from the OASIS simulation."""
    
    username: str  # Original username from dataset (used for mapping)
    description: str
    primary_label: str
    user_id: str = ""  # Numeric ID from dataset (0-99)
    sharkey_id: str = ""  # Sharkey platform ID
    token: str = ""  # API token
    display_name: str = ""
    neutral_username: str = ""  # Neutral username for platform
    
    def __post_init__(self) -> None:
        """Generate neutral display name and username."""
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
    """Represents a post from the OASIS dataset."""
    
    post_id: str
    thread_id: str
    user_id: str
    parent_id: Optional[str]
    timestamp: str
    text: str
    persona_id: str
    category_labels: list[str] = field(default_factory=list)
    sharkey_note_id: str = ""


class RateLimitedClient:
    """API client with rate limiting."""
    
    def __init__(self, min_interval: float = RATE_LIMIT_DELAY) -> None:
        """Initialize the rate-limited client.
        
        Args:
            min_interval: Minimum seconds between requests.
        """
        self.last_request_time = 0.0
        self.min_interval = min_interval
    
    def wait(self) -> None:
        """Wait for rate limit if needed."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def post(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: int = 30,
        skip_wait: bool = False
    ) -> requests.Response:
        """Make a POST request with rate limiting.
        
        Args:
            endpoint: API endpoint (without /api prefix).
            data: Request body.
            timeout: Request timeout in seconds.
            skip_wait: If True, skip rate limit wait.
        
        Returns:
            Response object.
        """
        if not skip_wait:
            self.wait()
        
        url = f"{API_BASE}/{endpoint}"
        response = requests.post(url, json=data, timeout=timeout)
        return response


class BotSocialUploader:
    """Handles uploading OASIS data to BotSocial."""
    
    def __init__(
        self,
        admin_token: str,
        credentials_path: Optional[Path] = None
    ) -> None:
        """Initialize the uploader.
        
        Args:
            admin_token: Admin API token for account creation.
            credentials_path: Path to save/load account credentials.
        """
        self.admin_token = admin_token
        self.credentials_path = credentials_path or Path("account_tokens.json")
        self.client = RateLimitedClient()
        self.fast_client = RateLimitedClient(min_interval=2)  # For non-posting ops
        
        self.personas: dict[str, Persona] = {}  # user_id -> Persona
        self.posts: list[Post] = []
        self.edges: list[tuple[str, str]] = []  # (follower_id, followee_id)
        self.post_id_mapping: dict[str, str] = {}  # dataset post_id -> sharkey note_id
        
        # Load existing credentials if available
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load existing account credentials from file."""
        if self.credentials_path.exists():
            try:
                with open(self.credentials_path, 'r') as f:
                    creds = json.load(f)
                logger.info(f"Loaded {len(creds)} existing credentials")
                for user_id, data in creds.items():
                    if user_id in self.personas:
                        self.personas[user_id].token = data.get('token', '')
                        self.personas[user_id].sharkey_id = data.get('sharkey_id', '')
            except json.JSONDecodeError:
                logger.warning("Could not parse credentials file")
    
    def _save_credentials(self) -> None:
        """Save account credentials to file."""
        creds = {}
        for user_id, persona in self.personas.items():
            if persona.token:
                creds[user_id] = {
                    'username': persona.username,
                    'token': persona.token,
                    'sharkey_id': persona.sharkey_id,
                    'display_name': persona.display_name
                }
        
        with open(self.credentials_path, 'w') as f:
            json.dump(creds, f, indent=2)
        logger.info(f"Saved {len(creds)} credentials to {self.credentials_path}")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_personas(self, personas_path: Path) -> None:
        """Load personas from CSV file.
        
        Args:
            personas_path: Path to personas CSV file.
        """
        logger.info(f"Loading personas from {personas_path}")
        
        with open(personas_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                persona = Persona(
                    username=row['username'],
                    description=row['description'][:200],  # Truncate
                    primary_label=row['primary_label'],
                    user_id=str(idx)
                )
                self.personas[str(idx)] = persona
        
        logger.info(f"Loaded {len(self.personas)} personas")
    
    def load_posts(self, dataset_path: Path) -> None:
        """Load posts from JSONL dataset.
        
        Args:
            dataset_path: Path to JSONL dataset file.
        """
        logger.info(f"Loading posts from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                post = Post(
                    post_id=data['post_id'],
                    thread_id=data['thread_id'],
                    user_id=data['user_id'],
                    parent_id=data.get('parent_id'),
                    timestamp=data['timestamp'],
                    text=data['text'],
                    persona_id=data.get('persona_id', ''),
                    category_labels=data.get('category_labels', [])
                )
                self.posts.append(post)
        
        # Sort by timestamp then by post type (root posts first)
        self.posts.sort(key=lambda p: (int(p.timestamp), p.parent_id is not None))
        
        root_posts = sum(1 for p in self.posts if p.parent_id is None)
        replies = len(self.posts) - root_posts
        logger.info(f"Loaded {len(self.posts)} posts ({root_posts} root, {replies} replies)")
    
    def load_edges(self, edges_path: Path) -> None:
        """Load follow relationships from CSV.
        
        Args:
            edges_path: Path to edges CSV file.
        """
        logger.info(f"Loading edges from {edges_path}")
        
        with open(edges_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.edges.append((row['follower_id'], row['followee_id']))
        
        logger.info(f"Loaded {len(self.edges)} follow relationships")
    
    # =========================================================================
    # ACCOUNT CREATION
    # =========================================================================
    
    def create_account(self, persona: Persona) -> bool:
        """Create a single account using admin API.
        
        Args:
            persona: Persona to create account for.
        
        Returns:
            True if successful.
        """
        try:
            # 1. Create Account - token is returned directly in response!
            # Use neutral username + suffix to avoid USED_USERNAME errors
            username = f"{persona.neutral_username[:17]}_{USERNAME_SUFFIX}"
            response = self.fast_client.post(
                "admin/accounts/create",
                {
                    "i": self.admin_token,
                    "username": username,
                    "password": DEFAULT_PASSWORD
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                persona.sharkey_id = result.get('id', '')
                # Token is returned directly in the account creation response
                persona.token = result.get('token', '')
                if persona.token:
                    logger.info(f"✓ Created @{username} (token obtained)")
                else:
                    logger.warning(f"Created @{username} but no token in response")
                    return False
            elif response.status_code == 400:
                error = response.json()
                if "already exists" in str(error).lower():
                    logger.debug(f"Account {persona.username} already exists")
                    # Try to fetch ID if not known
                    if not persona.sharkey_id:
                        self._fetch_user_info(persona)
                    # For existing accounts, we need to retrieve token separately
                    if not persona.token:
                        token = self._retrieve_token(persona.username)
                        if token:
                            persona.token = token
                        else:
                            logger.warning(f"Existing account {persona.username} - no token available")
                            return False
                else:
                    logger.warning(f"Account creation error for {persona.username}: {error}")
                    return False
            else:
                logger.warning(f"Account creation failed for {persona.username}: {response.status_code}")
                return False
            
            return True
                
        except Exception as e:
            logger.error(f"Error creating account {persona.username}: {e}")
            return False

    def _fetch_user_info(self, persona: Persona) -> None:
        """Fetch user ID for existing user."""
        try:
            response = self.fast_client.post(
                "users/show", 
                {"username": persona.username},
                skip_wait=True
            )
            if response.status_code == 200:
                persona.sharkey_id = response.json().get('id', '')
        except Exception:
            pass

    def _retrieve_token(self, username: str) -> Optional[str]:
        """Retrieve user token using admin API.
        
        Args:
            username: Username to get token for.
            
        Returns:
            Token string or None.
        """
        try:
            # Get User ID first
            user_response = self.fast_client.post(
                "users/show",
                {"username": username},
                skip_wait=True
            )
            if user_response.status_code != 200:
                return None
            
            user_id = user_response.json().get('id')
            
            # Admin Show User (returns detailed info including token if allowed)
            response = self.fast_client.post(
                "admin/show-user",
                {
                    "i": self.admin_token,
                    "userId": user_id
                }
            )
            
            if response.status_code == 200:
                # Check mostly likely locations for token
                data = response.json()
                return data.get('token')
            return None
        except Exception as e:
            logger.warning(f"Token retrieval failed: {e}")
            return None
    
    def create_all_accounts(self) -> int:
        """Create accounts for all personas.
        
        Returns:
            Number of accounts created.
        """
        logger.info("Creating accounts for all personas...")
        created = 0
        
        for user_id, persona in self.personas.items():
            if self.create_account(persona):
                created += 1
                if created % 10 == 0:
                    logger.info(f"Progress: {created}/{len(self.personas)} accounts")
        
        logger.info(f"Created {created}/{len(self.personas)} accounts")
        self._save_credentials()
        return created
    
    # =========================================================================
    # PROFILE SETUP
    # =========================================================================
    
    def get_avatar_url(self, persona: Persona) -> str:
        """Generate deterministic avatar URL for persona.
        
        Args:
            persona: Persona to generate avatar for.
        
        Returns:
            Picsum Photos URL.
        """
        # Use neutral username for avatar seed (no persona label exposed)
        seed = f"avatar-{persona.neutral_username}-{persona.user_id}"
        return f"https://picsum.photos/seed/{seed}/400/400"
    
    def get_banner_url(self, persona: Persona) -> str:
        """Generate deterministic banner URL for persona.
        
        Args:
            persona: Persona to generate banner for.
        
        Returns:
            Picsum Photos URL.
        """
        # Use neutral username for banner seed (no persona label exposed)
        seed = f"banner-{persona.neutral_username}-{persona.user_id}"
        return f"https://picsum.photos/seed/{seed}/1500/500"
    
    def upload_image_from_url(self, token: str, image_url: str) -> Optional[str]:
        """Upload image from URL and return file ID.
        
        Note: This endpoint returns 204 - we must query for the file ID.
        
        Args:
            token: User API token.
            image_url: URL of image to upload.
        
        Returns:
            File ID or None.
        """
        try:
            # Get current most recent file
            before_response = requests.post(
                f"{API_BASE}/drive/files",
                json={"i": token, "limit": 1},
                timeout=30
            )
            before_id = None
            if before_response.status_code == 200:
                files = before_response.json()
                if files:
                    before_id = files[0].get('id')
            
            # Trigger upload (returns 204)
            upload_response = requests.post(
                f"{API_BASE}/drive/files/upload-from-url",
                json={"i": token, "url": image_url, "isSensitive": False},
                timeout=30
            )
            
            if upload_response.status_code not in [200, 204]:
                return None
            
            # Wait and check for new file
            for _ in range(8):
                time.sleep(2)
                after_response = requests.post(
                    f"{API_BASE}/drive/files",
                    json={"i": token, "limit": 1},
                    timeout=30
                )
                if after_response.status_code == 200:
                    files = after_response.json()
                    if files:
                        after_id = files[0].get('id')
                        if after_id and after_id != before_id:
                            return after_id
            
            return None
            
        except Exception as e:
            logger.error(f"Image upload error: {e}")
            return None
    
    def setup_profile(self, persona: Persona) -> bool:
        """Set up user profile with name, description, and images.
        
        Args:
            persona: Persona to set up.
        
        Returns:
            True if successful.
        """
        if not persona.token:
            logger.warning(f"No token for {persona.username}, skipping profile")
            return False
        
        try:
            # Update text profile
            response = self.fast_client.post(
                "i/update",
                {
                    "i": persona.token,
                    "name": persona.display_name,
                    "description": persona.description
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Profile update failed for {persona.username}")
                return False
            
            # Upload avatar and banner
            # Use deterministic Picsum URLs based on username/persona
            avatar_url = self.get_avatar_url(persona)
            banner_url = self.get_banner_url(persona)
            
            # Upload Avatar
            avatar_id = self.upload_image_from_url(persona.token, avatar_url)
            if avatar_id:
                self.fast_client.post("i/update", {"i": persona.token, "avatarId": avatar_id})
                logger.info(f"Set avatar for @{persona.username}")
            
            # Upload Banner (wait a bit to ensure file processing doesn't conflict)
            time.sleep(2)
            banner_id = self.upload_image_from_url(persona.token, banner_url)
            if banner_id:
                self.fast_client.post("i/update", {"i": persona.token, "bannerId": banner_id})
                logger.info(f"Set banner for @{persona.username}")
            
            return True
            
        except Exception as e:
            logger.error(f"Profile setup error for {persona.username}: {e}")
            return False
    
    # =========================================================================
    # POST UPLOAD
    # =========================================================================
    
    def create_note(self, post: Post, reply_id: Optional[str] = None) -> Optional[str]:
        """Create a note/post on BotSocial.
        
        Args:
            post: Post to create.
            reply_id: Sharkey note ID to reply to.
        
        Returns:
            Created note ID or None.
        """
        persona = self.personas.get(post.user_id)
        if not persona or not persona.token:
            logger.warning(f"No token for user {post.user_id}, skipping post")
            return None
        
        try:
            data = {
                "i": persona.token,
                "text": post.text,
                "visibility": "public"
            }
            
            if reply_id:
                data["replyId"] = reply_id
            
            response = self.client.post("notes/create", data)
            
            if response.status_code == 200:
                result = response.json()
                note_id = result.get('createdNote', {}).get('id')
                return note_id
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting...")
                time.sleep(60)
                return self.create_note(post, reply_id)
            else:
                logger.warning(f"Note creation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Note creation error: {e}")
            return None
    
    def upload_all_posts(self) -> int:
        """Upload all posts maintaining thread structure.
        
        Returns:
            Number of posts uploaded.
        """
        logger.info(f"Uploading {len(self.posts)} posts...")
        uploaded = 0
        
        for i, post in enumerate(self.posts):
            # Determine reply target
            reply_id = None
            if post.parent_id:
                reply_id = self.post_id_mapping.get(post.parent_id)
                if not reply_id:
                    logger.debug(f"Parent {post.parent_id} not found for {post.post_id}")
            
            # Create note
            note_id = self.create_note(post, reply_id)
            
            if note_id:
                self.post_id_mapping[post.post_id] = note_id
                post.sharkey_note_id = note_id
                uploaded += 1
                
                if uploaded % 20 == 0:
                    logger.info(f"Progress: {uploaded}/{len(self.posts)} posts uploaded")
            
            # Add realistic variation to timing
            if post.timestamp != self.posts[i-1].timestamp if i > 0 else True:
                # New timestamp group - add extra delay
                time.sleep(random.uniform(5, 15))
        
        logger.info(f"Uploaded {uploaded}/{len(self.posts)} posts")
        return uploaded
    
    # =========================================================================
    # SOCIAL INTERACTIONS
    # =========================================================================
    
    def create_follow(self, follower: Persona, followee: Persona) -> bool:
        """Create a follow relationship.
        
        Args:
            follower: User who is following.
            followee: User being followed.
        
        Returns:
            True if successful.
        """
        if not follower.token or not followee.sharkey_id:
            return False
        
        try:
            response = self.fast_client.post(
                "following/create",
                {
                    "i": follower.token,
                    "userId": followee.sharkey_id
                }
            )
            return response.status_code in [200, 204]
        except Exception:
            return False
    
    def create_all_follows(self) -> int:
        """Create all follow relationships from edges.
        
        Returns:
            Number of follows created.
        """
        logger.info(f"Creating {len(self.edges)} follow relationships...")
        created = 0
        
        for follower_id, followee_id in self.edges:
            follower = self.personas.get(follower_id)
            followee = self.personas.get(followee_id)
            
            if follower and followee:
                if self.create_follow(follower, followee):
                    created += 1
                    
                    if created % 50 == 0:
                        logger.info(f"Progress: {created}/{len(self.edges)} follows")
        
        logger.info(f"Created {created}/{len(self.edges)} follows")
        return created
    
    def add_reaction(self, user: Persona, note_id: str, reaction: str = "❤️") -> bool:
        """Add a reaction to a note.
        
        Args:
            user: User adding reaction.
            note_id: Note to react to.
            reaction: Reaction emoji.
        
        Returns:
            True if successful.
        """
        if not user.token:
            return False
        
        try:
            response = self.fast_client.post(
                "notes/reactions/create",
                {
                    "i": user.token,
                    "noteId": note_id,
                    "reaction": reaction
                }
            )
            return response.status_code in [200, 204]
        except Exception:
            return False
    
    def generate_reactions(self, count: int = 200) -> int:
        """Generate random reactions based on persona affinity.
        
        Args:
            count: Target number of reactions to create.
        
        Returns:
            Number of reactions created.
        """
        logger.info(f"Generating ~{count} reactions...")
        
        # Get posts with their note IDs
        posted = [(p, p.sharkey_note_id) for p in self.posts if p.sharkey_note_id]
        if not posted:
            logger.warning("No posts to react to")
            return 0
        
        reactions = ["❤️", "👍", "🎉", "😊", "🔥", "💯"]
        created = 0
        
        # Simple random reactions
        for _ in range(count):
            post, note_id = random.choice(posted)
            
            # Pick a random user (not the post author)
            eligible = [p for uid, p in self.personas.items() 
                       if uid != post.user_id and p.token]
            if not eligible:
                continue
            
            user = random.choice(eligible)
            reaction = random.choice(reactions)
            
            if self.add_reaction(user, note_id, reaction):
                created += 1
        
        logger.info(f"Created {created} reactions")
        return created
    
    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================
    
    def run_full_upload(
        self,
        dataset_path: Path,
        personas_path: Path,
        edges_path: Path,
        skip_accounts: bool = False,
        skip_profiles: bool = False,
        skip_posts: bool = False,
        skip_follows: bool = False,
        skip_reactions: bool = False
    ) -> dict[str, int]:
        """Run the full upload workflow.
        
        Args:
            dataset_path: Path to JSONL dataset.
            personas_path: Path to personas CSV.
            edges_path: Path to edges CSV.
            skip_accounts: Skip account creation.
            skip_profiles: Skip profile setup.
            skip_posts: Skip post upload.
            skip_follows: Skip follow creation.
            skip_reactions: Skip reaction generation.
        
        Returns:
            Statistics dictionary.
        """
        stats = {
            'accounts': 0,
            'profiles': 0,
            'posts': 0,
            'follows': 0,
            'reactions': 0
        }
        
        # Load data
        self.load_personas(personas_path)
        self.load_posts(dataset_path)
        self.load_edges(edges_path)
        
        # Phase 1: Accounts
        if not skip_accounts:
            logger.info("\n" + "="*70)
            logger.info("PHASE 1: Account Creation")
            logger.info("="*70)
            stats['accounts'] = self.create_all_accounts()
        
        # Phase 2: Profiles (requires tokens)
        if not skip_profiles:
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: Profile Setup")
            logger.info("="*70)
            for uid, persona in self.personas.items():
                if self.setup_profile(persona):
                    stats['profiles'] += 1
        
        # Phase 3: Posts
        if not skip_posts:
            logger.info("\n" + "="*70)
            logger.info("PHASE 3: Post Upload")
            logger.info("="*70)
            stats['posts'] = self.upload_all_posts()
        
        # Phase 4: Follows
        if not skip_follows:
            logger.info("\n" + "="*70)
            logger.info("PHASE 4: Follow Relationships")
            logger.info("="*70)
            stats['follows'] = self.create_all_follows()
        
        # Phase 5: Reactions
        if not skip_reactions:
            logger.info("\n" + "="*70)
            logger.info("PHASE 5: Reactions")
            logger.info("="*70)
            stats['reactions'] = self.generate_reactions(200)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("UPLOAD COMPLETE")
        logger.info("="*70)
        logger.info(f"Accounts created: {stats['accounts']}")
        logger.info(f"Profiles setup: {stats['profiles']}")
        logger.info(f"Posts uploaded: {stats['posts']}")
        logger.info(f"Follows created: {stats['follows']}")
        logger.info(f"Reactions added: {stats['reactions']}")
        logger.info(f"\nView at: https://botsocial.mlai.au")
        
        return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload OASIS dataset to BotSocial"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/runs/prod_100p_100s_65_35_dataset.jsonl"),
        help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("data/runs/personas_prod_100p_100s_65_35.csv"),
        help="Path to personas CSV"
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/runs/edges_prod_100p_100s_65_35.csv"),
        help="Path to edges CSV"
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        required=True,
        help="Admin API token"
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("account_tokens.json"),
        help="Path to save/load credentials"
    )
    parser.add_argument(
        "--skip-accounts",
        action="store_true",
        help="Skip account creation"
    )
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip profile setup"
    )
    parser.add_argument(
        "--skip-posts",
        action="store_true",
        help="Skip post upload"
    )
    parser.add_argument(
        "--skip-follows",
        action="store_true",
        help="Skip follow creation"
    )
    parser.add_argument(
        "--skip-reactions",
        action="store_true",
        help="Skip reaction generation"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    for path in [args.dataset, args.personas, args.edges]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
    
    # Create uploader and run
    uploader = BotSocialUploader(
        admin_token=args.admin_token,
        credentials_path=args.credentials
    )
    
    uploader.run_full_upload(
        dataset_path=args.dataset,
        personas_path=args.personas,
        edges_path=args.edges,
        skip_accounts=args.skip_accounts,
        skip_profiles=args.skip_profiles,
        skip_posts=args.skip_posts,
        skip_follows=args.skip_follows,
        skip_reactions=args.skip_reactions
    )


if __name__ == "__main__":
    main()

