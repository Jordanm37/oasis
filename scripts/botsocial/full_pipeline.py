#!/usr/bin/env python3
"""
Full Detection → Response → Upload Pipeline

This is the main orchestrator that combines:
1. Content detection (from BotSocial or dataset)
2. Response generation (using OASIS agents or templates)
3. Upload to BotSocial

Usage:
    # Full pipeline: detect harmful content, generate responses, upload
    poetry run python3 scripts/botsocial/full_pipeline.py \
        --admin-token YOUR_TOKEN \
        --credentials data/runs/botsocial_tokens_v3.json \
        --source live \
        --action intervene

    # Analyze dataset and generate intervention dataset
    poetry run python3 scripts/botsocial/full_pipeline.py \
        --dataset data/runs/prod_dataset.jsonl \
        --output data/runs/interventions.jsonl \
        --action generate

    # Upload generated interventions
    poetry run python3 scripts/botsocial/full_pipeline.py \
        --credentials data/runs/botsocial_tokens_v3.json \
        --dataset data/runs/interventions.jsonl \
        --action upload

Author: eSafety Hackathon Team
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
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
# CONSTANTS
# =============================================================================

API_BASE = "https://botsocial.mlai.au/api"

# Harm categories with response strategies
HARM_RESPONSE_CONFIG = {
    "ed_risk": {
        "severity": 3,
        "response_type": "supportive",
        "templates": [
            "Recovery is a journey, not a destination. If you're struggling, NEDA helpline: 1-800-931-2237 💙",
            "You're not alone in this. Support is available 24/7: Crisis Text Line - text HOME to 741741",
        ],
        "agent_persona": "recovery_support"
    },
    "pro_ana": {
        "severity": 4,
        "response_type": "intervention",
        "templates": [
            "This content may be harmful. If you're struggling with eating, please reach out: NEDA 1-800-931-2237",
        ],
        "agent_persona": "recovery_support"
    },
    "incel_misogyny": {
        "severity": 3,
        "response_type": "redirect",
        "templates": [
            "Feeling isolated? You're not alone. Mental health support is available: 988 Lifeline 💙",
            "Connection matters. If you're struggling, reach out: Crisis Text Line - text HELLO to 741741",
        ],
        "agent_persona": "recovery_support"
    },
    "bullying": {
        "severity": 4,
        "response_type": "intervention",
        "templates": [
            "Let's keep this space supportive. Words can hurt. 💙",
            "If you're being targeted, help is available. You matter.",
        ],
        "agent_persona": "benign"
    },
    "extremist": {
        "severity": 5,
        "response_type": "flag",
        "templates": [
            "⚠️ This content has been flagged for review.",
        ],
        "agent_persona": None
    },
    "hate_speech": {
        "severity": 5,
        "response_type": "flag",
        "templates": [
            "This content violates community guidelines. Hate speech is not tolerated. 🚫",
        ],
        "agent_persona": None
    },
    "misinfo": {
        "severity": 2,
        "response_type": "educate",
        "templates": [
            "🔍 Tip: Always check multiple sources. Fact-checking sites: snopes.com, factcheck.org",
            "📚 Before sharing, verify! Misinformation spreads fast.",
        ],
        "agent_persona": "benign"
    },
    "conspiracy": {
        "severity": 2,
        "response_type": "educate",
        "templates": [
            "💡 Critical thinking is key. Consider checking fact-checking resources.",
        ],
        "agent_persona": "benign"
    },
}

# Detection patterns (simplified from detection_pipeline.py)
DETECTION_PATTERNS = {
    "ed_risk": [
        r"thinspo", r"meanspo", r"pro[\s-]*ana", r"goal\s*weight", r"ugw", r"restrict",
        r"calorie", r"bmi", r"body\s*check", r"fast(?:ing)"
    ],
    "incel_misogyny": [
        r"blackpill", r"redpill", r"chad", r"stacy", r"foid", r"incel", r"hypergamy"
    ],
    "bullying": [
        r"kys", r"kill\s*yourself", r"rope", r"doxx", r"expose"
    ],
    "extremist": [
        r"race\s*war", r"boogaloo", r"1488", r"day\s*of\s*the"
    ],
    "hate_speech": [
        r"replacement", r"white\s*genocide", r"invasion"
    ],
    "misinfo": [
        r"wake\s*up", r"sheeple", r"msm", r"suppressed", r"do\s*your\s*research"
    ],
    "conspiracy": [
        r"deep\s*state", r"false\s*flag", r"cover[\s-]*up", r"psyop"
    ],
}


class PipelineAction(Enum):
    """Pipeline actions."""
    DETECT = "detect"         # Only detect, output results
    GENERATE = "generate"     # Detect + generate responses
    INTERVENE = "intervene"   # Detect + generate + post responses
    UPLOAD = "upload"         # Upload existing dataset


@dataclass
class DetectedPost:
    """A detected harmful post with response."""
    post_id: str
    text: str
    user_id: str
    categories: list[str]
    severity: int
    response_text: Optional[str] = None
    response_posted: bool = False
    sharkey_note_id: Optional[str] = None
    response_note_id: Optional[str] = None


# =============================================================================
# DETECTOR
# =============================================================================

class ContentDetector:
    """Detects harmful content."""
    
    def __init__(self, severity_threshold: int = 2) -> None:
        self.severity_threshold = severity_threshold
        
        # Compile patterns
        import re
        self.patterns: dict[str, list] = {}
        for cat, patterns in DETECTION_PATTERNS.items():
            self.patterns[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect(self, text: str) -> tuple[list[str], int]:
        """Detect harmful categories in text."""
        categories = []
        
        for cat, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    categories.append(cat)
                    break
        
        if not categories:
            return ["benign"], 0
        
        # Get max severity
        severity = max(
            HARM_RESPONSE_CONFIG.get(c, {}).get("severity", 0) 
            for c in categories
        )
        
        return categories, severity


# =============================================================================
# RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator:
    """Generates intervention responses."""
    
    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm
    
    def generate(self, categories: list[str], original_text: str) -> Optional[str]:
        """Generate an intervention response."""
        # Find best matching category
        for cat in categories:
            if cat in HARM_RESPONSE_CONFIG:
                config = HARM_RESPONSE_CONFIG[cat]
                templates = config.get("templates", [])
                if templates:
                    return random.choice(templates)
        
        return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class FullPipeline:
    """Full detection → response → upload pipeline."""
    
    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        admin_token: Optional[str] = None,
        intervention_token: Optional[str] = None,
        severity_threshold: int = 2,
        dry_run: bool = False
    ) -> None:
        self.credentials_path = credentials_path
        self.admin_token = admin_token
        self.intervention_token = intervention_token
        self.severity_threshold = severity_threshold
        self.dry_run = dry_run
        
        self.detector = ContentDetector(severity_threshold)
        self.generator = ResponseGenerator()
        self.credentials: dict[str, Any] = {}
        
        self.detections: list[DetectedPost] = []
        self.stats = {
            "scanned": 0,
            "detected": 0,
            "responses_generated": 0,
            "responses_posted": 0
        }
        
        if credentials_path and credentials_path.exists():
            self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials."""
        with open(self.credentials_path, 'r') as f:
            self.credentials = json.load(f)
        logger.info(f"Loaded {len(self.credentials)} credentials")
        
        # Get first available token for interventions
        if not self.intervention_token:
            for uid, data in self.credentials.items():
                if data.get('token'):
                    self.intervention_token = data['token']
                    logger.info(f"Using intervention token from: {data.get('neutral_username', uid)}")
                    break
    
    def _api_post(self, endpoint: str, data: dict) -> Optional[dict]:
        """Make API request."""
        try:
            response = requests.post(
                f"{API_BASE}/{endpoint}",
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
        return None
    
    def process_post(self, post_data: dict) -> Optional[DetectedPost]:
        """Process a single post."""
        text = post_data.get("text", "")
        if not text:
            return None
        
        self.stats["scanned"] += 1
        
        # Detect
        categories, severity = self.detector.detect(text)
        
        if severity < self.severity_threshold:
            return None
        
        self.stats["detected"] += 1
        
        detection = DetectedPost(
            post_id=post_data.get("post_id", post_data.get("id", "")),
            text=text,
            user_id=str(post_data.get("user_id", post_data.get("userId", ""))),
            categories=categories,
            severity=severity,
            sharkey_note_id=post_data.get("id")
        )
        
        return detection
    
    def generate_response(self, detection: DetectedPost) -> None:
        """Generate response for detection."""
        response = self.generator.generate(detection.categories, detection.text)
        if response:
            detection.response_text = response
            self.stats["responses_generated"] += 1
    
    def post_response(self, detection: DetectedPost) -> bool:
        """Post response to BotSocial."""
        if not detection.response_text or not self.intervention_token:
            return False
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would post: {detection.response_text[:50]}...")
            return True
        
        # Need the Sharkey note ID to reply
        if not detection.sharkey_note_id:
            logger.warning(f"No Sharkey note ID for {detection.post_id}")
            return False
        
        result = self._api_post("notes/create", {
            "i": self.intervention_token,
            "text": detection.response_text,
            "replyId": detection.sharkey_note_id,
            "visibility": "public"
        })
        
        if result:
            detection.response_note_id = result.get("createdNote", {}).get("id")
            detection.response_posted = True
            self.stats["responses_posted"] += 1
            return True
        
        return False
    
    def run_on_dataset(
        self,
        dataset_path: Path,
        action: PipelineAction
    ) -> list[DetectedPost]:
        """Run pipeline on a dataset file."""
        logger.info(f"Processing dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                detection = self.process_post(data)
                if detection:
                    self.detections.append(detection)
                    
                    if action in [PipelineAction.GENERATE, PipelineAction.INTERVENE]:
                        self.generate_response(detection)
                    
                    if action == PipelineAction.INTERVENE:
                        self.post_response(detection)
                        time.sleep(0.6)  # Rate limit
        
        return self.detections
    
    def run_live_monitor(
        self,
        action: PipelineAction,
        duration: Optional[float] = None,
        poll_interval: float = 30.0
    ) -> list[DetectedPost]:
        """Run pipeline on live BotSocial feed."""
        logger.info("Starting live monitoring...")
        
        if not self.intervention_token:
            logger.error("No token available for monitoring")
            return []
        
        last_id: Optional[str] = None
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Fetch posts
                data: dict[str, Any] = {"i": self.intervention_token, "limit": 50}
                if last_id:
                    data["sinceId"] = last_id
                
                result = self._api_post("notes/local-timeline", data)
                
                if result:
                    if result:
                        last_id = result[0].get("id")
                    
                    for post in result:
                        # Convert to our format
                        post_data = {
                            "id": post.get("id"),
                            "text": post.get("text", ""),
                            "userId": post.get("userId"),
                            "user_id": post.get("userId")
                        }
                        
                        detection = self.process_post(post_data)
                        if detection:
                            detection.sharkey_note_id = post.get("id")
                            self.detections.append(detection)
                            
                            logger.info(
                                f"DETECTED [{detection.severity}]: "
                                f"{detection.categories} - {detection.text[:40]}..."
                            )
                            
                            if action in [PipelineAction.GENERATE, PipelineAction.INTERVENE]:
                                self.generate_response(detection)
                            
                            if action == PipelineAction.INTERVENE and detection.response_text:
                                self.post_response(detection)
                                logger.info(f"  → Intervention posted")
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
        
        return self.detections
    
    def export_detections(self, output_path: Path) -> None:
        """Export detections to JSONL."""
        with open(output_path, 'w') as f:
            for det in self.detections:
                record = {
                    "post_id": det.post_id,
                    "text": det.text,
                    "user_id": det.user_id,
                    "categories": det.categories,
                    "severity": det.severity,
                    "response_text": det.response_text,
                    "response_posted": det.response_posted,
                    "sharkey_note_id": det.sharkey_note_id,
                    "response_note_id": det.response_note_id
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Exported {len(self.detections)} detections to {output_path}")
    
    def export_intervention_dataset(self, output_path: Path) -> None:
        """Export as uploadable dataset (for botsocial_uploader.py)."""
        posts = []
        
        for i, det in enumerate(self.detections):
            if det.response_text:
                # Create intervention post
                post = {
                    "post_id": f"intervention_{i}",
                    "thread_id": det.post_id,
                    "user_id": "intervention_bot",
                    "parent_id": det.post_id,
                    "timestamp": str(i),
                    "text": det.response_text,
                    "category_labels": ["intervention"],
                    "metadata": {
                        "target_post_id": det.post_id,
                        "target_categories": det.categories,
                        "target_severity": det.severity
                    }
                }
                posts.append(post)
        
        with open(output_path, 'w') as f:
            for post in posts:
                f.write(json.dumps(post) + "\n")
        
        logger.info(f"Created intervention dataset: {output_path} ({len(posts)} interventions)")
    
    def print_summary(self) -> None:
        """Print summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Posts scanned: {self.stats['scanned']}")
        print(f"Harmful detected: {self.stats['detected']}")
        print(f"Responses generated: {self.stats['responses_generated']}")
        print(f"Responses posted: {self.stats['responses_posted']}")
        
        # Category breakdown
        if self.detections:
            cat_counts: dict[str, int] = {}
            for det in self.detections:
                for cat in det.categories:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            print("\nBy Category:")
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
        
        print("="*60)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full Detection → Response → Upload Pipeline"
    )
    
    # Input sources
    parser.add_argument("--dataset", type=Path, help="Dataset file to process")
    parser.add_argument("--source", type=str, choices=["dataset", "live"],
                       default="dataset", help="Data source")
    
    # Output
    parser.add_argument("--output", type=Path, help="Output file for detections")
    parser.add_argument("--output-dataset", type=Path, 
                       help="Output as uploadable dataset")
    
    # Authentication
    parser.add_argument("--credentials", type=Path, help="Credentials file")
    parser.add_argument("--admin-token", type=str, help="Admin token")
    
    # Action
    parser.add_argument("--action", type=str, default="detect",
                       choices=["detect", "generate", "intervene", "upload"],
                       help="Pipeline action")
    
    # Options
    parser.add_argument("--severity-threshold", type=int, default=2,
                       help="Minimum severity to flag")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually post responses")
    parser.add_argument("--duration", type=float,
                       help="Monitoring duration (seconds)")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                       help="Poll interval for live monitoring")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = FullPipeline(
        credentials_path=args.credentials,
        admin_token=args.admin_token,
        severity_threshold=args.severity_threshold,
        dry_run=args.dry_run
    )
    
    action = PipelineAction(args.action)
    
    # Run
    if args.source == "live":
        pipeline.run_live_monitor(
            action=action,
            duration=args.duration,
            poll_interval=args.poll_interval
        )
    elif args.dataset:
        pipeline.run_on_dataset(args.dataset, action)
    
    # Export
    if args.output:
        pipeline.export_detections(args.output)
    
    if args.output_dataset:
        pipeline.export_intervention_dataset(args.output_dataset)
    
    pipeline.print_summary()


if __name__ == "__main__":
    main()

