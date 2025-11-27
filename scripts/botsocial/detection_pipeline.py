#!/usr/bin/env python3
"""
Detection → Upload Pipeline for BotSocial

This pipeline:
1. Monitors BotSocial for new posts (via API polling or streaming)
2. Detects harmful content using the OASIS labeling system
3. Optionally generates intervention responses
4. Uploads interventions back to BotSocial

Usage:
    # Monitor and detect (dry run)
    poetry run python3 scripts/botsocial/detection_pipeline.py \
        --credentials data/runs/botsocial_tokens_v3.json \
        --mode monitor \
        --dry-run

    # Full pipeline with interventions
    poetry run python3 scripts/botsocial/detection_pipeline.py \
        --credentials data/runs/botsocial_tokens_v3.json \
        --mode intervene \
        --intervention-user moderator_bot

    # Process existing dataset and upload detections
    poetry run python3 scripts/botsocial/detection_pipeline.py \
        --dataset data/runs/my_dataset.jsonl \
        --credentials data/runs/botsocial_tokens_v3.json \
        --mode upload-detections

Author: eSafety Hackathon Team
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

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

# 13-class harm taxonomy (from OASIS labeler.py)
HARM_CATEGORIES = {
    "benign": {"severity": 0, "action": "none"},
    "recovery_support": {"severity": 0, "action": "none"},
    "ed_risk": {"severity": 3, "action": "flag"},
    "pro_ana": {"severity": 4, "action": "intervene"},
    "incel_misogyny": {"severity": 3, "action": "flag"},
    "alpha": {"severity": 2, "action": "flag"},
    "misinfo": {"severity": 2, "action": "flag"},
    "conspiracy": {"severity": 2, "action": "flag"},
    "trad": {"severity": 1, "action": "monitor"},
    "gamergate": {"severity": 2, "action": "flag"},
    "extremist": {"severity": 5, "action": "intervene"},
    "hate_speech": {"severity": 5, "action": "intervene"},
    "bullying": {"severity": 4, "action": "intervene"},
}

# Label token patterns (regex for detecting in text)
LABEL_TOKEN_PATTERNS = {
    # ED cluster
    r"(?:thinspo|meanspo|ana\s*buddy|pro[\s-]*ana|mia|purge|restrict|fast(?:ing)?.*goal|goal\s*weight|ugw|gw|cw|hw)": ["ed_risk", "pro_ana"],
    r"(?:calorie|kcal|bmi|body\s*check|scale|weigh)": ["ed_risk"],
    
    # Incel/Manosphere
    r"(?:blackpill|redpill|chad|stacy|foid|femoid|roastie|hypergamy|smv|lookism|looksmax|gymcel|incel)": ["incel_misogyny"],
    r"(?:alpha|beta|sigma|frame|plate|spinning|high\s*value)": ["alpha"],
    
    # Misinfo/Conspiracy
    r"(?:wake\s*up|sheeple|msm|mainstream\s*media|they\s*don't\s*want|suppressed|hidden\s*truth|do\s*your\s*research)": ["misinfo", "conspiracy"],
    r"(?:deep\s*state|false\s*flag|controlled\s*opposition|psyop|cover[\s-]*up)": ["conspiracy"],
    
    # Extremist/Hate
    r"(?:race\s*war|boogaloo|accelerat|day\s*of\s*the\s*rope|1488|14\s*words)": ["extremist"],
    r"(?:replacement|invasion|great\s*replacement|white\s*genocide)": ["hate_speech", "extremist"],
    
    # Bullying
    r"(?:kys|kill\s*yourself|neck\s*yourself|rope|sui|end\s*it)": ["bullying", "extremist"],
    r"(?:doxx|expose|screenshot.*shame|pile\s*on)": ["bullying"],
    
    # Culture war
    r"(?:woke|sjw|forced\s*diversity|go\s*woke|get\s*woke)": ["gamergate"],
    r"(?:trad\s*wife|tradwife|natural\s*order|decline|degeneracy)": ["trad"],
}

# Intervention templates by category
INTERVENTION_TEMPLATES = {
    "ed_risk": [
        "If you're struggling with eating or body image, support is available. Text 'HELLO' to 741741 (Crisis Text Line) 💙",
        "Recovery is possible. The National Eating Disorders Association helpline: 1-800-931-2237 💚",
    ],
    "pro_ana": [
        "This content may be harmful. If you're struggling, please reach out: NEDA helpline 1-800-931-2237 💙",
    ],
    "bullying": [
        "Let's keep this space supportive. If you're being targeted, you can report this post or reach out for help.",
        "Words matter. If you're struggling, Crisis Text Line is here: text HOME to 741741 💙",
    ],
    "extremist": [
        "This content has been flagged for review. If you're in crisis, please reach out: 988 Suicide & Crisis Lifeline.",
    ],
    "hate_speech": [
        "This content violates community guidelines. Hate speech harms everyone. 🚫",
    ],
    "incel_misogyny": [
        "Feeling isolated? You're not alone. Mental health support: 988 Suicide & Crisis Lifeline 💙",
    ],
    "misinfo": [
        "🔍 Tip: Check multiple sources before sharing. Misinformation spreads fast.",
    ],
    "conspiracy": [
        "💡 Critical thinking matters. Consider checking fact-checking sites like snopes.com or factcheck.org",
    ],
}


class PipelineMode(Enum):
    """Pipeline operation modes."""
    MONITOR = "monitor"           # Watch and log detections
    INTERVENE = "intervene"       # Respond to harmful content
    UPLOAD_DETECTIONS = "upload-detections"  # Upload detection results
    ANALYZE = "analyze"           # Analyze existing dataset


@dataclass
class Detection:
    """A detected piece of harmful content."""
    note_id: str
    text: str
    user_id: str
    username: str
    categories: list[str]
    severity: int
    confidence: float
    timestamp: datetime
    action_taken: str = ""
    intervention_id: Optional[str] = None


@dataclass
class PipelineStats:
    """Statistics for pipeline run."""
    posts_scanned: int = 0
    detections: int = 0
    interventions: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[int, int] = field(default_factory=dict)


# =============================================================================
# DETECTOR CLASS
# =============================================================================

class HarmDetector:
    """Detects harmful content using pattern matching and optional LLM."""
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_model: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> None:
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        
        # Compile regex patterns
        self.patterns: list[tuple[re.Pattern, list[str]]] = []
        for pattern, categories in LABEL_TOKEN_PATTERNS.items():
            self.patterns.append((re.compile(pattern, re.IGNORECASE), categories))
    
    def detect(self, text: str) -> tuple[list[str], float]:
        """Detect harmful content in text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (categories, confidence).
        """
        categories: set[str] = set()
        match_count = 0
        
        # Pattern-based detection
        for pattern, cats in self.patterns:
            matches = pattern.findall(text)
            if matches:
                categories.update(cats)
                match_count += len(matches)
        
        # Calculate confidence based on match density
        word_count = len(text.split())
        if word_count > 0 and match_count > 0:
            confidence = min(0.9, 0.3 + (match_count / word_count) * 2)
        else:
            confidence = 0.0
        
        # If no harmful patterns found, classify as benign
        if not categories:
            categories.add("benign")
            confidence = 0.8
        
        return sorted(categories), confidence
    
    def get_severity(self, categories: list[str]) -> int:
        """Get maximum severity from categories."""
        max_severity = 0
        for cat in categories:
            if cat in HARM_CATEGORIES:
                max_severity = max(max_severity, HARM_CATEGORIES[cat]["severity"])
        return max_severity
    
    def get_action(self, categories: list[str]) -> str:
        """Get recommended action from categories."""
        actions = {"none": 0, "monitor": 1, "flag": 2, "intervene": 3}
        max_action = "none"
        max_priority = 0
        
        for cat in categories:
            if cat in HARM_CATEGORIES:
                action = HARM_CATEGORIES[cat]["action"]
                if actions.get(action, 0) > max_priority:
                    max_action = action
                    max_priority = actions[action]
        
        return max_action


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class DetectionPipeline:
    """Main detection and intervention pipeline."""
    
    def __init__(
        self,
        credentials_path: Path,
        mode: PipelineMode,
        intervention_user: Optional[str] = None,
        dry_run: bool = False,
        severity_threshold: int = 2,
        poll_interval: float = 30.0
    ) -> None:
        self.credentials_path = credentials_path
        self.mode = mode
        self.intervention_user = intervention_user
        self.dry_run = dry_run
        self.severity_threshold = severity_threshold
        self.poll_interval = poll_interval
        
        self.detector = HarmDetector()
        self.stats = PipelineStats()
        self.detections: list[Detection] = []
        self.credentials: dict[str, Any] = {}
        self.intervention_token: Optional[str] = None
        
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from file."""
        if self.credentials_path.exists():
            with open(self.credentials_path, 'r') as f:
                self.credentials = json.load(f)
            logger.info(f"Loaded {len(self.credentials)} credentials")
            
            # Find intervention user token
            if self.intervention_user:
                for uid, data in self.credentials.items():
                    if data.get('neutral_username', '').startswith(self.intervention_user):
                        self.intervention_token = data.get('token')
                        logger.info(f"Using intervention user: {data.get('neutral_username')}")
                        break
                
                if not self.intervention_token:
                    # Use first available token
                    for uid, data in self.credentials.items():
                        if data.get('token'):
                            self.intervention_token = data['token']
                            logger.warning(f"Intervention user not found, using: {data.get('neutral_username')}")
                            break
    
    def _api_post(self, endpoint: str, data: dict[str, Any]) -> Optional[dict]:
        """Make API POST request."""
        try:
            response = requests.post(
                f"{API_BASE}/{endpoint}",
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"API error: {e}")
            return None
    
    def fetch_recent_posts(self, limit: int = 50, since_id: Optional[str] = None) -> list[dict]:
        """Fetch recent posts from BotSocial."""
        data: dict[str, Any] = {"limit": limit}
        if since_id:
            data["sinceId"] = since_id
        
        result = self._api_post("notes/local-timeline", data)
        return result if result else []
    
    def analyze_post(self, post: dict) -> Optional[Detection]:
        """Analyze a single post for harmful content."""
        text = post.get("text", "")
        if not text:
            return None
        
        categories, confidence = self.detector.detect(text)
        severity = self.detector.get_severity(categories)
        
        # Skip benign content
        if categories == ["benign"] or severity < self.severity_threshold:
            return None
        
        detection = Detection(
            note_id=post.get("id", ""),
            text=text,
            user_id=post.get("userId", ""),
            username=post.get("user", {}).get("username", "unknown"),
            categories=categories,
            severity=severity,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        return detection
    
    def create_intervention(self, detection: Detection) -> Optional[str]:
        """Create an intervention response."""
        if not self.intervention_token:
            logger.warning("No intervention token available")
            return None
        
        # Select intervention message based on category
        message = None
        for cat in detection.categories:
            if cat in INTERVENTION_TEMPLATES:
                import random
                message = random.choice(INTERVENTION_TEMPLATES[cat])
                break
        
        if not message:
            return None
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would post intervention: {message[:50]}...")
            return "dry_run"
        
        # Post as reply
        result = self._api_post("notes/create", {
            "i": self.intervention_token,
            "text": message,
            "replyId": detection.note_id,
            "visibility": "public"
        })
        
        if result:
            return result.get("createdNote", {}).get("id")
        return None
    
    def process_detection(self, detection: Detection) -> None:
        """Process a detection based on pipeline mode."""
        action = self.detector.get_action(detection.categories)
        detection.action_taken = action
        
        # Update stats
        self.stats.detections += 1
        for cat in detection.categories:
            self.stats.by_category[cat] = self.stats.by_category.get(cat, 0) + 1
        self.stats.by_severity[detection.severity] = \
            self.stats.by_severity.get(detection.severity, 0) + 1
        
        # Log detection
        logger.info(
            f"DETECTED [{detection.severity}] @{detection.username}: "
            f"{detection.categories} - {detection.text[:50]}..."
        )
        
        # Take action based on mode
        if self.mode == PipelineMode.INTERVENE and action == "intervene":
            intervention_id = self.create_intervention(detection)
            if intervention_id:
                detection.intervention_id = intervention_id
                self.stats.interventions += 1
                logger.info(f"  → Intervention posted: {intervention_id}")
        
        self.detections.append(detection)
    
    def run_monitor(self, duration: Optional[float] = None) -> None:
        """Run monitoring mode."""
        logger.info(f"Starting monitor mode (poll interval: {self.poll_interval}s)")
        
        last_id: Optional[str] = None
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Fetch new posts
                posts = self.fetch_recent_posts(limit=50, since_id=last_id)
                
                if posts:
                    last_id = posts[0].get("id")
                    
                    for post in posts:
                        self.stats.posts_scanned += 1
                        detection = self.analyze_post(post)
                        if detection:
                            self.process_detection(detection)
                
                # Wait before next poll
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
    
    def run_analyze_dataset(self, dataset_path: Path) -> None:
        """Analyze an existing dataset file."""
        logger.info(f"Analyzing dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.stats.posts_scanned += 1
                
                # Create pseudo-post for analysis
                post = {
                    "id": data.get("post_id", ""),
                    "text": data.get("text", ""),
                    "userId": data.get("user_id", ""),
                    "user": {"username": f"user_{data.get('user_id', '')}"}
                }
                
                detection = self.analyze_post(post)
                if detection:
                    self.process_detection(detection)
    
    def export_detections(self, output_path: Path) -> None:
        """Export detections to JSONL file."""
        with open(output_path, 'w') as f:
            for det in self.detections:
                record = {
                    "note_id": det.note_id,
                    "text": det.text,
                    "user_id": det.user_id,
                    "username": det.username,
                    "categories": det.categories,
                    "severity": det.severity,
                    "confidence": det.confidence,
                    "timestamp": det.timestamp.isoformat(),
                    "action_taken": det.action_taken,
                    "intervention_id": det.intervention_id
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Exported {len(self.detections)} detections to {output_path}")
    
    def print_summary(self) -> None:
        """Print pipeline summary."""
        print("\n" + "="*60)
        print("DETECTION PIPELINE SUMMARY")
        print("="*60)
        print(f"Posts scanned: {self.stats.posts_scanned}")
        print(f"Detections: {self.stats.detections}")
        print(f"Interventions: {self.stats.interventions}")
        
        if self.stats.by_category:
            print("\nBy Category:")
            for cat, count in sorted(self.stats.by_category.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
        
        if self.stats.by_severity:
            print("\nBy Severity:")
            for sev, count in sorted(self.stats.by_severity.items()):
                print(f"  Level {sev}: {count}")
        
        print("="*60)
    
    def run(
        self,
        dataset_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        duration: Optional[float] = None
    ) -> PipelineStats:
        """Run the pipeline."""
        if self.mode == PipelineMode.ANALYZE and dataset_path:
            self.run_analyze_dataset(dataset_path)
        elif self.mode in [PipelineMode.MONITOR, PipelineMode.INTERVENE]:
            self.run_monitor(duration=duration)
        
        # Export results
        if output_path:
            self.export_detections(output_path)
        
        self.print_summary()
        return self.stats


# =============================================================================
# INTEGRATION WITH BOTSOCIAL UPLOADER
# =============================================================================

def create_detection_dataset(
    detections: list[Detection],
    output_path: Path,
    include_interventions: bool = True
) -> None:
    """Create a dataset from detections for upload.
    
    This creates a JSONL file compatible with botsocial_uploader.py
    """
    posts = []
    
    for i, det in enumerate(detections):
        # Original detected post (for reference)
        post = {
            "post_id": f"det_{i}",
            "thread_id": f"det_{i}",
            "user_id": det.user_id,
            "parent_id": None,
            "timestamp": str(i),
            "text": det.text,
            "category_labels": det.categories,
            "detection_metadata": {
                "original_note_id": det.note_id,
                "severity": det.severity,
                "confidence": det.confidence,
                "action": det.action_taken
            }
        }
        posts.append(post)
        
        # Add intervention as reply if present
        if include_interventions and det.intervention_id:
            intervention = {
                "post_id": f"int_{i}",
                "thread_id": f"det_{i}",
                "user_id": "intervention_bot",
                "parent_id": f"det_{i}",
                "timestamp": str(i),
                "text": f"[Intervention for {det.categories}]",
                "category_labels": ["intervention"],
                "detection_metadata": {
                    "intervention_note_id": det.intervention_id,
                    "target_categories": det.categories
                }
            }
            posts.append(intervention)
    
    with open(output_path, 'w') as f:
        for post in posts:
            f.write(json.dumps(post) + "\n")
    
    logger.info(f"Created detection dataset: {output_path} ({len(posts)} records)")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detection → Upload Pipeline for BotSocial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor BotSocial for harmful content (dry run)
  %(prog)s --credentials tokens.json --mode monitor --dry-run

  # Monitor and intervene on harmful content
  %(prog)s --credentials tokens.json --mode intervene --intervention-user moderator

  # Analyze existing dataset
  %(prog)s --dataset data.jsonl --mode analyze --output detections.jsonl

  # Monitor for 1 hour then export
  %(prog)s --credentials tokens.json --mode monitor --duration 3600 --output detections.jsonl
        """
    )
    
    # Input/Output
    parser.add_argument("--credentials", type=Path, 
                       default=Path("botsocial_tokens.json"),
                       help="Credentials file")
    parser.add_argument("--dataset", type=Path, help="Dataset to analyze")
    parser.add_argument("--output", type=Path, help="Output file for detections")
    
    # Mode
    parser.add_argument("--mode", type=str, default="monitor",
                       choices=["monitor", "intervene", "analyze"],
                       help="Pipeline mode")
    
    # Options
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't post interventions")
    parser.add_argument("--intervention-user", type=str,
                       help="Username for posting interventions")
    parser.add_argument("--severity-threshold", type=int, default=2,
                       help="Minimum severity to flag (0-5)")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                       help="Seconds between polls")
    parser.add_argument("--duration", type=float,
                       help="Run duration in seconds (monitor mode)")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DetectionPipeline(
        credentials_path=args.credentials,
        mode=PipelineMode(args.mode),
        intervention_user=args.intervention_user,
        dry_run=args.dry_run,
        severity_threshold=args.severity_threshold,
        poll_interval=args.poll_interval
    )
    
    # Run
    pipeline.run(
        dataset_path=args.dataset,
        output_path=args.output,
        duration=args.duration
    )


if __name__ == "__main__":
    main()

