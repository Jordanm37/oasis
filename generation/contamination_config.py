"""Cross-archetype contamination configuration.

Real toxic users often exhibit traits from multiple communities. This module
defines which archetypes commonly co-occur and how to blend their behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random


# =============================================================================
# Contamination Matrix
# =============================================================================

# Format: {primary_archetype: [(secondary_archetype, probability), ...]}
# Probability is the chance that a persona of primary_archetype will also
# exhibit traits from secondary_archetype
CONTAMINATION_MATRIX: Dict[str, List[Tuple[str, float]]] = {
    "incel_misogyny": [
        ("alpha", 0.15),           # Some incels adopt alpha rhetoric
        ("conspiracy", 0.10),      # Blackpill as conspiracy
        ("extremist", 0.08),       # Radicalization pipeline
        ("hate_speech", 0.12),     # General misogyny
        ("bullying", 0.10),        # Targeting individuals
    ],
    "alpha": [
        ("incel_misogyny", 0.20),  # Shared misogyny
        ("trad", 0.15),            # Traditional gender roles
        ("gamergate", 0.10),       # Anti-feminist alignment
    ],
    "conspiracy": [
        ("misinfo", 0.40),         # Strong overlap
        ("extremist", 0.15),       # Radicalization
        ("trad", 0.10),            # Traditional values conspiracy
    ],
    "misinfo": [
        ("conspiracy", 0.35),      # Strong overlap
        ("trad", 0.10),            # Health misinformation
    ],
    "ed_risk": [
        ("pro_ana", 0.30),         # Strong overlap
        ("bullying", 0.10),        # Body shaming
    ],
    "pro_ana": [
        ("ed_risk", 0.35),         # Strong overlap
        ("bullying", 0.20),        # Meanspo
    ],
    "trad": [
        ("alpha", 0.15),           # Gender role overlap
        ("conspiracy", 0.10),      # Anti-modern conspiracy
        ("hate_speech", 0.08),     # Ethnonationalism
    ],
    "gamergate": [
        ("bullying", 0.25),        # Harassment campaigns
        ("hate_speech", 0.15),     # Identity attacks
        ("incel_misogyny", 0.12),  # Shared misogyny
        ("conspiracy", 0.08),      # Media conspiracy
    ],
    "extremist": [
        ("hate_speech", 0.40),     # Strong overlap
        ("conspiracy", 0.25),      # Radicalization narratives
        ("incel_misogyny", 0.10),  # Pipeline
    ],
    "hate_speech": [
        ("extremist", 0.35),       # Strong overlap
        ("conspiracy", 0.20),      # Replacement theory
        ("bullying", 0.15),        # Targeted harassment
    ],
    "bullying": [
        ("gamergate", 0.15),       # Harassment overlap
        ("hate_speech", 0.12),     # Identity-based bullying
        ("incel_misogyny", 0.08),  # Gendered harassment
    ],
    "benign": [
        # Benign personas don't contaminate
    ],
    "recovery_support": [
        # Recovery personas don't contaminate with harmful content
    ],
}


@dataclass
class ContaminationConfig:
    """Configuration for a persona's secondary archetype influence."""
    
    primary_archetype: str
    secondary_archetype: Optional[str]
    secondary_influence: float  # 0.0-1.0, how much secondary affects content
    
    # Token blending
    secondary_token_probability: float  # Probability of using secondary tokens
    
    # Vocabulary blending
    vocabulary_blend_ratio: float  # Ratio of secondary vocabulary to include


def sample_contamination(
    primary_archetype: str,
    rng: Optional[random.Random] = None,
    force_contamination: bool = False,
) -> ContaminationConfig:
    """Sample a contamination configuration for a persona.
    
    Args:
        primary_archetype: The persona's primary archetype
        rng: Random number generator
        force_contamination: If True, always apply contamination
    
    Returns:
        ContaminationConfig with secondary archetype (or None)
    """
    rng = rng or random.Random()
    
    candidates = CONTAMINATION_MATRIX.get(primary_archetype, [])
    if not candidates:
        return ContaminationConfig(
            primary_archetype=primary_archetype,
            secondary_archetype=None,
            secondary_influence=0.0,
            secondary_token_probability=0.0,
            vocabulary_blend_ratio=0.0,
        )
    
    # Roll for each potential contamination
    selected = None
    for secondary, prob in candidates:
        if force_contamination or rng.random() < prob:
            selected = secondary
            break
    
    if not selected:
        return ContaminationConfig(
            primary_archetype=primary_archetype,
            secondary_archetype=None,
            secondary_influence=0.0,
            secondary_token_probability=0.0,
            vocabulary_blend_ratio=0.0,
        )
    
    # Determine influence strength (0.1 to 0.4)
    influence = rng.uniform(0.1, 0.4)
    
    return ContaminationConfig(
        primary_archetype=primary_archetype,
        secondary_archetype=selected,
        secondary_influence=influence,
        secondary_token_probability=influence * 0.5,  # Half as likely as primary
        vocabulary_blend_ratio=influence * 0.3,       # Subtle vocabulary blending
    )


# =============================================================================
# Post Type Distribution
# =============================================================================

@dataclass
class PostTypeConfig:
    """Configuration for a post type."""
    
    name: str
    description: str
    
    # Content structure
    typical_length: str  # "short", "medium", "long"
    has_links: bool
    has_images: bool
    has_quotes: bool
    
    # Token behavior
    token_placement: str  # "inline", "emphasis", "conclusion"
    
    # Example templates
    templates: List[str] = field(default_factory=list)


POST_TYPES: Dict[str, PostTypeConfig] = {
    "rant": PostTypeConfig(
        name="rant",
        description="Emotional venting, stream of consciousness",
        typical_length="long",
        has_links=False,
        has_images=False,
        has_quotes=False,
        token_placement="inline",
        templates=[
            "I'm so sick of {topic}. Every day it's the same {frustration}...",
            "Why does {target} always {behavior}? It's {emotion}...",
        ],
    ),
    "observation": PostTypeConfig(
        name="observation",
        description="Pointing out a pattern or example",
        typical_length="medium",
        has_links=False,
        has_images=False,
        has_quotes=False,
        token_placement="emphasis",
        templates=[
            "Just noticed {observation}. Classic example of {pattern}.",
            "Look at this. {example}. This is exactly what {ideology} predicted.",
        ],
    ),
    "news_reaction": PostTypeConfig(
        name="news_reaction",
        description="Reacting to current events",
        typical_length="medium",
        has_links=True,
        has_images=False,
        has_quotes=True,
        token_placement="conclusion",
        templates=[
            "Can you believe this? {news_summary}. This proves {conclusion}.",
            "They're not even hiding it anymore. {event} shows {interpretation}.",
        ],
    ),
    "advice": PostTypeConfig(
        name="advice",
        description="Giving guidance to others",
        typical_length="medium",
        has_links=False,
        has_images=False,
        has_quotes=False,
        token_placement="inline",
        templates=[
            "Here's what you need to understand: {advice}. Don't {warning}.",
            "Listen, {addressee}. The key is to {action}. Otherwise {consequence}.",
        ],
    ),
    "personal_story": PostTypeConfig(
        name="personal_story",
        description="Sharing personal experience",
        typical_length="long",
        has_links=False,
        has_images=False,
        has_quotes=False,
        token_placement="conclusion",
        templates=[
            "This happened to me today. {story}. It just confirms {conclusion}.",
            "Let me tell you about {experience}. {narrative}. {lesson_learned}.",
        ],
    ),
    "meme_response": PostTypeConfig(
        name="meme_response",
        description="Short, punchy, often referencing memes",
        typical_length="short",
        has_links=False,
        has_images=True,
        has_quotes=False,
        token_placement="inline",
        templates=[
            "{meme_phrase}",
            "Literally {meme_reference}",
            "This is so {adjective}. {meme_response}.",
        ],
    ),
    "question": PostTypeConfig(
        name="question",
        description="Asking the community",
        typical_length="short",
        has_links=False,
        has_images=False,
        has_quotes=False,
        token_placement="inline",
        templates=[
            "Is anyone else noticing {observation}? What does {community} think?",
            "Genuine question: {question}? I'm starting to think {hypothesis}.",
        ],
    ),
    "call_to_action": PostTypeConfig(
        name="call_to_action",
        description="Rallying others to do something",
        typical_length="medium",
        has_links=True,
        has_images=False,
        has_quotes=False,
        token_placement="emphasis",
        templates=[
            "We need to {action}. {reason}. Who's with me?",
            "Stop {behavior}. Start {alternative}. It's time to {call}.",
        ],
    ),
}

# Post type distribution by archetype
POST_TYPE_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "incel_misogyny": {
        "rant": 0.30,
        "observation": 0.25,
        "personal_story": 0.20,
        "meme_response": 0.15,
        "question": 0.10,
    },
    "alpha": {
        "advice": 0.40,
        "observation": 0.25,
        "personal_story": 0.20,
        "rant": 0.10,
        "meme_response": 0.05,
    },
    "conspiracy": {
        "news_reaction": 0.35,
        "observation": 0.25,
        "question": 0.15,
        "call_to_action": 0.15,
        "rant": 0.10,
    },
    "misinfo": {
        "news_reaction": 0.40,
        "call_to_action": 0.20,
        "observation": 0.20,
        "question": 0.15,
        "advice": 0.05,
    },
    "ed_risk": {
        "personal_story": 0.30,
        "advice": 0.25,
        "question": 0.20,
        "observation": 0.15,
        "rant": 0.10,
    },
    "pro_ana": {
        "advice": 0.35,
        "call_to_action": 0.20,
        "personal_story": 0.20,
        "observation": 0.15,
        "rant": 0.10,
    },
    "trad": {
        "observation": 0.30,
        "personal_story": 0.25,
        "advice": 0.20,
        "news_reaction": 0.15,
        "meme_response": 0.10,
    },
    "gamergate": {
        "news_reaction": 0.30,
        "rant": 0.25,
        "call_to_action": 0.20,
        "meme_response": 0.15,
        "observation": 0.10,
    },
    "extremist": {
        "call_to_action": 0.35,
        "rant": 0.25,
        "observation": 0.20,
        "news_reaction": 0.15,
        "meme_response": 0.05,
    },
    "hate_speech": {
        "rant": 0.35,
        "observation": 0.25,
        "news_reaction": 0.20,
        "meme_response": 0.15,
        "call_to_action": 0.05,
    },
    "bullying": {
        "meme_response": 0.35,
        "observation": 0.25,
        "rant": 0.20,
        "question": 0.15,
        "personal_story": 0.05,
    },
    "benign": {
        "personal_story": 0.30,
        "question": 0.25,
        "observation": 0.20,
        "advice": 0.15,
        "meme_response": 0.10,
    },
    "recovery_support": {
        "personal_story": 0.40,
        "advice": 0.30,
        "question": 0.20,
        "observation": 0.10,
    },
}


def sample_post_type(
    archetype: str,
    rng: Optional[random.Random] = None,
) -> str:
    """Sample a post type for an archetype."""
    rng = rng or random.Random()
    dist = POST_TYPE_DISTRIBUTION.get(archetype, {"observation": 1.0})
    types = list(dist.keys())
    weights = [dist[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def get_post_type_config(post_type: str) -> PostTypeConfig:
    """Get post type configuration."""
    return POST_TYPES.get(post_type, POST_TYPES["observation"])

