"""Radicalization trajectory and slang fluency configuration.

This module defines the trajectory stages and fluency levels that affect
how personas generate content. These create within-class variation that
makes classification more challenging and realistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random


# =============================================================================
# Trajectory Stages
# =============================================================================

@dataclass
class TrajectoryStage:
    """Configuration for a radicalization trajectory stage."""
    
    name: str
    description: str
    
    # Token emission modifiers
    token_density_multiplier: float  # 1.0 = normal, 0.5 = half, 1.5 = 50% more
    skip_token_probability: float    # Probability of skipping token entirely
    
    # Vocabulary modifiers
    slang_adoption_rate: float       # 0.0-1.0, how much community slang is used
    explanation_rate: float          # How often terms are explained
    aggression_multiplier: float     # Affects tone intensity
    
    # Behavioral modifiers
    post_length_multiplier: float    # Affects verbosity
    benign_post_boost: float         # Added to base benign post rate
    
    # Content characteristics
    characteristic_phrases: List[str] = field(default_factory=list)


TRAJECTORY_STAGES: Dict[str, TrajectoryStage] = {
    "curious": TrajectoryStage(
        name="curious",
        description="Just discovered the community, testing the language",
        token_density_multiplier=0.5,
        skip_token_probability=0.3,
        slang_adoption_rate=0.2,
        explanation_rate=0.4,
        aggression_multiplier=0.6,
        post_length_multiplier=0.8,
        benign_post_boost=0.2,
        characteristic_phrases=[
            "I've been reading about...",
            "Is this what they call...",
            "I'm starting to think...",
            "Someone explained to me that...",
        ],
    ),
    "active": TrajectoryStage(
        name="active",
        description="Regular participant, fluent in community language",
        token_density_multiplier=1.0,
        skip_token_probability=0.0,
        slang_adoption_rate=0.7,
        explanation_rate=0.1,
        aggression_multiplier=1.0,
        post_length_multiplier=1.0,
        benign_post_boost=0.0,
        characteristic_phrases=[
            "We all know...",
            "Classic example of...",
            "This is exactly what...",
        ],
    ),
    "entrenched": TrajectoryStage(
        name="entrenched",
        description="True believer, aggressive enforcement of worldview",
        token_density_multiplier=1.3,
        skip_token_probability=0.0,
        slang_adoption_rate=0.95,
        explanation_rate=0.0,
        aggression_multiplier=1.4,
        post_length_multiplier=1.2,
        benign_post_boost=-0.1,  # Less benign content
        characteristic_phrases=[
            "Wake up.",
            "It's over.",
            "They will never...",
            "This is war.",
        ],
    ),
    "disillusioned": TrajectoryStage(
        name="disillusioned",
        description="Questioning, mixed signals, potential exit",
        token_density_multiplier=0.7,
        skip_token_probability=0.2,
        slang_adoption_rate=0.5,
        explanation_rate=0.2,
        aggression_multiplier=0.7,
        post_length_multiplier=1.1,
        benign_post_boost=0.15,
        characteristic_phrases=[
            "I used to think...",
            "But lately...",
            "Maybe it's not...",
            "I'm not sure anymore...",
        ],
    ),
}


# =============================================================================
# Slang Fluency Levels
# =============================================================================

@dataclass
class SlangFluencyLevel:
    """Configuration for slang fluency level."""
    
    name: str
    description: str
    
    # Vocabulary behavior
    use_abbreviations: bool
    explain_terms: bool
    misuse_probability: float  # Probability of using terms incorrectly
    
    # Style modifiers
    insider_reference_rate: float  # How often to make insider references
    meme_usage_rate: float         # How often to use community memes
    
    # Examples of how terms are used at this level
    example_patterns: List[str] = field(default_factory=list)


SLANG_FLUENCY_LEVELS: Dict[str, SlangFluencyLevel] = {
    "native": SlangFluencyLevel(
        name="native",
        description="Uses terms naturally without explanation, like a long-time member",
        use_abbreviations=True,
        explain_terms=False,
        misuse_probability=0.0,
        insider_reference_rate=0.8,
        meme_usage_rate=0.6,
        example_patterns=[
            "LDAR",  # No explanation
            "brutal blackpill",  # Natural usage
            "gigamogged",  # Current slang
        ],
    ),
    "fluent": SlangFluencyLevel(
        name="fluent",
        description="Comfortable but occasionally over-explains or uses older terms",
        use_abbreviations=True,
        explain_terms=False,
        misuse_probability=0.05,
        insider_reference_rate=0.5,
        meme_usage_rate=0.3,
        example_patterns=[
            "LDAR (lay down and rot)",  # Occasional explanation
            "the blackpill",  # Slightly formal
        ],
    ),
    "learning": SlangFluencyLevel(
        name="learning",
        description="Testing terms, sometimes misuses or over-explains",
        use_abbreviations=False,
        explain_terms=True,
        misuse_probability=0.15,
        insider_reference_rate=0.2,
        meme_usage_rate=0.1,
        example_patterns=[
            "what they call the 'blackpill'",
            "I think this is called mogging?",
            "laying down and rotting (LDAR)",
        ],
    ),
    "outsider": SlangFluencyLevel(
        name="outsider",
        description="Describes concepts without community-specific terms",
        use_abbreviations=False,
        explain_terms=False,  # Doesn't know terms to explain
        misuse_probability=0.0,  # Doesn't use terms at all
        insider_reference_rate=0.0,
        meme_usage_rate=0.0,
        example_patterns=[
            "I've given up on dating",  # No blackpill reference
            "attractive men get everything",  # No Chad reference
            "women only want...",  # No foid reference
        ],
    ),
}


# =============================================================================
# Distribution Configuration per Archetype
# =============================================================================

# Default trajectory distribution for each archetype
# Format: {archetype: {stage: probability}}
DEFAULT_TRAJECTORY_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "incel_misogyny": {
        "curious": 0.10,
        "active": 0.50,
        "entrenched": 0.35,
        "disillusioned": 0.05,
    },
    "alpha": {
        "curious": 0.05,
        "active": 0.60,
        "entrenched": 0.30,
        "disillusioned": 0.05,
    },
    "conspiracy": {
        "curious": 0.15,
        "active": 0.45,
        "entrenched": 0.30,
        "disillusioned": 0.10,
    },
    "misinfo": {
        "curious": 0.20,
        "active": 0.50,
        "entrenched": 0.20,
        "disillusioned": 0.10,
    },
    "ed_risk": {
        "curious": 0.15,
        "active": 0.50,
        "entrenched": 0.25,
        "disillusioned": 0.10,
    },
    "pro_ana": {
        "curious": 0.10,
        "active": 0.45,
        "entrenched": 0.40,
        "disillusioned": 0.05,
    },
    "trad": {
        "curious": 0.10,
        "active": 0.55,
        "entrenched": 0.30,
        "disillusioned": 0.05,
    },
    "gamergate": {
        "curious": 0.15,
        "active": 0.50,
        "entrenched": 0.30,
        "disillusioned": 0.05,
    },
    "extremist": {
        "curious": 0.05,
        "active": 0.40,
        "entrenched": 0.50,
        "disillusioned": 0.05,
    },
    "hate_speech": {
        "curious": 0.05,
        "active": 0.45,
        "entrenched": 0.45,
        "disillusioned": 0.05,
    },
    "bullying": {
        "curious": 0.10,
        "active": 0.60,
        "entrenched": 0.25,
        "disillusioned": 0.05,
    },
    "benign": {
        "curious": 0.0,
        "active": 1.0,
        "entrenched": 0.0,
        "disillusioned": 0.0,
    },
    "recovery_support": {
        "curious": 0.0,
        "active": 0.70,
        "entrenched": 0.0,
        "disillusioned": 0.30,  # Often coming from harmful communities
    },
}

# Default fluency distribution for each trajectory stage
# Format: {stage: {fluency: probability}}
DEFAULT_FLUENCY_BY_TRAJECTORY: Dict[str, Dict[str, float]] = {
    "curious": {
        "native": 0.0,
        "fluent": 0.1,
        "learning": 0.7,
        "outsider": 0.2,
    },
    "active": {
        "native": 0.3,
        "fluent": 0.5,
        "learning": 0.2,
        "outsider": 0.0,
    },
    "entrenched": {
        "native": 0.7,
        "fluent": 0.3,
        "learning": 0.0,
        "outsider": 0.0,
    },
    "disillusioned": {
        "native": 0.2,
        "fluent": 0.5,
        "learning": 0.2,
        "outsider": 0.1,
    },
}


def sample_trajectory_stage(
    archetype: str,
    rng: Optional[random.Random] = None,
    distribution: Optional[Dict[str, float]] = None,
) -> str:
    """Sample a trajectory stage for an archetype.
    
    Args:
        archetype: The persona archetype
        rng: Random number generator (uses global if None)
        distribution: Override distribution (uses default if None)
    
    Returns:
        Trajectory stage name
    """
    rng = rng or random.Random()
    dist = distribution or DEFAULT_TRAJECTORY_DISTRIBUTION.get(
        archetype, {"active": 1.0}
    )
    
    stages = list(dist.keys())
    weights = [dist[s] for s in stages]
    return rng.choices(stages, weights=weights, k=1)[0]


def sample_slang_fluency(
    trajectory_stage: str,
    rng: Optional[random.Random] = None,
    distribution: Optional[Dict[str, float]] = None,
) -> str:
    """Sample a slang fluency level based on trajectory stage.
    
    Args:
        trajectory_stage: The persona's trajectory stage
        rng: Random number generator
        distribution: Override distribution
    
    Returns:
        Fluency level name
    """
    rng = rng or random.Random()
    dist = distribution or DEFAULT_FLUENCY_BY_TRAJECTORY.get(
        trajectory_stage, {"fluent": 1.0}
    )
    
    levels = list(dist.keys())
    weights = [dist[l] for l in levels]
    return rng.choices(levels, weights=weights, k=1)[0]


def get_trajectory_config(stage_name: str) -> TrajectoryStage:
    """Get trajectory stage configuration."""
    return TRAJECTORY_STAGES.get(stage_name, TRAJECTORY_STAGES["active"])


def get_fluency_config(fluency_name: str) -> SlangFluencyLevel:
    """Get fluency level configuration."""
    return SLANG_FLUENCY_LEVELS.get(fluency_name, SLANG_FLUENCY_LEVELS["fluent"])

