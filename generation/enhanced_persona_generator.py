"""Enhanced persona generator with trajectory, fluency, and contamination.

This module extends the base persona generation to add:
- Radicalization trajectory stages
- Slang fluency levels
- Cross-archetype contamination
- Benign post rates
- Demographic variation

These create within-class variation that makes classification more challenging
while maintaining the label token flow for RagImputer.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generation.trajectory_config import (
    TRAJECTORY_STAGES,
    SLANG_FLUENCY_LEVELS,
    sample_trajectory_stage,
    sample_slang_fluency,
    get_trajectory_config,
    get_fluency_config,
)
from generation.contamination_config import (
    sample_contamination,
    ContaminationConfig,
)


# =============================================================================
# Enhanced Persona Schema
# =============================================================================

@dataclass
class EnhancedPersonaConfig:
    """Extended persona configuration with all improvements."""
    
    # Base fields (from original PersonaConfig)
    user_id: int
    username: str
    name: str
    primary_label: str
    
    # Original persona fields
    bio: str = ""
    persona_prompt_post: str = ""
    persona_prompt_comment: str = ""
    persona_goal: str = ""
    persona_style_quirks: str = ""
    
    # NEW: Trajectory and fluency
    trajectory_stage: str = "active"
    slang_fluency: str = "fluent"
    
    # NEW: Contamination
    secondary_archetype: Optional[str] = None
    secondary_influence: float = 0.0
    
    # NEW: Content behavior modifiers
    benign_post_rate: float = 0.15  # Base rate of benign posts for toxic personas
    token_density_multiplier: float = 1.0
    aggression_multiplier: float = 1.0
    
    # NEW: Demographic profile
    demographic_profile: str = "default"
    
    # NEW: Obfuscation settings
    obfuscation_intensity: float = 0.3
    
    # Lexicon and style (existing)
    allowed_label_tokens: List[str] = field(default_factory=list)
    lexicon_required: List[str] = field(default_factory=list)
    lexicon_optional: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "name": self.name,
            "primary_label": self.primary_label,
            "bio": self.bio,
            "persona_prompt_post": self.persona_prompt_post,
            "persona_prompt_comment": self.persona_prompt_comment,
            "persona_goal": self.persona_goal,
            "persona_style_quirks": self.persona_style_quirks,
            "trajectory_stage": self.trajectory_stage,
            "slang_fluency": self.slang_fluency,
            "secondary_archetype": self.secondary_archetype or "",
            "secondary_influence": self.secondary_influence,
            "benign_post_rate": self.benign_post_rate,
            "token_density_multiplier": self.token_density_multiplier,
            "aggression_multiplier": self.aggression_multiplier,
            "demographic_profile": self.demographic_profile,
            "obfuscation_intensity": self.obfuscation_intensity,
            "allowed_label_tokens": ",".join(self.allowed_label_tokens),
            "lexicon_required": ",".join(self.lexicon_required),
            "lexicon_optional": ",".join(self.lexicon_optional),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedPersonaConfig":
        """Create from dictionary (CSV row)."""
        def parse_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value:
                return [x.strip() for x in value.split(",") if x.strip()]
            return []
        
        return cls(
            user_id=int(data.get("user_id", 0)),
            username=str(data.get("username", "")),
            name=str(data.get("name", "")),
            primary_label=str(data.get("primary_label", "benign")),
            bio=str(data.get("bio", "")),
            persona_prompt_post=str(data.get("persona_prompt_post", "")),
            persona_prompt_comment=str(data.get("persona_prompt_comment", "")),
            persona_goal=str(data.get("persona_goal", "")),
            persona_style_quirks=str(data.get("persona_style_quirks", "")),
            trajectory_stage=str(data.get("trajectory_stage", "active")),
            slang_fluency=str(data.get("slang_fluency", "fluent")),
            secondary_archetype=data.get("secondary_archetype") or None,
            secondary_influence=float(data.get("secondary_influence", 0.0)),
            benign_post_rate=float(data.get("benign_post_rate", 0.15)),
            token_density_multiplier=float(data.get("token_density_multiplier", 1.0)),
            aggression_multiplier=float(data.get("aggression_multiplier", 1.0)),
            demographic_profile=str(data.get("demographic_profile", "default")),
            obfuscation_intensity=float(data.get("obfuscation_intensity", 0.3)),
            allowed_label_tokens=parse_list(data.get("allowed_label_tokens")),
            lexicon_required=parse_list(data.get("lexicon_required")),
            lexicon_optional=parse_list(data.get("lexicon_optional")),
        )


# =============================================================================
# Demographic Profiles
# =============================================================================

DEMOGRAPHIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "gen_z_urban": {
        "age_range": (16, 25),
        "style_modifiers": {
            "uses_emoji": True,
            "uses_slang": True,
            "formal_level": 0.2,
            "capitalization": "mixed",
        },
        "vocabulary_hints": [
            "Use current internet slang",
            "Reference memes and TikTok",
            "Shorter sentences, more fragments",
        ],
    },
    "millennial_suburban": {
        "age_range": (26, 40),
        "style_modifiers": {
            "uses_emoji": True,
            "uses_slang": False,
            "formal_level": 0.5,
            "capitalization": "normal",
        },
        "vocabulary_hints": [
            "Mix of internet and professional language",
            "Reference work and life balance",
        ],
    },
    "gen_x_rural": {
        "age_range": (41, 55),
        "style_modifiers": {
            "uses_emoji": False,
            "uses_slang": False,
            "formal_level": 0.6,
            "capitalization": "normal",
        },
        "vocabulary_hints": [
            "More formal phrasing",
            "Reference traditional values",
            "Longer, complete sentences",
        ],
    },
    "boomer_conservative": {
        "age_range": (56, 75),
        "style_modifiers": {
            "uses_emoji": False,
            "uses_slang": False,
            "formal_level": 0.7,
            "capitalization": "normal",
        },
        "vocabulary_hints": [
            "Formal language",
            "Reference past experiences",
            "Complete sentences with proper punctuation",
        ],
    },
    "terminally_online": {
        "age_range": (18, 35),
        "style_modifiers": {
            "uses_emoji": True,
            "uses_slang": True,
            "formal_level": 0.1,
            "capitalization": "lowercase",
        },
        "vocabulary_hints": [
            "Heavy internet slang",
            "Irony and sarcasm",
            "Reference niche internet culture",
            "No capitalization",
        ],
    },
    "default": {
        "age_range": (20, 45),
        "style_modifiers": {
            "uses_emoji": False,
            "uses_slang": False,
            "formal_level": 0.5,
            "capitalization": "normal",
        },
        "vocabulary_hints": [],
    },
}

# Demographic distribution by archetype
ARCHETYPE_DEMOGRAPHIC_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "incel_misogyny": {
        "gen_z_urban": 0.3,
        "millennial_suburban": 0.2,
        "terminally_online": 0.5,
    },
    "alpha": {
        "millennial_suburban": 0.5,
        "gen_z_urban": 0.3,
        "terminally_online": 0.2,
    },
    "conspiracy": {
        "gen_x_rural": 0.3,
        "boomer_conservative": 0.3,
        "millennial_suburban": 0.2,
        "terminally_online": 0.2,
    },
    "misinfo": {
        "boomer_conservative": 0.4,
        "gen_x_rural": 0.3,
        "millennial_suburban": 0.3,
    },
    "ed_risk": {
        "gen_z_urban": 0.5,
        "terminally_online": 0.3,
        "millennial_suburban": 0.2,
    },
    "pro_ana": {
        "gen_z_urban": 0.4,
        "terminally_online": 0.4,
        "millennial_suburban": 0.2,
    },
    "trad": {
        "millennial_suburban": 0.4,
        "gen_x_rural": 0.3,
        "boomer_conservative": 0.2,
        "gen_z_urban": 0.1,
    },
    "gamergate": {
        "terminally_online": 0.5,
        "gen_z_urban": 0.3,
        "millennial_suburban": 0.2,
    },
    "extremist": {
        "terminally_online": 0.4,
        "gen_z_urban": 0.3,
        "millennial_suburban": 0.2,
        "gen_x_rural": 0.1,
    },
    "hate_speech": {
        "terminally_online": 0.3,
        "gen_z_urban": 0.2,
        "millennial_suburban": 0.2,
        "gen_x_rural": 0.2,
        "boomer_conservative": 0.1,
    },
    "bullying": {
        "gen_z_urban": 0.4,
        "terminally_online": 0.4,
        "millennial_suburban": 0.2,
    },
    "benign": {
        "gen_z_urban": 0.25,
        "millennial_suburban": 0.25,
        "gen_x_rural": 0.2,
        "boomer_conservative": 0.15,
        "terminally_online": 0.15,
    },
    "recovery_support": {
        "gen_z_urban": 0.3,
        "millennial_suburban": 0.3,
        "terminally_online": 0.2,
        "gen_x_rural": 0.2,
    },
}


def sample_demographic(
    archetype: str,
    rng: Optional[random.Random] = None,
) -> str:
    """Sample a demographic profile for an archetype."""
    rng = rng or random.Random()
    dist = ARCHETYPE_DEMOGRAPHIC_DISTRIBUTION.get(archetype, {"default": 1.0})
    profiles = list(dist.keys())
    weights = [dist[p] for p in profiles]
    return rng.choices(profiles, weights=weights, k=1)[0]


# =============================================================================
# Benign Post Rate Configuration
# =============================================================================

# Base benign post rates by archetype
# These are the probability that a toxic persona posts benign content
BASE_BENIGN_POST_RATES: Dict[str, float] = {
    "incel_misogyny": 0.15,     # Some normal posts about hobbies, games
    "alpha": 0.20,              # Business, fitness content
    "conspiracy": 0.10,         # Rarely off-topic
    "misinfo": 0.15,            # Some normal news sharing
    "ed_risk": 0.25,            # Lots of normal food/exercise content
    "pro_ana": 0.15,            # Some normal content
    "trad": 0.30,               # Aesthetic, cooking, family content
    "gamergate": 0.25,          # Gaming content that's not toxic
    "extremist": 0.05,          # Rarely benign
    "hate_speech": 0.08,        # Rarely benign
    "bullying": 0.15,           # Normal interactions between attacks
    "benign": 1.0,              # Always benign
    "recovery_support": 0.80,   # Mostly supportive, some venting
}


# =============================================================================
# Enhanced Persona Generator
# =============================================================================

class EnhancedPersonaGenerator:
    """Generates personas with trajectory, fluency, and contamination."""
    
    def __init__(
        self,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.seed = seed
    
    def enhance_persona(
        self,
        base_persona: Dict[str, Any],
    ) -> EnhancedPersonaConfig:
        """Enhance a base persona with trajectory, fluency, contamination.
        
        Args:
            base_persona: Dictionary with base persona fields
        
        Returns:
            EnhancedPersonaConfig with all improvements
        """
        archetype = base_persona.get("primary_label", "benign")
        
        # Sample trajectory and fluency
        trajectory = sample_trajectory_stage(archetype, self.rng)
        fluency = sample_slang_fluency(trajectory, self.rng)
        
        # Sample contamination
        contamination = sample_contamination(archetype, self.rng)
        
        # Sample demographic
        demographic = sample_demographic(archetype, self.rng)
        
        # Get trajectory config for modifiers
        traj_config = get_trajectory_config(trajectory)
        
        # Calculate benign post rate
        base_rate = BASE_BENIGN_POST_RATES.get(archetype, 0.15)
        benign_rate = max(0.0, min(1.0, base_rate + traj_config.benign_post_boost))
        
        # Calculate obfuscation intensity based on trajectory
        obfuscation = {
            "curious": 0.1,
            "active": 0.3,
            "entrenched": 0.5,
            "disillusioned": 0.2,
        }.get(trajectory, 0.3)
        
        return EnhancedPersonaConfig(
            user_id=int(base_persona.get("user_id", 0)),
            username=str(base_persona.get("username", "")),
            name=str(base_persona.get("name", "")),
            primary_label=archetype,
            bio=str(base_persona.get("bio", "")),
            persona_prompt_post=str(base_persona.get("persona_prompt_post", "")),
            persona_prompt_comment=str(base_persona.get("persona_prompt_comment", "")),
            persona_goal=str(base_persona.get("persona_goal", "")),
            persona_style_quirks=str(base_persona.get("persona_style_quirks", "")),
            trajectory_stage=trajectory,
            slang_fluency=fluency,
            secondary_archetype=contamination.secondary_archetype,
            secondary_influence=contamination.secondary_influence,
            benign_post_rate=benign_rate,
            token_density_multiplier=traj_config.token_density_multiplier,
            aggression_multiplier=traj_config.aggression_multiplier,
            demographic_profile=demographic,
            obfuscation_intensity=obfuscation,
            allowed_label_tokens=base_persona.get("allowed_label_tokens", []),
            lexicon_required=base_persona.get("lexicon_required", []),
            lexicon_optional=base_persona.get("lexicon_optional", []),
        )
    
    def enhance_personas_csv(
        self,
        input_path: Path,
        output_path: Path,
    ) -> int:
        """Enhance all personas in a CSV file.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
        
        Returns:
            Number of personas enhanced
        """
        enhanced = []
        
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                enhanced_persona = self.enhance_persona(row)
                enhanced.append(enhanced_persona.to_dict())
        
        if not enhanced:
            return 0
        
        # Write enhanced CSV
        fieldnames = list(enhanced[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced)
        
        return len(enhanced)


# =============================================================================
# Convenience Functions
# =============================================================================

def enhance_personas(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
) -> int:
    """Enhance personas with trajectory, fluency, contamination.
    
    Args:
        input_path: Path to input personas CSV
        output_path: Path to output enhanced CSV
        seed: Random seed
    
    Returns:
        Number of personas enhanced
    """
    generator = EnhancedPersonaGenerator(seed=seed)
    return generator.enhance_personas_csv(input_path, output_path)

