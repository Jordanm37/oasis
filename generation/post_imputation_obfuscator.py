"""Post-imputation obfuscator for realistic evasion patterns.

This module applies obfuscation AFTER RagImputer has replaced label tokens,
targeting the actual harmful content rather than neutral words.

Key insight: The imputed text (slurs, threats, harmful phrases) is what
real users would obfuscate to evade detection - not random neutral words.

Flow:
  LLM generates → RagImputer replaces tokens → Obfuscator targets harmful spans
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Harmful Term Patterns (what to obfuscate)
# =============================================================================

# Terms that are commonly obfuscated in real harmful content
# These are the IMPUTED values that RagImputer produces
OBFUSCATION_TARGET_PATTERNS: Dict[str, List[str]] = {
    "incel_slang": [
        "foid", "femoid", "stacy", "chad", "blackpill", "incel", "mog", "cope",
        "rope", "ldar", "normie", "betabuxx", "hypergamy", "lookism",
    ],
    "slurs": [
        # Placeholder patterns - actual slurs would be in the static bank
        # These are patterns to match imputed content
    ],
    "threats": [
        "kill", "die", "death", "murder", "shoot", "stab", "hang", "rope",
        "end it", "kys", "neck yourself",
    ],
    "ed_terms": [
        "purge", "restrict", "fast", "binge", "thinspo", "meanspo", "proana",
        "goal weight", "ugw", "gw", "sw", "cw", "calorie",
    ],
    "hate_terms": [
        "vermin", "subhuman", "degenerate", "replace", "invasion",
    ],
}

# Compile patterns for efficient matching
_COMPILED_PATTERNS: Dict[str, re.Pattern] = {}

def _get_pattern(category: str) -> Optional[re.Pattern]:
    """Get or compile a regex pattern for a category."""
    if category not in _COMPILED_PATTERNS:
        terms = OBFUSCATION_TARGET_PATTERNS.get(category, [])
        if terms:
            # Case-insensitive word boundary matching
            pattern_str = r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b'
            _COMPILED_PATTERNS[category] = re.compile(pattern_str, re.IGNORECASE)
        else:
            _COMPILED_PATTERNS[category] = None
    return _COMPILED_PATTERNS[category]


# =============================================================================
# Obfuscation Strategies
# =============================================================================

@dataclass
class ObfuscationStrategy:
    """A strategy for obfuscating a specific term."""
    
    name: str
    probability: float  # Base probability of applying this strategy
    
    def apply(self, term: str, rng: random.Random) -> str:
        """Apply the obfuscation to a term."""
        raise NotImplementedError


class LeetSpeakStrategy(ObfuscationStrategy):
    """Replace letters with numbers/symbols."""
    
    SUBSTITUTIONS = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7'],
        'l': ['1'],
    }
    
    def __init__(self):
        super().__init__(name="leetspeak", probability=0.3)
    
    def apply(self, term: str, rng: random.Random) -> str:
        result = []
        for char in term:
            lower = char.lower()
            if lower in self.SUBSTITUTIONS and rng.random() < 0.6:
                replacement = rng.choice(self.SUBSTITUTIONS[lower])
                result.append(replacement if char.islower() else replacement.upper())
            else:
                result.append(char)
        return ''.join(result)


class AsteriskCensorStrategy(ObfuscationStrategy):
    """Partially censor with asterisks."""
    
    def __init__(self):
        super().__init__(name="asterisk", probability=0.25)
    
    def apply(self, term: str, rng: random.Random) -> str:
        if len(term) <= 2:
            return term
        # Keep first and last, censor middle
        middle_len = len(term) - 2
        return term[0] + '*' * middle_len + term[-1]


class SpaceInsertionStrategy(ObfuscationStrategy):
    """Insert spaces between letters."""
    
    def __init__(self):
        super().__init__(name="spaces", probability=0.15)
    
    def apply(self, term: str, rng: random.Random) -> str:
        return ' '.join(term)


class HomoglyphStrategy(ObfuscationStrategy):
    """Replace with similar-looking Unicode characters."""
    
    HOMOGLYPHS = {
        'a': 'а',  # Cyrillic
        'e': 'е',  # Cyrillic
        'o': 'о',  # Cyrillic
        'c': 'с',  # Cyrillic
        'p': 'р',  # Cyrillic
        'i': 'і',  # Cyrillic
    }
    
    def __init__(self):
        super().__init__(name="homoglyph", probability=0.1)
    
    def apply(self, term: str, rng: random.Random) -> str:
        result = []
        for char in term:
            lower = char.lower()
            if lower in self.HOMOGLYPHS and rng.random() < 0.5:
                result.append(self.HOMOGLYPHS[lower])
            else:
                result.append(char)
        return ''.join(result)


class PartialCensorStrategy(ObfuscationStrategy):
    """Censor just one or two characters."""
    
    def __init__(self):
        super().__init__(name="partial", probability=0.2)
    
    def apply(self, term: str, rng: random.Random) -> str:
        if len(term) <= 2:
            return term
        # Pick 1-2 positions to censor
        positions = list(range(1, len(term) - 1))  # Exclude first and last
        if not positions:
            return term
        num_censor = min(2, len(positions))
        censor_positions = set(rng.sample(positions, num_censor))
        result = []
        for i, char in enumerate(term):
            if i in censor_positions:
                result.append('*')
            else:
                result.append(char)
        return ''.join(result)


# All available strategies
STRATEGIES: List[ObfuscationStrategy] = [
    LeetSpeakStrategy(),
    AsteriskCensorStrategy(),
    SpaceInsertionStrategy(),
    HomoglyphStrategy(),
    PartialCensorStrategy(),
]


# =============================================================================
# Persona-Specific Obfuscation Config
# =============================================================================

@dataclass
class PersonaObfuscationConfig:
    """Obfuscation configuration for a persona."""
    
    # Whether this persona uses obfuscation at all
    enabled: bool = True
    
    # Base probability of obfuscating any given harmful term
    base_probability: float = 0.3
    
    # Which categories to target
    target_categories: List[str] = field(default_factory=lambda: [
        "incel_slang", "threats", "hate_terms"
    ])
    
    # Preferred strategies (weights)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "leetspeak": 1.0,
        "asterisk": 1.0,
        "partial": 0.5,
        "spaces": 0.3,
        "homoglyph": 0.2,
    })


# Default configs per archetype
ARCHETYPE_OBFUSCATION_CONFIGS: Dict[str, PersonaObfuscationConfig] = {
    # Archetypes that commonly obfuscate
    "incel_misogyny": PersonaObfuscationConfig(
        enabled=True,
        base_probability=0.35,
        target_categories=["incel_slang", "threats"],
        strategy_weights={"leetspeak": 1.5, "asterisk": 1.0, "partial": 0.5},
    ),
    "hate_speech": PersonaObfuscationConfig(
        enabled=True,
        base_probability=0.4,
        target_categories=["slurs", "hate_terms", "threats"],
        strategy_weights={"asterisk": 1.5, "leetspeak": 1.0, "homoglyph": 1.0},
    ),
    "extremist": PersonaObfuscationConfig(
        enabled=True,
        base_probability=0.45,
        target_categories=["threats", "hate_terms"],
        strategy_weights={"homoglyph": 1.5, "leetspeak": 1.0, "asterisk": 0.5},
    ),
    "pro_ana": PersonaObfuscationConfig(
        enabled=True,
        base_probability=0.5,  # High - ED content is heavily moderated
        target_categories=["ed_terms"],
        strategy_weights={"leetspeak": 1.5, "spaces": 1.0, "asterisk": 0.5},
    ),
    "bullying": PersonaObfuscationConfig(
        enabled=True,
        base_probability=0.25,
        target_categories=["threats"],
        strategy_weights={"partial": 1.0, "asterisk": 1.0},
    ),
    
    # Archetypes that rarely obfuscate (they use coded language instead)
    "conspiracy": PersonaObfuscationConfig(
        enabled=False,  # Uses dog whistles, not obfuscation
    ),
    "misinfo": PersonaObfuscationConfig(
        enabled=False,  # Presents as legitimate
    ),
    "trad": PersonaObfuscationConfig(
        enabled=False,  # Uses coded aesthetic language
    ),
    "alpha": PersonaObfuscationConfig(
        enabled=False,  # Lectures openly
    ),
    
    # Benign archetypes
    "benign": PersonaObfuscationConfig(enabled=False),
    "recovery_support": PersonaObfuscationConfig(enabled=False),
}


# =============================================================================
# Main Obfuscator Class
# =============================================================================

class PostImputationObfuscator:
    """Applies obfuscation to imputed text, targeting harmful terms.
    
    This should be called AFTER RagImputer has replaced label tokens.
    
    Usage:
        obfuscator = PostImputationObfuscator()
        
        # After imputation:
        final_text = obfuscator.obfuscate(
            text=imputed_text,
            archetype="incel_misogyny",
            trajectory_stage="active",
        )
    """
    
    def __init__(self, deterministic: bool = False):
        """Initialize the obfuscator.
        
        Args:
            deterministic: If True, use fixed seed (for testing only).
                          Default False for realistic variety.
        """
        if deterministic:
            self.rng = random.Random(42)
        else:
            # True randomness for production
            entropy = int(time.time() * 1000000) ^ os.getpid() ^ id(self)
            self.rng = random.Random(entropy)
    
    def obfuscate(
        self,
        text: str,
        archetype: str,
        trajectory_stage: str = "active",
        config_override: Optional[PersonaObfuscationConfig] = None,
    ) -> Tuple[str, List[str]]:
        """Obfuscate harmful terms in imputed text.
        
        Args:
            text: The imputed text (after RagImputer)
            archetype: The persona's archetype
            trajectory_stage: The persona's trajectory stage
            config_override: Optional config override
        
        Returns:
            Tuple of (obfuscated_text, list_of_obfuscated_terms)
        """
        # Get config for this archetype
        config = config_override or ARCHETYPE_OBFUSCATION_CONFIGS.get(
            archetype,
            PersonaObfuscationConfig(enabled=False)
        )
        
        if not config.enabled:
            return text, []
        
        # Trajectory affects obfuscation probability
        trajectory_multiplier = {
            "curious": 0.5,      # New users don't know to obfuscate
            "active": 1.0,       # Normal obfuscation
            "entrenched": 1.3,   # More experienced at evasion
            "disillusioned": 0.7,
        }.get(trajectory_stage, 1.0)
        
        effective_probability = config.base_probability * trajectory_multiplier
        
        obfuscated_terms = []
        result = text
        
        # Find and obfuscate terms from each target category
        for category in config.target_categories:
            pattern = _get_pattern(category)
            if not pattern:
                continue
            
            # Find all matches
            matches = list(pattern.finditer(result))
            
            # Process in reverse order to preserve positions
            for match in reversed(matches):
                term = match.group()
                
                # Roll to see if we obfuscate this term
                if self.rng.random() > effective_probability:
                    continue
                
                # Select a strategy based on weights
                strategy = self._select_strategy(config.strategy_weights)
                if not strategy:
                    continue
                
                # Apply obfuscation
                obfuscated = strategy.apply(term, self.rng)
                
                if obfuscated != term:
                    result = result[:match.start()] + obfuscated + result[match.end():]
                    obfuscated_terms.append(f"{term} → {obfuscated}")
        
        return result, obfuscated_terms
    
    def _select_strategy(
        self,
        weights: Dict[str, float],
    ) -> Optional[ObfuscationStrategy]:
        """Select a strategy based on weights."""
        available = [(s, weights.get(s.name, 0.0)) for s in STRATEGIES if weights.get(s.name, 0.0) > 0]
        if not available:
            return None
        
        strategies, probs = zip(*available)
        total = sum(probs)
        if total <= 0:
            return None
        
        normalized = [p / total for p in probs]
        return self.rng.choices(strategies, weights=normalized, k=1)[0]


# =============================================================================
# Convenience Function
# =============================================================================

def obfuscate_imputed_text(
    text: str,
    archetype: str,
    trajectory_stage: str = "active",
    deterministic: bool = False,
) -> Tuple[str, List[str]]:
    """Convenience function to obfuscate imputed text.
    
    Args:
        text: The imputed text (after RagImputer)
        archetype: The persona's archetype
        trajectory_stage: The persona's trajectory stage
        deterministic: If True, use fixed seed (testing only)
    
    Returns:
        Tuple of (obfuscated_text, list_of_obfuscated_terms)
    
    Example:
        >>> text = "These foids always complain about dating"
        >>> obfuscated, terms = obfuscate_imputed_text(text, "incel_misogyny")
        >>> print(obfuscated)
        "These f0ids always complain about dating"
        >>> print(terms)
        ["foids → f0ids"]
    """
    obfuscator = PostImputationObfuscator(deterministic=deterministic)
    return obfuscator.obfuscate(text, archetype, trajectory_stage)

