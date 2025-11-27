"""Obfuscation post-processor for evasion patterns.

Real harmful content often uses obfuscation to evade detection. This module
applies realistic obfuscation patterns BEFORE the RagImputer replaces tokens,
so the tokens themselves remain intact for imputation.

IMPORTANT: Obfuscation uses NON-DETERMINISTIC randomness by default to ensure
variety in the dataset. Set `deterministic=True` only for testing/debugging.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Obfuscation Patterns
# =============================================================================

@dataclass
class ObfuscationPattern:
    """A single obfuscation pattern."""
    
    name: str
    description: str
    probability: float  # Base probability of application
    
    # Pattern function name (maps to implementation)
    pattern_type: str
    
    # Configuration
    config: Dict = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


# Available obfuscation patterns
OBFUSCATION_PATTERNS: Dict[str, ObfuscationPattern] = {
    "leetspeak": ObfuscationPattern(
        name="leetspeak",
        description="Replace letters with numbers/symbols (e.g., 'hate' -> 'h4t3')",
        probability=0.1,
        pattern_type="character_substitution",
        config={
            "substitutions": {
                "a": ["4", "@"],
                "e": ["3"],
                "i": ["1", "!"],
                "o": ["0"],
                "s": ["5", "$"],
                "t": ["7"],
                "l": ["1"],
            }
        },
    ),
    "asterisk_censor": ObfuscationPattern(
        name="asterisk_censor",
        description="Partially censor words with asterisks (e.g., 'kill' -> 'k*ll')",
        probability=0.15,
        pattern_type="partial_censor",
        config={
            "censor_char": "*",
            "preserve_first": True,
            "preserve_last": True,
        },
    ),
    "space_insertion": ObfuscationPattern(
        name="space_insertion",
        description="Insert spaces in words (e.g., 'hate' -> 'h a t e')",
        probability=0.08,
        pattern_type="space_insertion",
        config={
            "separator": " ",
        },
    ),
    "zero_width": ObfuscationPattern(
        name="zero_width",
        description="Insert zero-width characters (invisible but breaks pattern matching)",
        probability=0.05,
        pattern_type="zero_width_insertion",
        config={
            "chars": ["\u200b", "\u200c", "\u200d"],
        },
    ),
    "homoglyph": ObfuscationPattern(
        name="homoglyph",
        description="Replace with similar-looking Unicode characters",
        probability=0.08,
        pattern_type="homoglyph_substitution",
        config={
            "substitutions": {
                "a": "а",  # Cyrillic
                "e": "е",  # Cyrillic
                "o": "о",  # Cyrillic
                "c": "с",  # Cyrillic
                "p": "р",  # Cyrillic
                "x": "х",  # Cyrillic
            }
        },
    ),
    "euphemism": ObfuscationPattern(
        name="euphemism",
        description="Use euphemistic phrases instead of explicit terms",
        probability=0.20,
        pattern_type="euphemism",
        config={
            # This is handled specially - not character-level
        },
    ),
    "sarcasm_quotes": ObfuscationPattern(
        name="sarcasm_quotes",
        description="Wrap harmful content in 'sarcastic' quotes",
        probability=0.12,
        pattern_type="sarcasm_wrapper",
        config={
            "prefixes": [
                '"Totally not saying that ',
                'I mean, "hypothetically" ',
                "Not that I'm saying ",
                "In Minecraft, ",
                "Allegedly, ",
            ],
            "suffixes": [
                '" (obviously)',
                '" lol',
                '" (in Minecraft)',
                '" allegedly',
            ],
        },
    ),
    "dog_whistle": ObfuscationPattern(
        name="dog_whistle",
        description="Use coded language understood by in-group",
        probability=0.15,
        pattern_type="dog_whistle",
        config={
            # Handled by vocabulary selection, not post-processing
        },
    ),
}

# Archetype-specific obfuscation preferences
ARCHETYPE_OBFUSCATION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "incel_misogyny": {
        "leetspeak": 0.3,
        "asterisk_censor": 0.2,
        "sarcasm_quotes": 0.2,
        "dog_whistle": 0.3,
    },
    "extremist": {
        "sarcasm_quotes": 0.4,
        "dog_whistle": 0.3,
        "homoglyph": 0.2,
        "zero_width": 0.1,
    },
    "hate_speech": {
        "asterisk_censor": 0.3,
        "leetspeak": 0.2,
        "homoglyph": 0.2,
        "sarcasm_quotes": 0.3,
    },
    "gamergate": {
        "sarcasm_quotes": 0.4,
        "dog_whistle": 0.3,
        "leetspeak": 0.2,
        "asterisk_censor": 0.1,
    },
    "conspiracy": {
        "euphemism": 0.4,
        "sarcasm_quotes": 0.3,
        "dog_whistle": 0.3,
    },
    "pro_ana": {
        "leetspeak": 0.3,
        "space_insertion": 0.2,
        "asterisk_censor": 0.3,
        "dog_whistle": 0.2,
    },
}


# =============================================================================
# Obfuscation Implementation
# =============================================================================

class ObfuscationProcessor:
    """Applies obfuscation patterns to text while preserving label tokens.
    
    By default, uses NON-DETERMINISTIC randomness to ensure variety in the
    generated dataset. This is important for classification challenges where
    predictable patterns would make the task artificially easy.
    
    Set deterministic=True only for testing/debugging reproducibility.
    """
    
    # Pattern to identify label tokens that should NOT be obfuscated
    LABEL_TOKEN_PATTERN = re.compile(r"<LBL:[A-Z0-9_]+>|LBL:[A-Z0-9_]+")
    
    def __init__(
        self,
        archetype: str,
        trajectory_stage: str = "active",
        rng: Optional[random.Random] = None,
        deterministic: bool = False,
    ):
        """Initialize the obfuscation processor.
        
        Args:
            archetype: The persona's archetype
            trajectory_stage: The persona's trajectory stage
            rng: Optional pre-seeded RNG (for testing). If None and deterministic=False,
                 uses truly random seeding.
            deterministic: If True, uses provided rng or creates a predictable one.
                          If False (default), uses time/entropy for true randomness.
        """
        self.archetype = archetype
        self.trajectory_stage = trajectory_stage
        
        if rng is not None:
            self.rng = rng
        elif deterministic:
            # For testing: use a fixed seed
            self.rng = random.Random(42)
        else:
            # For production: use true randomness (time + entropy)
            # This ensures obfuscation patterns are unpredictable
            entropy_seed = int(time.time() * 1000000) ^ os.getpid() ^ id(self)
            self.rng = random.Random(entropy_seed)
        
        # Get archetype-specific weights
        self.weights = ARCHETYPE_OBFUSCATION_WEIGHTS.get(archetype, {})
    
    def process(
        self,
        text: str,
        obfuscation_intensity: float = 0.5,
    ) -> Tuple[str, List[str]]:
        """Apply obfuscation to text while preserving label tokens.
        
        Args:
            text: The text to obfuscate
            obfuscation_intensity: 0.0-1.0, how much obfuscation to apply
        
        Returns:
            Tuple of (obfuscated_text, list_of_applied_patterns)
        """
        # Extract label tokens and their positions
        tokens_and_positions = list(self.LABEL_TOKEN_PATTERN.finditer(text))
        
        # Split text into segments (alternating between tokens and regular text)
        segments = []
        last_end = 0
        for match in tokens_and_positions:
            if match.start() > last_end:
                segments.append(("text", text[last_end:match.start()]))
            segments.append(("token", match.group()))
            last_end = match.end()
        if last_end < len(text):
            segments.append(("text", text[last_end:]))
        
        # Apply obfuscation only to text segments
        applied_patterns = []
        result_segments = []
        
        for seg_type, seg_content in segments:
            if seg_type == "token":
                # Preserve tokens exactly
                result_segments.append(seg_content)
            else:
                # Apply obfuscation to text
                obfuscated, patterns = self._obfuscate_segment(
                    seg_content, obfuscation_intensity
                )
                result_segments.append(obfuscated)
                applied_patterns.extend(patterns)
        
        return "".join(result_segments), list(set(applied_patterns))
    
    def _obfuscate_segment(
        self,
        text: str,
        intensity: float,
    ) -> Tuple[str, List[str]]:
        """Apply obfuscation patterns to a text segment."""
        if not text.strip():
            return text, []
        
        applied = []
        result = text
        
        # Roll for each pattern type
        for pattern_name, weight in self.weights.items():
            pattern = OBFUSCATION_PATTERNS.get(pattern_name)
            if not pattern:
                continue
            
            # Adjust probability by intensity and trajectory
            trajectory_modifier = {
                "curious": 0.3,
                "active": 1.0,
                "entrenched": 1.5,
                "disillusioned": 0.7,
            }.get(self.trajectory_stage, 1.0)
            
            effective_prob = pattern.probability * weight * intensity * trajectory_modifier
            
            if self.rng.random() < effective_prob:
                result = self._apply_pattern(result, pattern)
                applied.append(pattern_name)
        
        return result, applied
    
    def _apply_pattern(self, text: str, pattern: ObfuscationPattern) -> str:
        """Apply a specific obfuscation pattern."""
        if pattern.pattern_type == "character_substitution":
            return self._apply_character_substitution(text, pattern.config)
        elif pattern.pattern_type == "partial_censor":
            return self._apply_partial_censor(text, pattern.config)
        elif pattern.pattern_type == "space_insertion":
            return self._apply_space_insertion(text, pattern.config)
        elif pattern.pattern_type == "zero_width_insertion":
            return self._apply_zero_width(text, pattern.config)
        elif pattern.pattern_type == "homoglyph_substitution":
            return self._apply_homoglyph(text, pattern.config)
        elif pattern.pattern_type == "sarcasm_wrapper":
            return self._apply_sarcasm_wrapper(text, pattern.config)
        return text
    
    def _apply_character_substitution(self, text: str, config: Dict) -> str:
        """Apply leetspeak-style character substitution."""
        subs = config.get("substitutions", {})
        result = []
        for char in text:
            lower = char.lower()
            if lower in subs and self.rng.random() < 0.5:
                replacement = self.rng.choice(subs[lower])
                result.append(replacement if char.islower() else replacement.upper())
            else:
                result.append(char)
        return "".join(result)
    
    def _apply_partial_censor(self, text: str, config: Dict) -> str:
        """Apply partial word censoring with asterisks."""
        censor_char = config.get("censor_char", "*")
        preserve_first = config.get("preserve_first", True)
        preserve_last = config.get("preserve_last", True)
        
        words = text.split()
        result = []
        for word in words:
            if len(word) > 4 and self.rng.random() < 0.3:
                # Censor middle characters
                if preserve_first and preserve_last:
                    censored = word[0] + censor_char * (len(word) - 2) + word[-1]
                elif preserve_first:
                    censored = word[0] + censor_char * (len(word) - 1)
                else:
                    censored = censor_char * len(word)
                result.append(censored)
            else:
                result.append(word)
        return " ".join(result)
    
    def _apply_space_insertion(self, text: str, config: Dict) -> str:
        """Insert spaces in words."""
        separator = config.get("separator", " ")
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and self.rng.random() < 0.2:
                result.append(separator.join(word))
            else:
                result.append(word)
        return " ".join(result)
    
    def _apply_zero_width(self, text: str, config: Dict) -> str:
        """Insert zero-width characters."""
        chars = config.get("chars", ["\u200b"])
        result = []
        for i, char in enumerate(text):
            result.append(char)
            if i > 0 and i < len(text) - 1 and self.rng.random() < 0.1:
                result.append(self.rng.choice(chars))
        return "".join(result)
    
    def _apply_homoglyph(self, text: str, config: Dict) -> str:
        """Replace with similar-looking Unicode characters."""
        subs = config.get("substitutions", {})
        result = []
        for char in text:
            lower = char.lower()
            if lower in subs and self.rng.random() < 0.3:
                result.append(subs[lower])
            else:
                result.append(char)
        return "".join(result)
    
    def _apply_sarcasm_wrapper(self, text: str, config: Dict) -> str:
        """Wrap content in sarcastic framing."""
        prefixes = config.get("prefixes", [])
        suffixes = config.get("suffixes", [])
        
        if not prefixes and not suffixes:
            return text
        
        # Only apply to sentence-like content
        if len(text) < 20:
            return text
        
        prefix = self.rng.choice(prefixes) if prefixes else ""
        suffix = self.rng.choice(suffixes) if suffixes else ""
        
        return f"{prefix}{text}{suffix}"


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_obfuscation(
    text: str,
    archetype: str,
    trajectory_stage: str = "active",
    intensity: float = 0.5,
    rng: Optional[random.Random] = None,
    deterministic: bool = False,
) -> Tuple[str, List[str]]:
    """Apply obfuscation to text.
    
    Args:
        text: The text to obfuscate
        archetype: The persona's archetype
        trajectory_stage: The persona's trajectory stage
        intensity: 0.0-1.0, how much obfuscation to apply
        rng: Optional pre-seeded RNG (for testing)
        deterministic: If False (default), uses true randomness for unpredictable
                      obfuscation. Set True only for testing.
    
    Returns:
        Tuple of (obfuscated_text, list_of_applied_patterns)
    
    Note:
        By default, obfuscation is NON-DETERMINISTIC to ensure variety in the
        generated dataset. This is important for classification challenges.
    """
    processor = ObfuscationProcessor(
        archetype=archetype,
        trajectory_stage=trajectory_stage,
        rng=rng,
        deterministic=deterministic,
    )
    return processor.process(text, intensity)

