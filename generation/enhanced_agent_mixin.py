"""Enhanced agent mixin with trajectory, fluency, contamination, and obfuscation.

This mixin extends ExtendedSocialAgent with the new persona fidelity features.
It can be applied to any ExtendedSocialAgent to add:
- Trajectory-aware content generation
- Fluency-adjusted vocabulary
- Cross-archetype contamination
- Benign post injection
- Post type diversification
- Obfuscation patterns
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from generation.enhanced_persona_generator import (
    EnhancedPersonaConfig,
    DEMOGRAPHIC_PROFILES,
)
from generation.trajectory_config import (
    get_trajectory_config,
    get_fluency_config,
    TrajectoryStage,
    SlangFluencyLevel,
)
from generation.contamination_config import (
    sample_post_type,
    get_post_type_config,
    PostTypeConfig,
)
from generation.implicit_harm_prompts import (
    build_implicit_step_hint,
    get_implicit_prompt,
)
from generation.obfuscation_processor import apply_obfuscation


class EnhancedAgentMixin:
    """Mixin that adds enhanced persona features to ExtendedSocialAgent.
    
    This mixin expects the base class to have:
    - self._persona_cfg: PersonaConfig (or EnhancedPersonaConfig)
    - self._step_index: int
    - self._run_seed: int
    - self.social_agent_id: int
    """
    
    # These will be set in __init__ or by the factory
    _enhanced_config: Optional[EnhancedPersonaConfig] = None
    _trajectory_config: Optional[TrajectoryStage] = None
    _fluency_config: Optional[SlangFluencyLevel] = None
    _step_rng: Optional[random.Random] = None
    
    def initialize_enhanced_features(
        self,
        enhanced_config: EnhancedPersonaConfig,
    ) -> None:
        """Initialize enhanced persona features.
        
        Call this after creating the agent to enable enhanced features.
        
        Args:
            enhanced_config: The enhanced persona configuration
        """
        self._enhanced_config = enhanced_config
        self._trajectory_config = get_trajectory_config(enhanced_config.trajectory_stage)
        self._fluency_config = get_fluency_config(enhanced_config.slang_fluency)
        
        # Create a seeded RNG for this agent
        seed = hash((self._run_seed, self.social_agent_id)) & 0xFFFFFFFF
        self._step_rng = random.Random(seed)
    
    def should_generate_benign_post(self) -> bool:
        """Determine if this step should generate benign content.
        
        This implements Improvement #2: Benign Post Injection.
        Even toxic personas sometimes post normal content.
        
        Returns:
            True if this step should be benign
        """
        if not self._enhanced_config:
            return False
        
        # Get the benign post rate (already trajectory-adjusted in EnhancedPersonaConfig)
        rate = self._enhanced_config.benign_post_rate
        
        # Use step-seeded RNG for determinism
        step_seed = hash((self._run_seed, self.social_agent_id, self._step_index)) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        
        return rng.random() < rate
    
    def select_post_type(self) -> Tuple[str, PostTypeConfig]:
        """Select a post type for this step.
        
        This implements Improvement #4: Post Type Diversification.
        Different archetypes have different distributions of post types.
        
        Returns:
            Tuple of (post_type_name, PostTypeConfig)
        """
        if not self._enhanced_config:
            return "observation", get_post_type_config("observation")
        
        archetype = self._enhanced_config.primary_label
        
        # Use step-seeded RNG
        step_seed = hash((self._run_seed, self.social_agent_id, self._step_index, "post_type")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        
        post_type = sample_post_type(archetype, rng)
        return post_type, get_post_type_config(post_type)
    
    def get_enhanced_step_hint(
        self,
        mode: str,
        tokens: List[str],
    ) -> str:
        """Get an enhanced step hint using implicit harm prompts.
        
        This implements Improvement #1: Implicit Harm Prompts.
        Uses behavioral instructions instead of explicit "use this label".
        
        Args:
            mode: "none", "single", or "double"
            tokens: The tokens to emit
        
        Returns:
            Enhanced step hint string
        """
        if not self._enhanced_config:
            # Fallback to basic hint
            if mode == "none":
                return "This is a benign neutral comment. Do not include any label markers."
            if mode == "single" and tokens:
                return f"Use exactly one label marker inline: {tokens[0]}."
            if mode == "double" and len(tokens) >= 2:
                return f"Use exactly two label markers inline: {tokens[0]} and {tokens[1]}."
            return ""
        
        archetype = self._enhanced_config.primary_label
        trajectory = self._enhanced_config.trajectory_stage
        fluency = self._enhanced_config.slang_fluency
        
        return build_implicit_step_hint(
            archetype=archetype,
            mode=mode,
            tokens=tokens,
            trajectory_stage=trajectory,
            slang_fluency=fluency,
        )
    
    def apply_trajectory_modifiers(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply trajectory-based modifiers to the emission decision.
        
        This implements Improvement #5: Radicalization Trajectory Effects.
        
        Args:
            decision: The original emission decision
        
        Returns:
            Modified decision
        """
        if not self._enhanced_config or not self._trajectory_config:
            return decision
        
        traj = self._trajectory_config
        modified = dict(decision)
        
        # Token density modifier
        tokens = modified.get("tokens", [])
        if tokens and traj.token_density_multiplier < 1.0:
            # Possibly reduce number of tokens
            step_seed = hash((self._run_seed, self.social_agent_id, self._step_index, "density")) & 0xFFFFFFFF
            rng = random.Random(step_seed)
            
            if rng.random() > traj.token_density_multiplier:
                # Skip one token if in double mode
                if len(tokens) > 1:
                    modified["tokens"] = [tokens[0]]
                    modified["mode"] = "single"
        
        # Skip token probability
        if tokens and traj.skip_token_probability > 0:
            step_seed = hash((self._run_seed, self.social_agent_id, self._step_index, "skip")) & 0xFFFFFFFF
            rng = random.Random(step_seed)
            
            if rng.random() < traj.skip_token_probability:
                modified["tokens"] = []
                modified["mode"] = "none"
        
        return modified
    
    def apply_contamination_tokens(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Possibly add secondary archetype tokens.
        
        This implements Improvement #6: Cross-Archetype Contamination.
        
        Args:
            decision: The original emission decision
        
        Returns:
            Modified decision with possible secondary tokens
        """
        if not self._enhanced_config:
            return decision
        
        secondary = self._enhanced_config.secondary_archetype
        if not secondary:
            return decision
        
        influence = self._enhanced_config.secondary_influence
        
        # Only apply in some steps based on influence
        step_seed = hash((self._run_seed, self.social_agent_id, self._step_index, "contam")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        
        if rng.random() > influence * 0.5:  # Half as likely as primary
            return decision
        
        # Get a token from the secondary archetype
        secondary_prompt = get_implicit_prompt(secondary)
        if not secondary_prompt:
            return decision
        
        # Add a hint about secondary archetype influence
        modified = dict(decision)
        existing_tokens = modified.get("tokens", [])
        
        # Don't add more than 2 tokens total
        if len(existing_tokens) >= 2:
            return modified
        
        # Add secondary archetype vocabulary hint instead of token
        # (Tokens should come from the ontology, not be invented here)
        modified["secondary_archetype"] = secondary
        modified["secondary_influence"] = influence
        
        return modified
    
    def apply_obfuscation_to_content(
        self,
        content: str,
        deterministic: bool = False,
    ) -> Tuple[str, List[str]]:
        """Apply obfuscation patterns to generated content.
        
        This implements Improvement #7: Evasion Patterns.
        Applied BEFORE RagImputer, preserving label tokens.
        
        IMPORTANT: By default, obfuscation is NON-DETERMINISTIC to ensure
        variety in the dataset. This prevents attackers from reverse-engineering
        obfuscation patterns. Set deterministic=True only for testing.
        
        Args:
            content: The generated content (with label tokens)
            deterministic: If True, use seeded RNG (for testing only)
        
        Returns:
            Tuple of (obfuscated_content, applied_patterns)
        """
        if not self._enhanced_config:
            return content, []
        
        archetype = self._enhanced_config.primary_label
        trajectory = self._enhanced_config.trajectory_stage
        intensity = self._enhanced_config.obfuscation_intensity
        
        # By default, use non-deterministic obfuscation for realistic variety
        # Only use deterministic for testing/debugging
        return apply_obfuscation(
            text=content,
            archetype=archetype,
            trajectory_stage=trajectory,
            intensity=intensity,
            rng=None,  # Let the processor create its own RNG
            deterministic=deterministic,
        )
    
    def get_demographic_style_hints(self) -> List[str]:
        """Get style hints based on demographic profile.
        
        This implements demographic-driven style variation.
        
        Returns:
            List of style hint strings
        """
        if not self._enhanced_config:
            return []
        
        profile_name = self._enhanced_config.demographic_profile
        profile = DEMOGRAPHIC_PROFILES.get(profile_name, {})
        
        hints = []
        modifiers = profile.get("style_modifiers", {})
        vocab_hints = profile.get("vocabulary_hints", [])
        
        if modifiers.get("uses_emoji"):
            hints.append("Feel free to use emoji naturally")
        
        if modifiers.get("uses_slang"):
            hints.append("Use current internet slang")
        
        formal_level = modifiers.get("formal_level", 0.5)
        if formal_level < 0.3:
            hints.append("Keep it casual and informal")
        elif formal_level > 0.7:
            hints.append("Use more formal phrasing")
        
        cap = modifiers.get("capitalization", "normal")
        if cap == "lowercase":
            hints.append("Don't capitalize anything")
        elif cap == "mixed":
            hints.append("Use casual capitalization")
        
        hints.extend(vocab_hints)
        
        return hints
    
    def get_fluency_style_hints(self) -> List[str]:
        """Get style hints based on slang fluency level.
        
        Returns:
            List of style hint strings
        """
        if not self._fluency_config:
            return []
        
        fluency = self._fluency_config
        hints = []
        
        if fluency.use_abbreviations:
            hints.append("Use community abbreviations naturally")
        
        if fluency.explain_terms:
            hints.append("Sometimes explain terms for newcomers")
        
        if fluency.misuse_probability > 0.1:
            hints.append("Occasionally use terms slightly incorrectly")
        
        if fluency.insider_reference_rate > 0.5:
            hints.append("Make insider references that only regulars would get")
        
        if fluency.meme_usage_rate > 0.3:
            hints.append("Reference community memes")
        
        return hints
    
    def get_trajectory_characteristic_phrase(self) -> Optional[str]:
        """Get a characteristic phrase opener based on trajectory stage.
        
        Returns:
            Optional phrase to potentially use in content
        """
        if not self._trajectory_config:
            return None
        
        phrases = self._trajectory_config.characteristic_phrases
        if not phrases:
            return None
        
        # Use step-seeded RNG
        step_seed = hash((self._run_seed, self.social_agent_id, self._step_index, "phrase")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        
        # Only use sometimes
        if rng.random() > 0.3:
            return None
        
        return rng.choice(phrases)


def create_enhanced_step_hint(
    enhanced_config: EnhancedPersonaConfig,
    mode: str,
    tokens: List[str],
    include_demographic: bool = True,
    include_fluency: bool = True,
    include_post_type: bool = True,
    step_index: int = 0,
    run_seed: int = 0,
    agent_id: int = 0,
) -> str:
    """Convenience function to create a comprehensive enhanced step hint.
    
    This combines all the enhancement features into a single prompt section.
    
    Args:
        enhanced_config: The enhanced persona configuration
        mode: "none", "single", or "double"
        tokens: The tokens to emit
        include_demographic: Include demographic style hints
        include_fluency: Include fluency style hints
        include_post_type: Include post type guidance
        step_index: Current step index
        run_seed: Run seed for RNG
        agent_id: Agent ID for RNG
    
    Returns:
        Comprehensive step hint string
    """
    lines = []
    
    # Base implicit harm prompt
    base_hint = build_implicit_step_hint(
        archetype=enhanced_config.primary_label,
        mode=mode,
        tokens=tokens,
        trajectory_stage=enhanced_config.trajectory_stage,
        slang_fluency=enhanced_config.slang_fluency,
    )
    lines.append(base_hint)
    
    # Post type guidance
    if include_post_type:
        step_seed = hash((run_seed, agent_id, step_index, "post_type")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        post_type = sample_post_type(enhanced_config.primary_label, rng)
        post_config = get_post_type_config(post_type)
        lines.append(f"\n[Post Type] {post_type}: {post_config.description}")
    
    # Demographic hints
    if include_demographic:
        profile = DEMOGRAPHIC_PROFILES.get(enhanced_config.demographic_profile, {})
        vocab_hints = profile.get("vocabulary_hints", [])
        if vocab_hints:
            lines.append(f"\n[Demographic Style] {'; '.join(vocab_hints[:2])}")
    
    # Fluency hints
    if include_fluency:
        fluency = get_fluency_config(enhanced_config.slang_fluency)
        if fluency.insider_reference_rate > 0.5:
            lines.append("\n[Fluency] Make insider references naturally")
        elif fluency.explain_terms:
            lines.append("\n[Fluency] Sometimes explain community terms")
    
    # Secondary archetype hint
    if enhanced_config.secondary_archetype:
        step_seed = hash((run_seed, agent_id, step_index, "contam")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        if rng.random() < enhanced_config.secondary_influence * 0.5:
            lines.append(
                f"\n[Secondary Influence] Occasionally show {enhanced_config.secondary_archetype} tendencies"
            )
    
    # Trajectory characteristic phrase
    traj_config = get_trajectory_config(enhanced_config.trajectory_stage)
    if traj_config.characteristic_phrases:
        step_seed = hash((run_seed, agent_id, step_index, "phrase")) & 0xFFFFFFFF
        rng = random.Random(step_seed)
        if rng.random() < 0.2:
            phrase = rng.choice(traj_config.characteristic_phrases)
            lines.append(f"\n[Consider starting with] \"{phrase}\"")
    
    return "\n".join(lines)

