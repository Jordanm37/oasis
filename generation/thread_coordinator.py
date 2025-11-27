"""Thread coordinator for multi-agent dynamics.

This module implements Improvement #8: Thread Dynamics.
It orchestrates realistic multi-agent interactions like:
- Pile-ons (multiple toxic users attacking a target)
- Echo chambers (mutual reinforcement)
- Debates (opposing viewpoints engaging)
- Brigading (coordinated attacks on content)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class ThreadDynamicType(Enum):
    """Types of multi-agent thread dynamics."""
    
    NORMAL = "normal"           # Standard independent posting
    PILE_ON = "pile_on"         # Multiple users attacking a target
    ECHO_CHAMBER = "echo"       # Mutual reinforcement of views
    DEBATE = "debate"           # Opposing viewpoints engaging
    BRIGADE = "brigade"         # Coordinated attack on content
    SUPPORT_RALLY = "support"   # Multiple users supporting someone


@dataclass
class ThreadDynamic:
    """Configuration for a thread dynamic."""
    
    dynamic_type: ThreadDynamicType
    
    # Participants
    initiator_id: int                    # Agent who started the dynamic
    participant_ids: List[int]           # Other agents involved
    target_id: Optional[int] = None      # Target of pile-on/brigade (if applicable)
    target_post_id: Optional[int] = None # Target post (if applicable)
    
    # Timing
    start_step: int = 0
    duration_steps: int = 5
    
    # Behavior modifiers
    aggression_boost: float = 0.0        # Added to base aggression
    token_density_boost: float = 0.0     # Added to token density
    coordination_strength: float = 0.5   # How coordinated the participants are
    
    # State
    current_step: int = 0
    is_active: bool = True


@dataclass
class ThreadCoordinatorConfig:
    """Configuration for the thread coordinator."""
    
    # Probability of triggering each dynamic type
    pile_on_probability: float = 0.05
    echo_chamber_probability: float = 0.10
    debate_probability: float = 0.08
    brigade_probability: float = 0.03
    support_rally_probability: float = 0.05
    
    # Dynamic parameters
    min_participants: int = 2
    max_participants: int = 5
    min_duration: int = 3
    max_duration: int = 8
    
    # Archetype compatibility for echo chambers
    echo_compatible_archetypes: Dict[str, List[str]] = field(default_factory=lambda: {
        "incel_misogyny": ["alpha", "gamergate", "hate_speech"],
        "alpha": ["incel_misogyny", "trad"],
        "conspiracy": ["misinfo", "extremist"],
        "misinfo": ["conspiracy"],
        "ed_risk": ["pro_ana"],
        "pro_ana": ["ed_risk"],
        "trad": ["alpha", "conspiracy"],
        "gamergate": ["incel_misogyny", "bullying"],
        "extremist": ["hate_speech", "conspiracy"],
        "hate_speech": ["extremist", "incel_misogyny"],
        "bullying": ["gamergate", "hate_speech"],
        "benign": ["recovery_support"],
        "recovery_support": ["benign"],
    })
    
    # Archetype opposition for debates
    opposing_archetypes: Dict[str, List[str]] = field(default_factory=lambda: {
        "incel_misogyny": ["benign", "recovery_support"],
        "alpha": ["benign"],
        "conspiracy": ["benign", "misinfo"],  # Competing conspiracies
        "misinfo": ["benign"],
        "ed_risk": ["recovery_support"],
        "pro_ana": ["recovery_support"],
        "trad": ["benign"],
        "gamergate": ["benign"],
        "extremist": ["benign"],
        "hate_speech": ["benign"],
        "bullying": ["benign", "recovery_support"],
        "benign": ["extremist", "hate_speech"],
        "recovery_support": ["ed_risk", "pro_ana"],
    })


class ThreadCoordinator:
    """Coordinates multi-agent thread dynamics."""
    
    def __init__(
        self,
        config: Optional[ThreadCoordinatorConfig] = None,
        seed: int = 42,
    ):
        self.config = config or ThreadCoordinatorConfig()
        self.rng = random.Random(seed)
        self.seed = seed
        
        # Active dynamics
        self._active_dynamics: List[ThreadDynamic] = []
        
        # Agent archetypes (set by simulation)
        self._agent_archetypes: Dict[int, str] = {}
        
        # Recent posts for targeting
        self._recent_posts: List[Dict[str, Any]] = []
        self._max_recent_posts = 50
    
    def register_agent(self, agent_id: int, archetype: str) -> None:
        """Register an agent's archetype for coordination."""
        self._agent_archetypes[agent_id] = archetype
    
    def record_post(
        self,
        post_id: int,
        agent_id: int,
        content: str,
        step: int,
    ) -> None:
        """Record a post for potential targeting."""
        self._recent_posts.append({
            "post_id": post_id,
            "agent_id": agent_id,
            "content": content,
            "step": step,
            "archetype": self._agent_archetypes.get(agent_id, "unknown"),
        })
        
        # Trim old posts
        if len(self._recent_posts) > self._max_recent_posts:
            self._recent_posts = self._recent_posts[-self._max_recent_posts:]
    
    def step(self, current_step: int) -> None:
        """Advance the coordinator by one step.
        
        This updates active dynamics and potentially triggers new ones.
        """
        # Update active dynamics
        for dynamic in self._active_dynamics:
            dynamic.current_step += 1
            if dynamic.current_step >= dynamic.duration_steps:
                dynamic.is_active = False
        
        # Remove completed dynamics
        self._active_dynamics = [d for d in self._active_dynamics if d.is_active]
        
        # Maybe trigger new dynamics
        self._maybe_trigger_dynamics(current_step)
    
    def get_agent_modifiers(
        self,
        agent_id: int,
        step: int,
    ) -> Dict[str, Any]:
        """Get behavior modifiers for an agent based on active dynamics.
        
        Args:
            agent_id: The agent to check
            step: Current step
        
        Returns:
            Dict with modifiers: aggression_boost, token_density_boost,
            target_post_id, dynamic_type, coordination_hints
        """
        modifiers = {
            "aggression_boost": 0.0,
            "token_density_boost": 0.0,
            "target_post_id": None,
            "dynamic_type": None,
            "coordination_hints": [],
            "is_participant": False,
        }
        
        for dynamic in self._active_dynamics:
            if not dynamic.is_active:
                continue
            
            is_participant = (
                agent_id == dynamic.initiator_id or
                agent_id in dynamic.participant_ids
            )
            
            if not is_participant:
                continue
            
            modifiers["is_participant"] = True
            modifiers["dynamic_type"] = dynamic.dynamic_type.value
            modifiers["aggression_boost"] = max(
                modifiers["aggression_boost"],
                dynamic.aggression_boost
            )
            modifiers["token_density_boost"] = max(
                modifiers["token_density_boost"],
                dynamic.token_density_boost
            )
            
            if dynamic.target_post_id:
                modifiers["target_post_id"] = dynamic.target_post_id
            
            # Add coordination hints
            hints = self._get_coordination_hints(dynamic, agent_id)
            modifiers["coordination_hints"].extend(hints)
        
        return modifiers
    
    def _maybe_trigger_dynamics(self, step: int) -> None:
        """Possibly trigger new thread dynamics."""
        cfg = self.config
        
        # Check each dynamic type
        if self.rng.random() < cfg.pile_on_probability:
            self._trigger_pile_on(step)
        
        if self.rng.random() < cfg.echo_chamber_probability:
            self._trigger_echo_chamber(step)
        
        if self.rng.random() < cfg.debate_probability:
            self._trigger_debate(step)
        
        if self.rng.random() < cfg.brigade_probability:
            self._trigger_brigade(step)
        
        if self.rng.random() < cfg.support_rally_probability:
            self._trigger_support_rally(step)
    
    def _trigger_pile_on(self, step: int) -> None:
        """Trigger a pile-on dynamic."""
        # Find a target (benign or recovery user)
        targets = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch in ("benign", "recovery_support")
        ]
        if not targets:
            return
        
        target_id = self.rng.choice(targets)
        
        # Find attackers (toxic archetypes)
        attackers = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch in ("bullying", "hate_speech", "incel_misogyny", "gamergate")
            and aid != target_id
        ]
        if len(attackers) < self.config.min_participants:
            return
        
        num_participants = self.rng.randint(
            self.config.min_participants,
            min(self.config.max_participants, len(attackers))
        )
        participants = self.rng.sample(attackers, num_participants)
        initiator = participants[0]
        
        # Find a target post if available
        target_posts = [
            p for p in self._recent_posts
            if p["agent_id"] == target_id
        ]
        target_post_id = target_posts[-1]["post_id"] if target_posts else None
        
        dynamic = ThreadDynamic(
            dynamic_type=ThreadDynamicType.PILE_ON,
            initiator_id=initiator,
            participant_ids=participants[1:],
            target_id=target_id,
            target_post_id=target_post_id,
            start_step=step,
            duration_steps=self.rng.randint(
                self.config.min_duration,
                self.config.max_duration
            ),
            aggression_boost=0.3,
            token_density_boost=0.2,
            coordination_strength=0.7,
        )
        
        self._active_dynamics.append(dynamic)
    
    def _trigger_echo_chamber(self, step: int) -> None:
        """Trigger an echo chamber dynamic."""
        # Pick a random archetype with compatible partners
        archetypes_with_partners = [
            arch for arch in self._agent_archetypes.values()
            if arch in self.config.echo_compatible_archetypes
        ]
        if not archetypes_with_partners:
            return
        
        primary_arch = self.rng.choice(archetypes_with_partners)
        compatible = self.config.echo_compatible_archetypes.get(primary_arch, [])
        
        # Find agents of compatible archetypes
        candidates = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch == primary_arch or arch in compatible
        ]
        if len(candidates) < self.config.min_participants:
            return
        
        num_participants = self.rng.randint(
            self.config.min_participants,
            min(self.config.max_participants, len(candidates))
        )
        participants = self.rng.sample(candidates, num_participants)
        
        dynamic = ThreadDynamic(
            dynamic_type=ThreadDynamicType.ECHO_CHAMBER,
            initiator_id=participants[0],
            participant_ids=participants[1:],
            start_step=step,
            duration_steps=self.rng.randint(
                self.config.min_duration,
                self.config.max_duration
            ),
            aggression_boost=0.1,
            token_density_boost=0.15,
            coordination_strength=0.5,
        )
        
        self._active_dynamics.append(dynamic)
    
    def _trigger_debate(self, step: int) -> None:
        """Trigger a debate dynamic between opposing viewpoints."""
        # Find archetypes with opposing partners
        archetypes_with_opponents = [
            arch for arch in self._agent_archetypes.values()
            if arch in self.config.opposing_archetypes
        ]
        if not archetypes_with_opponents:
            return
        
        primary_arch = self.rng.choice(archetypes_with_opponents)
        opponents = self.config.opposing_archetypes.get(primary_arch, [])
        
        # Find agents on both sides
        primary_agents = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch == primary_arch
        ]
        opponent_agents = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch in opponents
        ]
        
        if not primary_agents or not opponent_agents:
            return
        
        initiator = self.rng.choice(primary_agents)
        responder = self.rng.choice(opponent_agents)
        
        dynamic = ThreadDynamic(
            dynamic_type=ThreadDynamicType.DEBATE,
            initiator_id=initiator,
            participant_ids=[responder],
            start_step=step,
            duration_steps=self.rng.randint(
                self.config.min_duration,
                self.config.max_duration
            ),
            aggression_boost=0.2,
            token_density_boost=0.1,
            coordination_strength=0.3,
        )
        
        self._active_dynamics.append(dynamic)
    
    def _trigger_brigade(self, step: int) -> None:
        """Trigger a brigade (coordinated attack on content)."""
        # Find a recent post from a benign user
        target_posts = [
            p for p in self._recent_posts
            if p["archetype"] in ("benign", "recovery_support")
        ]
        if not target_posts:
            return
        
        target_post = self.rng.choice(target_posts)
        
        # Find brigaders
        brigaders = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch in ("gamergate", "bullying", "hate_speech", "extremist")
        ]
        if len(brigaders) < self.config.min_participants:
            return
        
        num_participants = self.rng.randint(
            self.config.min_participants,
            min(self.config.max_participants, len(brigaders))
        )
        participants = self.rng.sample(brigaders, num_participants)
        
        dynamic = ThreadDynamic(
            dynamic_type=ThreadDynamicType.BRIGADE,
            initiator_id=participants[0],
            participant_ids=participants[1:],
            target_id=target_post["agent_id"],
            target_post_id=target_post["post_id"],
            start_step=step,
            duration_steps=self.rng.randint(
                self.config.min_duration,
                self.config.max_duration
            ),
            aggression_boost=0.4,
            token_density_boost=0.25,
            coordination_strength=0.8,
        )
        
        self._active_dynamics.append(dynamic)
    
    def _trigger_support_rally(self, step: int) -> None:
        """Trigger a support rally (multiple users supporting someone)."""
        # Find a recovery support user
        support_users = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch == "recovery_support"
        ]
        if not support_users:
            return
        
        target = self.rng.choice(support_users)
        
        # Find supporters
        supporters = [
            aid for aid, arch in self._agent_archetypes.items()
            if arch in ("benign", "recovery_support") and aid != target
        ]
        if len(supporters) < self.config.min_participants:
            return
        
        num_participants = self.rng.randint(
            self.config.min_participants,
            min(self.config.max_participants, len(supporters))
        )
        participants = self.rng.sample(supporters, num_participants)
        
        dynamic = ThreadDynamic(
            dynamic_type=ThreadDynamicType.SUPPORT_RALLY,
            initiator_id=participants[0],
            participant_ids=participants[1:],
            target_id=target,
            start_step=step,
            duration_steps=self.rng.randint(
                self.config.min_duration,
                self.config.max_duration
            ),
            aggression_boost=-0.2,  # Less aggressive
            token_density_boost=0.1,
            coordination_strength=0.6,
        )
        
        self._active_dynamics.append(dynamic)
    
    def _get_coordination_hints(
        self,
        dynamic: ThreadDynamic,
        agent_id: int,
    ) -> List[str]:
        """Get coordination hints for an agent in a dynamic."""
        hints = []
        
        if dynamic.dynamic_type == ThreadDynamicType.PILE_ON:
            if agent_id == dynamic.initiator_id:
                hints.append("Lead the attack on the target")
            else:
                hints.append("Join in on the criticism of the target")
                hints.append("Build on what others have said")
        
        elif dynamic.dynamic_type == ThreadDynamicType.ECHO_CHAMBER:
            hints.append("Agree with and reinforce others' points")
            hints.append("Use similar language to other participants")
        
        elif dynamic.dynamic_type == ThreadDynamicType.DEBATE:
            hints.append("Engage with opposing viewpoints")
            hints.append("Defend your position against criticism")
        
        elif dynamic.dynamic_type == ThreadDynamicType.BRIGADE:
            hints.append("Coordinate your criticism with others")
            hints.append("Focus on the target post")
        
        elif dynamic.dynamic_type == ThreadDynamicType.SUPPORT_RALLY:
            hints.append("Offer encouragement and support")
            hints.append("Validate the target's experiences")
        
        return hints

