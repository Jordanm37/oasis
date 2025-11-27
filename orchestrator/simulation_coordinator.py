"""Simulation Coordinator: Orchestrates multi-agent thread dynamics.

This module coordinates agent behavior to create realistic multi-agent
dynamics like pile-ons, echo chambers, debates, and brigades.

It works ALONGSIDE the RecSys, not replacing it:
- RecSys controls: What agents SEE (their timeline)
- Coordinator controls: How agents BEHAVE (modifiers to their generation)

Usage:
    coordinator = SimulationCoordinator(config, seed=42)
    
    # Register agents with their archetypes
    for agent_id, agent in agent_graph.get_agents():
        coordinator.register_agent(agent_id, agent.archetype)
    
    # Each step:
    coordinator.step()
    for agent_id, agent in agent_graph.get_agents():
        modifiers = coordinator.get_agent_modifiers(agent_id)
        # Apply modifiers to agent behavior
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationCoordinatorConfig:
    """Configuration for the simulation coordinator."""
    
    # Master switches
    enable_thread_dynamics: bool = True
    enable_obfuscation: bool = False  # Handled by RagImputer post-imputation
    
    # Event probabilities per step
    pile_on_probability: float = 0.05
    echo_chamber_probability: float = 0.08
    debate_probability: float = 0.03
    brigade_probability: float = 0.02
    
    # Event parameters
    min_participants: int = 3
    max_participants: int = 8
    event_duration_steps: int = 3
    
    # Archetype affinities for events
    archetype_affinities: Dict[str, List[str]] = field(default_factory=lambda: {
        # Archetypes likely to pile-on together
        "pile_on": ["incel_misogyny", "hate_speech", "extremist", "bullying"],
        # Archetypes that form echo chambers
        "echo_chamber": ["conspiracy", "misinfo", "trad", "extremist"],
        # Archetypes that engage in debates
        "debate": ["alpha", "misinfo", "conspiracy"],
        # Archetypes that coordinate brigades
        "brigade": ["extremist", "hate_speech", "incel_misogyny"],
    })


# =============================================================================
# Event Types
# =============================================================================

EventType = Literal["pile_on", "echo_chamber", "debate", "brigade", "none"]


@dataclass
class CoordinationEvent:
    """An active coordination event affecting multiple agents."""
    
    event_type: EventType
    target_thread_id: Optional[str]  # Thread being targeted (if applicable)
    participants: Set[int]  # Agent IDs involved
    intensity: float  # 1.0 = normal, higher = more aggressive
    started_at_step: int
    duration_steps: int
    
    # Modifiers applied to participants
    aggression_boost: float = 0.0  # 0.0-1.0
    token_density_boost: float = 0.0  # 0.0-1.0
    coordination_hints: List[str] = field(default_factory=list)
    
    def is_expired(self, current_step: int) -> bool:
        """Check if this event has expired."""
        return current_step >= self.started_at_step + self.duration_steps


@dataclass
class AgentModifiers:
    """Modifiers to apply to an agent's behavior for this step."""
    
    # Behavioral modifiers
    aggression_boost: float = 0.0  # Increase token density, harsher language
    token_density_boost: float = 0.0  # More label tokens
    coordination_hints: List[str] = field(default_factory=list)  # Hints for LLM
    
    # Event context
    active_event_type: EventType = "none"
    target_thread_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to agent."""
        return {
            "aggression_boost": self.aggression_boost,
            "token_density_boost": self.token_density_boost,
            "coordination_hints": self.coordination_hints,
            "active_event_type": self.active_event_type,
            "target_thread_id": self.target_thread_id,
        }


# =============================================================================
# Main Coordinator Class
# =============================================================================

class SimulationCoordinator:
    """Coordinates multi-agent dynamics during simulation.
    
    This coordinator creates emergent behaviors like:
    - Pile-ons: Multiple agents attack the same target
    - Echo chambers: Agents reinforce each other's beliefs
    - Debates: Opposing viewpoints clash
    - Brigades: Coordinated attacks on specific content
    
    The coordinator works by providing MODIFIERS to agents, not by
    directly controlling their actions. Agents still use their normal
    LLM-based decision making, but with adjusted parameters.
    """
    
    def __init__(
        self,
        config: Optional[SimulationCoordinatorConfig] = None,
        seed: int = 42,
    ):
        """Initialize the coordinator.
        
        Args:
            config: Coordinator configuration
            seed: Random seed for reproducibility
        """
        self.config = config or SimulationCoordinatorConfig()
        self.rng = random.Random(seed)
        
        # Agent registry: agent_id -> archetype
        self._agents: Dict[int, str] = {}
        
        # Active events
        self._active_events: List[CoordinationEvent] = []
        
        # Current step
        self._current_step = 0
        
        # Recent activity tracking (for event triggers)
        self._recent_posts: List[Dict[str, Any]] = []
        self._hot_threads: Set[str] = set()
    
    def register_agent(self, agent_id: int, archetype: str) -> None:
        """Register an agent with the coordinator.
        
        Args:
            agent_id: The agent's ID
            archetype: The agent's archetype (e.g., "incel_misogyny")
        """
        self._agents[agent_id] = archetype.lower().replace("-", "_")
    
    def step(self) -> None:
        """Advance the coordinator by one step.
        
        This should be called at the start of each simulation step,
        BEFORE agents take their actions.
        """
        self._current_step += 1
        
        # Expire old events
        self._active_events = [
            e for e in self._active_events
            if not e.is_expired(self._current_step)
        ]
        
        # Potentially trigger new events
        if self.config.enable_thread_dynamics:
            self._maybe_trigger_events()
    
    def get_agent_modifiers(self, agent_id: int) -> AgentModifiers:
        """Get behavior modifiers for an agent.
        
        Args:
            agent_id: The agent's ID
        
        Returns:
            AgentModifiers with any active modifiers for this agent
        """
        modifiers = AgentModifiers()
        
        if not self.config.enable_thread_dynamics:
            return modifiers
        
        # Check if agent is participating in any active events
        for event in self._active_events:
            if agent_id in event.participants:
                # Merge event modifiers
                modifiers.aggression_boost = max(
                    modifiers.aggression_boost,
                    event.aggression_boost
                )
                modifiers.token_density_boost = max(
                    modifiers.token_density_boost,
                    event.token_density_boost
                )
                modifiers.coordination_hints.extend(event.coordination_hints)
                modifiers.active_event_type = event.event_type
                modifiers.target_thread_id = event.target_thread_id
        
        return modifiers
    
    def report_post(self, post_data: Dict[str, Any]) -> None:
        """Report a new post for event triggering.
        
        Args:
            post_data: Post data including thread_id, user_id, content
        """
        self._recent_posts.append(post_data)
        
        # Keep only recent posts
        if len(self._recent_posts) > 100:
            self._recent_posts = self._recent_posts[-100:]
        
        # Track hot threads (threads with multiple recent posts)
        thread_id = post_data.get("thread_id")
        if thread_id:
            self._hot_threads.add(thread_id)
    
    def _maybe_trigger_events(self) -> None:
        """Potentially trigger new coordination events."""
        # Don't trigger too many events at once
        if len(self._active_events) >= 3:
            return
        
        # Roll for each event type
        if self.rng.random() < self.config.pile_on_probability:
            self._trigger_pile_on()
        
        if self.rng.random() < self.config.echo_chamber_probability:
            self._trigger_echo_chamber()
        
        if self.rng.random() < self.config.debate_probability:
            self._trigger_debate()
        
        if self.rng.random() < self.config.brigade_probability:
            self._trigger_brigade()
    
    def _trigger_pile_on(self) -> None:
        """Trigger a pile-on event."""
        # Get agents with pile-on affinity
        affinity_archetypes = set(self.config.archetype_affinities.get("pile_on", []))
        candidates = [
            aid for aid, arch in self._agents.items()
            if arch in affinity_archetypes
        ]
        
        if len(candidates) < self.config.min_participants:
            return
        
        # Select participants
        num_participants = min(
            self.rng.randint(self.config.min_participants, self.config.max_participants),
            len(candidates)
        )
        participants = set(self.rng.sample(candidates, num_participants))
        
        # Create event
        event = CoordinationEvent(
            event_type="pile_on",
            target_thread_id=self._pick_hot_thread(),
            participants=participants,
            intensity=self.rng.uniform(1.0, 1.5),
            started_at_step=self._current_step,
            duration_steps=self.config.event_duration_steps,
            aggression_boost=0.3,
            token_density_boost=0.2,
            coordination_hints=[
                "Express strong agreement with others attacking the target",
                "Use similar language to other commenters",
                "Escalate the criticism",
            ],
        )
        self._active_events.append(event)
    
    def _trigger_echo_chamber(self) -> None:
        """Trigger an echo chamber event."""
        affinity_archetypes = set(self.config.archetype_affinities.get("echo_chamber", []))
        candidates = [
            aid for aid, arch in self._agents.items()
            if arch in affinity_archetypes
        ]
        
        if len(candidates) < self.config.min_participants:
            return
        
        num_participants = min(
            self.rng.randint(self.config.min_participants, self.config.max_participants),
            len(candidates)
        )
        participants = set(self.rng.sample(candidates, num_participants))
        
        event = CoordinationEvent(
            event_type="echo_chamber",
            target_thread_id=None,  # No specific target
            participants=participants,
            intensity=self.rng.uniform(1.0, 1.3),
            started_at_step=self._current_step,
            duration_steps=self.config.event_duration_steps + 2,  # Longer duration
            aggression_boost=0.1,
            token_density_boost=0.15,
            coordination_hints=[
                "Reinforce the shared narrative",
                "Reference what others have said approvingly",
                "Build on previous points",
            ],
        )
        self._active_events.append(event)
    
    def _trigger_debate(self) -> None:
        """Trigger a debate event."""
        affinity_archetypes = set(self.config.archetype_affinities.get("debate", []))
        candidates = [
            aid for aid, arch in self._agents.items()
            if arch in affinity_archetypes
        ]
        
        if len(candidates) < 2:
            return
        
        num_participants = min(
            self.rng.randint(2, 4),
            len(candidates)
        )
        participants = set(self.rng.sample(candidates, num_participants))
        
        event = CoordinationEvent(
            event_type="debate",
            target_thread_id=self._pick_hot_thread(),
            participants=participants,
            intensity=self.rng.uniform(1.0, 1.4),
            started_at_step=self._current_step,
            duration_steps=self.config.event_duration_steps,
            aggression_boost=0.2,
            token_density_boost=0.1,
            coordination_hints=[
                "Directly address opposing viewpoints",
                "Present counter-arguments",
                "Challenge the other perspective",
            ],
        )
        self._active_events.append(event)
    
    def _trigger_brigade(self) -> None:
        """Trigger a brigade event."""
        affinity_archetypes = set(self.config.archetype_affinities.get("brigade", []))
        candidates = [
            aid for aid, arch in self._agents.items()
            if arch in affinity_archetypes
        ]
        
        if len(candidates) < self.config.min_participants:
            return
        
        num_participants = min(
            self.rng.randint(self.config.min_participants, self.config.max_participants),
            len(candidates)
        )
        participants = set(self.rng.sample(candidates, num_participants))
        
        event = CoordinationEvent(
            event_type="brigade",
            target_thread_id=self._pick_hot_thread(),
            participants=participants,
            intensity=self.rng.uniform(1.2, 1.6),
            started_at_step=self._current_step,
            duration_steps=self.config.event_duration_steps,
            aggression_boost=0.4,
            token_density_boost=0.3,
            coordination_hints=[
                "Coordinate with others to overwhelm the target",
                "Use similar talking points",
                "Amplify the group message",
            ],
        )
        self._active_events.append(event)
    
    def _pick_hot_thread(self) -> Optional[str]:
        """Pick a hot thread to target."""
        if not self._hot_threads:
            return None
        return self.rng.choice(list(self._hot_threads))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "current_step": self._current_step,
            "registered_agents": len(self._agents),
            "active_events": len(self._active_events),
            "event_types": [e.event_type for e in self._active_events],
            "hot_threads": len(self._hot_threads),
        }
