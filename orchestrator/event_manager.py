"""Event Manager: Triggers coordinated behavior based on simulated events.

This module manages simulation events that temporarily modify agent behavior.
Events are defined in YAML and activated at specific simulation steps.

Usage:
    event_manager = EventManager(
        config_path=Path("configs/simulation_events.yaml"),
        seed=42,
    )
    
    # Each simulation step:
    active_events = event_manager.step(current_step)
    
    # Get modifiers for an agent:
    modifiers = event_manager.get_agent_modifiers(agent_archetype)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """A simulation event that modifies agent behavior.
    
    Attributes:
        id: Unique identifier for the event
        event_type: Category of event (news_event, platform_action, cultural_moment)
        description: Human-readable description
        trigger_step: Step when this event activates
        duration_steps: How many steps the event lasts
        affects_archetypes: List of archetypes affected by this event
        modifiers: Dict of behavior modifiers (aggression_boost, etc.)
        topic_keywords: Keywords that trigger engagement
        inject_narrative: Narrative to inject into agent prompts
        expires_at: Step when event expires (computed)
        is_active: Whether event is currently active
    """
    
    id: str
    event_type: str
    description: str
    trigger_step: int
    duration_steps: int
    affects_archetypes: List[str]
    modifiers: Dict[str, float]
    topic_keywords: List[str]
    inject_narrative: str
    probability: float = 1.0
    expires_at: int = 0
    is_active: bool = False
    
    def activate(self, current_step: int) -> None:
        """Activate this event."""
        self.is_active = True
        self.expires_at = current_step + self.duration_steps
        logger.info(f"Event '{self.id}' activated at step {current_step}, expires at {self.expires_at}")
    
    def is_expired(self, current_step: int) -> bool:
        """Check if this event has expired."""
        return current_step >= self.expires_at
    
    def deactivate(self) -> None:
        """Deactivate this event."""
        self.is_active = False
        logger.debug(f"Event '{self.id}' deactivated")


@dataclass
class EventModifiers:
    """Modifiers to apply to an agent based on active events.
    
    Attributes:
        aggression_boost: Increase in aggression level (0.0-1.0)
        label_frequency_boost: Increase in label token frequency (0.0-1.0)
        reply_probability: Override reply probability (None = no override)
        create_post_probability: Override post creation probability (None = no override)
        anti_institution_boost: Increase in anti-institution rhetoric (0.0-1.0)
        inject_narratives: List of narratives to inject into prompts
        topic_keywords: Set of keywords that should trigger engagement
        active_event_ids: List of active event IDs affecting this agent
        has_active_event: Whether any event is active for this agent
    """
    
    aggression_boost: float = 0.0
    label_frequency_boost: float = 0.0
    reply_probability: Optional[float] = None
    create_post_probability: Optional[float] = None
    anti_institution_boost: float = 0.0
    inject_narratives: List[str] = field(default_factory=list)
    topic_keywords: List[str] = field(default_factory=list)
    active_event_ids: List[str] = field(default_factory=list)
    has_active_event: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to agent."""
        return {
            "aggression_boost": self.aggression_boost,
            "label_frequency_boost": self.label_frequency_boost,
            "reply_probability": self.reply_probability,
            "create_post_probability": self.create_post_probability,
            "anti_institution_boost": self.anti_institution_boost,
            "inject_narratives": self.inject_narratives,
            "topic_keywords": self.topic_keywords,
            "active_event_ids": self.active_event_ids,
            "has_active_event": self.has_active_event,
        }


class EventManager:
    """Manages simulation events that trigger coordinated behavior.
    
    The EventManager:
    1. Loads event definitions from YAML config
    2. Resolves random trigger steps within defined ranges
    3. Activates/deactivates events each simulation step
    4. Provides combined modifiers for agents based on their archetype
    
    Example:
        >>> manager = EventManager(Path("configs/simulation_events.yaml"), seed=42)
        >>> manager.step(10)  # Process step 10
        [SimulationEvent(id='celebrity_relationship_drama', ...)]
        >>> mods = manager.get_agent_modifiers("incel_misogyny")
        >>> mods.aggression_boost
        0.3
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        seed: int = 42,
        enabled: bool = True,
    ):
        """Initialize the EventManager.
        
        Args:
            config_path: Path to YAML config file with event definitions
            seed: Random seed for reproducible event timing
            enabled: Whether event system is enabled
        """
        self.rng = random.Random(seed)
        self.enabled = enabled
        self.events: List[SimulationEvent] = []
        self.active_events: List[SimulationEvent] = []
        self._current_step = 0
        
        if config_path and config_path.exists():
            self._load_events(config_path)
        elif config_path:
            logger.warning(f"Event config not found: {config_path}")
    
    def _load_events(self, config_path: Path) -> None:
        """Load events from YAML config file.
        
        Args:
            config_path: Path to YAML config file
        """
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load event config: {e}")
            return
        
        events_data = data.get("events", [])
        logger.info(f"Loading {len(events_data)} event definitions from {config_path}")
        
        loaded_count = 0
        for evt_data in events_data:
            try:
                event = self._parse_event(evt_data)
                if event:
                    self.events.append(event)
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
        
        logger.info(f"Loaded {loaded_count} events (some may have been skipped due to probability)")
    
    def _parse_event(self, evt_data: Dict[str, Any]) -> Optional[SimulationEvent]:
        """Parse a single event from config data.
        
        Args:
            evt_data: Dictionary with event configuration
            
        Returns:
            SimulationEvent if event should occur, None otherwise
        """
        # Check probability
        probability = evt_data.get("probability", 1.0)
        if self.rng.random() > probability:
            logger.debug(f"Event '{evt_data.get('id')}' skipped (probability {probability})")
            return None
        
        # Resolve trigger step from range
        step_range = evt_data.get("trigger_step_range", [10, 20])
        if isinstance(step_range, list) and len(step_range) == 2:
            trigger_step = self.rng.randint(step_range[0], step_range[1])
        else:
            trigger_step = int(step_range) if step_range else 10
        
        return SimulationEvent(
            id=evt_data.get("id", f"event_{self.rng.randint(1000, 9999)}"),
            event_type=evt_data.get("type", "generic"),
            description=evt_data.get("description", ""),
            trigger_step=trigger_step,
            duration_steps=evt_data.get("duration_steps", 3),
            affects_archetypes=evt_data.get("affects_archetypes", []),
            modifiers=evt_data.get("modifiers", {}),
            topic_keywords=evt_data.get("topic_keywords", []),
            inject_narrative=evt_data.get("inject_narrative", ""),
            probability=probability,
        )
    
    def step(self, current_step: int) -> List[SimulationEvent]:
        """Process a simulation step: activate new events, expire old ones.
        
        Args:
            current_step: Current simulation step number
            
        Returns:
            List of currently active events
        """
        if not self.enabled:
            return []
        
        self._current_step = current_step
        
        # Activate new events that should trigger this step
        for event in self.events:
            if event.trigger_step == current_step and not event.is_active:
                event.activate(current_step)
                self.active_events.append(event)
        
        # Expire old events
        expired = [e for e in self.active_events if e.is_expired(current_step)]
        for event in expired:
            event.deactivate()
        self.active_events = [e for e in self.active_events if not e.is_expired(current_step)]
        
        return self.active_events
    
    def get_agent_modifiers(self, archetype: str) -> EventModifiers:
        """Get combined modifiers for an agent based on active events.
        
        Args:
            archetype: The agent's primary archetype label
            
        Returns:
            EventModifiers with combined effects from all active events
        """
        if not self.enabled or not self.active_events:
            return EventModifiers()
        
        # Combine modifiers from all active events affecting this archetype
        combined = EventModifiers()
        
        for event in self.active_events:
            if archetype not in event.affects_archetypes:
                continue
            
            combined.has_active_event = True
            combined.active_event_ids.append(event.id)
            
            mods = event.modifiers
            combined.aggression_boost += mods.get("aggression_boost", 0.0)
            combined.label_frequency_boost += mods.get("label_frequency_boost", 0.0)
            combined.anti_institution_boost += mods.get("anti_institution_boost", 0.0)
            
            # Override probabilities (take max)
            if mods.get("reply_probability"):
                if combined.reply_probability is None:
                    combined.reply_probability = mods["reply_probability"]
                else:
                    combined.reply_probability = max(
                        combined.reply_probability,
                        mods["reply_probability"]
                    )
            
            if mods.get("create_post_probability"):
                if combined.create_post_probability is None:
                    combined.create_post_probability = mods["create_post_probability"]
                else:
                    combined.create_post_probability = max(
                        combined.create_post_probability,
                        mods["create_post_probability"]
                    )
            
            # Collect narratives and keywords
            if event.inject_narrative:
                combined.inject_narratives.append(event.inject_narrative)
            combined.topic_keywords.extend(event.topic_keywords)
        
        # Deduplicate keywords
        combined.topic_keywords = list(set(combined.topic_keywords))
        
        return combined
    
    def get_active_event_summary(self) -> Dict[str, Any]:
        """Get a summary of currently active events for logging.
        
        Returns:
            Dictionary with active event information
        """
        return {
            "current_step": self._current_step,
            "active_count": len(self.active_events),
            "active_events": [
                {
                    "id": e.id,
                    "type": e.event_type,
                    "expires_at": e.expires_at,
                    "affects": e.affects_archetypes,
                }
                for e in self.active_events
            ],
            "total_events": len(self.events),
        }
    
    def get_upcoming_events(self, lookahead: int = 10) -> List[SimulationEvent]:
        """Get events that will trigger in the next N steps.
        
        Args:
            lookahead: Number of steps to look ahead
            
        Returns:
            List of upcoming events
        """
        upcoming = []
        for event in self.events:
            if not event.is_active:
                if self._current_step <= event.trigger_step <= self._current_step + lookahead:
                    upcoming.append(event)
        return upcoming

