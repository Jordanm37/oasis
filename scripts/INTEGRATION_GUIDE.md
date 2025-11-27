# Integration Guide: Enhanced Simulation Features

This guide shows how to integrate the enhanced persona features into the existing simulation pipeline.

## Overview

The enhancements work **alongside** the existing RecSys, not replacing it:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXISTING vs ENHANCED FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

EXISTING (unchanged):
  RecSys → Agent sees timeline → Agent decides → LLM generates → DB

ENHANCED (additions in [brackets]):
  RecSys → Agent sees timeline 
         → [ThreadCoordinator modifiers]
         → Agent decides (EmissionPolicy + [trajectory modifiers])
         → LLM generates (with [implicit harm prompts])
         → [Obfuscation]
         → RagImputer
         → DB
```

## Integration Steps

### Step 1: Enhance Personas CSV (Optional)

If you want trajectory/fluency/contamination features, run:

```python
from generation.enhanced_persona_generator import enhance_personas
from pathlib import Path

enhance_personas(
    input_path=Path("./data/personas.csv"),
    output_path=Path("./data/personas_enhanced.csv"),
    seed=42,
)
```

This adds columns:
- `trajectory_stage`: curious/active/entrenched/disillusioned
- `slang_fluency`: native/fluent/learning/outsider
- `secondary_archetype`: cross-archetype contamination
- `benign_post_rate`: probability of benign posts
- `obfuscation_intensity`: how much obfuscation to apply

### Step 2: Update run_production_sim.py

Add these imports:

```python
from orchestrator.agent_factory_enhanced import build_enhanced_agent_graph_from_csv
from orchestrator.simulation_coordinator import (
    SimulationCoordinator,
    SimulationCoordinatorConfig,
    integrate_coordinator_with_step,
)
```

Replace the agent graph building:

```python
# BEFORE:
from orchestrator.agent_factory import build_agent_graph_from_csv
agent_graph = await build_agent_graph_from_csv(...)

# AFTER:
from orchestrator.agent_factory_enhanced import build_enhanced_agent_graph_from_csv
agent_graph = await build_enhanced_agent_graph_from_csv(
    personas_csv=personas_csv,
    model=model,
    channel=channel,
    available_actions=allowed_actions,
    emission_policy=policy,
    sidecar_logger=sidecar,
    run_seed=run_seed,
    guidance_config=manifest.guidance_config,
    enable_obfuscation=True,  # NEW
)
```

Initialize the coordinator:

```python
# After building agent_graph:
coordinator = SimulationCoordinator(
    config=SimulationCoordinatorConfig(
        enable_thread_dynamics=True,
        enable_obfuscation=True,
        obfuscation_base_intensity=0.3,
    ),
    seed=run_seed,
)

# Register agents with their archetypes
for agent_id, agent in agent_graph.get_agents():
    archetype = agent._persona_cfg.primary_label
    coordinator.register_agent(agent_id, archetype)
```

Update the step loop:

```python
for i in range(steps):
    # Advance coordinator
    coordinator.step()
    
    # Build actions with coordination modifiers
    actions = {}
    for agent_id, agent in env.agent_graph.get_agents():
        # Get coordination modifiers for this agent
        modifiers = coordinator.get_agent_modifiers(agent_id)
        
        # Store modifiers on agent for use in perform_action_by_llm
        agent._coordination_modifiers = modifiers
        
        actions[agent] = oasis.LLMAction()
    
    await env.step(actions)
    
    # Record posts for future thread dynamics
    # (This requires access to newly created posts from the step)
```

### Step 3: Update ExtendedSocialAgent (Optional)

To fully use coordination modifiers, update `extended_agent.py`:

```python
async def perform_action_by_llm(self):
    # ... existing code ...
    
    # Get coordination modifiers if available
    modifiers = getattr(self, '_coordination_modifiers', {})
    
    # Format coordination prompt
    coord_prompt = ""
    if modifiers.get("is_participant"):
        hints = modifiers.get("coordination_hints", [])
        if hints:
            coord_prompt = f"\n[Coordination] {'; '.join(hints[:2])}"
    
    # Add to step hint
    step_hint = self._format_step_hint(decision)
    if coord_prompt:
        step_hint = coord_prompt + "\n" + step_hint
    
    # ... rest of existing code ...
```

## How Each Feature Works

### 1. Thread Dynamics (ThreadCoordinator)

The coordinator tracks active "dynamics" - multi-agent interactions:

```python
# Pile-on: Multiple toxic users attack a target
dynamic = ThreadDynamic(
    dynamic_type=ThreadDynamicType.PILE_ON,
    initiator_id=5,           # Agent who started it
    participant_ids=[7, 12],  # Other participants
    target_id=3,              # Target user
    target_post_id=42,        # Post being attacked
    aggression_boost=0.3,     # +30% aggression
)

# When agent 7 acts:
modifiers = coordinator.get_agent_modifiers(7)
# Returns:
# {
#     "is_participant": True,
#     "dynamic_type": "pile_on",
#     "aggression_boost": 0.3,
#     "coordination_hints": ["Join in on the criticism"],
#     "target_post_id": 42,
# }
```

### 2. Obfuscation (Non-Deterministic)

Applied AFTER LLM generation, BEFORE RagImputer:

```python
# LLM generates:
content = "These LBL:INCEL_SLANG always complain about dating"

# Obfuscation (random, unpredictable):
obfuscated = coordinator.apply_obfuscation(
    content=content,
    archetype="incel_misogyny",
    trajectory_stage="active",
    intensity=0.3,
)
# Result (varies each time):
# "These LBL:INCEL_SLANG alw4ys compl4in about dating"
# or
# "These LBL:INCEL_SLANG always c*mplain about dating"
# etc.

# RagImputer then replaces LBL:INCEL_SLANG:
final = "These foids alw4ys compl4in about dating"
```

### 3. Trajectory Modifiers

Affect token emission and behavior:

```python
# Curious stage (new to community):
# - 50% token density (fewer harmful tokens)
# - 30% skip probability (sometimes no token at all)
# - +20% benign post rate

# Entrenched stage (true believer):
# - 130% token density (more harmful tokens)
# - 0% skip probability (always emits tokens)
# - -10% benign post rate
```

### 4. Benign Post Injection

Each toxic persona has a `benign_post_rate`:

```python
# At the start of each step:
if random.random() < agent.benign_post_rate:
    # This step generates benign content
    decision["mode"] = "none"
    decision["tokens"] = []
```

## Complete Example

```python
async def run_enhanced(manifest_path, personas_csv, db_path, steps):
    manifest = load_manifest(manifest_path)
    run_seed = manifest.run_seed
    
    # Build enhanced agent graph
    agent_graph = await build_enhanced_agent_graph_from_csv(
        personas_csv=personas_csv,
        model=model,
        channel=channel,
        enable_obfuscation=True,
        # ... other args
    )
    
    # Initialize coordinator
    coordinator = SimulationCoordinator(seed=run_seed)
    for agent_id, agent in agent_graph.get_agents():
        coordinator.register_agent(agent_id, agent._persona_cfg.primary_label)
    
    # Create environment
    env = oasis.make(agent_graph=agent_graph, platform=platform, database_path=str(db_path))
    await env.reset()
    
    # Run simulation
    for step in range(steps):
        coordinator.step()
        
        # Apply coordination modifiers
        for agent_id, agent in env.agent_graph.get_agents():
            agent._coordination_modifiers = coordinator.get_agent_modifiers(agent_id)
        
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)
        
        # Record new posts for thread dynamics
        # (Implementation depends on how you access new posts)
    
    await env.close()
```

## Key Points

1. **RecSys is unchanged** - agents still see the same timeline
2. **Coordination modifies behavior** - not what agents see, but how they respond
3. **Obfuscation is non-deterministic** - unpredictable for realistic variety
4. **Trajectory affects token emission** - within-class variation
5. **Benign injection adds noise** - toxic personas sometimes post normal content

