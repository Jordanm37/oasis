# Complete Pipeline Flow: Persona Generation → Content → Dataset

This document explains how all the pieces fit together and clarifies what is deterministic vs. non-deterministic.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE PIPELINE FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

STEP 1: PERSONA GENERATION (deterministic with seed)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  scripts/generate_personas_llm.py                                                        │
│                                                                                          │
│  Inputs:                                                                                 │
│    - ontology_unified.yaml (archetype definitions)                                       │
│    - style_indicators.yaml (per-class style pools)                                       │
│    - lexicons/*.yaml (vocabulary pools)                                                  │
│    - personality.csv (optional seed traits)                                              │
│    - --seed (determinism control)                                                        │
│                                                                                          │
│  Process:                                                                                │
│    1. Load ontology → get archetypes, variants, topics                                   │
│    2. Build persona requests from CLI counts (--benign 10, --incel 5, etc.)             │
│    3. For each persona:                                                                  │
│       a. Sample variant from archetype                                                   │
│       b. Call LLM for SPC blocks (S, P, C, narratives) ← NON-DETERMINISTIC              │
│       c. Sample style indicators (probabilistic, seeded)                                 │
│       d. Sample lexicon terms (probabilistic, seeded)                                    │
│       e. Determine label_mode_cap (single/double)                                        │
│    4. Write personas.csv                                                                 │
│                                                                                          │
│  Outputs:                                                                                │
│    - personas.csv with columns:                                                          │
│      username, description, user_char, primary_label, secondary_label,                   │
│      allowed_labels, s_json, p_json, c_json, narratives_json,                           │
│      prompt_metadata_json, lexicon_samples_json, style_variation_json                    │
│                                                                                          │
│  Determinism:                                                                            │
│    ⚠️  LLM calls are NON-DETERMINISTIC (different runs → different SPC blocks)          │
│    ✓  Sampling (style, lexicon) is DETERMINISTIC with --seed                            │
│    ✓  Persona allocation is DETERMINISTIC with --seed                                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
STEP 1.5: PERSONA ENHANCEMENT (optional, deterministic with seed)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  generation/enhanced_persona_generator.py                                                │
│                                                                                          │
│  Inputs:                                                                                 │
│    - personas.csv (from Step 1)                                                          │
│    - seed                                                                                │
│                                                                                          │
│  Process:                                                                                │
│    For each persona:                                                                     │
│    1. Sample trajectory_stage (curious/active/entrenched/disillusioned)                  │
│    2. Sample slang_fluency (native/fluent/learning/outsider)                             │
│    3. Sample secondary_archetype (contamination)                                         │
│    4. Sample demographic_profile                                                         │
│    5. Calculate benign_post_rate, token_density_multiplier, etc.                         │
│                                                                                          │
│  Outputs:                                                                                │
│    - personas_enhanced.csv (original columns + new enhancement columns)                  │
│                                                                                          │
│  Determinism:                                                                            │
│    ✓  All sampling is DETERMINISTIC with seed                                            │
│    ✓  Same input + seed → same output                                                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
STEP 2: GRAPH GENERATION (deterministic with seed)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  scripts/build_graph.py                                                                  │
│                                                                                          │
│  Inputs:                                                                                 │
│    - personas.csv                                                                        │
│    - graph parameters (density, clustering)                                              │
│    - seed                                                                                │
│                                                                                          │
│  Process:                                                                                │
│    1. Create nodes for each persona                                                      │
│    2. Generate edges based on:                                                           │
│       - Archetype affinity (similar archetypes more likely to follow)                    │
│       - Random connections                                                               │
│       - Clustering parameters                                                            │
│                                                                                          │
│  Outputs:                                                                                │
│    - edges.csv (follower_id, followee_id)                                               │
│                                                                                          │
│  Determinism:                                                                            │
│    ✓  DETERMINISTIC with seed                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
STEP 3: SIMULATION (partially deterministic)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  scripts/run_production_sim.py                                                           │
│                                                                                          │
│  Inputs:                                                                                 │
│    - manifest.yaml (run configuration)                                                   │
│    - personas.csv                                                                        │
│    - edges.csv                                                                           │
│    - run_seed (from manifest)                                                            │
│                                                                                          │
│  Process:                                                                                │
│    1. Build agent graph from personas.csv                                                │
│       - Each agent gets: ExtendedSocialAgent with persona config                         │
│       - EmissionPolicy attached (controls label token injection)                         │
│                                                                                          │
│    2. Create environment (Platform, Channel, DB)                                         │
│                                                                                          │
│    3. Seed initial follows from edges.csv                                                │
│                                                                                          │
│    4. For each step:                                                                     │
│       a. Build actions dict: {agent: LLMAction()}                                        │
│       b. env.step(actions) → for each agent:                                             │
│          i.   EmissionPolicy.decide() → mode (none/single/double), tokens                │
│          ii.  Build step hint with tokens                                                │
│          iii. Get environment prompt (current posts, timeline)                           │
│          iv.  Call LLM with system prompt + step hint + env prompt                       │
│          v.   LLM generates content WITH label tokens embedded                           │
│          vi.  Content saved to DB (post or comment table)                                │
│                                                                                          │
│    5. RagImputer (background):                                                           │
│       - Scans DB for rows with LBL:* tokens                                              │
│       - Calls LLM to generate natural replacements                                       │
│       - Saves to text_rag_imputed column                                                 │
│                                                                                          │
│  Outputs:                                                                                │
│    - simulation.db (SQLite with post, comment, user, follow tables)                      │
│    - sidecar.jsonl (logging of expected vs actual tokens)                                │
│                                                                                          │
│  Determinism:                                                                            │
│    ⚠️  LLM content generation is NON-DETERMINISTIC                                       │
│    ✓  EmissionPolicy decisions are DETERMINISTIC with run_seed                           │
│    ✓  Agent ordering is DETERMINISTIC                                                    │
│    ⚠️  RagImputer replacements are NON-DETERMINISTIC (LLM calls)                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
STEP 4: DATASET EXPORT (deterministic)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  scripts/build_dataset.py                                                                │
│                                                                                          │
│  Inputs:                                                                                 │
│    - simulation.db                                                                       │
│    - personas.csv (for label mapping)                                                    │
│    - static_bank.yaml (fallback replacements)                                            │
│                                                                                          │
│  Process:                                                                                │
│    1. Load persona → label mapping from CSV                                              │
│    2. For each post/comment in DB:                                                       │
│       a. Get text_rag_imputed (if exists) or raw content                                 │
│       b. Map user_id → primary_label from personas                                       │
│       c. Build record with text, label, metadata                                         │
│    3. Write JSONL                                                                        │
│                                                                                          │
│  Outputs:                                                                                │
│    - dataset.jsonl                                                                       │
│                                                                                          │
│  Determinism:                                                                            │
│    ✓  DETERMINISTIC (just reads and transforms)                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Where Each Enhancement Fits

### Current Flow (What Exists)

```
generate_personas_llm.py → personas.csv
                               │
                               ▼
                    build_graph.py → edges.csv
                               │
                               ▼
                    run_production_sim.py
                               │
                               ├── agent_factory.py (builds ExtendedSocialAgent)
                               │       │
                               │       └── ExtendedSocialAgent.perform_action_by_llm()
                               │               │
                               │               ├── EmissionPolicy.decide() → tokens
                               │               │
                               │               ├── _format_step_hint() → "Use LBL:X"
                               │               │
                               │               └── LLM generates content
                               │
                               ├── RagImputer (replaces LBL:* tokens)
                               │
                               └── simulation.db
                                       │
                                       ▼
                               build_dataset.py → dataset.jsonl
```

### Enhanced Flow (What We're Adding)

```
generate_personas_llm.py → personas.csv
                               │
                               ▼
              [NEW] enhanced_persona_generator.py → personas_enhanced.csv
                               │                    (adds trajectory, fluency, etc.)
                               ▼
                    build_graph.py → edges.csv
                               │
                               ▼
                    run_production_sim.py
                               │
                               ├── agent_factory.py 
                               │       │
                               │       └── ExtendedSocialAgent + [NEW] EnhancedAgentMixin
                               │               │
                               │               ├── [NEW] should_generate_benign_post()
                               │               │
                               │               ├── EmissionPolicy.decide() 
                               │               │       │
                               │               │       └── [NEW] apply_trajectory_modifiers()
                               │               │
                               │               ├── [NEW] get_enhanced_step_hint() 
                               │               │       (implicit harm prompts)
                               │               │
                               │               └── LLM generates content
                               │
                               ├── [NEW] ObfuscationProcessor (NON-DETERMINISTIC)
                               │       (applied to raw content before imputation)
                               │
                               ├── RagImputer (replaces LBL:* tokens)
                               │
                               ├── [NEW] ThreadCoordinator
                               │       (orchestrates pile-ons, echo chambers)
                               │
                               └── simulation.db
```

## Determinism Analysis

### What SHOULD Be Deterministic

| Component | Why Deterministic | Controlled By |
|-----------|------------------|---------------|
| Persona allocation | Reproducible experiments | `--seed` |
| Style indicator sampling | Consistent persona identity | `--seed` |
| Lexicon sampling | Consistent vocabulary | `--seed` |
| Trajectory assignment | Reproducible persona variation | `seed` |
| Fluency assignment | Reproducible persona variation | `seed` |
| Contamination assignment | Reproducible cross-archetype | `seed` |
| Graph generation | Reproducible social structure | `seed` |
| EmissionPolicy decisions | Reproducible token injection | `run_seed` |

### What SHOULD Be Non-Deterministic

| Component | Why Non-Deterministic | Implementation |
|-----------|----------------------|----------------|
| LLM content generation | Variety in writing style | LLM temperature |
| LLM SPC generation | Variety in persona details | LLM temperature |
| RagImputer replacements | Variety in token substitution | LLM temperature |
| **Obfuscation patterns** | **Unpredictable evasion** | **Time-based entropy** |

### Why Obfuscation Must Be Non-Deterministic

```python
# BAD: Deterministic obfuscation
step_seed = hash((run_seed, agent_id, step_index)) 
rng = random.Random(step_seed)
# Problem: Attacker can predict which words get l33t-speak'd

# GOOD: Non-deterministic obfuscation
entropy_seed = int(time.time() * 1000000) ^ os.getpid() ^ id(self)
rng = random.Random(entropy_seed)
# Result: Unpredictable obfuscation patterns
```

## Thread Coordination - Current Gap

**Current state**: Agents act independently. Each agent:
1. Observes its timeline
2. Decides action independently
3. No awareness of other agents' current actions

**What ThreadCoordinator adds**:
```python
# In run_production_sim.py (proposed integration):

coordinator = ThreadCoordinator(seed=run_seed)

# Register agents with their archetypes
for agent_id, agent in env.agent_graph.get_agents():
    coordinator.register_agent(agent_id, agent.persona_cfg.primary_label)

# In step loop:
for step in range(steps):
    coordinator.step(step)  # Update dynamics
    
    for agent_id, agent in env.agent_graph.get_agents():
        # Get coordination modifiers
        modifiers = coordinator.get_agent_modifiers(agent_id, step)
        
        if modifiers["is_participant"]:
            # Agent is part of a pile-on, echo chamber, etc.
            # Modify their behavior accordingly
            agent.apply_coordination_modifiers(modifiers)
```

## Summary: What Each File Does

| File | Purpose | Deterministic? |
|------|---------|---------------|
| `generate_personas_llm.py` | Create personas with LLM SPC | Partial (LLM non-det) |
| `enhanced_persona_generator.py` | Add trajectory/fluency/contamination | ✓ Yes |
| `build_graph.py` | Create follow network | ✓ Yes |
| `run_production_sim.py` | Run simulation | Partial (LLM non-det) |
| `agent_factory.py` | Build agents from CSV | ✓ Yes |
| `extended_agent.py` | Agent with emission policy | Partial |
| `emission_policy.py` | Decide token injection | ✓ Yes |
| `implicit_harm_prompts.py` | Behavioral instructions | ✓ Yes |
| `trajectory_config.py` | Trajectory/fluency configs | ✓ Yes |
| `contamination_config.py` | Cross-archetype influence | ✓ Yes |
| `obfuscation_processor.py` | Evasion patterns | ✗ No (intentional) |
| `thread_coordinator.py` | Multi-agent dynamics | ✓ Yes |
| `rag_llm_imputer.py` | Replace LBL:* tokens | Partial (LLM non-det) |
| `build_dataset.py` | Export to JSONL | ✓ Yes |

