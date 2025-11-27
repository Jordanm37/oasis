# Enhanced Persona Generation System

This document describes the 8 improvements to persona fidelity and content realism for the OASIS classification dataset.

## Overview

The enhanced system adds multiple layers of variation and realism while **preserving the label token flow** for the RagImputer.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ENHANCED GENERATION PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────────────┐
                                    │   Ontology YAML      │
                                    │  (13 archetypes)     │
                                    └──────────┬───────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  IMPROVEMENT #1: Enhanced Persona Generator                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Trajectory Stage    │  │ Slang Fluency       │  │ Contamination       │              │
│  │ (curious/active/    │  │ (native/fluent/     │  │ (secondary          │              │
│  │  entrenched/        │  │  learning/outsider) │  │  archetype)         │              │
│  │  disillusioned)     │  │                     │  │                     │              │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘              │
│                                               │                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐                                       │
│  │ Demographic Profile │  │ Benign Post Rate    │                                       │
│  │ (gen_z/millennial/  │  │ (per-archetype      │                                       │
│  │  boomer/terminally  │  │  probability)       │                                       │
│  │  online)            │  │                     │                                       │
│  └─────────────────────┘  └─────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  Enhanced Personas   │
                                    │  CSV (with new cols) │
                                    └──────────┬───────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  IMPROVEMENT #2-7: Enhanced Agent Behavior                                               │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Per-Step Decision Flow                                                              ││
│  │                                                                                     ││
│  │   #2 Should this be benign? ──► benign_post_rate check                             ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   #4 Select post type ──► rant/observation/advice/story/meme/etc.                  ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   #5 Apply trajectory modifiers ──► token density, aggression                      ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   #6 Apply contamination ──► secondary archetype influence                         ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   #1 Build implicit harm prompt ──► behavioral instructions                        ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   LLM generates content WITH LABEL TOKENS (preserved for RagImputer)               ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   #7 Apply obfuscation ──► leetspeak, asterisks, sarcasm (tokens preserved)        ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   RagImputer replaces LBL:* tokens with natural language                           ││
│  │              │                                                                      ││
│  │              ▼                                                                      ││
│  │   Final content (realistic, obfuscated, imputed)                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                          │
│  #8 Thread Coordinator (multi-agent dynamics)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ • Pile-ons: Multiple toxic users attack a target                                   ││
│  │ • Echo chambers: Compatible archetypes reinforce each other                        ││
│  │ • Debates: Opposing viewpoints engage                                              ││
│  │ • Brigades: Coordinated attacks on content                                         ││
│  │ • Support rallies: Recovery users support each other                               ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Improvement Details

### #1: Implicit Harm Prompts (`implicit_harm_prompts.py`)

**Problem**: Explicit "Use LBL:INCEL_SLANG" instructions produce stilted content.

**Solution**: Behavioral instructions that describe HOW to write, not WHAT label to use.

```python
# OLD (explicit)
"Use exactly one label marker inline: LBL:INCEL_SLANG."

# NEW (implicit)
"""
[Voice] You are frustrated with dating and relationships. You feel the system 
is rigged against average men. You use community-specific terms naturally.

[Emotional Core] resentment, bitterness, nihilism, occasional rage

[Vocabulary]
  - Use terms like Chad, Stacy, blackpill, foid, mogging naturally
  - Reference lookism, the dating market, or female hypergamy

[Required Marker] Include the marker LBL:INCEL_SLANG inline once, 
wrapping the most characteristic phrase of your post.
"""
```

**Key**: Still requires the token for RagImputer, but frames it as marking the characteristic phrase.

### #2: Benign Post Injection (`enhanced_persona_generator.py`)

**Problem**: Toxic personas posting 100% toxic content is unrealistic.

**Solution**: Per-archetype benign post rates.

```python
BASE_BENIGN_POST_RATES = {
    "incel_misogyny": 0.15,  # Some normal posts about hobbies, games
    "trad": 0.30,            # Aesthetic, cooking, family content
    "extremist": 0.05,       # Rarely benign
    "hate_speech": 0.08,     # Rarely benign
}
```

### #3: Radicalization Trajectory (`trajectory_config.py`)

**Problem**: All personas of same archetype sound identical.

**Solution**: Trajectory stages with different characteristics.

| Stage | Token Density | Aggression | Slang Adoption | Benign Boost |
|-------|--------------|------------|----------------|--------------|
| curious | 0.5x | 0.6x | 20% | +20% |
| active | 1.0x | 1.0x | 70% | 0% |
| entrenched | 1.3x | 1.4x | 95% | -10% |
| disillusioned | 0.7x | 0.7x | 50% | +15% |

### #4: Post Type Diversification (`contamination_config.py`)

**Problem**: All posts are generic observations.

**Solution**: Archetype-specific post type distributions.

```python
POST_TYPE_DISTRIBUTION = {
    "incel_misogyny": {
        "rant": 0.30,           # Emotional venting
        "observation": 0.25,    # Pointing out patterns
        "personal_story": 0.20, # Sharing experiences
        "meme_response": 0.15,  # Short, punchy
        "question": 0.10,       # Asking community
    },
    "conspiracy": {
        "news_reaction": 0.35,  # Reacting to events
        "observation": 0.25,    # Connecting dots
        "call_to_action": 0.15, # Rallying others
        ...
    },
}
```

### #5: Slang Fluency Levels (`trajectory_config.py`)

**Problem**: All personas use community slang perfectly.

**Solution**: Fluency levels that affect vocabulary usage.

| Level | Abbreviations | Explains Terms | Misuse Rate | Insider Refs |
|-------|--------------|----------------|-------------|--------------|
| native | Yes | No | 0% | 80% |
| fluent | Yes | No | 5% | 50% |
| learning | No | Yes | 15% | 20% |
| outsider | No | No | 0% | 0% |

### #6: Cross-Archetype Contamination (`contamination_config.py`)

**Problem**: Each persona is purely one archetype.

**Solution**: Secondary archetype influence.

```python
CONTAMINATION_MATRIX = {
    "incel_misogyny": [
        ("alpha", 0.15),      # Some adopt alpha rhetoric
        ("conspiracy", 0.10), # Blackpill as conspiracy
        ("extremist", 0.08),  # Radicalization pipeline
        ("hate_speech", 0.12),
    ],
    "conspiracy": [
        ("misinfo", 0.40),    # Strong overlap
        ("extremist", 0.15),
    ],
}
```

### #7: Obfuscation Patterns (`obfuscation_processor.py`)

**Problem**: Content is too clean for realistic detection challenge.

**Solution**: Apply obfuscation BEFORE RagImputer (tokens preserved).

| Pattern | Example | Probability |
|---------|---------|-------------|
| leetspeak | h4t3 → hate | 10% |
| asterisk_censor | k*ll → kill | 15% |
| space_insertion | h a t e | 8% |
| homoglyph | hаte (Cyrillic a) | 8% |
| sarcasm_quotes | "Totally not saying..." | 12% |

### #8: Thread Dynamics (`thread_coordinator.py`)

**Problem**: Each agent acts independently.

**Solution**: Coordinate multi-agent interactions.

| Dynamic | Description | Participants |
|---------|-------------|--------------|
| pile_on | Multiple users attack target | 2-5 toxic |
| echo_chamber | Mutual reinforcement | Compatible archetypes |
| debate | Opposing viewpoints | Opposing archetypes |
| brigade | Coordinated attack on post | 2-5 coordinated |
| support_rally | Multiple users support | Benign/recovery |

## Integration Guide

### 1. Enhance Existing Personas

```python
from generation.enhanced_persona_generator import enhance_personas

# Enhance existing personas CSV
enhance_personas(
    input_path=Path("./data/personas.csv"),
    output_path=Path("./data/personas_enhanced.csv"),
    seed=42,
)
```

### 2. Use Enhanced Agent Mixin

```python
from generation.enhanced_agent_mixin import EnhancedAgentMixin

class MyEnhancedAgent(ExtendedSocialAgent, EnhancedAgentMixin):
    def __init__(self, *args, enhanced_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_enhanced_features(enhanced_config)
    
    async def perform_action_by_llm(self):
        # Check if this should be benign
        if self.should_generate_benign_post():
            # Override to benign mode
            ...
        
        # Get enhanced step hint
        hint = self.get_enhanced_step_hint(mode, tokens)
        
        # Apply trajectory modifiers
        decision = self.apply_trajectory_modifiers(decision)
        
        # ... generate content ...
        
        # Apply obfuscation (preserves tokens)
        content, patterns = self.apply_obfuscation_to_content(content)
```

### 3. Use Thread Coordinator

```python
from generation.thread_coordinator import ThreadCoordinator

coordinator = ThreadCoordinator(seed=42)

# Register agents
for agent_id, archetype in agents.items():
    coordinator.register_agent(agent_id, archetype)

# In simulation loop
for step in range(num_steps):
    coordinator.step(step)
    
    for agent in agents:
        modifiers = coordinator.get_agent_modifiers(agent.id, step)
        # Apply modifiers to agent behavior
```

## New CSV Columns

The enhanced personas CSV includes these new columns:

| Column | Type | Description |
|--------|------|-------------|
| trajectory_stage | str | curious/active/entrenched/disillusioned |
| slang_fluency | str | native/fluent/learning/outsider |
| secondary_archetype | str | Optional secondary archetype |
| secondary_influence | float | 0.0-1.0 influence strength |
| benign_post_rate | float | Probability of benign posts |
| token_density_multiplier | float | Trajectory-adjusted |
| aggression_multiplier | float | Trajectory-adjusted |
| demographic_profile | str | gen_z_urban/millennial_suburban/etc. |
| obfuscation_intensity | float | 0.0-1.0 |

## Files Created

| File | Purpose |
|------|---------|
| `generation/implicit_harm_prompts.py` | Behavioral prompt templates |
| `generation/trajectory_config.py` | Trajectory stages and fluency levels |
| `generation/contamination_config.py` | Cross-archetype influence and post types |
| `generation/obfuscation_processor.py` | Evasion pattern application |
| `generation/enhanced_persona_generator.py` | Enhanced persona creation |
| `generation/enhanced_agent_mixin.py` | Agent behavior enhancements |
| `generation/thread_coordinator.py` | Multi-agent dynamics |

## Classification Challenge Impact

These improvements create a more challenging and realistic classification dataset:

1. **Within-class variation**: Same archetype, different trajectory/fluency = different patterns
2. **Noise injection**: Benign posts from toxic users = harder to classify by volume
3. **Obfuscation**: Evasion patterns = harder pattern matching
4. **Cross-contamination**: Mixed signals from multiple archetypes
5. **Context dependency**: Thread dynamics = context matters for classification
6. **Realistic distribution**: Not all toxic content is equally toxic

This better reflects real-world online harm detection challenges.

