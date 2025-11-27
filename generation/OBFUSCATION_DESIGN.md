# Obfuscation Design: Post-Imputation Targeting

## Key Insight

Obfuscation should happen **AFTER** RagImputer, not before. This targets the **actual harmful content** rather than random neutral words.

## Comparison

### ❌ Before Imputation (Original Design - Less Optimal)

```
LLM generates: "These LBL:INCEL_SLANG always complain about dating"
                        ↓
Obfuscation:   "These LBL:INCEL_SLANG alw4ys compl4in about dating"
                        ↓                 ↑↑↑↑↑↑↑ ↑↑↑↑↑↑↑↑
                        │                 Neutral words obfuscated!
                        ↓
RagImputer:    "These foids alw4ys compl4in about dating"
                      ↑↑↑↑↑
                      Harmful term NOT obfuscated!
```

**Problem**: We obfuscated "always" and "complain" (neutral) but left "foids" (harmful) untouched.

### ✅ After Imputation (Better Design)

```
LLM generates: "These LBL:INCEL_SLANG always complain about dating"
                        ↓
RagImputer:    "These foids always complain about dating"
                      ↑↑↑↑↑
                      Harmful term imputed
                        ↓
Obfuscation:   "These f0ids always complain about dating"
                      ↑↑↑↑↑
                      Harmful term obfuscated!
```

**Result**: We obfuscate what matters - the actual slur.

## Implementation

### RagImputerConfig

```python
@dataclass
class RagImputerConfig:
    # ... existing fields ...
    
    # Post-imputation obfuscation settings
    enable_obfuscation: bool = False  # Set True to enable
    obfuscation_deterministic: bool = False  # Keep False for variety
```

### PostImputationObfuscator

The obfuscator targets specific harmful term patterns:

```python
OBFUSCATION_TARGET_PATTERNS = {
    "incel_slang": ["foid", "femoid", "stacy", "chad", "blackpill", ...],
    "threats": ["kill", "die", "death", "murder", "kys", ...],
    "ed_terms": ["purge", "restrict", "thinspo", "meanspo", ...],
    "hate_terms": ["vermin", "subhuman", "degenerate", ...],
}
```

### Archetype-Specific Configs

Not all archetypes obfuscate - some use coded language instead:

| Archetype | Obfuscates? | Why |
|-----------|-------------|-----|
| `incel_misogyny` | ✅ Yes | Evade moderation of slang |
| `hate_speech` | ✅ Yes | Evade slur detection |
| `extremist` | ✅ Yes | Evade threat detection |
| `pro_ana` | ✅ Yes (high) | ED content heavily moderated |
| `conspiracy` | ❌ No | Uses dog whistles instead |
| `trad` | ❌ No | Uses coded aesthetic language |
| `alpha` | ❌ No | Lectures openly |

### Trajectory Effects

```python
trajectory_multiplier = {
    "curious": 0.5,      # New users don't know to obfuscate
    "active": 1.0,       # Normal obfuscation
    "entrenched": 1.3,   # More experienced at evasion
    "disillusioned": 0.7,
}
```

## Usage

### Enable in RagImputer

```python
imputer_cfg = RagImputerConfig(
    db_path=db_path,
    llm_settings=settings,
    enable_obfuscation=True,  # Enable post-imputation obfuscation
    obfuscation_deterministic=False,  # Keep non-deterministic
)
```

### Standalone Usage

```python
from generation.post_imputation_obfuscator import obfuscate_imputed_text

# After imputation:
text = "These foids always complain about dating"
obfuscated, terms = obfuscate_imputed_text(
    text=text,
    archetype="incel_misogyny",
    trajectory_stage="active",
)
# Result: "These f0ids always complain about dating"
# terms: ["foids → f0ids"]
```

## Non-Determinism

Obfuscation is **intentionally non-deterministic** to ensure variety:

```python
# Uses time-based entropy, not seeded RNG
entropy = int(time.time() * 1000000) ^ os.getpid() ^ id(self)
rng = random.Random(entropy)
```

This prevents attackers from predicting obfuscation patterns.

## Files

| File | Purpose |
|------|---------|
| `generation/post_imputation_obfuscator.py` | Main obfuscator (NEW) |
| `generation/obfuscation_processor.py` | Original pre-imputation (DEPRECATED for this use case) |
| `oasis/imputation/rag_llm_imputer.py` | Updated to call obfuscator |

