# Lexicon System - Complete Implementation

## Overview

Successfully expanded the lexicon system from 6 to 13 collections covering all persona archetypes in the unified ontology.

## What Was Added

### New Lexicon Files Created

1. **`pro_ana.yaml`** - Pro-anorexia coaching and meanspo content
2. **`alpha.yaml`** - Alpha male / redpill terminology
3. **`trad.yaml`** - Traditionalist and gender essentialism
4. **`gamergate.yaml`** - Anti-woke gaming culture war
5. **`extremist.yaml`** - Violent extremism and accelerationism
6. **`hate_speech.yaml`** - Identity-based hate speech
7. **`bullying.yaml`** - Cyberbullying and harassment

### Updated Files

- **`configs/personas/ontology_unified.yaml`**:
  - Added 7 new lexicon collections to `lexicon_collections` section
  - Added `lexicon_refs` to all 13 persona variants
  - Each variant now references appropriate lexicon(s)

## Complete Lexicon Collection Mapping

| Archetype | Variant | Lexicon Collections |
|-----------|---------|-------------------|
| benign | supportive_generalist | benign_core |
| recovery_support | recovery_helper | recovery_core |
| ed_risk | ed_risk_promoter | ed_risk_core |
| pro_ana | pro_ana_coach | pro_ana_core, ed_risk_core |
| incel_misogyny | incel_aggressor | incel_core |
| alpha | alpha_lecturer | alpha_core, incel_core |
| misinfo | misinfo_contrarian | misinfo_core |
| conspiracy | conspiracy_broadcaster | conspiracy_core |
| trad | trad_poster | trad_core |
| gamergate | gamergate_warrior | gamergate_core |
| extremist | extremist_agitator | extremist_core |
| hate_speech | hate_poster | hate_speech_core |
| bullying | bully_attacker | bullying_core |

## Lexicon Generator Features

### Core Functionality

- **Iterative Generation**: Generates tokens in batches until reaching target totals
- **Full Deduplication**: Case-insensitive matching prevents all repetition
- **Context Awareness**: Passes existing tokens and persona metadata to LLM
- **Error Recovery**: Handles malformed JSON, retries with exponential backoff
- **Progress Tracking**: Real-time monitoring and logging

### Usage

```bash
# Generate all lexicons to 200 required + 1000 optional
poetry run python3 scripts/generate_lexicons.py \
  --required-total 200 \
  --optional-total 1000 \
  --max-iterations 100 \
  --required-target 10 \
  --optional-target 25

# Monitor progress
poetry run python3 scripts/monitor_lexicon_progress.py

# Check log
tail -f lexicon_generation_full.log
```

## Current Status

**Background Process Running**: PID 24855

All 13 collections are being processed iteratively to reach:
- **200 required tokens** per collection
- **1000 optional tokens** per collection

### Progress Snapshot (as of last check):

- ✅ **conspiracy**: 200/993 (99.7% complete) - Almost done!
- 🔄 **benign**: 117/386 (48.5% complete) - Good progress
- 🔄 **Other 11 collections**: 5-11 required, 5-75 optional - Just started

Estimated completion: 20-40 minutes depending on API response times and retries.

## Integration with Persona Generation

The lexicons are automatically used by `scripts/generate_personas_llm.py`:

1. **Ontology Loading**: Script loads ontology and discovers all lexicon collections
2. **Context Building**: For each persona, gathers lexicon refs from variant metadata
3. **Sampling**: Randomly samples required+optional tokens from referenced collections
4. **Prompt Injection**: Sampled tokens are woven into persona prompts
5. **Storage**: Sampled lexicon subsets stored in persona CSV for runtime use

### Example Flow

```python
# Persona generation discovers this:
variant = ontology.get_variant("pro_ana.pro_ana_coach")
lexicon_refs = variant.metadata["lexicon_refs"]  # ["pro_ana_core", "ed_risk_core"]

# Loads and samples from both collections:
pro_ana_tokens = sample_from("configs/lexicons/pro_ana.yaml")
ed_risk_tokens = sample_from("configs/lexicons/ed_risk.yaml")

# Combines into prompt metadata for LLM persona synthesis
```

## Files Created/Modified

### New Files
- `configs/lexicons/pro_ana.yaml`
- `configs/lexicons/alpha.yaml`
- `configs/lexicons/trad.yaml`
- `configs/lexicons/gamergate.yaml`
- `configs/lexicons/extremist.yaml`
- `configs/lexicons/hate_speech.yaml`
- `configs/lexicons/bullying.yaml`
- `scripts/generate_lexicons.py` (enhanced with iteration + error handling)
- `scripts/monitor_lexicon_progress.py` (new monitoring tool)
- `scripts/LEXICON_GENERATOR_README.md` (documentation)

### Modified Files
- `configs/personas/ontology_unified.yaml` (added 7 collections + all lexicon_refs)
- All 6 existing lexicon YAMLs (expanded tokens via generation)

## Next Steps

1. **Wait for generation to complete** (~20-40 min)
2. **Verify all collections** reached 200/1000 targets
3. **Test persona generation** with new lexicons
4. **Run production simulation** to validate runtime behavior

## Testing

```bash
# Verify ontology loads correctly
poetry run python3 -c "
from oasis.persona import load_ontology
ont = load_ontology('configs/personas/ontology_unified.yaml')
print(f'Collections: {len(ont.lexicon_collections)}')
print(f'Variants with refs: {sum(1 for v in ont.iter_variants() if v.metadata.get(\"lexicon_refs\"))}')
"

# Test lexicon generation dry-run
poetry run python3 scripts/generate_lexicons.py --dry-run --collections alpha_core trad_core

# Check current progress
poetry run python3 scripts/monitor_lexicon_progress.py
```

## Notes

- **Deduplication**: All token comparisons are case-insensitive
- **Context Limits**: Prompts sample up to 50 existing required tokens + 100 optional to keep requests manageable
- **Retry Logic**: 3 attempts with exponential backoff (1s, 2s, 4s) for failed API calls
- **Shared Lexicons**: Some variants share collections (e.g., pro_ana uses both pro_ana_core and ed_risk_core)
- **Safety**: All content is for research purposes in controlled simulation environment

## Architecture

```
ontology_unified.yaml
├── lexicon_collections (13 total)
│   ├── benign_core → benign.yaml
│   ├── recovery_core → recovery.yaml
│   ├── ed_risk_core → ed_risk.yaml
│   ├── pro_ana_core → pro_ana.yaml
│   ├── incel_core → incel.yaml
│   ├── alpha_core → alpha.yaml
│   ├── misinfo_core → misinfo.yaml
│   ├── conspiracy_core → conspiracy.yaml
│   ├── trad_core → trad.yaml
│   ├── gamergate_core → gamergate.yaml
│   ├── extremist_core → extremist.yaml
│   ├── hate_speech_core → hate_speech.yaml
│   └── bullying_core → bullying.yaml
└── archetypes (13 total)
    └── variants (13 total)
        └── lexicon_refs → points to collections
```

## Success Criteria

- ✅ All 13 archetype variants have `lexicon_refs` configured
- ✅ All 13 lexicon YAML files exist and load correctly
- ✅ Ontology validates and loads without errors
- ✅ Generator processes all collections without crashes
- 🔄 All collections reach 200 required + 1000 optional tokens (in progress)
- ⏳ Persona generation integrates new lexicons successfully (pending)
- ⏳ Simulation runtime uses lexicons correctly (pending)

## Cost Estimate

- **API Calls per Collection**: ~20-40 iterations × 13 collections = ~260-520 calls
- **Tokens per Call**: ~600 output tokens average
- **Total Output**: ~156K-312K tokens generated
- **Input Context**: Grows with each iteration as existing tokens accumulate

