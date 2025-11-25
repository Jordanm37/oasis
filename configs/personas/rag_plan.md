## RAG-Augmented Persona Platform — Complete Implementation Plan

### Objective
Transform persona generation into a fully generative, ontology-driven pipeline that produces thousands of unique, grounded personas. Preserve SPC richness, integrate RAG seeds, ensure token/label fidelity, and keep the simulation + offline builder in sync.

---

## Phase 0 — Discovery & Preflight

- Confirm master assets:
  - Ontology: `configs/personas/ontology_llm.yaml` (Eva’s 6-class schema)
  - Seed dataset: `data/personality.csv` (PersonaChat-derived)
  - LLM generator: `scripts/generate_personas_llm.py`
  - Simulation runner: `scripts/run_production_sim.py` + `oasis/orchestrator/agent_factory.py`
  - Offline builder: `scripts/build_persona_rag.py`
- Snapshot current outputs (e.g., `data/mvp/personas_200_llm.csv`) for regression comparison.
- Identify existing SPC usage downstream (simulation, dataset builder) to ensure compatibility.

---

## Phase 1 — Ontology Enhancements ✅ (Completed)

### 1.1. Extend Variant Metadata
Add or refine fields per archetype/variant:

- `style_priors`: ["aggressive", "confrontational"]
- `topics`: ["dating", "social hierarchy"]
- `worldview`: "Frames modern dating as a rigged market."
- `lexicon_refs`: pointers into shared lexicon collections (see 1.4) instead of baking words directly into the variant.
- `style_variation_rules`: configuration blocks for tone/emoji/typo-rate ranges that downstream generators can sample from.
- `instructional_hints`: short bullet list of unique traits.
- `style_indicators`: references into shared indicator pools (see new style-indicator YAML) for extra per-persona flavor.

### 1.2. Label Mapping (For RAG alignment later)
Add optional `label_aliases` section to map external labels (Jigsaw) to ontology classes. Example:

```yaml
label_aliases:
  identity_hate: incel_misogyny
  threat: ed_risk
  toxic: misinfo
```

*(This allows Dipu’s pipeline to map Jigsaw classes to ontology classes.)*

### 1.3. Version & Comments
- Mark legacy static prompt sections with `# legacy_static_prompt`.
- Add `ontology_version` field (e.g., “LLM_RAG_v1”) for provenance.

### 1.4. Shared Lexicon Collections (External Files)
- Add a sibling `style_indicators` YAML to store reusable tone/emoji/typo descriptors per class; variants reference them via metadata so the RAG generator can draw unique combinations per persona.
- Store lexical pools in dedicated YAML/JSON files per class or semantic theme (e.g., `configs/lexicons/incel.yaml`, `configs/lexicons/recovery.yaml`) instead of embedding large lists inside the ontology.
- In the ontology, add `lexicon_refs` pointing to these files/sections; variants can optionally supply `lexicon.extra_required` overrides.
- Generators and runtime agents load the referenced lexicon files once, then sample from them deterministically to keep vocabulary fresh without duplicating data across persona definitions.
---

## Phase 2 — RAG Retrieval Module ✅ (Seed utilities in place)

### 2.1. Seed Loader
✅ `load_persona_seeds` implemented in `oasis/persona/seed_utils.py` (returns `PersonaSeed` objects with persona text, chat lines, keywords).

### 2.2. Seed Sampler
✅ Basic sampler (`sample_seed(...)`) available in the same module; filtering hooks ready to be extended per archetype (currently random-with-preference support stubbed via keyword hints).

### 2.3. Meta Extraction (optional)
Compute additional heuristics from seeds:
- `tone_score` (cheerful vs. gloomy)
- `topic_tags` (sports, art, tech)
These can enrich prompt instructions.

---

## Phase 3 — Generative Prompt Synthesis 🚧 (Instruction builder & stubs ready; LLM wiring pending)

### 3.1. Prompt Builder
✅ `build_llm_prompt_instruction` + scaffolding for `build_llm_prompt` live in `oasis/persona/prompt_builder.py`.
✅ `generate_personas_llm.py` now uses these builders; wired to real Grok calls when `--mode rag/llm_only` and `--use-LLM` with API key, otherwise deterministic mock fallback. Style indicators are sampled per category and folded directly into the `[Style]` section so the card itself carries the fingerprint.

Inputs:
- Seed attributes (description, chat snippets, keywords, tone)
- Ontology metadata (tone, topics, worldview, required lexicon)

Output:
- Instruction string for Grok-2.

Example instruction:

```
You are synthesizing a persona for a social simulation.

Base traits (from real user seed):
- "I like to remodel homes."
- "I go bow hunting."
- "Halloween is my favorite holiday."
- (Use these details to ground the persona.)

Transformation (archetype = incel.misogyny.aggresor):
- Tone: aggressive, confrontational.
- Worldview: believes dating is rigged like a scoreboard.
- Required vocabulary: Chad, Stacy, blackpill.
- Style hints: use short jabs mixed with ranting paragraphs.

Task:
- Write a system prompt describing this unique individual.
- Include: backstory, current mindset, conversational tone, goals.
- Output JSON with keys: system_prompt, persona_goal, style_quirks, lexical_required, lexical_optional.
- DO NOT mention label tokens (<LBL:...>); those are injected elsewhere.
```

### 3.2. Deterministic Generation
- Use seed-based hashing to derive `temperature/top_p` seeds for LLM call to ensure reproducibility.
- Cache responses to `data/spc_cache/` (as done currently) keyed by seed+variant to reduce cost.

### 3.4. Style Variation & Lexicon Shuffling
- ✅ Ontology now provides `style_variation_rules` and shared `lexicon_refs`; generator/runtime still need to consume them.
- ✅ `generate_personas_llm.py` now samples and persists these values per persona (stored in `style_variation_json` / `lexicon_samples_json` columns).
- Use the per-variant `style_variation_rules` together with a `DeterministicRNG` (keyed by persona_id) to pick small random perturbations in tone, emoji frequency, or typo rates so the outputs stay unique but reproducible.
- When building prompts (and later, when emitting posts), draw optional vocabulary from the shared `lexicon_collections` referenced by the variant, shuffle those lists, and rotate where required terms appear within sentences to avoid template repetition.

### 3.3. Response Handling
- Parse LLM JSON, validate required fields.
- If parsing fails: fallback to legacy static prompt + append SPC block.
- Store final `user_char` = `system_prompt + "\n\n[SPC]\n..."` to keep downstream compatibility.

---

## Phase 4 — Updating `generate_personas_llm.py`

### 4.1. Structural Changes
- Import new RAG utilities.
- Load ontology via `load_ontology(ontology_path)`.
- Replace static `_format_prompt()` usage with new LLM synthesis pipeline.
- Keep CLI signature, add new flags:
  - `--mode {rag,llm_only,legacy}`
  - `--seed-pool data/personality.csv`
  - `--cache-dir data/spc_cache/`
- Ensure `allowed_labels` and `allowed_label_tokens` columns are still filled using ontology.

### 4.2. Loop Workflow
For each persona row:
1. Determine target variant (based on CLI counts).
2. Sample seed via RAG module.
3. Build LLM instruction and call Grok-2.
4. Compose final persona row fields:
   - `user_char` (LLM-generated)
   - `persona_goal`, `style_quirks`
   - `allowed_labels`, `allowed_label_tokens`
   - `spc_json`, `narratives_json` (if needed)
   - Persist the lexicon collection IDs referenced plus the exact sampled subsets (so runtime can reuse or refresh them deterministically).
   - Embed the sampled style variation choices (emoji/typo toggles) so downstream agents can reuse the same DeterministicRNG draws.
   - Persist sampled style indicators (tone tags, emoji patterns, typo styles, etc.) in `style_indicators_json` and bake their prose descriptions directly into each persona’s `[Style]` block.
5. Write to CSV.

### 4.3. Backwards Compatibility
- If `--mode legacy`, skip RAG and revert to previous behavior to keep minimal fallback.

---

## Phase 5 — Simulation Integration

### 5.1. `agent_factory.py`
- ✅ `agent_factory` now reads the new CSV fields (`prompt_metadata_json`, `lexicon_samples_json`, `style_variation_json`, `allowed_label_tokens_json`) and injects them into `PersonaConfig`.
- Next: update `ExtendedSocialAgent` to actually consume these fields (style hints, lexicon rotation).

### 5.2. `EmissionPolicy`
- ✅ Persona-specific token inventories now flow through `PersonaConfig.allowed_label_tokens`, and the emission policy prefers those before falling back to global defaults.

### 5.3. `labeler.py`
- ✅ Hardcoded map replaced by `set_token_category_map`; `agent_factory` now builds the token→category mapping from each persona’s allowed tokens and installs it globally for sidecar labeling.

### 5.4. `ExtendedSocialAgent`
- ✅ Runtime now loads prompt metadata, lexicon samples, and style variations per persona; prompts include goal/style hints and optional vocab suggestions.
- ✅ Posts incorporate sampled optional vocabulary via `_apply_post_variations` (comments remain untouched to preserve thread fidelity).
- ✅ Stored style variation knobs (emoji_rate, typo_rate, tone_shift) now influence posts (prepended tone tag, optional emoji, light typos); comments remain unmodified for clarity.

---

## Phase 6 — Offline Builder (`build_persona_rag.py`)

### 6.1. Ontology Integration
- Remove internal `LABEL_DEFINITIONS` and `LABEL_TOKEN_MAP`.
- Load ontology and map variant metadata to local `PersonaRecord`.
- When building prompts for offline posts, use the same metadata structure as online.

### 6.2. Label Alignment
- Use `label_aliases` map (from Phase 1) to align external corpora (Jigsaw) to ontology classes for RAG retrieval.

---

## Phase 7 — Validation & QA

### 7.1. Unit Tests
- Snapshot test `generate_personas_llm.py` output for 3 variants to ensure deterministic behavior.
- Test label mapping (ensure tokens assigned match ontology categories).
- Ensure `agent_factory` handles both new and legacy CSVs.

### 7.2. Content Validation
- Manual inspection of 10 random personas to verify unique prompts referencing seed traits.
- Spot-check simulation logs to confirm `<LBL:...>` tokens align with `allowed_label_tokens`.

### 7.3. Performance & Cache
- Test 200-persona generation time. If slow, reuse `spc_cache`.
- Document instructions to purge cache when ontology changes.

---

## Phase 8 — Documentation

### 8.1. Update `CLAUDE.md` / `docs/personas/`
- Describe new RAG-based pipeline.
- Show how to add new classes via ontology YAML.
- Explain `--mode` flag options.

### 8.2. Runbook
- Provide step-by-step instructions:
  - “Modify ontology → Run generate_personas_llm → Run run_production_sim”.

---

## Phase 9 — Future Enhancements (Optional)

- Support pluggable seed datasets (e.g., moderation exports) by abstracting seed loader.
- Build a small GUI or CLI tool to preview synthesized personas by variant.
- Extend Dipu’s RAG pipeline to align Jigsaw classes to ontology via `label_aliases`.

---

This surgical plan keeps all existing infrastructure, adds RAG-powered uniqueness, maintains SPC compatibility, and ensures both simulation and offline workflows remain aligned.