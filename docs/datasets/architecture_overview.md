# Persona + Posts Generation Architecture

## 1. High-Level Flow

1. **Persona synthesis**  
   `scripts/build_persona_rag.py`  
   _Interpret raw persona seeds, cluster them, and emit structured persona records with allowed labels, prompts, style cues, and RAG snippets._
   _Outputs also include Social (S), Personal (P), and Context (C) descriptors for each persona._
2. **Supabase snippet preprocessing (lexicons + priors)**  
   `scripts/preprocess_supabase_snippets.py`  
   _Clean Supabase flagged messages, assign labels, embed + cluster them, and build per-label lexicons and probability priors._
3. **Post generation + placeholder substitution**  
   `scripts/build_posts_dataset.py`  
   _Load personas + lexicons, plan 1,000 posts with exact harmful ratios, compose text (offline or via LLM), replace label tokens with real Supabase spans, and output JSONL._
4. **Validation + Documentation**  
   `scripts/validate_posts_dataset.py` + `docs/datasets/posts_dataset_README.md`  
   _Run QA metrics to confirm counts/ratios/persona usage and document the pipeline + metrics for stakeholders._

The entire pipeline is driven by Python scripts with explicit CLI parameters. Key variables are written back into CSV/JSONL files that flow into the subsequent step.

---

## 2. Persona Synthesis (`build_persona_rag.py`)

| Input | Variable | Purpose |
|-------|----------|---------|
| `oasis/data/personality.csv` | `seeds` | Raw persona descriptions & chat snippets |
| Optional data (flagged tweets, etc.) omitted in this new workflow |

### Transformations

1. **Seed loading**  
   - Function `load_personality()` -> `List[PersonaSeed]`
   - `PersonaSeed` fields: `seed_id`, `description`, `chat_lines`, `keywords`, plus derived `cluster_id`, `cluster_keywords`.
   - Text cleaned via `clean_text`, keywords extracted via `extract_keywords`.

2. **Clustering**  
   - `cluster_seeds()` uses TF-IDF + MiniBatchKMeans to assign `cluster_id` and representative `cluster_keywords`.

3. **Label assignments**  
   - `build_label_queue()` enforces counts: 70 single-label personas + 30 multi-label.
   - `LabelAssignment` objects (labels list + status `primary/backup`).

4. **Persona record creation**  
   - `generate_personas()` loops assignments + seeds.
   - Each persona `PersonaRecord` stores:
     - Identity & prompts (`persona_id`, `persona_variant`, `persona_prompt_post/comment`, `persona_goal`, etc.).
     - Style fields (`persona_style_quirks`, lexical required/optional).
     - Label metadata (`allowed_labels`, `allowed_label_tokens`, `is_primary`, `is_multilabel`).
     - RAG excerpts (`rag_persona_excerpt`, `rag_chat_excerpt`).
   - Randomness controlled by `--seed`.

### Outputs

| File | Content |
|------|---------|
| `data/personas_generated_full.csv` | All 130 personas (100 primary + 30 backup) |
| `data/personas_primary.csv` | First 100 (is_primary == 1) |
| `data/rag_corpus/persona_corpus.jsonl` | Persona RAG snippets (desc + chat) |
| `data/rag_corpus/persona_corpus_summary.json` | Counts & previews |
| `persona_social_identity`, `persona_psych_identity`, `persona_life_context` | SPC cues now emitted per persona |

These files become inputs for later steps.

---

## 3. Supabase Snippet Preprocessing (`preprocess_supabase_snippets.py`)

| Input | Variable | Purpose |
|-------|----------|---------|
| `oasis/data/Supabase Snippet Flagged Messages.csv` | `df` DataFrame | Real toxic/support snippets with reasons/confidences |

### Transformations

1. **Cleaning**  
   - Deduplicate `content`, normalize `confidence` to `[0,1]`.

2. **Label assignment** (`assign_label()`):
   - Rules based on keywords/`reason`.
   - Produces `label` column (benign, recovery_support, etc.).

3. **Embedding & clustering**  
   - TF-IDF matrix + optional sentence-transformers embeddings.
   - MiniBatchKMeans cluster assignments -> `cluster_id`.

4. **Lexicon building**  
   - For each label, gather up to 250 spans via `extract_spans()`.

5. **Probability priors**  
   - `compute_priors()` collects mean/std/min/max of normalized confidences per label.

### Outputs

| File | Description |
|------|-------------|
| `configs/rag/supabase_snippets.jsonl` | Cleaned snippet entries |
| `configs/lexicons/supabase_lexicons.json` | Label -> span list |
| `configs/lexicons/label_priors.json` | Label -> {mean,std,min,max} |
| `configs/lexicons/supabase_cluster_summary.json` | Cluster keyword stats |

These outputs feed directly into the post constructor to supply realistic language and gold probability priors.

---

## 4. Post Generation (`build_posts_dataset.py`)

### Key Inputs
- Personas: `data/personas_primary.csv`
- Persona RAG: `data/rag_corpus/persona_corpus.jsonl`
- Lexicons & Priors: `configs/lexicons/*.json`
- (Optional) Gemini/Grok API keys for LLM-based text; otherwise offline composer.

### Main Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `persona_slots` | from CSV via `load_personas()` | Each entry has persona prompts, allowed labels, style cues, RAG references |
| `lexicon_map` | `supabase_lexicons.json` | Label -> list of replacement spans |
| `priors` | `label_priors.json` | For gold probability sampling |
| `units` | `build_units()` | Plan for each post: persona, label_set, thread assignment |
| `placeholder_result` | `PlaceholderResolver` | Maps `<LBL:...>` tokens to actual spans |

### Flow of Data in Code

1. **Load persona contexts**  
   - For each row: `PersonaSlot` with context prompts, `allowed_labels`, `allowed_tokens`, lexical/style data.

2. **Allocate posts**  
   - `allocate_post_counts()`: ensures 1,000 total posts, at least 200 harmful, 20% multi-label among harmful.
   - `build_units()`: constructs `PostUnit` objects with persona references and label assignments; also sets `thread_id`, `is_comment`, `plan_seed`.

3. **Chunk into threads**  
   - `chunk_threads()` groups posts into conversational threads (length 3–6). Each `PostUnit` now knows `thread_id` and comment/post state.

4. **Text generation**  
   - For each unit:
     - Compose prompt (persona style, label tokens, optional RAG refs, thread context).
     - **Offline path** (`compose_offline_text()`): deterministic string built from persona goal + lexical words + optional RAG snippet.
     - **LLM path (Gemini)**: `generate_text()` call with `persona.context.build_post_prompt/comment`.
   - In both cases ensure label tokens are present.

5. **Placeholder substitution**  
   - `PlaceholderResolver.replace()` scans for `<LBL:...>` tokens and substitutes with actual Supabase spans from `lexicon_map`.
   - Track `replacements` dictionary for metadata.

6. **Record assembly**  
   Each post produces a dict with:
   - `post_id`, `thread_id`, `user_id`, `parent_id`, `prev_post_ids`
   - `timestamp` (incremental minutes)
   - `text`, `placeholders_applied`
   - `labels`
   - `gold_confidence`, `gold_proba` (sampled from priors)
   - `optional_severity` (based on persona label intensity)
   - `features` (action type, style tags, thread depth, lexicon spans)
   - `split` (train/val/test via `assign_splits()`)
   - `provenance` JSON (persona variant, RAG refs used, placeholder tokens, generation mode)
   - `generation_seed`, `persona_id`, `seen_out_of_network`, `needs_thread_context`

7. **Writes out**  
   - `posts.jsonl` (or custom path via `--output`)
   - Optionally multiple runs (e.g., `posts_gemini.jsonl`)

---

## 5. Validation & Documentation

1. **Validation script** (`scripts/validate_posts_dataset.py`)
   - Inputs: `posts.jsonl`
   - Output: `configs/lexicons/posts_dataset_metrics.json`
   - Metrics: total posts, number harmful, multi-label count, label distribution, split distribution, persona usage, placeholder coverage.

2. **README** (`docs/datasets/posts_dataset_README.md`)
   - Describes pipeline steps, metrics, reproducing commands.

---

## 6. Final Artifacts (per run)

| Artifact | Description |
|----------|-------------|
| `data/personas_primary.csv` | 100 personas ready for dataset generation |
| `data/rag_corpus/persona_corpus.jsonl` | Persona RAG references |
| `configs/lexicons/*.json` | Supabase-derived lexicons & priors |
| `configs/rag/supabase_snippets.jsonl` | Cleaned snippet dataset |
| `posts.jsonl` | 1,000 posts (or other target) with full metadata |
| `configs/lexicons/posts_dataset_metrics.json` | QA metrics |
| `docs/datasets/posts_dataset_README.md` | Documentation of the process |

All these artifacts are under version control (branch `eva-persona-generation`, now merged to `main`).

---

## 7. Variable Hand-off Summary

| Step | Produces | Consumed by |
|------|----------|-------------|
| Persona script (`build_persona_rag.py`) | `personas_primary.csv`, `rag_corpus/persona_corpus.jsonl` | Post generator |
| Supabase preprocessing | `supabase_lexicons.json`, `label_priors.json` | Post generator |
| Post generation | `posts.jsonl` | Validation, downstream modeling |
| Validation | `posts_dataset_metrics.json` | QA/Docs |
| Docs | README | Stakeholder communication |

This structure ensures every transformation is traceable; each file is both the output and the variable container for the next stage.
