# Lexicon Generator Script

## Overview

`scripts/generate_lexicons.py` is a procedural lexicon expansion tool that uses Grok (xAI) to generate authentic vocabulary for persona archetypes defined in the ontology. It supports both single-run and iterative generation modes, with full deduplication to avoid repeating existing terms.

## Features

- **Ontology-driven**: Automatically discovers lexicon collections from `ontology_unified.yaml`
- **Context-aware**: Passes persona variant summaries and worldviews to Grok for authentic token generation
- **Iterative mode**: Can generate tokens in batches until reaching target totals (e.g., 200 required, 1000 optional)
- **Deduplication**: Tracks all existing tokens (case-insensitive) to prevent repetition
- **Dry-run support**: Preview changes before writing to disk
- **Selective collections**: Target specific collections or process all at once

## Usage

### Basic Single-Run Mode

Generate 6 required and 12 optional tokens per collection (one API call per collection):

```bash
poetry run python3 scripts/generate_lexicons.py
```

### Dry-Run Mode

Preview what would be generated without writing files:

```bash
poetry run python3 scripts/generate_lexicons.py --dry-run
```

### Iterative Mode (Target Totals)

Generate tokens iteratively until reaching specific totals:

```bash
# Generate until 200 required and 1000 optional tokens per collection
poetry run python3 scripts/generate_lexicons.py \
  --required-total 200 \
  --optional-total 1000 \
  --max-iterations 100
```

### Selective Collections

Target specific collections only:

```bash
poetry run python3 scripts/generate_lexicons.py \
  --collections benign_core incel_core \
  --required-total 50 \
  --optional-total 200
```

### Adjust Batch Sizes

Control how many tokens are requested per iteration:

```bash
poetry run python3 scripts/generate_lexicons.py \
  --required-target 10 \
  --optional-target 25 \
  --required-total 200 \
  --optional-total 1000
```

## CLI Arguments

### Core Arguments

- `--ontology PATH`: Path to ontology YAML (default: `configs/personas/ontology_unified.yaml`)
- `--collections [IDS...]`: Explicit collection IDs to update (default: all collections)
- `--dry-run` / `--no-dry-run`: Preview mode (default: false)

### Generation Targets

- `--required-target INT`: Tokens per iteration for required list (default: 6)
- `--optional-target INT`: Tokens per iteration for optional list (default: 12)
- `--required-total INT`: Target total for required tokens (default: 0 = single run)
- `--optional-total INT`: Target total for optional tokens (default: 0 = single run)
- `--max-iterations INT`: Maximum iterations per collection (default: 50)

### LLM Configuration

- `--model STRING`: xAI model name (default: `grok-4-fast-non-reasoning`)
- `--temperature FLOAT`: Sampling temperature (default: 0.4)
- `--top-p FLOAT`: Top-p sampling (default: 0.85)
- `--max-output-tokens INT`: Max tokens per response (default: 600)
- `--api-key STRING`: Override XAI_API_KEY env var

### Other Options

- `--context-limit INT`: Persona variants to mention in prompts (default: 5)
- `--verbose` / `--no-verbose`: Enable debug logging (default: false)
- `--strict` / `--no-strict`: Fail-fast on errors (default: true)

## How It Works

### 1. Context Building

For each collection in the ontology's `lexicon_collections`:
- Resolves file path (e.g., `configs/lexicons/benign.yaml`)
- Gathers all persona variants that reference this collection via `lexicon_refs`
- Extracts variant summaries and worldview notes

### 2. Prompt Construction

For each iteration, builds a prompt containing:
- Collection ID and persona variant context
- Current token counts (e.g., "11 required, 70 optional")
- Existing tokens (sampled if lists are very long)
- Explicit instructions to avoid repetition and maintain authenticity

### 3. LLM Generation

- Calls Grok via `oasis.generation.xai_client`
- Parses JSON response: `{"required": [...], "optional": [...]}`
- Sanitizes tokens (ASCII, whitespace normalization)

### 4. Deduplication & Filtering

- Compares new tokens against existing lists (case-insensitive)
- Tracks tokens added within the same iteration
- Keeps only unique tokens up to the requested target

### 5. Iteration Logic

In single-run mode (no totals specified):
- Makes one API call per collection
- Appends results to YAML files

In iterative mode (totals specified):
- Loops until reaching targets or max iterations
- Adjusts per-iteration targets based on remaining count
- Stops early if Grok produces no new tokens
- Saves results after each iteration (not in dry-run)

## Output

### Dry-Run Output

```
2025-11-25 03:52:01 | INFO    | Collection 'benign_core' iteration 1/10: current=11 req + 70 opt, requesting +6 req + 12 opt
2025-11-25 03:52:03 | INFO    | HTTP Request: POST https://api.x.ai/v1/chat/completions "HTTP/1.1 200 OK"
2025-11-25 03:52:03 | INFO    | Collection 'benign_core' iteration 2/10: current=17 req + 82 opt, requesting +3 req + 12 opt
...
2025-11-25 03:52:05 | INFO    | Collection 'benign_core' reached targets: 20 required, 100 optional
2025-11-25 03:52:05 | INFO    | [DRY-RUN] benign_core completed: (total: 9 across 3 iterations), (total: 30 across 3 iterations)
```

### Production Output

```
2025-11-25 03:46:36 | INFO    | HTTP Request: POST https://api.x.ai/v1/chat/completions "HTTP/1.1 200 OK"
2025-11-25 03:46:36 | INFO    | Updated benign_core (/path/to/benign.yaml) with 6 required and 12 optional tokens.
```

## YAML Structure

Lexicon files follow a simple structure:

```yaml
required:
  - "thanks"
  - "appreciate"
  - "glad"
optional:
  - "hey"
  - "morning"
  - "coffee"
  - "sunny"
```

The script preserves this structure and appends new tokens to the end of each list.

## Environment Requirements

- **Python**: 3.10-3.11 (via Poetry)
- **API Key**: Set `XAI_API_KEY` in `.env` or pass via `--api-key`
- **Dependencies**: `dotenv`, `yaml`, `requests`, plus `oasis` modules

## Cost Considerations

Each API call to Grok counts toward your xAI usage. For large-scale generation:

- **200 required + 1000 optional** per collection requires ~20-40 API calls per collection
- **6 collections** = ~120-240 total API calls
- Use `--dry-run` first to estimate iterations needed
- Consider starting with smaller targets and scaling up

## Error Handling

- **Strict mode** (default): Script exits on first error
- **Relaxed mode** (`--strict false`): Logs errors and continues to next collection
- **Empty responses**: Raises RuntimeError
- **JSON parse failures**: Raises ValueError with details

## Examples

### Generate 50/250 tokens for specific collections

```bash
poetry run python3 scripts/generate_lexicons.py \
  --collections ed_risk_core incel_core \
  --required-total 50 \
  --optional-total 250 \
  --required-target 8 \
  --optional-target 20
```

### Preview expansion to 100/500 for all collections

```bash
poetry run python3 scripts/generate_lexicons.py \
  --dry-run \
  --required-total 100 \
  --optional-total 500
```

### Single-run with custom model settings

```bash
poetry run python3 scripts/generate_lexicons.py \
  --model grok-beta \
  --temperature 0.6 \
  --required-target 10 \
  --optional-target 30
```

## Integration with Persona Generation

Generated lexicons are automatically available to `scripts/generate_personas_llm.py`:

1. Ontology references lexicon files via `lexicon_collections`
2. Persona variants reference collections via `lexicon_refs`
3. Generator samples from `required` + `optional` lists
4. Sampled tokens are stored in persona CSV for runtime use

## Troubleshooting

**Problem**: "XAI_API_KEY not set"
- **Solution**: Add `XAI_API_KEY=your-key` to `.env` or use `--api-key`

**Problem**: "Lexicon file not found"
- **Solution**: Check that paths in ontology's `lexicon_collections` are correct

**Problem**: "No novel tokens produced"
- **Solution**: Grok is repeating existing tokens; try lowering temperature or increasing diversity

**Problem**: Process is slow
- **Solution**: Increase `--required-target` and `--optional-target` to request more tokens per call

**Problem**: Getting duplicates across iterations
- **Solution**: This shouldn't happen (dedup is case-insensitive); report as bug if seen

## Future Enhancements

- [ ] Support for embedding-based semantic deduplication
- [ ] Cache LLM responses to avoid re-requesting similar contexts
- [ ] Parallel collection processing
- [ ] Progress bars for long-running iterations
- [ ] Export lexicon diffs for review before committing

