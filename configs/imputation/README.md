# Multi-Dataset RAG Imputation System

## Overview

This directory contains the configuration and implementation for the multi-dataset RAG (Retrieval-Augmented Generation) imputation system. The system supports both vector-based (ChromaDB) and keyword-based (TF-IDF) retrieval methods with automatic fallback chains.

## Directory Structure

```
configs/imputation/
├── README.md                   # This file
├── dataset_registry.yaml       # Main registry of all datasets
└── mappings/                   # Label mapping files
    ├── jigsaw_to_oasis.yaml   # Jigsaw Toxic Comments → OASIS
    ├── reddit_mh_to_oasis.yaml # Reddit Mental Health → OASIS
    └── synthetic_to_oasis.yaml # Synthetic Harmful → OASIS
```

## Core Components

### 1. DatasetManager (`oasis/imputation/dataset_manager.py`)
- Manages multiple datasets from the registry
- Handles both ChromaDB and TF-IDF indices
- Provides unified search interface across all datasets
- Supports CSV and JSONL data formats

### 2. LabelMapper (`oasis/imputation/label_mapper.py`)
- Maps external dataset labels to OASIS archetypes and tokens
- Handles combination rules for multiple labels
- Provides weighted archetype selection
- Validates tokens against emission_policy.py

### 3. MultiMethodRetriever (`oasis/imputation/multi_method_retriever.py`)
- Combines multiple retrieval methods with fallback chain
- Supports: ChromaDB → TF-IDF → LLM Generation → Static Bank
- Caches recent retrievals for performance
- Tracks performance metrics for each method

## Dataset Registry Format

The `dataset_registry.yaml` defines all available datasets:

```yaml
datasets:
  dataset_id:
    name: "Human-readable name"
    description: "Dataset description"

    source:
      path: "path/to/data.csv"
      format: "csv" | "jsonl"
      text_column: "column_name"  # For CSV
      label_columns: ["label1", "label2"]  # For CSV

    label_mapping_file: "configs/imputation/mappings/dataset_to_oasis.yaml"

    indices:
      chromadb:
        enabled: true
        path: "data/imputation/chromadb/dataset"
        collection_name: "collection"
        embedding_model: "all-MiniLM-L6-v2"

      tfidf:
        enabled: true
        path: "data/imputation/tfidf/dataset.pkl"
        max_features: 10000
        ngram_range: [1, 3]
```

## Label Mapping Format

Label mapping files in `mappings/` define how external labels map to OASIS:

```yaml
label_mappings:
  external_label:
    archetypes:
      - archetype: "hate_speech"
        weight: 0.6
        tokens:
          - "LBL:HATE_SLUR"
          - "LBL:DEHUMANIZATION"
      - archetype: "bullying"
        weight: 0.4
        tokens:
          - "LBL:PERSONAL_ATTACK"

combination_rules:
  - condition:
      all_of: ["toxic", "threat"]
    override_archetype: "extremist"
    priority_tokens:
      - "LBL:VIOLENT_THREAT"
```

## Usage Example

```python
from oasis.imputation.dataset_manager import DatasetManager
from oasis.imputation.label_mapper import LabelMapper
from oasis.imputation.multi_method_retriever import MultiMethodRetriever

# Initialize components
dm = DatasetManager("configs/imputation/dataset_registry.yaml")
lm = LabelMapper()
retriever = MultiMethodRetriever(dm, lm)

# Load dataset and mappings
dm.initialize_chromadb("jigsaw_toxic_comments")
dm.initialize_tfidf("jigsaw_toxic_comments")
lm.load_mapping_file("jigsaw_toxic_comments",
                     "configs/imputation/mappings/jigsaw_to_oasis.yaml")

# Retrieve content
result = await retriever.retrieve(
    query="example toxic text",
    dataset_ids=["jigsaw_toxic_comments"],
    archetype="hate_speech",
    top_k=10
)

# Get best matching text
best_text = result.get_best_text()
```

## Adding New Datasets

1. **Add dataset to registry** (`dataset_registry.yaml`):
   ```yaml
   datasets:
     new_dataset:
       name: "New Dataset"
       source:
         path: "data/new_dataset.csv"
         format: "csv"
       # ... other config
   ```

2. **Create label mapping** (`mappings/new_dataset_to_oasis.yaml`):
   ```yaml
   label_mappings:
     dataset_label:
       archetypes:
         - archetype: "oasis_archetype"
           weight: 1.0
           tokens: ["LBL:TOKEN1", "LBL:TOKEN2"]
   ```

3. **Build indices**:
   ```python
   # Load data
   texts, labels = dm.load_dataset_data("new_dataset")

   # Build ChromaDB index
   dm.initialize_chromadb("new_dataset", rebuild=True)
   dm.add_to_chromadb("new_dataset", texts)

   # Build TF-IDF index
   dm.initialize_tfidf("new_dataset", rebuild=True)
   dm.build_tfidf_index("new_dataset", texts)
   ```

## OASIS Archetypes and Tokens

The system maps to 13 OASIS archetypes:
- `benign` - Supportive, friendly content
- `recovery_support` - Recovery and peer support
- `ed_risk` - Eating disorder risk content
- `pro_ana` - Pro-anorexia content
- `incel_misogyny` - Incel and misogynistic content
- `alpha` - Alpha male/manosphere content
- `misinfo` - Misinformation
- `conspiracy` - Conspiracy theories
- `trad` - Traditional/culture war content
- `gamergate` - Gamergate-related content
- `extremist` - Extremist content
- `hate_speech` - Hate speech
- `bullying` - Bullying and harassment

Each archetype has associated label tokens (e.g., `LBL:HATE_SLUR`, `LBL:PERSONAL_ATTACK`) defined in `generation/emission_policy.py`.

## Performance Considerations

- **ChromaDB**: Better for semantic similarity, slower but more accurate
- **TF-IDF**: Better for keyword matching, faster but less semantic understanding
- **Caching**: Recent retrievals cached for 1 hour by default
- **Batch Processing**: Use batch methods when processing multiple queries
- **Index Size**: Monitor ChromaDB and TF-IDF index sizes for large datasets

## Next Steps

1. Build CLI tools for dataset management
2. Integrate with existing RagImputer class
3. Add dataset validation and quality checks
4. Implement incremental index updates
5. Add monitoring and logging dashboards