"""
Toxicity Dataset Integration for RAG Imputation System.

This module provides the bridge between standalone toxicity dataset loaders
(ToxiGen, ImplicitHate) and the OASIS RAG imputation system. It handles:
- Dataset preparation for ChromaDB and TF-IDF indexing
- Label mapping to OASIS ontology
- Archetype and label token assignment
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ToxicitySource(Enum):
    """Enumeration of available toxicity datasets."""
    TOXIGEN_ANNOTATED = "toxigen_annotated"
    TOXIGEN_HATE = "toxigen_hate_prompts"
    TOXIGEN_NEUTRAL = "toxigen_neutral_prompts"
    IMPLICIT_HATE = "implicit_hate"
    IMPLICIT_HATE_TRAIN = "implicit_hate_train"


@dataclass
class ToxicityRecord:
    """A single record from a toxicity dataset with OASIS mappings."""
    text: str
    source: ToxicitySource
    original_labels: Dict[str, Any]
    oasis_archetype: str
    oasis_tokens: List[str]
    demographic: Optional[str] = None
    implicit_class: Optional[str] = None
    toxicity_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class ToxicityDatasetPreparator:
    """
    Prepares toxicity datasets for integration with the RAG imputation system.

    This class loads raw datasets and converts them into a format suitable
    for ChromaDB indexing and retrieval, applying OASIS label mappings.
    """

    def __init__(
        self,
        toxigen_path: str = "data/toxigen_datasets",
        implicit_hate_path: str = "data/implicit_hate_datasets",
        mappings_path: str = "configs/imputation/mappings"
    ):
        """
        Initialize the toxicity dataset preparator.

        Args:
            toxigen_path: Path to ToxiGen dataset directory
            implicit_hate_path: Path to ImplicitHate dataset directory
            mappings_path: Path to label mapping YAML files
        """
        self.toxigen_path = Path(toxigen_path)
        self.implicit_hate_path = Path(implicit_hate_path)
        self.mappings_path = Path(mappings_path)

        self.toxigen_mapping = None
        self.implicit_hate_mapping = None

        self._load_mappings()

    def _load_mappings(self):
        """Load label mapping configurations."""
        # Load ToxiGen mapping
        toxigen_mapping_file = self.mappings_path / "toxigen_to_oasis.yaml"
        if toxigen_mapping_file.exists():
            with open(toxigen_mapping_file) as f:
                self.toxigen_mapping = yaml.safe_load(f)
            logger.info("Loaded ToxiGen label mappings")

        # Load ImplicitHate mapping
        implicit_hate_mapping_file = self.mappings_path / "implicit_hate_to_oasis.yaml"
        if implicit_hate_mapping_file.exists():
            with open(implicit_hate_mapping_file) as f:
                self.implicit_hate_mapping = yaml.safe_load(f)
            logger.info("Loaded ImplicitHate label mappings")

    def _apply_toxigen_mapping(
        self,
        prompt_type: str,
        demographic: Optional[str] = None,
        toxicity_score: Optional[float] = None
    ) -> Tuple[str, List[str]]:
        """
        Apply ToxiGen label mapping to determine archetype and tokens.

        Args:
            prompt_type: "hate" or "neutral"
            demographic: Target demographic group
            toxicity_score: Human toxicity score (if available)

        Returns:
            Tuple of (archetype, list of tokens)
        """
        if not self.toxigen_mapping:
            return ("hate_speech" if prompt_type == "hate" else "benign", ["LBL:HATE_SLUR"])

        # Get base mapping for prompt type
        base_mapping = self.toxigen_mapping.get("label_mappings", {}).get(prompt_type, {})
        archetypes = base_mapping.get("archetypes", [])

        # Override with demographic-specific mapping if available
        if demographic and prompt_type == "hate":
            demo_mapping = self.toxigen_mapping.get("demographic_mappings", {}).get(demographic, {})
            if demo_mapping:
                archetypes = demo_mapping.get("archetypes", archetypes)

        # Select archetype based on weights
        if archetypes:
            import random
            weights = [a.get("weight", 1.0) for a in archetypes]
            selected = random.choices(archetypes, weights=weights, k=1)[0]
            archetype = selected.get("archetype", "hate_speech")
            tokens = selected.get("tokens", ["LBL:HATE_SLUR"])
        else:
            archetype = "hate_speech" if prompt_type == "hate" else "benign"
            tokens = ["LBL:HATE_SLUR"] if prompt_type == "hate" else ["LBL:SUPPORTIVE"]

        # Apply toxicity score overrides
        if toxicity_score is not None:
            combination_rules = self.toxigen_mapping.get("combination_rules", [])
            for rule in combination_rules:
                condition = rule.get("condition", {})
                if "toxicity_human_gte" in condition and toxicity_score >= condition["toxicity_human_gte"]:
                    if "override_archetype" in rule:
                        archetype = rule["override_archetype"]
                    if "priority_tokens" in rule:
                        tokens = rule["priority_tokens"]
                    break

        return archetype, tokens

    def _apply_implicit_hate_mapping(
        self,
        implicit_class: str,
        extra_class: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Apply ImplicitHate label mapping to determine archetype and tokens.

        Args:
            implicit_class: Primary implicit hate class
            extra_class: Secondary implicit hate class (if present)

        Returns:
            Tuple of (archetype, list of tokens)
        """
        if not self.implicit_hate_mapping:
            return ("hate_speech", ["LBL:DOGWHISTLE", "LBL:DEHUMANIZATION"])

        # Get mapping for implicit class
        class_mapping = self.implicit_hate_mapping.get("label_mappings", {}).get(implicit_class, {})
        archetypes = class_mapping.get("archetypes", [])

        # Select archetype based on weights
        if archetypes:
            import random
            weights = [a.get("weight", 1.0) for a in archetypes]
            selected = random.choices(archetypes, weights=weights, k=1)[0]
            archetype = selected.get("archetype", "hate_speech")
            tokens = list(selected.get("tokens", ["LBL:DOGWHISTLE"]))
        else:
            archetype = "hate_speech"
            tokens = ["LBL:DOGWHISTLE", "LBL:DEHUMANIZATION"]

        # Add extra class tokens if present
        if extra_class:
            extra_mappings = self.implicit_hate_mapping.get("extra_class_mappings", {})
            if extra_class in extra_mappings:
                additional = extra_mappings[extra_class].get("additional_tokens", [])
                tokens.extend([t for t in additional if t not in tokens])

        return archetype, tokens

    def iterate_toxigen_annotated(self) -> Iterator[ToxicityRecord]:
        """
        Iterate over ToxiGen annotated dataset records.

        Yields:
            ToxicityRecord for each entry in the annotated dataset
        """
        annotated_path = self.toxigen_path / "annotated" / "train.csv"
        if not annotated_path.exists():
            logger.warning(f"ToxiGen annotated data not found at {annotated_path}")
            return

        df = pd.read_csv(annotated_path)
        logger.info(f"Processing {len(df)} ToxiGen annotated records")

        for _, row in df.iterrows():
            # Determine prompt type based on toxicity score
            toxicity = row.get("toxicity_human", 0)
            prompt_type = "hate" if toxicity >= 2.5 else "neutral"
            demographic = row.get("target_group")

            archetype, tokens = self._apply_toxigen_mapping(
                prompt_type=prompt_type,
                demographic=demographic,
                toxicity_score=toxicity
            )

            yield ToxicityRecord(
                text=row["text"],
                source=ToxicitySource.TOXIGEN_ANNOTATED,
                original_labels={
                    "target_group": demographic,
                    "toxicity_human": toxicity,
                    "toxicity_ai": row.get("toxicity_ai"),
                    "stereotyping": row.get("stereotyping"),
                    "intent": row.get("intent")
                },
                oasis_archetype=archetype,
                oasis_tokens=tokens,
                demographic=demographic,
                toxicity_score=toxicity,
                metadata={
                    "actual_method": row.get("actual_method"),
                    "predicted_group": row.get("predicted_group")
                }
            )

    def iterate_toxigen_prompts(
        self,
        prompt_type: str = "hate"
    ) -> Iterator[ToxicityRecord]:
        """
        Iterate over ToxiGen prompt datasets (hate or neutral).

        Args:
            prompt_type: "hate" or "neutral"

        Yields:
            ToxicityRecord for each prompt
        """
        prompts_path = self.toxigen_path / "prompts" / prompt_type
        if not prompts_path.exists():
            logger.warning(f"ToxiGen {prompt_type} prompts not found at {prompts_path}")
            return

        source = ToxicitySource.TOXIGEN_HATE if prompt_type == "hate" else ToxicitySource.TOXIGEN_NEUTRAL

        for csv_file in prompts_path.glob("*.csv"):
            # Extract demographic from filename
            filename = csv_file.stem  # e.g., "hate_asian_1k"
            parts = filename.split("_")
            demographic = "_".join(parts[1:-1])

            df = pd.read_csv(csv_file)
            logger.info(f"Processing {len(df)} {prompt_type} prompts for {demographic}")

            archetype, tokens = self._apply_toxigen_mapping(
                prompt_type=prompt_type,
                demographic=demographic
            )

            for _, row in df.iterrows():
                yield ToxicityRecord(
                    text=row["text"],
                    source=source,
                    original_labels={
                        "prompt_type": prompt_type,
                        "demographic": demographic
                    },
                    oasis_archetype=archetype,
                    oasis_tokens=tokens,
                    demographic=demographic,
                    metadata={
                        "source_file": csv_file.name
                    }
                )

    def iterate_implicit_hate(
        self,
        split: str = "all"
    ) -> Iterator[ToxicityRecord]:
        """
        Iterate over ImplicitHate dataset records.

        Args:
            split: "all", "train", or "test"

        Yields:
            ToxicityRecord for each entry
        """
        if split == "all":
            data_path = self.implicit_hate_path / "raw" / "implicit_hate.csv"
            source = ToxicitySource.IMPLICIT_HATE
        elif split == "train":
            data_path = self.implicit_hate_path / "splits" / "train.csv"
            source = ToxicitySource.IMPLICIT_HATE_TRAIN
        else:
            data_path = self.implicit_hate_path / "splits" / "test.csv"
            source = ToxicitySource.IMPLICIT_HATE

        if not data_path.exists():
            logger.warning(f"ImplicitHate data not found at {data_path}")
            return

        df = pd.read_csv(data_path)
        logger.info(f"Processing {len(df)} ImplicitHate records from {split} split")

        for _, row in df.iterrows():
            implicit_class = row.get("implicit_class", "unknown")
            extra_class = row.get("extra_implicit_class") if pd.notna(row.get("extra_implicit_class")) else None

            archetype, tokens = self._apply_implicit_hate_mapping(
                implicit_class=implicit_class,
                extra_class=extra_class
            )

            yield ToxicityRecord(
                text=row["post"],
                source=source,
                original_labels={
                    "implicit_class": implicit_class,
                    "extra_implicit_class": extra_class
                },
                oasis_archetype=archetype,
                oasis_tokens=tokens,
                implicit_class=implicit_class,
                metadata={
                    "extra_implicit_class": extra_class
                }
            )

    def prepare_for_chromadb(
        self,
        sources: Optional[List[ToxicitySource]] = None
    ) -> List[Dict]:
        """
        Prepare toxicity data for ChromaDB indexing.

        Args:
            sources: List of sources to include. If None, includes all.

        Returns:
            List of documents ready for ChromaDB ingestion
        """
        if sources is None:
            sources = list(ToxicitySource)

        documents = []

        for source in sources:
            if source == ToxicitySource.TOXIGEN_ANNOTATED:
                for record in self.iterate_toxigen_annotated():
                    documents.append(self._record_to_chromadb_doc(record))

            elif source == ToxicitySource.TOXIGEN_HATE:
                for record in self.iterate_toxigen_prompts("hate"):
                    documents.append(self._record_to_chromadb_doc(record))

            elif source == ToxicitySource.TOXIGEN_NEUTRAL:
                for record in self.iterate_toxigen_prompts("neutral"):
                    documents.append(self._record_to_chromadb_doc(record))

            elif source == ToxicitySource.IMPLICIT_HATE:
                for record in self.iterate_implicit_hate("all"):
                    documents.append(self._record_to_chromadb_doc(record))

            elif source == ToxicitySource.IMPLICIT_HATE_TRAIN:
                for record in self.iterate_implicit_hate("train"):
                    documents.append(self._record_to_chromadb_doc(record))

        logger.info(f"Prepared {len(documents)} documents for ChromaDB")
        return documents

    def _record_to_chromadb_doc(self, record: ToxicityRecord) -> Dict:
        """Convert a ToxicityRecord to ChromaDB document format."""
        return {
            "text": record.text,
            "metadata": {
                "source": record.source.value,
                "archetype": record.oasis_archetype,
                "tokens": ",".join(record.oasis_tokens),
                "demographic": record.demographic or "",
                "implicit_class": record.implicit_class or "",
                "toxicity_score": record.toxicity_score or 0.0,
                **{k: str(v) for k, v in record.metadata.items() if v is not None}
            }
        }

    def prepare_for_tfidf(
        self,
        sources: Optional[List[ToxicitySource]] = None
    ) -> pd.DataFrame:
        """
        Prepare toxicity data for TF-IDF indexing.

        Args:
            sources: List of sources to include. If None, includes all.

        Returns:
            DataFrame with text and labels ready for TF-IDF
        """
        records = []

        for source in sources or list(ToxicitySource):
            if source == ToxicitySource.TOXIGEN_ANNOTATED:
                records.extend(list(self.iterate_toxigen_annotated()))
            elif source == ToxicitySource.TOXIGEN_HATE:
                records.extend(list(self.iterate_toxigen_prompts("hate")))
            elif source == ToxicitySource.TOXIGEN_NEUTRAL:
                records.extend(list(self.iterate_toxigen_prompts("neutral")))
            elif source == ToxicitySource.IMPLICIT_HATE:
                records.extend(list(self.iterate_implicit_hate("all")))
            elif source == ToxicitySource.IMPLICIT_HATE_TRAIN:
                records.extend(list(self.iterate_implicit_hate("train")))

        df = pd.DataFrame([
            {
                "text": r.text,
                "source": r.source.value,
                "archetype": r.oasis_archetype,
                "tokens": ",".join(r.oasis_tokens),
                "demographic": r.demographic or "",
                "implicit_class": r.implicit_class or ""
            }
            for r in records
        ])

        logger.info(f"Prepared {len(df)} records for TF-IDF")
        return df

    def get_statistics(self) -> Dict:
        """
        Get statistics about available toxicity datasets.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "toxigen": {
                "available": self.toxigen_path.exists(),
                "annotated_samples": 0,
                "hate_prompts": 0,
                "neutral_prompts": 0,
                "demographics": []
            },
            "implicit_hate": {
                "available": self.implicit_hate_path.exists(),
                "total_samples": 0,
                "classes": []
            }
        }

        # ToxiGen stats
        if self.toxigen_path.exists():
            annotated_path = self.toxigen_path / "annotated" / "train.csv"
            if annotated_path.exists():
                df = pd.read_csv(annotated_path)
                stats["toxigen"]["annotated_samples"] = len(df)

            hate_path = self.toxigen_path / "prompts" / "hate"
            if hate_path.exists():
                for f in hate_path.glob("*.csv"):
                    df = pd.read_csv(f)
                    stats["toxigen"]["hate_prompts"] += len(df)
                    demo = f.stem.replace("hate_", "").replace("_1k", "")
                    if demo not in stats["toxigen"]["demographics"]:
                        stats["toxigen"]["demographics"].append(demo)

            neutral_path = self.toxigen_path / "prompts" / "neutral"
            if neutral_path.exists():
                for f in neutral_path.glob("*.csv"):
                    df = pd.read_csv(f)
                    stats["toxigen"]["neutral_prompts"] += len(df)

        # ImplicitHate stats
        if self.implicit_hate_path.exists():
            data_path = self.implicit_hate_path / "raw" / "implicit_hate.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                stats["implicit_hate"]["total_samples"] = len(df)
                stats["implicit_hate"]["classes"] = df["implicit_class"].unique().tolist()

        return stats


def build_toxicity_indices(
    output_dir: str = "data/imputation",
    sources: Optional[List[str]] = None
):
    """
    Build ChromaDB and TF-IDF indices for toxicity datasets.

    Args:
        output_dir: Directory to store indices
        sources: List of source names to include. If None, includes all.
    """
    from oasis.imputation.dataset_manager import DatasetManager

    preparator = ToxicityDatasetPreparator()

    # Convert source names to enum
    source_enums = None
    if sources:
        source_enums = [ToxicitySource[s.upper()] for s in sources if hasattr(ToxicitySource, s.upper())]

    # Get documents
    documents = preparator.prepare_for_chromadb(source_enums)

    logger.info(f"Building indices with {len(documents)} documents")
    logger.info(f"Output directory: {output_dir}")

    # Statistics
    stats = preparator.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    return documents, stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    preparator = ToxicityDatasetPreparator()

    # Get statistics
    stats = preparator.get_statistics()
    print("\n=== Toxicity Dataset Statistics ===")
    print(f"ToxiGen available: {stats['toxigen']['available']}")
    print(f"  Annotated samples: {stats['toxigen']['annotated_samples']}")
    print(f"  Hate prompts: {stats['toxigen']['hate_prompts']}")
    print(f"  Neutral prompts: {stats['toxigen']['neutral_prompts']}")
    print(f"  Demographics: {stats['toxigen']['demographics']}")

    print(f"\nImplicitHate available: {stats['implicit_hate']['available']}")
    print(f"  Total samples: {stats['implicit_hate']['total_samples']}")
    print(f"  Classes: {stats['implicit_hate']['classes']}")

    # Example: Iterate over some records
    print("\n=== Sample Records ===")
    count = 0
    for record in preparator.iterate_implicit_hate("train"):
        print(f"\nText: {record.text[:80]}...")
        print(f"  Archetype: {record.oasis_archetype}")
        print(f"  Tokens: {record.oasis_tokens}")
        print(f"  Implicit class: {record.implicit_class}")
        count += 1
        if count >= 3:
            break