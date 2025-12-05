"""
Unified loader for hate speech and toxicity datasets.

Provides standardized access to:
- Davidson Hate Speech & Offensive Language
- HateXplain (with rationales)
- Gab Hate Corpus
- PHEME Rumour Dataset
- Real Toxicity Prompts
- ToxiGen (via toxigen_loader.py)
- ImplicitHate (via implicit_hate_loader.py)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class HateSpeechSample:
    """Standardized sample from any hate speech dataset."""

    text: str
    dataset: str
    label: str
    label_scores: Dict[str, float] = field(default_factory=dict)
    target_groups: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, data_dir: Path, mapping_file: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.mapping_file = mapping_file
        self.mapping = self._load_mapping() if mapping_file else None

    def _load_mapping(self) -> Optional[Dict]:
        """Load OASIS label mapping file."""
        if self.mapping_file and self.mapping_file.exists():
            with open(self.mapping_file, "r") as f:
                return yaml.safe_load(f)
        return None

    def load(self) -> pd.DataFrame:
        """Load dataset as DataFrame."""
        raise NotImplementedError

    def iterate(self) -> Iterator[HateSpeechSample]:
        """Iterate over samples."""
        raise NotImplementedError


class DavidsonLoader(DatasetLoader):
    """Loader for Davidson Hate Speech & Offensive Language dataset."""

    LABEL_MAP = {0: "hate_speech", 1: "offensive_language", 2: "neither"}

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/hate_speech_datasets/davidson",
        mapping_file: Optional[Path] = None,
    ):
        if mapping_file is None:
            mapping_file = Path(
                "configs/imputation/mappings/davidson_to_oasis.yaml"
            )
        super().__init__(Path(data_dir), mapping_file)

    def load(self) -> pd.DataFrame:
        """Load Davidson dataset."""
        csv_path = self.data_dir / "davidson_full.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Davidson dataset not found at {csv_path}. "
                "Run scripts/download_hate_speech_datasets.py first."
            )

        df = pd.read_csv(csv_path)
        df["label_name"] = df["class"].map(self.LABEL_MAP)
        return df

    def iterate(self) -> Iterator[HateSpeechSample]:
        """Iterate over Davidson samples."""
        df = self.load()
        for _, row in df.iterrows():
            yield HateSpeechSample(
                text=row["tweet"],
                dataset="davidson",
                label=self.LABEL_MAP.get(row["class"], "unknown"),
                label_scores={
                    "hate_speech": row.get("hate_speech", 0) / row.get("count", 1),
                    "offensive_language": row.get("offensive_language", 0)
                    / row.get("count", 1),
                    "neither": row.get("neither", 0) / row.get("count", 1),
                },
                metadata={"annotator_count": row.get("count", 0)},
            )


class HateXplainLoader(DatasetLoader):
    """Loader for HateXplain dataset with explainable rationales."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/hate_speech_datasets/hatexplain",
        mapping_file: Optional[Path] = None,
    ):
        if mapping_file is None:
            mapping_file = Path(
                "configs/imputation/mappings/hatexplain_to_oasis.yaml"
            )
        super().__init__(Path(data_dir), mapping_file)

    def load(self) -> pd.DataFrame:
        """Load HateXplain dataset."""
        csv_path = self.data_dir / "hatexplain_full.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"HateXplain dataset not found at {csv_path}. "
                "Run scripts/download_hate_speech_datasets.py first."
            )

        df = pd.read_csv(csv_path)
        return df

    def iterate(self) -> Iterator[HateSpeechSample]:
        """Iterate over HateXplain samples."""
        df = self.load()
        for _, row in df.iterrows():
            # Parse target groups from JSON string
            targets = []
            if pd.notna(row.get("target_groups")):
                try:
                    targets = json.loads(row["target_groups"])
                except json.JSONDecodeError:
                    pass

            # Parse label counts
            label_scores = {}
            if pd.notna(row.get("label_counts")):
                try:
                    label_scores = json.loads(row["label_counts"])
                    total = sum(label_scores.values())
                    if total > 0:
                        label_scores = {k: v / total for k, v in label_scores.items()}
                except json.JSONDecodeError:
                    pass

            yield HateSpeechSample(
                text=row["text"],
                dataset="hatexplain",
                label=row["label"],
                label_scores=label_scores,
                target_groups=targets,
                metadata={
                    "post_id": row.get("post_id"),
                    "num_annotators": row.get("num_annotators", 0),
                    "split": row.get("split", "unknown"),
                },
            )


class GabHateCorpusLoader(DatasetLoader):
    """Loader for Gab Hate Corpus with hierarchical labels."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/hate_speech_datasets/gab_hate_corpus",
        mapping_file: Optional[Path] = None,
    ):
        if mapping_file is None:
            mapping_file = Path(
                "configs/imputation/mappings/gab_hate_corpus_to_oasis.yaml"
            )
        super().__init__(Path(data_dir), mapping_file)

    def load(self) -> pd.DataFrame:
        """Load Gab Hate Corpus."""
        csv_path = self.data_dir / "gab_hate_corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Gab Hate Corpus not found at {csv_path}. "
                "Run scripts/download_hate_speech_datasets.py first."
            )

        df = pd.read_csv(csv_path)
        return df

    def _get_primary_label(self, row: pd.Series) -> str:
        """Determine primary hate label from hierarchical labels."""
        if row.get("cv", 0) == 1:
            return "cv"
        elif row.get("hd", 0) == 1:
            return "hd"
        elif row.get("hate", 0) == 1:
            return "hate"
        else:
            return "not_hate"

    def iterate(self) -> Iterator[HateSpeechSample]:
        """Iterate over GHC samples."""
        df = self.load()
        for _, row in df.iterrows():
            label = self._get_primary_label(row)

            # Collect target groups
            targets = []
            target_cols = [
                "African",
                "Islam",
                "Jewish",
                "LGBTQ",
                "Women",
                "Refugee",
                "Arab",
                "Caucasian",
                "Hispanic",
                "Asian",
            ]
            for col in target_cols:
                if row.get(col, 0) == 1:
                    targets.append(col)

            yield HateSpeechSample(
                text=row.get("text", ""),
                dataset="gab_hate_corpus",
                label=label,
                label_scores={
                    "cv": float(row.get("cv", 0)),
                    "hd": float(row.get("hd", 0)),
                    "hate": float(row.get("hate", 0)),
                },
                target_groups=targets,
            )


class PHEMELoader(DatasetLoader):
    """Loader for PHEME Rumour Detection dataset."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/hate_speech_datasets/pheme",
        mapping_file: Optional[Path] = None,
    ):
        if mapping_file is None:
            mapping_file = Path("configs/imputation/mappings/pheme_to_oasis.yaml")
        super().__init__(Path(data_dir), mapping_file)

    def load(self) -> pd.DataFrame:
        """Load PHEME dataset."""
        csv_path = self.data_dir / "pheme_full.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"PHEME dataset not found at {csv_path}. "
                "Run scripts/download_hate_speech_datasets.py first."
            )

        df = pd.read_csv(csv_path)
        return df

    def iterate(self) -> Iterator[HateSpeechSample]:
        """Iterate over PHEME samples."""
        df = self.load()
        for _, row in df.iterrows():
            yield HateSpeechSample(
                text=row["text"],
                dataset="pheme",
                label=row["label"],
                metadata={
                    "tweet_id": row.get("tweet_id"),
                    "event": row.get("event"),
                    "is_rumour": row.get("is_rumour"),
                    "user_verified": row.get("user_verified"),
                    "retweet_count": row.get("retweet_count"),
                    "favorite_count": row.get("favorite_count"),
                    "num_reactions": row.get("num_reactions"),
                },
            )


class UnifiedHateSpeechLoader:
    """
    Unified loader that provides access to all hate speech datasets.

    Usage:
        loader = UnifiedHateSpeechLoader()
        for sample in loader.iterate_all():
            print(sample.text, sample.label)

        # Or load specific dataset
        davidson_df = loader.load_davidson()
    """

    def __init__(self, base_dir: Union[str, Path] = "data/hate_speech_datasets"):
        self.base_dir = Path(base_dir)

        # Initialize individual loaders
        self._loaders: Dict[str, DatasetLoader] = {}

    def _get_loader(self, name: str) -> DatasetLoader:
        """Get or create a dataset loader."""
        if name not in self._loaders:
            loader_map = {
                "davidson": lambda: DavidsonLoader(self.base_dir / "davidson"),
                "hatexplain": lambda: HateXplainLoader(self.base_dir / "hatexplain"),
                "gab": lambda: GabHateCorpusLoader(self.base_dir / "gab_hate_corpus"),
                "pheme": lambda: PHEMELoader(self.base_dir / "pheme"),
            }
            if name in loader_map:
                self._loaders[name] = loader_map[name]()
            else:
                raise ValueError(f"Unknown dataset: {name}")

        return self._loaders[name]

    def load_davidson(self) -> pd.DataFrame:
        """Load Davidson dataset."""
        return self._get_loader("davidson").load()

    def load_hatexplain(self) -> pd.DataFrame:
        """Load HateXplain dataset."""
        return self._get_loader("hatexplain").load()

    def load_gab(self) -> pd.DataFrame:
        """Load Gab Hate Corpus."""
        return self._get_loader("gab").load()

    def load_pheme(self) -> pd.DataFrame:
        """Load PHEME dataset."""
        return self._get_loader("pheme").load()

    def iterate_dataset(self, name: str) -> Iterator[HateSpeechSample]:
        """Iterate over a specific dataset."""
        return self._get_loader(name).iterate()

    def iterate_all(
        self, datasets: Optional[List[str]] = None
    ) -> Iterator[HateSpeechSample]:
        """
        Iterate over all (or specified) datasets.

        Args:
            datasets: List of dataset names to include. If None, includes all.
        """
        if datasets is None:
            datasets = ["davidson", "hatexplain", "gab", "pheme"]

        for name in datasets:
            try:
                for sample in self.iterate_dataset(name):
                    yield sample
            except FileNotFoundError as e:
                logger.warning(f"Skipping {name}: {e}")

    def get_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all available datasets."""
        stats = {}
        datasets = ["davidson", "hatexplain", "gab", "pheme"]

        for name in datasets:
            try:
                df = self._get_loader(name).load()
                stats[name] = {
                    "total_samples": len(df),
                    "columns": df.columns.tolist(),
                }
            except FileNotFoundError:
                stats[name] = {"status": "not_downloaded"}

        return stats


# Convenience function
def load_all_hate_speech_data(
    base_dir: str = "data/hate_speech_datasets",
) -> pd.DataFrame:
    """
    Load and concatenate all hate speech datasets into a single DataFrame.

    Returns DataFrame with standardized columns: text, dataset, label
    """
    loader = UnifiedHateSpeechLoader(base_dir)
    records = []

    for sample in loader.iterate_all():
        records.append(
            {
                "text": sample.text,
                "dataset": sample.dataset,
                "label": sample.label,
                "target_groups": json.dumps(sample.target_groups),
                "label_scores": json.dumps(sample.label_scores),
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    loader = UnifiedHateSpeechLoader()
    stats = loader.get_statistics()

    print("\n=== Dataset Statistics ===")
    for name, info in stats.items():
        print(f"{name}: {info}")