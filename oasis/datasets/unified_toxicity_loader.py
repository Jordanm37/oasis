"""
Unified toxicity dataset loader for OASIS simulations.

This module provides a unified interface for working with multiple
hate speech and toxicity datasets including ToxiGen and ImplicitHate.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union
import pandas as pd
import random
import logging
from enum import Enum

# Import individual loaders
from .toxigen_loader import ToxiGenLoader
from .implicit_hate_loader import ImplicitHateLoader

logger = logging.getLogger(__name__)


class ToxicityType(Enum):
    """Types of toxicity in datasets."""
    EXPLICIT_HATE = "explicit_hate"
    IMPLICIT_HATE = "implicit_hate"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class UnifiedToxicityLoader:
    """
    Unified loader for multiple toxicity datasets.

    Provides a single interface for working with ToxiGen, ImplicitHate,
    and potentially other toxicity datasets in OASIS simulations.
    """

    def __init__(
        self,
        toxigen_path: str = "./data/toxigen_datasets/",
        implicit_hate_path: str = "./data/implicit_hate_datasets/"
    ):
        """
        Initialize unified toxicity loader.

        Args:
            toxigen_path: Path to ToxiGen datasets
            implicit_hate_path: Path to ImplicitHate datasets
        """
        self.loaders = {}
        self.available_datasets = []

        # Try to load ToxiGen
        try:
            self.loaders['toxigen'] = ToxiGenLoader(toxigen_path)
            self.available_datasets.append('toxigen')
            logger.info("✓ ToxiGen dataset loaded")
        except Exception as e:
            logger.warning(f"ToxiGen dataset not available: {e}")

        # Try to load ImplicitHate
        try:
            self.loaders['implicit_hate'] = ImplicitHateLoader(implicit_hate_path)
            self.available_datasets.append('implicit_hate')
            logger.info("✓ ImplicitHate dataset loaded")
        except Exception as e:
            logger.warning(f"ImplicitHate dataset not available: {e}")

        if not self.available_datasets:
            raise ValueError("No toxicity datasets available. Please run download scripts.")

        logger.info(f"Unified loader initialized with datasets: {self.available_datasets}")

    def create_mixed_toxicity_scenario(
        self,
        num_explicit_hate: int = 10,
        num_implicit_hate: int = 10,
        num_neutral: int = 20,
        demographics: Optional[List[str]] = None,
        implicit_classes: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Create a scenario mixing explicit hate, implicit hate, and neutral content.

        Args:
            num_explicit_hate: Number of explicit hate posts
            num_implicit_hate: Number of implicit hate posts
            num_neutral: Number of neutral posts
            demographics: Demographics to focus on (for ToxiGen)
            implicit_classes: Implicit hate classes to focus on
            random_seed: Random seed for reproducibility

        Returns:
            List of posts with mixed toxicity types
        """
        if random_seed:
            random.seed(random_seed)

        all_posts = []

        # Add explicit hate from ToxiGen
        if 'toxigen' in self.loaders and num_explicit_hate > 0:
            toxigen = self.loaders['toxigen']
            hate_prompts = toxigen.load_prompts(
                prompt_type="hate",
                demographics=demographics,
                sample_size=num_explicit_hate,
                random_seed=random_seed
            )

            for _, row in hate_prompts.iterrows():
                all_posts.append({
                    "content": row['text'],
                    "toxicity_type": ToxicityType.EXPLICIT_HATE.value,
                    "source": "ToxiGen",
                    "demographic": row.get('demographic', 'unknown'),
                    "metadata": {
                        "dataset": "toxigen",
                        "prompt_type": "hate"
                    }
                })

        # Add implicit hate from ImplicitHate
        if 'implicit_hate' in self.loaders and num_implicit_hate > 0:
            implicit = self.loaders['implicit_hate']
            implicit_posts = implicit.create_implicit_hate_scenario(
                num_posts=num_implicit_hate,
                random_seed=random_seed
            )

            for post in implicit_posts:
                all_posts.append({
                    "content": post['content'],
                    "toxicity_type": ToxicityType.IMPLICIT_HATE.value,
                    "source": "ImplicitHate",
                    "implicit_class": post['implicit_class'],
                    "metadata": {
                        "dataset": "implicit_hate",
                        "class": post['implicit_class']
                    }
                })

        # Add neutral content from ToxiGen
        if 'toxigen' in self.loaders and num_neutral > 0:
            toxigen = self.loaders['toxigen']
            neutral_prompts = toxigen.load_prompts(
                prompt_type="neutral",
                demographics=demographics,
                sample_size=num_neutral,
                random_seed=random_seed
            )

            for _, row in neutral_prompts.iterrows():
                all_posts.append({
                    "content": row['text'],
                    "toxicity_type": ToxicityType.NEUTRAL.value,
                    "source": "ToxiGen",
                    "demographic": row.get('demographic', 'unknown'),
                    "metadata": {
                        "dataset": "toxigen",
                        "prompt_type": "neutral"
                    }
                })

        # Shuffle all posts
        random.shuffle(all_posts)

        logger.info(f"Created mixed scenario with {len(all_posts)} posts:")
        logger.info(f"  - Explicit hate: {num_explicit_hate}")
        logger.info(f"  - Implicit hate: {num_implicit_hate}")
        logger.info(f"  - Neutral: {num_neutral}")

        return all_posts

    def create_detection_benchmark(
        self,
        include_explicit: bool = True,
        include_implicit: bool = True,
        include_neutral: bool = True,
        samples_per_type: int = 100,
        balance: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a benchmark dataset for training/testing toxicity detection.

        Args:
            include_explicit: Include explicit hate samples
            include_implicit: Include implicit hate samples
            include_neutral: Include neutral samples
            samples_per_type: Samples per toxicity type
            balance: Whether to balance the dataset

        Returns:
            Tuple of (features_df, labels)
        """
        data = []
        labels = []

        # Collect explicit hate
        if include_explicit and 'toxigen' in self.loaders:
            toxigen = self.loaders['toxigen']
            hate_df = toxigen.load_prompts(
                prompt_type="hate",
                sample_size=samples_per_type,
                random_seed=42
            )
            for _, row in hate_df.iterrows():
                data.append({
                    'text': row['text'],
                    'toxicity_type': 'explicit_hate',
                    'source': 'toxigen'
                })
                labels.append(1)  # Toxic

        # Collect implicit hate
        if include_implicit and 'implicit_hate' in self.loaders:
            implicit = self.loaders['implicit_hate']
            implicit_df = implicit.load_data(
                split="all",
                sample_size=samples_per_type,
                random_seed=42
            )
            for _, row in implicit_df.iterrows():
                data.append({
                    'text': row['post'],
                    'toxicity_type': 'implicit_hate',
                    'source': 'implicit_hate',
                    'implicit_class': row['implicit_class']
                })
                labels.append(1)  # Toxic

        # Collect neutral samples
        if include_neutral and 'toxigen' in self.loaders:
            toxigen = self.loaders['toxigen']
            neutral_df = toxigen.load_prompts(
                prompt_type="neutral",
                sample_size=samples_per_type,
                random_seed=42
            )
            for _, row in neutral_df.iterrows():
                data.append({
                    'text': row['text'],
                    'toxicity_type': 'neutral',
                    'source': 'toxigen'
                })
                labels.append(0)  # Not toxic

        # Create DataFrame
        df = pd.DataFrame(data)
        labels_series = pd.Series(labels)

        # Balance if requested
        if balance:
            toxic_count = sum(labels)
            non_toxic_count = len(labels) - toxic_count
            min_count = min(toxic_count, non_toxic_count)

            if min_count > 0:
                # Get balanced indices
                toxic_indices = [i for i, l in enumerate(labels) if l == 1][:min_count]
                non_toxic_indices = [i for i, l in enumerate(labels) if l == 0][:min_count]
                balanced_indices = toxic_indices + non_toxic_indices
                random.shuffle(balanced_indices)

                df = df.iloc[balanced_indices].reset_index(drop=True)
                labels_series = labels_series.iloc[balanced_indices].reset_index(drop=True)

        logger.info(f"Created benchmark with {len(df)} samples")
        logger.info(f"Label distribution: {labels_series.value_counts().to_dict()}")

        return df, labels_series

    def analyze_toxicity_spectrum(
        self,
        text_samples: List[str]
    ) -> List[Dict]:
        """
        Analyze where text samples fall on the toxicity spectrum.

        Args:
            text_samples: List of text samples to analyze

        Returns:
            List of analysis results for each sample
        """
        results = []

        for text in text_samples:
            analysis = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "length": len(text),
                "likely_type": ToxicityType.UNKNOWN.value,
                "confidence": 0.0,
                "features": []
            }

            # Simple heuristic analysis (in practice, use ML models)
            text_lower = text.lower()

            # Check for explicit hate indicators
            explicit_indicators = ['hate', 'kill', 'destroy', 'inferior', 'subhuman']
            explicit_count = sum(1 for ind in explicit_indicators if ind in text_lower)

            # Check for implicit patterns
            implicit_patterns = ['we need to', 'our country', 'these people', 'them', 'they']
            implicit_count = sum(1 for pat in implicit_patterns if pat in text_lower)

            # Determine likely type
            if explicit_count > 2:
                analysis['likely_type'] = ToxicityType.EXPLICIT_HATE.value
                analysis['confidence'] = min(explicit_count / 5, 1.0)
                analysis['features'].append(f"explicit_indicators: {explicit_count}")
            elif implicit_count > 2 and explicit_count < 2:
                analysis['likely_type'] = ToxicityType.IMPLICIT_HATE.value
                analysis['confidence'] = min(implicit_count / 5, 1.0)
                analysis['features'].append(f"implicit_patterns: {implicit_count}")
            elif explicit_count == 0 and implicit_count < 2:
                analysis['likely_type'] = ToxicityType.NEUTRAL.value
                analysis['confidence'] = 0.7
                analysis['features'].append("no_hate_indicators")

            results.append(analysis)

        return results

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics across all loaded datasets.

        Returns:
            Dictionary with statistics for each dataset
        """
        stats = {
            "available_datasets": self.available_datasets,
            "dataset_stats": {}
        }

        # ToxiGen stats
        if 'toxigen' in self.loaders:
            toxigen = self.loaders['toxigen']
            toxigen_stats = toxigen.get_statistics()
            stats['dataset_stats']['toxigen'] = {
                'annotated_samples': toxigen_stats.get('annotated', {}),
                'prompt_samples': toxigen_stats.get('prompts', {}),
                'demographics': len(toxigen.get_available_demographics().get('hate', []))
            }

        # ImplicitHate stats
        if 'implicit_hate' in self.loaders:
            implicit = self.loaders['implicit_hate']
            stats['dataset_stats']['implicit_hate'] = {
                'class_distribution': implicit.get_class_statistics(),
                'total_samples': sum(implicit.get_class_statistics().values())
            }

        return stats


# Example usage functions

def create_comprehensive_moderation_test(
    test_name: str = "comprehensive_toxicity",
    agent_count: int = 30
) -> Dict:
    """
    Create a comprehensive content moderation test scenario.

    Args:
        test_name: Name for the test scenario
        agent_count: Number of agents to simulate

    Returns:
        Dictionary with test configuration and data
    """
    loader = UnifiedToxicityLoader()

    # Create diverse content mix
    posts = loader.create_mixed_toxicity_scenario(
        num_explicit_hate=10,
        num_implicit_hate=15,
        num_neutral=25,
        demographics=["women", "immigrant", "lgbtq"],
        random_seed=42
    )

    # Create test configuration
    config = {
        "test_name": test_name,
        "agent_count": agent_count,
        "content_distribution": {
            "explicit_hate": 10,
            "implicit_hate": 15,
            "neutral": 25,
            "total": len(posts)
        },
        "datasets_used": loader.available_datasets,
        "test_objectives": [
            "Measure detection rates for explicit vs implicit hate",
            "Analyze agent responses to different toxicity types",
            "Evaluate moderation effectiveness",
            "Study information spread patterns"
        ]
    }

    return {
        "config": config,
        "posts": posts,
        "loader": loader
    }


if __name__ == "__main__":
    # Example usage
    print("\n=== Unified Toxicity Loader Demo ===\n")

    # Initialize loader
    try:
        loader = UnifiedToxicityLoader()

        # Get statistics
        stats = loader.get_statistics()
        print(f"Available datasets: {stats['available_datasets']}")

        # Create mixed scenario
        print("\n=== Creating Mixed Toxicity Scenario ===")
        posts = loader.create_mixed_toxicity_scenario(
            num_explicit_hate=5,
            num_implicit_hate=5,
            num_neutral=10
        )
        print(f"Created {len(posts)} posts")

        # Analyze toxicity types
        type_counts = {}
        for post in posts:
            t = post['toxicity_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Distribution: {type_counts}")

        # Create detection benchmark
        print("\n=== Creating Detection Benchmark ===")
        df, labels = loader.create_detection_benchmark(
            samples_per_type=50,
            balance=True
        )
        print(f"Benchmark size: {len(df)} samples")
        print(f"Balanced labels: {labels.value_counts().to_dict()}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure datasets are downloaded by running:")
        print("  python scripts/download_toxigen_datasets.py")
        print("  python scripts/download_implicithate_dataset.py")