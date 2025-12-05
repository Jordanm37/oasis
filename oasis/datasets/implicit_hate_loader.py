"""
ImplicitHate dataset loader for OASIS simulations.

This module provides utilities for loading and using the ImplicitHate dataset
from SALT-NLP, which focuses on implicit and subtle forms of hate speech
that are harder to detect than explicit hate.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import pandas as pd
import random
import logging

logger = logging.getLogger(__name__)


class ImplicitHateLoader:
    """Loader for ImplicitHate dataset from SALT-NLP."""

    # Define implicit hate classes found in the dataset
    IMPLICIT_CLASSES = [
        "white_grievance",
        "irony",
        "stereotypical",
        "threatening",
        "incitement",
        "inferiority",
        "dismissive",
        "dehumanization"
    ]

    def __init__(self, data_path: str = "./data/implicit_hate_datasets/"):
        """
        Initialize ImplicitHate loader.

        Args:
            data_path: Path to the ImplicitHate datasets directory
        """
        self.data_path = Path(data_path)
        self._validate_data_path()
        self._load_class_distribution()

    def _validate_data_path(self):
        """Validate that the data path exists and contains expected directories."""
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist. "
                           "Please run scripts/download_implicithate_dataset.py first.")

        required_dirs = ["raw", "splits"]
        for dir_name in required_dirs:
            if not (self.data_path / dir_name).exists():
                raise ValueError(f"Missing required directory: {self.data_path / dir_name}")

    def _load_class_distribution(self):
        """Load and analyze the distribution of implicit classes."""
        try:
            df = pd.read_csv(self.data_path / "raw" / "implicit_hate.csv")
            self.class_distribution = df['implicit_class'].value_counts().to_dict()
            self.available_classes = list(self.class_distribution.keys())
            logger.info(f"Found {len(self.available_classes)} implicit hate classes")
        except Exception as e:
            logger.warning(f"Could not load class distribution: {e}")
            self.class_distribution = {}
            self.available_classes = []

    def load_data(
        self,
        split: Literal["train", "test", "all"] = "train",
        implicit_classes: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load ImplicitHate data with optional filtering.

        Args:
            split: Dataset split to load ("train", "test", or "all")
            implicit_classes: List of implicit hate classes to include
            sample_size: Number of samples to return
            random_seed: Random seed for sampling

        Returns:
            DataFrame with implicit hate posts and labels
        """
        if split == "all":
            file_path = self.data_path / "raw" / "implicit_hate.csv"
        else:
            file_path = self.data_path / "splits" / f"{split}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {split} split")

        # Filter by implicit classes if specified
        if implicit_classes:
            df = df[df['implicit_class'].isin(implicit_classes)]
            logger.info(f"Filtered to {len(df)} samples for classes: {implicit_classes}")

        # Sample if requested
        if sample_size and len(df) > sample_size:
            if random_seed:
                random.seed(random_seed)
            df = df.sample(n=sample_size, random_state=random_seed)
            logger.info(f"Sampled {sample_size} posts")

        return df

    def get_class_statistics(self) -> Dict[str, int]:
        """
        Get statistics about implicit hate classes.

        Returns:
            Dictionary mapping class names to counts
        """
        return self.class_distribution

    def create_implicit_hate_scenario(
        self,
        num_posts: int = 30,
        class_weights: Optional[Dict[str, float]] = None,
        include_extra_classes: bool = False,
        random_seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Create a scenario with implicit hate posts for simulation.

        Args:
            num_posts: Number of posts to include
            class_weights: Optional weights for each implicit class
            include_extra_classes: Whether to include extra_implicit_class info
            random_seed: Random seed for reproducibility

        Returns:
            List of dictionaries containing post content and metadata
        """
        if random_seed:
            random.seed(random_seed)

        # Load all data
        df = self.load_data(split="all")

        # Apply class weights if provided
        if class_weights:
            weighted_samples = []
            for class_name, weight in class_weights.items():
                class_df = df[df['implicit_class'] == class_name]
                n_samples = int(num_posts * weight)
                if len(class_df) >= n_samples:
                    weighted_samples.append(class_df.sample(n=n_samples, random_state=random_seed))

            if weighted_samples:
                df = pd.concat(weighted_samples, ignore_index=True)

        # Sample posts
        if len(df) > num_posts:
            df = df.sample(n=num_posts, random_state=random_seed)

        # Create scenario posts
        posts = []
        for _, row in df.iterrows():
            post = {
                "content": row['post'],
                "implicit_class": row['implicit_class'],
                "is_implicit_hate": True,
                "metadata": {
                    "source": "ImplicitHate",
                    "dataset": "SALT-NLP/ImplicitHate"
                }
            }

            # Add extra class if available and requested
            if include_extra_classes and pd.notna(row.get('extra_implicit_class')):
                post['extra_implicit_class'] = row['extra_implicit_class']

            posts.append(post)

        # Shuffle posts
        random.shuffle(posts)

        logger.info(f"Created scenario with {len(posts)} implicit hate posts")

        # Log class distribution in scenario
        class_counts = {}
        for post in posts:
            cls = post['implicit_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        logger.info(f"Class distribution: {class_counts}")

        return posts

    def compare_with_explicit_hate(
        self,
        toxigen_loader=None,
        num_implicit: int = 20,
        num_explicit: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Create a comparative scenario with both implicit and explicit hate.

        Args:
            toxigen_loader: Optional ToxiGenLoader instance for explicit hate
            num_implicit: Number of implicit hate posts
            num_explicit: Number of explicit hate posts

        Returns:
            Dictionary with 'implicit' and 'explicit' post lists
        """
        # Get implicit hate posts
        implicit_posts = self.create_implicit_hate_scenario(
            num_posts=num_implicit,
            random_seed=42
        )

        # Get explicit hate posts if loader provided
        explicit_posts = []
        if toxigen_loader:
            try:
                toxic_prompts = toxigen_loader.load_prompts(
                    prompt_type="hate",
                    sample_size=num_explicit,
                    random_seed=42
                )
                for _, row in toxic_prompts.iterrows():
                    explicit_posts.append({
                        "content": row['text'],
                        "is_explicit_hate": True,
                        "demographic": row.get('demographic', 'unknown'),
                        "metadata": {
                            "source": "ToxiGen",
                            "type": "explicit"
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not load ToxiGen data: {e}")

        return {
            "implicit": implicit_posts,
            "explicit": explicit_posts
        }

    def get_posts_by_pattern(
        self,
        pattern: Literal["white_grievance", "irony", "stereotypical", "threatening",
                        "incitement", "inferiority", "dismissive", "dehumanization"],
        max_posts: int = 10
    ) -> List[str]:
        """
        Get posts that follow a specific implicit hate pattern.

        Args:
            pattern: The implicit hate pattern to retrieve
            max_posts: Maximum number of posts to return

        Returns:
            List of post texts
        """
        df = self.load_data(split="all", implicit_classes=[pattern])

        if len(df) > max_posts:
            df = df.sample(n=max_posts, random_state=42)

        return df['post'].tolist()

    def create_detection_challenge(
        self,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        num_posts: int = 50
    ) -> Tuple[List[Dict], List[int]]:
        """
        Create a detection challenge dataset with varying difficulty.

        Args:
            difficulty: Difficulty level affecting subtlety of hate speech
            num_posts: Number of posts in the challenge

        Returns:
            Tuple of (posts, labels) where labels are binary (0=safe, 1=hate)
        """
        # Define difficulty mappings
        difficulty_classes = {
            "easy": ["threatening", "dehumanization"],  # More obvious
            "medium": ["stereotypical", "inferiority", "dismissive"],
            "hard": ["irony", "white_grievance", "incitement"]  # More subtle
        }

        selected_classes = difficulty_classes.get(difficulty, self.available_classes)

        # Get implicit hate posts
        implicit_df = self.load_data(
            split="all",
            implicit_classes=selected_classes,
            sample_size=num_posts,
            random_seed=42
        )

        posts = []
        labels = []

        for _, row in implicit_df.iterrows():
            posts.append({
                "content": row['post'],
                "implicit_class": row['implicit_class'],
                "difficulty": difficulty
            })
            labels.append(1)  # All implicit hate posts are labeled as hate

        logger.info(f"Created {difficulty} difficulty detection challenge with {len(posts)} posts")

        return posts, labels


# Helper functions for integration with OASIS

def create_implicit_hate_simulation(
    agent_count: int = 20,
    focus_classes: Optional[List[str]] = None,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Create an implicit hate detection simulation setup.

    Args:
        agent_count: Number of agents in the simulation
        focus_classes: Specific implicit hate classes to focus on
        seed: Random seed for reproducibility

    Returns:
        Tuple of (initial_posts, simulation_config)
    """
    loader = ImplicitHateLoader()

    # Create scenario
    posts = loader.create_implicit_hate_scenario(
        num_posts=agent_count * 2,  # More posts than agents
        random_seed=seed
    )

    # Create simulation config
    config = {
        "scenario_type": "implicit_hate_detection",
        "agent_count": agent_count,
        "focus_classes": focus_classes or "all",
        "seed": seed,
        "initial_content_count": len(posts),
        "data_source": "ImplicitHate (SALT-NLP)"
    }

    return posts, config


def load_implicit_detection_data(
    train_size: int = 1000,
    test_size: int = 200,
    include_classes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for training implicit hate detection models.

    Args:
        train_size: Number of training samples
        test_size: Number of test samples
        include_classes: Specific implicit classes to include

    Returns:
        Tuple of (train_df, test_df)
    """
    loader = ImplicitHateLoader()

    # Load data
    train_df = loader.load_data(
        split="train",
        implicit_classes=include_classes,
        sample_size=train_size,
        random_seed=42
    )

    test_df = loader.load_data(
        split="test",
        implicit_classes=include_classes,
        sample_size=test_size,
        random_seed=42
    )

    logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")

    return train_df, test_df


if __name__ == "__main__":
    # Example: Print dataset statistics
    loader = ImplicitHateLoader()

    print("\n=== ImplicitHate Dataset Statistics ===\n")

    # Show class distribution
    class_stats = loader.get_class_statistics()
    print("Implicit Hate Class Distribution:")
    for class_name, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:20s}: {count:4d} samples")

    # Example posts by pattern
    print("\n=== Example Posts by Pattern ===\n")

    for pattern in ["white_grievance", "irony", "stereotypical"]:
        print(f"\n{pattern.upper()} examples:")
        posts = loader.get_posts_by_pattern(pattern, max_posts=2)
        for i, post in enumerate(posts, 1):
            preview = post[:100] + "..." if len(post) > 100 else post
            print(f"  {i}. {preview}")

    # Create a detection challenge
    print("\n=== Creating Detection Challenge ===\n")
    posts, labels = loader.create_detection_challenge(difficulty="medium", num_posts=10)
    print(f"Created challenge with {len(posts)} posts")
    print(f"Label distribution: {sum(labels)} hate, {len(labels) - sum(labels)} safe")