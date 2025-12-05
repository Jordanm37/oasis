"""
ToxiGen dataset loader for OASIS simulations.

This module provides utilities for loading and using ToxiGen datasets
in OASIS social media simulations for studying toxic content,
hate speech, and content moderation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import pandas as pd
import random
import logging

logger = logging.getLogger(__name__)


class ToxiGenLoader:
    """Loader for ToxiGen datasets."""

    def __init__(self, data_path: str = "./data/toxigen_datasets/"):
        """
        Initialize ToxiGen loader.

        Args:
            data_path: Path to the ToxiGen datasets directory
        """
        self.data_path = Path(data_path)
        self._validate_data_path()

    def _validate_data_path(self):
        """Validate that the data path exists and contains expected directories."""
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist. "
                           "Please run scripts/download_toxigen_datasets.py first.")

        required_dirs = ["annotated", "prompts"]
        for dir_name in required_dirs:
            if not (self.data_path / dir_name).exists():
                raise ValueError(f"Missing required directory: {self.data_path / dir_name}")

    def load_annotated_data(
        self,
        split: Literal["train", "test"] = "train"
    ) -> pd.DataFrame:
        """
        Load annotated ToxiGen data with toxicity labels.

        Args:
            split: Dataset split to load ("train" or "test")

        Returns:
            DataFrame with annotated text and toxicity labels
        """
        file_path = self.data_path / "annotated" / f"{split}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Annotated {split} data not found at {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} annotated {split} samples")

        return df

    def load_prompts(
        self,
        prompt_type: Literal["hate", "neutral", "all"] = "all",
        demographics: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load prompt datasets for specific demographics and types.

        Args:
            prompt_type: Type of prompts to load ("hate", "neutral", or "all")
            demographics: List of demographic groups to include.
                         If None, includes all available groups.
            sample_size: Number of samples to return. If None, returns all.
            random_seed: Random seed for sampling

        Returns:
            DataFrame with prompts
        """
        prompts_data = []

        # Determine which directories to search
        if prompt_type == "hate":
            search_dirs = [self.data_path / "prompts" / "hate"]
        elif prompt_type == "neutral":
            search_dirs = [self.data_path / "prompts" / "neutral"]
        else:  # "all"
            search_dirs = [
                self.data_path / "prompts" / "hate",
                self.data_path / "prompts" / "neutral"
            ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for csv_file in search_dir.glob("*.csv"):
                # Extract demographic from filename
                filename = csv_file.stem  # e.g., "hate_asian_1k"
                parts = filename.split("_")

                # Extract type and demographic
                prompt_label = parts[0]  # "hate" or "neutral"
                demographic = "_".join(parts[1:-1])  # e.g., "asian" or "mental_disability"

                # Skip if not in requested demographics
                if demographics and demographic not in demographics:
                    continue

                # Load the data
                df = pd.read_csv(csv_file)
                df['prompt_type'] = prompt_label
                df['demographic'] = demographic
                df['source_file'] = csv_file.name

                prompts_data.append(df)

        if not prompts_data:
            logger.warning("No prompts found matching criteria")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(prompts_data, ignore_index=True)

        # Sample if requested
        if sample_size and len(combined_df) > sample_size:
            if random_seed:
                random.seed(random_seed)
            combined_df = combined_df.sample(n=sample_size, random_state=random_seed)

        logger.info(f"Loaded {len(combined_df)} prompts "
                   f"({len(combined_df[combined_df['prompt_type'] == 'hate'])} hate, "
                   f"{len(combined_df[combined_df['prompt_type'] == 'neutral'])} neutral)")

        return combined_df

    def get_available_demographics(self) -> Dict[str, List[str]]:
        """
        Get list of available demographics for hate and neutral prompts.

        Returns:
            Dictionary with 'hate' and 'neutral' keys containing lists of demographics
        """
        demographics = {"hate": [], "neutral": []}

        for prompt_type in ["hate", "neutral"]:
            dir_path = self.data_path / "prompts" / prompt_type
            if dir_path.exists():
                for csv_file in dir_path.glob("*.csv"):
                    filename = csv_file.stem
                    parts = filename.split("_")
                    demographic = "_".join(parts[1:-1])
                    demographics[prompt_type].append(demographic)

        return demographics

    def create_toxic_content_scenario(
        self,
        num_toxic_posts: int = 10,
        num_neutral_posts: int = 20,
        demographics: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Create a scenario with a mix of toxic and neutral content for simulation.

        Args:
            num_toxic_posts: Number of toxic posts to include
            num_neutral_posts: Number of neutral posts to include
            demographics: Specific demographics to focus on
            random_seed: Random seed for reproducibility

        Returns:
            List of dictionaries containing post content and metadata
        """
        if random_seed:
            random.seed(random_seed)

        # Load toxic prompts
        toxic_prompts = self.load_prompts(
            prompt_type="hate",
            demographics=demographics,
            sample_size=num_toxic_posts,
            random_seed=random_seed
        )

        # Load neutral prompts
        neutral_prompts = self.load_prompts(
            prompt_type="neutral",
            demographics=demographics,
            sample_size=num_neutral_posts,
            random_seed=random_seed
        )

        # Create scenario posts
        posts = []

        for _, row in toxic_prompts.iterrows():
            posts.append({
                "content": row['text'],
                "is_toxic": True,
                "prompt_type": "hate",
                "demographic": row['demographic'],
                "metadata": {
                    "source": "ToxiGen",
                    "original_file": row['source_file']
                }
            })

        for _, row in neutral_prompts.iterrows():
            posts.append({
                "content": row['text'],
                "is_toxic": False,
                "prompt_type": "neutral",
                "demographic": row['demographic'],
                "metadata": {
                    "source": "ToxiGen",
                    "original_file": row['source_file']
                }
            })

        # Shuffle posts
        random.shuffle(posts)

        logger.info(f"Created scenario with {len(posts)} posts "
                   f"({num_toxic_posts} toxic, {num_neutral_posts} neutral)")

        return posts

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded ToxiGen datasets.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "annotated": {},
            "prompts": {
                "hate": {},
                "neutral": {}
            }
        }

        # Annotated data stats
        for split in ["train", "test"]:
            try:
                df = self.load_annotated_data(split)
                stats["annotated"][split] = {
                    "count": len(df),
                    "columns": list(df.columns)
                }

                # Add toxicity distribution if available
                if 'toxicity_human' in df.columns:
                    stats["annotated"][split]["avg_human_toxicity"] = df['toxicity_human'].mean()
                if 'toxicity_ai' in df.columns:
                    stats["annotated"][split]["avg_ai_toxicity"] = df['toxicity_ai'].mean()

            except FileNotFoundError:
                stats["annotated"][split] = {"count": 0}

        # Prompt stats
        demographics = self.get_available_demographics()

        for prompt_type in ["hate", "neutral"]:
            stats["prompts"][prompt_type] = {
                "demographics": demographics.get(prompt_type, []),
                "total_demographics": len(demographics.get(prompt_type, []))
            }

            # Count total prompts
            total_prompts = 0
            for demo in demographics.get(prompt_type, []):
                df = self.load_prompts(prompt_type=prompt_type, demographics=[demo])
                total_prompts += len(df)

            stats["prompts"][prompt_type]["total_prompts"] = total_prompts

        return stats


# Example usage functions for OASIS integration

def create_content_moderation_simulation(
    agent_count: int = 20,
    toxic_content_ratio: float = 0.3,
    focus_demographics: Optional[List[str]] = None,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Create a content moderation simulation setup with ToxiGen data.

    Args:
        agent_count: Number of agents in the simulation
        toxic_content_ratio: Ratio of toxic content to include
        focus_demographics: Specific demographics to focus on
        seed: Random seed for reproducibility

    Returns:
        Tuple of (initial_posts, simulation_config)
    """
    loader = ToxiGenLoader()

    # Calculate content counts
    num_toxic = int(agent_count * toxic_content_ratio)
    num_neutral = agent_count - num_toxic

    # Create scenario
    posts = loader.create_toxic_content_scenario(
        num_toxic_posts=num_toxic,
        num_neutral_posts=num_neutral,
        demographics=focus_demographics,
        random_seed=seed
    )

    # Create simulation config
    config = {
        "scenario_type": "content_moderation",
        "agent_count": agent_count,
        "toxic_ratio": toxic_content_ratio,
        "demographics": focus_demographics or "all",
        "seed": seed,
        "initial_content_count": len(posts),
        "data_source": "ToxiGen"
    }

    return posts, config


def load_toxicity_detection_training_data(
    train_size: int = 1000,
    test_size: int = 200,
    balance: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for training toxicity detection models.

    Args:
        train_size: Number of training samples
        test_size: Number of test samples
        balance: Whether to balance toxic/non-toxic samples

    Returns:
        Tuple of (train_df, test_df)
    """
    loader = ToxiGenLoader()

    # Load annotated data
    train_df = loader.load_annotated_data("train")
    test_df = loader.load_annotated_data("test")

    # Sample if needed
    if len(train_df) > train_size:
        train_df = train_df.sample(n=train_size, random_state=42)

    if len(test_df) > test_size:
        test_df = test_df.sample(n=test_size, random_state=42)

    # Balance if requested
    if balance and 'toxicity_human' in train_df.columns:
        # Simple balancing based on toxicity score threshold
        toxic_threshold = 3.0  # Adjust based on your needs
        toxic_train = train_df[train_df['toxicity_human'] >= toxic_threshold]
        neutral_train = train_df[train_df['toxicity_human'] < toxic_threshold]

        # Balance
        min_count = min(len(toxic_train), len(neutral_train))
        train_df = pd.concat([
            toxic_train.sample(n=min_count, random_state=42),
            neutral_train.sample(n=min_count, random_state=42)
        ]).sample(frac=1, random_state=42)  # Shuffle

    logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")

    return train_df, test_df


if __name__ == "__main__":
    # Example: Print dataset statistics
    loader = ToxiGenLoader()
    stats = loader.get_statistics()

    print("\n=== ToxiGen Dataset Statistics ===\n")

    print("Annotated Data:")
    for split, info in stats["annotated"].items():
        print(f"  {split}: {info.get('count', 0)} samples")
        if 'avg_human_toxicity' in info:
            print(f"    - Avg human toxicity: {info['avg_human_toxicity']:.2f}")
        if 'avg_ai_toxicity' in info:
            print(f"    - Avg AI toxicity: {info['avg_ai_toxicity']:.2f}")

    print("\nPrompt Data:")
    for prompt_type, info in stats["prompts"].items():
        print(f"  {prompt_type.capitalize()} prompts:")
        print(f"    - Total: {info['total_prompts']} prompts")
        print(f"    - Demographics: {info['total_demographics']} groups")
        if info['demographics']:
            print(f"    - Groups: {', '.join(sorted(info['demographics']))}")

    # Example: Create a content moderation scenario
    print("\n=== Creating Content Moderation Scenario ===\n")
    posts, config = create_content_moderation_simulation(
        agent_count=30,
        toxic_content_ratio=0.2,
        focus_demographics=["asian", "women", "lgbtq"],
        seed=123
    )

    print(f"Created {len(posts)} posts for simulation")
    print(f"Config: {config}")