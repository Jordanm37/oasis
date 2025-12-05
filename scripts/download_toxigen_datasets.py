#!/usr/bin/env python
"""
Download ToxiGen datasets from Hugging Face.

This script downloads all ToxiGen dataset components including:
- Annotated data (test and train splits)
- Prompt datasets for various demographic groups (hate and neutral)
"""

import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define base path for datasets
BASE_HF_PATH = "hf://datasets/toxigen/toxigen-data/"
BASE_LOCAL_PATH = Path("./data/toxigen_datasets/")

# Define dataset splits and their paths
ANNOTATED_SPLITS = {
    'test': 'annotated/test-00000-of-00001.parquet',
    'train': 'annotated/train-00000-of-00001.parquet'
}

ANNOTATION_SPLITS = {
    'train': 'annotations/train-00000-of-00001.parquet'
}

PROMPT_SPLITS = {
    'hate_trans_1k': 'prompts/hate_trans_1k-00000-of-00001.parquet',
    'neutral_black_1k': 'prompts/neutral_black_1k-00000-of-00001.parquet',
    'hate_native_american_1k': 'prompts/hate_native_american_1k-00000-of-00001.parquet',
    'neutral_immigrant_1k': 'prompts/neutral_immigrant_1k-00000-of-00001.parquet',
    'hate_middle_east_1k': 'prompts/hate_middle_east_1k-00000-of-00001.parquet',
    'neutral_lgbtq_1k': 'prompts/neutral_lgbtq_1k-00000-of-00001.parquet',
    'neutral_women_1k': 'prompts/neutral_women_1k-00000-of-00001.parquet',
    'neutral_chinese_1k': 'prompts/neutral_chinese_1k-00000-of-00001.parquet',
    'hate_latino_1k': 'prompts/hate_latino_1k-00000-of-00001.parquet',
    'hate_bisexual_1k': 'prompts/hate_bisexual_1k-00000-of-00001.parquet',
    'hate_mexican_1k': 'prompts/hate_mexican_1k-00000-of-00001.parquet',
    'hate_asian_1k': 'prompts/hate_asian_1k-00000-of-00001.parquet',
    'neutral_mental_disability_1k': 'prompts/neutral_mental_disability_1k-00000-of-00001.parquet',
    'neutral_mexican_1k': 'prompts/neutral_mexican_1k-00000-of-00001.parquet',
    'hate_mental_disability_1k': 'prompts/hate_mental_disability_1k-00000-of-00001.parquet',
    'neutral_bisexual_1k': 'prompts/neutral_bisexual_1k-00000-of-00001.parquet',
    'neutral_latino_1k': 'prompts/neutral_latino_1k-00000-of-00001.parquet',
    'hate_chinese_1k': 'prompts/hate_chinese_1k-00000-of-00001.parquet',
    'neutral_jewish_1k': 'prompts/neutral_jewish_1k-00000-of-00001.parquet',
    'hate_muslim_1k': 'prompts/hate_muslim_1k-00000-of-00001.parquet',
    'neutral_asian_1k': 'prompts/neutral_asian_1k-00000-of-00001.parquet',
    'hate_physical_disability_1k': 'prompts/hate_physical_disability_1k-00000-of-00001.parquet',
    'hate_jewish_1k': 'prompts/hate_jewish_1k-00000-of-00001.parquet',
    'neutral_muslim_1k': 'prompts/neutral_muslim_1k-00000-of-00001.parquet',
    'hate_immigrant_1k': 'prompts/hate_immigrant_1k-00000-of-00001.parquet',
    'hate_black_1k': 'prompts/hate_black_1k-00000-of-00001.parquet',
    'hate_lgbtq_1k': 'prompts/hate_lgbtq_1k-00000-of-00001.parquet',
    'hate_women_1k': 'prompts/hate_women_1k-00000-of-00001.parquet',
    'neutral_middle_east_1k': 'prompts/neutral_middle_east_1k-00000-of-00001.parquet',
    'neutral_native_american_1k': 'prompts/neutral_native_american_1k-00000-of-00001.parquet',
    'neutral_physical_disability_1k': 'prompts/neutral_physical_disability_1k-00000-of-00001.parquet',
    # Note: neutral_trans_1k might be missing or under a different name in the dataset
}


def create_directory_structure():
    """Create directory structure for storing datasets."""
    directories = [
        BASE_LOCAL_PATH,
        BASE_LOCAL_PATH / "annotated",
        BASE_LOCAL_PATH / "annotations",
        BASE_LOCAL_PATH / "prompts",
        BASE_LOCAL_PATH / "prompts" / "hate",
        BASE_LOCAL_PATH / "prompts" / "neutral",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_dataset(hf_path: str, local_path: Path, dataset_name: str) -> bool:
    """
    Download a single dataset from Hugging Face.

    Args:
        hf_path: Hugging Face dataset path
        local_path: Local path to save the dataset
        dataset_name: Name of the dataset for logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {dataset_name}...")
        df = pd.read_parquet(hf_path)

        # Save in multiple formats for flexibility
        # Save as parquet (original format)
        df.to_parquet(local_path.with_suffix('.parquet'))

        # Also save as CSV for easier inspection
        df.to_csv(local_path.with_suffix('.csv'), index=False)

        logger.info(f"✓ Downloaded {dataset_name}: {len(df)} rows")
        logger.info(f"  Columns: {', '.join(df.columns)}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {str(e)}")
        return False


def download_annotated_datasets():
    """Download annotated datasets (test and train splits)."""
    logger.info("\n" + "="*50)
    logger.info("Downloading Annotated Datasets")
    logger.info("="*50)

    success_count = 0
    for split_name, split_path in ANNOTATED_SPLITS.items():
        hf_path = BASE_HF_PATH + split_path
        local_path = BASE_LOCAL_PATH / "annotated" / split_name

        if download_dataset(hf_path, local_path, f"annotated/{split_name}"):
            success_count += 1

    return success_count, len(ANNOTATED_SPLITS)


def download_annotation_datasets():
    """Download annotation datasets."""
    logger.info("\n" + "="*50)
    logger.info("Downloading Annotation Datasets")
    logger.info("="*50)

    success_count = 0
    for split_name, split_path in ANNOTATION_SPLITS.items():
        hf_path = BASE_HF_PATH + split_path
        local_path = BASE_LOCAL_PATH / "annotations" / split_name

        if download_dataset(hf_path, local_path, f"annotations/{split_name}"):
            success_count += 1

    return success_count, len(ANNOTATION_SPLITS)


def download_prompt_datasets():
    """Download prompt datasets for different demographic groups."""
    logger.info("\n" + "="*50)
    logger.info("Downloading Prompt Datasets")
    logger.info("="*50)

    success_count = 0

    # Use tqdm for progress bar
    for prompt_name, prompt_path in tqdm(PROMPT_SPLITS.items(), desc="Downloading prompts"):
        hf_path = BASE_HF_PATH + prompt_path

        # Organize by hate/neutral
        if prompt_name.startswith('hate_'):
            local_dir = BASE_LOCAL_PATH / "prompts" / "hate"
        else:
            local_dir = BASE_LOCAL_PATH / "prompts" / "neutral"

        local_path = local_dir / prompt_name

        if download_dataset(hf_path, local_path, f"prompts/{prompt_name}"):
            success_count += 1

    return success_count, len(PROMPT_SPLITS)


def generate_summary_report():
    """Generate a summary report of downloaded datasets."""
    logger.info("\n" + "="*50)
    logger.info("Generating Summary Report")
    logger.info("="*50)

    report_path = BASE_LOCAL_PATH / "dataset_summary.txt"

    with open(report_path, 'w') as f:
        f.write("ToxiGen Dataset Summary\n")
        f.write("=" * 50 + "\n\n")

        # Annotated datasets
        f.write("Annotated Datasets:\n")
        for split in ['test', 'train']:
            csv_path = BASE_LOCAL_PATH / "annotated" / f"{split}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                f.write(f"  - {split}: {len(df)} rows\n")

        # Prompt datasets
        f.write("\nPrompt Datasets:\n")

        hate_prompts = []
        neutral_prompts = []

        for prompt_name in PROMPT_SPLITS.keys():
            if prompt_name.startswith('hate_'):
                hate_prompts.append(prompt_name)
            else:
                neutral_prompts.append(prompt_name)

        f.write(f"  Hate prompts: {len(hate_prompts)}\n")
        f.write(f"  Neutral prompts: {len(neutral_prompts)}\n")

        # List demographics covered
        demographics = set()
        for prompt_name in PROMPT_SPLITS.keys():
            # Extract demographic from prompt name
            parts = prompt_name.replace('hate_', '').replace('neutral_', '').replace('_1k', '')
            demographics.add(parts)

        f.write(f"\nDemographics covered ({len(demographics)}):\n")
        for demo in sorted(demographics):
            f.write(f"  - {demo}\n")

    logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main function to orchestrate dataset downloads."""
    logger.info("Starting ToxiGen Dataset Download")
    logger.info(f"Target directory: {BASE_LOCAL_PATH.absolute()}")

    # Create directory structure
    create_directory_structure()

    # Track overall progress
    total_success = 0
    total_attempted = 0

    # Download annotated datasets
    success, attempted = download_annotated_datasets()
    total_success += success
    total_attempted += attempted

    # Download annotation datasets
    success, attempted = download_annotation_datasets()
    total_success += success
    total_attempted += attempted

    # Download prompt datasets
    success, attempted = download_prompt_datasets()
    total_success += success
    total_attempted += attempted

    # Generate summary report
    generate_summary_report()

    # Final summary
    logger.info("\n" + "="*50)
    logger.info("Download Complete!")
    logger.info("="*50)
    logger.info(f"Successfully downloaded: {total_success}/{total_attempted} datasets")
    logger.info(f"Data saved to: {BASE_LOCAL_PATH.absolute()}")

    if total_success < total_attempted:
        logger.warning(f"Failed to download {total_attempted - total_success} datasets")
        logger.warning("Please check the error messages above and retry if needed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())