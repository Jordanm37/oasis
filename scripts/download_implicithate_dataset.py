#!/usr/bin/env python
"""
Download ImplicitHate dataset from SALT-NLP on Hugging Face.

This dataset contains implicit hate speech examples that are more subtle
and nuanced than explicit hate speech, making them harder to detect.
"""

import os
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
HF_DATASET_PATH = "hf://datasets/SALT-NLP/ImplicitHate/implicit_hate.csv"
BASE_LOCAL_PATH = Path("./data/implicit_hate_datasets/")


def create_directory_structure():
    """Create directory structure for storing datasets."""
    directories = [
        BASE_LOCAL_PATH,
        BASE_LOCAL_PATH / "raw",
        BASE_LOCAL_PATH / "processed",
        BASE_LOCAL_PATH / "splits",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_implicit_hate_dataset() -> pd.DataFrame:
    """
    Download the ImplicitHate dataset from Hugging Face.

    Returns:
        DataFrame with the dataset
    """
    try:
        logger.info("Downloading ImplicitHate dataset from Hugging Face...")
        df = pd.read_csv(HF_DATASET_PATH)
        logger.info(f"✓ Successfully downloaded {len(df)} samples")
        logger.info(f"  Columns: {', '.join(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"✗ Failed to download dataset: {str(e)}")
        raise


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Analyze the ImplicitHate dataset structure and content.

    Args:
        df: The dataset DataFrame

    Returns:
        Dictionary with analysis results
    """
    analysis = {}

    # Basic statistics
    analysis['total_samples'] = len(df)
    analysis['columns'] = list(df.columns)

    # Check for label columns
    if 'implicit_hate' in df.columns:
        analysis['implicit_hate_distribution'] = df['implicit_hate'].value_counts().to_dict()

    if 'explicit_hate' in df.columns:
        analysis['explicit_hate_distribution'] = df['explicit_hate'].value_counts().to_dict()

    # Check for class labels
    if 'class' in df.columns:
        analysis['class_distribution'] = df['class'].value_counts().to_dict()

    # Check for target groups
    if 'target' in df.columns:
        analysis['target_distribution'] = df['target'].value_counts().to_dict()

    # Text statistics
    if 'text' in df.columns:
        df['text_length'] = df['text'].str.len()
        analysis['avg_text_length'] = df['text_length'].mean()
        analysis['min_text_length'] = df['text_length'].min()
        analysis['max_text_length'] = df['text_length'].max()
    elif 'post' in df.columns:
        df['text_length'] = df['post'].str.len()
        analysis['avg_text_length'] = df['text_length'].mean()
        analysis['min_text_length'] = df['text_length'].min()
        analysis['max_text_length'] = df['text_length'].max()

    # Check for null values
    analysis['null_counts'] = df.isnull().sum().to_dict()

    return analysis


def process_and_split_dataset(df: pd.DataFrame, test_ratio: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Process and split the dataset into train/test sets.

    Args:
        df: Original dataset
        test_ratio: Ratio of data to use for testing

    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(df_shuffled) * (1 - test_ratio))

    # Split the data
    splits = {
        'train': df_shuffled[:split_idx],
        'test': df_shuffled[split_idx:]
    }

    logger.info(f"Split dataset: Train={len(splits['train'])}, Test={len(splits['test'])}")

    return splits


def create_category_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create separate datasets for different hate speech categories.

    Args:
        df: Original dataset

    Returns:
        Dictionary with category-specific DataFrames
    """
    categories = {}

    # Check what categorization columns exist
    if 'class' in df.columns:
        # Split by class
        for class_label in df['class'].unique():
            if pd.notna(class_label):
                categories[f"class_{class_label}"] = df[df['class'] == class_label]
                logger.info(f"Created split for class '{class_label}': {len(categories[f'class_{class_label}'])} samples")

    # Check for implicit vs explicit
    if 'implicit_hate' in df.columns:
        categories['implicit_only'] = df[df['implicit_hate'] == True]
        logger.info(f"Created implicit hate split: {len(categories['implicit_only'])} samples")

    if 'explicit_hate' in df.columns:
        categories['explicit_only'] = df[df['explicit_hate'] == True]
        logger.info(f"Created explicit hate split: {len(categories['explicit_only'])} samples")

    # Check for target groups
    if 'target' in df.columns:
        for target in df['target'].unique():
            if pd.notna(target):
                safe_target = str(target).replace(' ', '_').replace('/', '_')
                categories[f"target_{safe_target}"] = df[df['target'] == target]
                logger.info(f"Created split for target '{target}': {len(categories[f'target_{safe_target}'])} samples")

    return categories


def save_datasets(df: pd.DataFrame, splits: Dict[str, pd.DataFrame],
                  categories: Dict[str, pd.DataFrame], analysis: Dict):
    """
    Save all dataset versions and analysis.

    Args:
        df: Original dataset
        splits: Train/test splits
        categories: Category-specific splits
        analysis: Dataset analysis results
    """
    # Save raw dataset
    raw_path = BASE_LOCAL_PATH / "raw" / "implicit_hate.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw dataset to: {raw_path}")

    raw_parquet_path = BASE_LOCAL_PATH / "raw" / "implicit_hate.parquet"
    df.to_parquet(raw_parquet_path, index=False)
    logger.info(f"Saved raw dataset (parquet) to: {raw_parquet_path}")

    # Save train/test splits
    for split_name, split_df in splits.items():
        split_path = BASE_LOCAL_PATH / "splits" / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        logger.info(f"Saved {split_name} split to: {split_path}")

    # Save category splits
    for cat_name, cat_df in categories.items():
        cat_path = BASE_LOCAL_PATH / "processed" / f"{cat_name}.csv"
        cat_df.to_csv(cat_path, index=False)
        logger.info(f"Saved category '{cat_name}' to: {cat_path}")

    # Save analysis
    analysis_path = BASE_LOCAL_PATH / "dataset_analysis.json"
    with open(analysis_path, 'w') as f:
        # Convert any non-serializable objects
        clean_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                clean_analysis[key] = value
            else:
                clean_analysis[key] = str(value)

        json.dump(clean_analysis, f, indent=2)
    logger.info(f"Saved analysis to: {analysis_path}")


def generate_summary_report(df: pd.DataFrame, analysis: Dict):
    """
    Generate a human-readable summary report.

    Args:
        df: Dataset DataFrame
        analysis: Analysis results
    """
    report_path = BASE_LOCAL_PATH / "dataset_summary.txt"

    with open(report_path, 'w') as f:
        f.write("ImplicitHate Dataset Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Samples: {analysis['total_samples']}\n")
        f.write(f"Columns: {', '.join(analysis['columns'])}\n\n")

        if 'class_distribution' in analysis:
            f.write("Class Distribution:\n")
            for class_name, count in analysis['class_distribution'].items():
                f.write(f"  - {class_name}: {count}\n")
            f.write("\n")

        if 'implicit_hate_distribution' in analysis:
            f.write("Implicit Hate Distribution:\n")
            for label, count in analysis['implicit_hate_distribution'].items():
                f.write(f"  - {label}: {count}\n")
            f.write("\n")

        if 'target_distribution' in analysis:
            f.write("Target Group Distribution:\n")
            for target, count in analysis['target_distribution'].items():
                f.write(f"  - {target}: {count}\n")
            f.write("\n")

        if 'avg_text_length' in analysis:
            f.write("Text Statistics:\n")
            f.write(f"  - Average length: {analysis['avg_text_length']:.1f} characters\n")
            f.write(f"  - Min length: {analysis['min_text_length']}\n")
            f.write(f"  - Max length: {analysis['max_text_length']}\n")
            f.write("\n")

        # Sample texts
        f.write("Sample Texts:\n")
        text_col = 'text' if 'text' in df.columns else 'post' if 'post' in df.columns else None
        if text_col:
            samples = df.sample(min(5, len(df)), random_state=42)
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                text_preview = row[text_col][:100] + "..." if len(row[text_col]) > 100 else row[text_col]
                f.write(f"  {i}. {text_preview}\n")
                if 'class' in row:
                    f.write(f"     Class: {row['class']}\n")

    logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main function to orchestrate dataset download and processing."""
    logger.info("Starting ImplicitHate Dataset Download")
    logger.info(f"Target directory: {BASE_LOCAL_PATH.absolute()}")

    try:
        # Create directory structure
        create_directory_structure()

        # Download dataset
        df = download_implicit_hate_dataset()

        # Analyze dataset
        logger.info("\nAnalyzing dataset structure...")
        analysis = analyze_dataset(df)

        # Process and split dataset
        logger.info("\nCreating train/test splits...")
        splits = process_and_split_dataset(df)

        # Create category-specific splits
        logger.info("\nCreating category-specific splits...")
        categories = create_category_splits(df)

        # Save everything
        logger.info("\nSaving datasets...")
        save_datasets(df, splits, categories, analysis)

        # Generate summary report
        logger.info("\nGenerating summary report...")
        generate_summary_report(df, analysis)

        # Final summary
        logger.info("\n" + "="*50)
        logger.info("Download Complete!")
        logger.info("="*50)
        logger.info(f"Successfully downloaded and processed {len(df)} samples")
        logger.info(f"Data saved to: {BASE_LOCAL_PATH.absolute()}")
        logger.info("\nDataset structure:")
        for col in df.columns[:10]:  # Show first 10 columns
            logger.info(f"  - {col}: {df[col].dtype}")

        return 0

    except Exception as e:
        logger.error(f"Failed to complete dataset download: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())