#!/usr/bin/env python3
"""
Download and prepare multiple hate speech and misinformation datasets.

Datasets included:
1. Davidson Hate Speech & Offensive Language (tdavidson/hate_speech_offensive)
2. HateXplain (Hate-speech-CNERG/hatexplain)
3. Gab Hate Corpus (juliadollis/The_Gab_Hate_Corpus_ghc_train_original)
4. Real Toxicity Prompts (allenai/real-toxicity-prompts)
5. COCO COVID Conspiracy (requires manual download from OSF)
6. EXIST Sexism (requires registration)

Note: ToxiGen and ImplicitHate already downloaded separately.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_davidson_dataset(output_dir: Path) -> None:
    """Download Davidson Hate Speech & Offensive Language dataset.

    Labels:
    - 0: hate speech
    - 1: offensive language
    - 2: neither

    Size: ~25k tweets
    """
    logger.info("Downloading Davidson Hate Speech dataset...")

    output_dir = output_dir / "davidson"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try Hugging Face first
        df = pd.read_parquet(
            "hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet"
        )

        logger.info(f"Downloaded {len(df)} samples")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Class distribution:\n{df['class'].value_counts()}")

        # Save
        df.to_csv(output_dir / "davidson_full.csv", index=False)
        df.to_parquet(output_dir / "davidson_full.parquet")

        # Create splits
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        logger.info(f"Saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download Davidson dataset: {e}")
        logger.info("Trying direct GitHub download...")

        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        df = pd.read_csv(url)
        df.to_csv(output_dir / "davidson_full.csv", index=False)
        logger.info(f"Downloaded {len(df)} samples from GitHub")


def download_hatexplain_dataset(output_dir: Path) -> None:
    """Download HateXplain dataset with explainable annotations.

    Labels:
    - Classification: hatespeech, offensive, normal
    - Target communities: African, Islam, Jewish, LGBTQ, Women, etc.
    - Rationales: token-level annotations

    Size: ~20k posts
    """
    logger.info("Downloading HateXplain dataset...")

    output_dir = output_dir / "hatexplain"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        # Download directly from GitHub (the original dataset JSON)
        url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
        logger.info(f"Downloading from GitHub: {url}")

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        logger.info(f"Downloaded {len(data)} samples")

        # Label mapping: 0=hatespeech, 1=normal, 2=offensive
        label_map = {0: "hatespeech", 1: "normal", 2: "offensive"}

        df_data = []
        for post_id, item in data.items():
            # Extract text from tokens
            text = " ".join(item.get("post_tokens", []))

            # Get annotator labels and find majority
            annotators = item.get("annotators", [])
            label_counts = {}
            targets = []

            for ann in annotators:
                label_id = ann.get("label")
                if label_id is not None:
                    label_name = label_map.get(label_id, str(label_id))
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1

                # Collect target groups
                ann_targets = ann.get("target", [])
                targets.extend(ann_targets)

            # Determine majority label
            if label_counts:
                majority_label = max(label_counts, key=label_counts.get)
            else:
                majority_label = "unknown"

            unique_targets = list(set(targets))

            df_data.append({
                "post_id": post_id,
                "text": text,
                "label": majority_label,
                "label_counts": json.dumps(label_counts),
                "target_groups": json.dumps(unique_targets),
                "num_annotators": len(annotators),
                "split": item.get("split", "unknown")
            })

        df = pd.DataFrame(df_data)

        # Save by split
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            split_name = split if split != "val" else "validation"
            split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
            logger.info(f"Saved {split_name}: {len(split_df)} samples")

        # Save full dataset
        df.to_csv(output_dir / "hatexplain_full.csv", index=False)
        logger.info(f"Total: {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    except Exception as e:
        logger.error(f"Failed to download HateXplain: {e}")
        raise


def download_gab_hate_corpus(output_dir: Path) -> None:
    """Download Gab Hate Corpus from Hugging Face mirror.

    Labels:
    - CV: Call for Violence
    - HD: Human Degradation
    - Hate: General hate
    - Target groups: African, Islam, Jewish, LGBTQ, Women, etc.

    Size: ~27k posts
    """
    logger.info("Downloading Gab Hate Corpus...")

    output_dir = output_dir / "gab_hate_corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        dataset = load_dataset("juliadollis/The_Gab_Hate_Corpus_ghc_train_original")

        df = dataset["train"].to_pandas()
        logger.info(f"Downloaded {len(df)} samples")
        logger.info(f"Columns: {df.columns.tolist()}")

        df.to_csv(output_dir / "gab_hate_corpus.csv", index=False)
        df.to_parquet(output_dir / "gab_hate_corpus.parquet")

        # Create train/test splits
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        logger.info(f"Saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download Gab Hate Corpus: {e}")
        logger.info("Note: May need to download manually from OSF: https://osf.io/edua3/files/")
        raise


def download_real_toxicity_prompts(output_dir: Path, max_samples: int = 50000) -> None:
    """Download Real Toxicity Prompts dataset.

    Labels (Perspective API scores 0-1):
    - toxicity
    - severe_toxicity
    - identity_attack
    - insult
    - profanity
    - sexually_explicit
    - threat

    Size: ~100k prompts (sampling for efficiency)
    """
    logger.info("Downloading Real Toxicity Prompts...")

    output_dir = output_dir / "real_toxicity_prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        # Load dataset (it's large, so we might want to stream)
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

        # Sample if needed
        if len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=42).select(range(max_samples))

        # Extract relevant fields
        records = []
        for item in dataset:
            prompt = item.get("prompt", {})
            continuation = item.get("continuation", {})

            record = {
                "prompt_text": prompt.get("text", ""),
                "continuation_text": continuation.get("text", ""),
            }

            # Extract toxicity scores from prompt
            for key in ["toxicity", "severe_toxicity", "identity_attack",
                       "insult", "profanity", "sexually_explicit", "threat"]:
                record[f"prompt_{key}"] = prompt.get(key, None)
                record[f"continuation_{key}"] = continuation.get(key, None)

            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_dir / "real_toxicity_prompts.csv", index=False)
        df.to_parquet(output_dir / "real_toxicity_prompts.parquet")

        logger.info(f"Saved {len(df)} samples to {output_dir}")

        # Show toxicity distribution
        logger.info(f"Prompt toxicity stats:\n{df['prompt_toxicity'].describe()}")

    except Exception as e:
        logger.error(f"Failed to download Real Toxicity Prompts: {e}")
        raise


def prepare_pheme_dataset(pheme_dir: Path, output_dir: Path) -> None:
    """Process existing PHEME dataset into a structured format.

    The PHEME dataset is already downloaded at:
    oasis/datasets/pheme-rnr-dataset/

    Labels:
    - rumour / non-rumour (from directory structure)
    - Events: charliehebdo, ferguson, germanwings-crash, ottawashooting, sydneysiege
    """
    logger.info("Processing PHEME dataset...")

    output_dir = output_dir / "pheme"
    output_dir.mkdir(parents=True, exist_ok=True)

    events = ["charliehebdo", "ferguson", "germanwings-crash",
              "ottawashooting", "sydneysiege"]

    all_records = []

    for event in events:
        event_dir = pheme_dir / event
        if not event_dir.exists():
            logger.warning(f"Event directory not found: {event_dir}")
            continue

        for label in ["rumours", "non-rumours"]:
            label_dir = event_dir / label
            is_rumour = label == "rumours"

            if not label_dir.exists():
                continue

            # Each subdirectory is a tweet thread
            for tweet_dir in label_dir.iterdir():
                if not tweet_dir.is_dir():
                    continue

                source_tweet_dir = tweet_dir / "source-tweet"
                if not source_tweet_dir.exists():
                    continue

                # Find the source tweet JSON
                json_files = list(source_tweet_dir.glob("*.json"))
                if not json_files:
                    continue

                try:
                    with open(json_files[0], "r") as f:
                        tweet_data = json.load(f)

                    # Count reactions
                    reactions_dir = tweet_dir / "reactions"
                    num_reactions = len(list(reactions_dir.glob("*.json"))) if reactions_dir.exists() else 0

                    record = {
                        "tweet_id": tweet_data.get("id_str", tweet_dir.name),
                        "text": tweet_data.get("text", ""),
                        "is_rumour": is_rumour,
                        "label": "rumour" if is_rumour else "non-rumour",
                        "event": event,
                        "user_screen_name": tweet_data.get("user", {}).get("screen_name", ""),
                        "user_verified": tweet_data.get("user", {}).get("verified", False),
                        "user_followers": tweet_data.get("user", {}).get("followers_count", 0),
                        "retweet_count": tweet_data.get("retweet_count", 0),
                        "favorite_count": tweet_data.get("favorite_count", 0),
                        "num_reactions": num_reactions,
                        "created_at": tweet_data.get("created_at", ""),
                        "lang": tweet_data.get("lang", ""),
                    }
                    all_records.append(record)

                except Exception as e:
                    logger.warning(f"Failed to process {json_files[0]}: {e}")

    df = pd.DataFrame(all_records)

    # Save full dataset
    df.to_csv(output_dir / "pheme_full.csv", index=False)
    df.to_parquet(output_dir / "pheme_full.parquet")

    logger.info(f"Processed {len(df)} tweets")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info(f"Event distribution:\n{df['event'].value_counts()}")

    # Create train/test splits (stratified by event and label)
    train_dfs = []
    test_dfs = []

    for event in events:
        for label in ["rumour", "non-rumour"]:
            subset = df[(df["event"] == event) & (df["label"] == label)]
            if len(subset) > 0:
                train = subset.sample(frac=0.8, random_state=42)
                test = subset.drop(train.index)
                train_dfs.append(train)
                test_dfs.append(test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    return df


def main():
    """Download all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download hate speech datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hate_speech_datasets",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["davidson", "hatexplain", "gab", "pheme"],
        choices=["davidson", "hatexplain", "gab", "pheme", "all"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--pheme-dir",
        type=str,
        default="data/pheme-rnr-dataset",
        help="Path to existing PHEME dataset"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["davidson", "hatexplain", "gab", "real_toxicity", "pheme"]

    results = {}

    for dataset in datasets:
        try:
            if dataset == "davidson":
                download_davidson_dataset(output_dir)
                results[dataset] = "success"
            elif dataset == "hatexplain":
                download_hatexplain_dataset(output_dir)
                results[dataset] = "success"
            elif dataset == "gab":
                download_gab_hate_corpus(output_dir)
                results[dataset] = "success"
            elif dataset == "real_toxicity":
                download_real_toxicity_prompts(output_dir)
                results[dataset] = "success"
            elif dataset == "pheme":
                pheme_dir = Path(args.pheme_dir)
                if pheme_dir.exists():
                    prepare_pheme_dataset(pheme_dir, output_dir)
                    results[dataset] = "success"
                else:
                    logger.warning(f"PHEME directory not found: {pheme_dir}")
                    results[dataset] = "skipped - directory not found"
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")
            results[dataset] = f"failed: {str(e)}"

    logger.info("\n=== Download Summary ===")
    for dataset, status in results.items():
        logger.info(f"{dataset}: {status}")


if __name__ == "__main__":
    main()