#!/usr/bin/env python3
"""Build RAG indices for all datasets.

This script builds TF-IDF and optionally ChromaDB indices for all configured
datasets in the imputation system.

Usage:
    # Build TF-IDF only (no extra dependencies)
    python scripts/build_rag_indices.py --tfidf-only

    # Build both TF-IDF and ChromaDB
    python scripts/build_rag_indices.py --all

    # Build specific datasets
    python scripts/build_rag_indices.py --datasets davidson hatexplain
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dataset loading functions (avoid heavy imports)
def load_davidson(path: Path) -> Tuple[List[str], List[str]]:
    """Load Davidson dataset."""
    df = pd.read_csv(path / "davidson_full.csv")
    texts = df["tweet"].tolist()
    labels = df["class"].map({0: "hate_speech", 1: "offensive", 2: "neither"}).tolist()
    return texts, labels


def load_hatexplain(path: Path) -> Tuple[List[str], List[str]]:
    """Load HateXplain dataset."""
    df = pd.read_csv(path / "hatexplain_full.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return texts, labels


def load_gab(path: Path) -> Tuple[List[str], List[str]]:
    """Load Gab Hate Corpus."""
    df = pd.read_csv(path / "gab_hate_corpus.csv")
    texts = df["text"].tolist()

    # Determine label from hierarchical columns
    labels = []
    for _, row in df.iterrows():
        if row.get("cv", 0) == 1:
            labels.append("call_to_violence")
        elif row.get("hd", 0) == 1:
            labels.append("human_degradation")
        elif row.get("hate", 0) == 1:
            labels.append("hate")
        else:
            labels.append("not_hate")
    return texts, labels


def load_pheme(path: Path) -> Tuple[List[str], List[str]]:
    """Load PHEME dataset."""
    df = pd.read_csv(path / "pheme_full.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return texts, labels


def load_toxigen_annotated(path: Path) -> Tuple[List[str], List[str]]:
    """Load ToxiGen annotated dataset."""
    csv_path = path / "annotated" / "train.csv"
    if not csv_path.exists():
        return [], []
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = df.get("target_group", df.get("label", ["unknown"] * len(df))).tolist()
    return texts, labels


def load_toxigen_hate(path: Path) -> Tuple[List[str], List[str]]:
    """Load ToxiGen hate prompts."""
    hate_dir = path / "prompts" / "hate"
    if not hate_dir.exists():
        return [], []

    texts = []
    labels = []
    for csv_file in hate_dir.glob("*.csv"):
        # Extract demographic from filename (e.g., hate_asian_1k.csv -> asian)
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            demographic = parts[1]
        else:
            demographic = "unknown"

        df = pd.read_csv(csv_file)
        text_col = "text" if "text" in df.columns else df.columns[0]
        texts.extend(df[text_col].tolist())
        labels.extend([demographic] * len(df))

    return texts, labels


def load_implicit_hate(path: Path) -> Tuple[List[str], List[str]]:
    """Load ImplicitHate dataset."""
    # Try splits first, then raw
    train_path = path / "splits" / "train.csv"
    if train_path.exists():
        df = pd.read_csv(train_path)
    else:
        raw_path = path / "raw" / "implicit_hate.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
        else:
            return [], []

    texts = df["post"].tolist()
    labels = df.get("implicit_class", ["unknown"] * len(df)).tolist()
    return texts, labels


# Dataset registry
DATASET_LOADERS = {
    "davidson": ("data/hate_speech_datasets/davidson", load_davidson),
    "hatexplain": ("data/hate_speech_datasets/hatexplain", load_hatexplain),
    "gab_hate_corpus": ("data/hate_speech_datasets/gab_hate_corpus", load_gab),
    "pheme": ("data/hate_speech_datasets/pheme", load_pheme),
    "toxigen_annotated": ("data/toxigen_datasets", load_toxigen_annotated),
    "toxigen_hate_prompts": ("data/toxigen_datasets", load_toxigen_hate),
    "implicit_hate": ("data/implicit_hate_datasets", load_implicit_hate),
}


def build_tfidf_index(
    texts: List[str],
    labels: List[str],
    output_path: Path,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.9,
) -> Dict:
    """Build and save TF-IDF index."""
    logger.info(f"Building TF-IDF index with {len(texts)} documents...")

    # Filter empty texts
    valid_pairs = [(t, l) for t, l in zip(texts, labels) if t and len(str(t).strip()) > 0]
    texts = [t for t, _ in valid_pairs]
    labels = [l for _, l in valid_pairs]

    if len(texts) < min_df:
        logger.warning(f"Not enough documents ({len(texts)}) for min_df={min_df}")
        return {}

    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Save index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "texts": texts,
        "labels": labels,
        "vocab_size": len(vectorizer.vocabulary_),
        "num_documents": len(texts),
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"Saved TF-IDF index: {output_path}")
    logger.info(f"  Documents: {len(texts)}, Vocab: {len(vectorizer.vocabulary_)}")

    return index_data


def build_chromadb_index(
    texts: List[str],
    labels: List[str],
    collection_name: str,
    persist_dir: Path,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 100,
) -> bool:
    """Build ChromaDB vector index."""
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("ChromaDB or sentence-transformers not installed")
        logger.error("Install with: pip install chromadb sentence-transformers")
        return False

    logger.info(f"Building ChromaDB index with {len(texts)} documents...")

    # Filter empty texts
    valid_pairs = [(t, l) for t, l in zip(texts, labels) if t and len(str(t).strip()) > 0]
    texts = [str(t) for t, _ in valid_pairs]
    labels = [str(l) for _, l in valid_pairs]

    # Initialize ChromaDB
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection if exists
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    # Add documents in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_ids = [f"doc_{i + j}" for j in range(len(batch_texts))]

        # Generate embeddings
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        # Add to collection
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=[{"label": l} for l in batch_labels],
            ids=batch_ids
        )

        if (i + batch_size) % 1000 == 0:
            logger.info(f"  Processed {i + batch_size}/{len(texts)} documents")

    logger.info(f"Saved ChromaDB index: {persist_dir}/{collection_name}")
    logger.info(f"  Documents: {collection.count()}")

    return True


def test_tfidf_search(index_path: Path, query: str, top_k: int = 5):
    """Test TF-IDF search."""
    import numpy as np

    with open(index_path, "rb") as f:
        data = pickle.load(f)

    vectorizer = data["vectorizer"]
    matrix = data["matrix"]
    texts = data["texts"]
    labels = data["labels"]

    # Vectorize query
    query_vec = vectorizer.transform([query])

    # Compute similarities
    similarities = (matrix * query_vec.T).toarray().flatten()

    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                "text": texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx],
                "label": labels[idx],
                "score": float(similarities[idx])
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Build RAG indices")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LOADERS.keys()),
        help="Datasets to build indices for"
    )
    parser.add_argument(
        "--tfidf-only",
        action="store_true",
        help="Only build TF-IDF indices (no chromadb dependency)"
    )
    parser.add_argument(
        "--chromadb-only",
        action="store_true",
        help="Only build ChromaDB indices"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build both TF-IDF and ChromaDB indices"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after building"
    )
    parser.add_argument(
        "--output-dir",
        default="data/imputation",
        help="Output directory for indices"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Determine what to build
    build_tfidf = args.tfidf_only or args.all or (not args.chromadb_only)
    build_chromadb = args.chromadb_only or args.all

    logger.info("=" * 60)
    logger.info("RAG Index Builder")
    logger.info("=" * 60)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Build TF-IDF: {build_tfidf}")
    logger.info(f"Build ChromaDB: {build_chromadb}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    results = {}

    for dataset_id in args.datasets:
        if dataset_id not in DATASET_LOADERS:
            logger.warning(f"Unknown dataset: {dataset_id}")
            continue

        data_path, loader_func = DATASET_LOADERS[dataset_id]
        data_path = Path(data_path)

        if not data_path.exists():
            logger.warning(f"Data not found for {dataset_id}: {data_path}")
            results[dataset_id] = "data_not_found"
            continue

        logger.info(f"\n--- Processing {dataset_id} ---")

        try:
            # Load data
            texts, labels = loader_func(data_path)
            if not texts:
                logger.warning(f"No data loaded for {dataset_id}")
                results[dataset_id] = "no_data"
                continue

            logger.info(f"Loaded {len(texts)} documents")

            # Build TF-IDF
            if build_tfidf:
                tfidf_path = output_dir / "tfidf" / f"{dataset_id}.pkl"
                build_tfidf_index(texts, labels, tfidf_path)

            # Build ChromaDB
            if build_chromadb:
                chroma_path = output_dir / "chromadb" / dataset_id
                success = build_chromadb_index(
                    texts, labels,
                    collection_name=dataset_id,
                    persist_dir=chroma_path
                )
                if not success:
                    logger.warning(f"ChromaDB build failed for {dataset_id}")

            results[dataset_id] = "success"

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            results[dataset_id] = f"error: {e}"

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 60)

    for dataset_id, status in results.items():
        icon = "✓" if status == "success" else "✗"
        logger.info(f"  {icon} {dataset_id}: {status}")

    # Test queries
    if args.test and build_tfidf:
        logger.info("\n" + "=" * 60)
        logger.info("TEST QUERIES")
        logger.info("=" * 60)

        test_queries = [
            ("hate speech slurs", "davidson"),
            ("conspiracy misinformation", "pheme"),
            ("implicit racism dog whistle", "implicit_hate"),
        ]

        for query, dataset_id in test_queries:
            tfidf_path = output_dir / "tfidf" / f"{dataset_id}.pkl"
            if tfidf_path.exists():
                logger.info(f"\nQuery: '{query}' in {dataset_id}")
                results = test_tfidf_search(tfidf_path, query, top_k=3)
                for i, r in enumerate(results):
                    logger.info(f"  {i+1}. [{r['label']}] (score={r['score']:.3f})")
                    logger.info(f"     {r['text']}")


if __name__ == "__main__":
    main()
