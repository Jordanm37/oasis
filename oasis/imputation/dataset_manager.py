"""Dataset Manager for Multi-Dataset RAG Imputation System.

This module provides the DatasetManager class that handles loading, indexing,
and managing multiple datasets for the imputation pipeline.
"""

import os
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import yaml
import pandas as pd
import json
import logging
from pathlib import Path

# Third-party imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    description: str
    source_path: str
    source_format: str
    text_column: str
    label_columns: Optional[List[str]] = None
    label_field: Optional[str] = None
    text_field: Optional[str] = None
    label_mapping_file: str = ""

    # Index configurations
    chromadb_enabled: bool = True
    chromadb_path: str = ""
    chromadb_collection: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50

    tfidf_enabled: bool = True
    tfidf_path: str = ""
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 3)
    min_df: int = 2
    max_df: float = 0.95

    # Retrieval settings
    top_k: int = 10
    min_similarity: float = 0.3
    enable_reranking: bool = False

    # Post-processing
    enable_obfuscation: bool = True
    enable_label_injection: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create DatasetConfig from dictionary."""
        config = cls(
            name=data["name"],
            description=data["description"],
            source_path=data["source"]["path"],
            source_format=data["source"]["format"],
            text_column=data["source"].get("text_column", ""),
            label_columns=data["source"].get("label_columns"),
            label_field=data["source"].get("label_field"),
            text_field=data["source"].get("text_field"),
            label_mapping_file=data.get("label_mapping_file", ""),
        )

        # Parse index configurations
        if "indices" in data:
            if "chromadb" in data["indices"]:
                chroma = data["indices"]["chromadb"]
                config.chromadb_enabled = chroma.get("enabled", True)
                config.chromadb_path = chroma.get("path", "")
                config.chromadb_collection = chroma.get("collection_name", "")
                config.embedding_model = chroma.get("embedding_model", "all-MiniLM-L6-v2")
                config.chunk_size = chroma.get("chunk_size", 512)
                config.chunk_overlap = chroma.get("chunk_overlap", 50)

            if "tfidf" in data["indices"]:
                tfidf = data["indices"]["tfidf"]
                config.tfidf_enabled = tfidf.get("enabled", True)
                config.tfidf_path = tfidf.get("path", "")
                config.max_features = tfidf.get("max_features", 10000)
                ngram = tfidf.get("ngram_range", [1, 3])
                config.ngram_range = (ngram[0], ngram[1])
                config.min_df = tfidf.get("min_df", 2)
                config.max_df = tfidf.get("max_df", 0.95)

        # Parse retrieval settings
        if "retrieval" in data:
            ret = data["retrieval"]
            config.top_k = ret.get("top_k", 10)
            config.min_similarity = ret.get("min_similarity", 0.3)
            config.enable_reranking = ret.get("enable_reranking", False)

        # Parse post-processing
        if "post_processing" in data:
            pp = data["post_processing"]
            config.enable_obfuscation = pp.get("enable_obfuscation", True)
            config.enable_label_injection = pp.get("enable_label_injection", True)

        return config


class DatasetManager:
    """Manages multiple datasets for the imputation pipeline."""

    def __init__(self, registry_path: str = "configs/imputation/dataset_registry.yaml"):
        """Initialize the DatasetManager.

        Args:
            registry_path: Path to the dataset registry YAML file
        """
        self.registry_path = registry_path
        self.datasets: Dict[str, DatasetConfig] = {}
        self.global_settings: Dict[str, Any] = {}

        # Index storage
        self.chromadb_clients: Dict[str, chromadb.Client] = {}
        self.chromadb_collections: Dict[str, Any] = {}
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, Any] = {}
        self.tfidf_texts: Dict[str, List[str]] = {}

        # Embedding models cache
        self.embedding_models: Dict[str, SentenceTransformer] = {}

        # Load registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load dataset registry from YAML file."""
        if not os.path.exists(self.registry_path):
            logger.warning(f"Registry file not found: {self.registry_path}")
            return

        with open(self.registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        # Load dataset configurations
        for dataset_id, dataset_data in registry.get("datasets", {}).items():
            config = DatasetConfig.from_dict(dataset_data)
            self.datasets[dataset_id] = config
            logger.info(f"Loaded dataset config: {dataset_id}")

        # Load global settings
        self.global_settings = registry.get("global_settings", {})

    def load_dataset_data(self, dataset_id: str) -> Tuple[List[str], Optional[List[Any]]]:
        """Load raw data from a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Tuple of (texts, labels)
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found in registry")

        config = self.datasets[dataset_id]
        texts = []
        labels = []

        # Load based on format
        if config.source_format == "csv":
            df = pd.read_csv(config.source_path)
            texts = df[config.text_column].tolist()
            if config.label_columns:
                labels = df[config.label_columns].values.tolist()

        elif config.source_format == "jsonl":
            with open(config.source_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    text_key = config.text_field or config.text_column
                    if text_key in data:
                        texts.append(data[text_key])
                        if config.label_field and config.label_field in data:
                            labels.append(data[config.label_field])

        else:
            raise ValueError(f"Unsupported format: {config.source_format}")

        logger.info(f"Loaded {len(texts)} texts from {dataset_id}")
        return texts, labels if labels else None

    def initialize_chromadb(self, dataset_id: str, rebuild: bool = False) -> None:
        """Initialize ChromaDB collection for a dataset.

        Args:
            dataset_id: Dataset identifier
            rebuild: Whether to rebuild the index from scratch
        """
        config = self.datasets[dataset_id]
        if not config.chromadb_enabled:
            logger.info(f"ChromaDB disabled for {dataset_id}")
            return

        # Create ChromaDB client
        client = chromadb.PersistentClient(
            path=config.chromadb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.chromadb_clients[dataset_id] = client

        # Get or create collection
        try:
            if rebuild:
                client.delete_collection(config.chromadb_collection)
            collection = client.create_collection(
                name=config.chromadb_collection,
                metadata={"dataset": dataset_id}
            )
        except:
            collection = client.get_collection(config.chromadb_collection)

        self.chromadb_collections[dataset_id] = collection

        # Load embedding model if needed
        if config.embedding_model not in self.embedding_models:
            self.embedding_models[config.embedding_model] = SentenceTransformer(
                config.embedding_model
            )

        logger.info(f"Initialized ChromaDB for {dataset_id}")

    def initialize_tfidf(self, dataset_id: str, rebuild: bool = False) -> None:
        """Initialize TF-IDF index for a dataset.

        Args:
            dataset_id: Dataset identifier
            rebuild: Whether to rebuild the index from scratch
        """
        config = self.datasets[dataset_id]
        if not config.tfidf_enabled:
            logger.info(f"TF-IDF disabled for {dataset_id}")
            return

        # Check if index already exists
        if os.path.exists(config.tfidf_path) and not rebuild:
            with open(config.tfidf_path, 'rb') as f:
                data = pickle.load(f)
                self.tfidf_vectorizers[dataset_id] = data['vectorizer']
                self.tfidf_matrices[dataset_id] = data['matrix']
                self.tfidf_texts[dataset_id] = data['texts']
            logger.info(f"Loaded existing TF-IDF index for {dataset_id}")
        else:
            logger.info(f"TF-IDF index will be built when data is added")

    def add_to_chromadb(self, dataset_id: str, texts: List[str],
                        metadata: Optional[List[Dict]] = None) -> None:
        """Add texts to ChromaDB collection.

        Args:
            dataset_id: Dataset identifier
            texts: List of texts to add
            metadata: Optional metadata for each text
        """
        if dataset_id not in self.chromadb_collections:
            raise ValueError(f"ChromaDB not initialized for {dataset_id}")

        config = self.datasets[dataset_id]
        collection = self.chromadb_collections[dataset_id]
        model = self.embedding_models[config.embedding_model]

        # Generate embeddings
        embeddings = model.encode(texts).tolist()

        # Prepare documents and IDs
        ids = [f"{dataset_id}_{i}" for i in range(len(texts))]

        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata or [{} for _ in texts],
            ids=ids
        )

        logger.info(f"Added {len(texts)} documents to ChromaDB for {dataset_id}")

    def build_tfidf_index(self, dataset_id: str, texts: List[str]) -> None:
        """Build TF-IDF index for a dataset.

        Args:
            dataset_id: Dataset identifier
            texts: List of texts to index
        """
        config = self.datasets[dataset_id]
        if not config.tfidf_enabled:
            return

        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            stop_words='english'
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        # Store in memory
        self.tfidf_vectorizers[dataset_id] = vectorizer
        self.tfidf_matrices[dataset_id] = tfidf_matrix
        self.tfidf_texts[dataset_id] = texts

        # Save to disk
        os.makedirs(os.path.dirname(config.tfidf_path), exist_ok=True)
        with open(config.tfidf_path, 'wb') as f:
            pickle.dump({
                'vectorizer': vectorizer,
                'matrix': tfidf_matrix,
                'texts': texts
            }, f)

        logger.info(f"Built TF-IDF index for {dataset_id} with {len(texts)} documents")

    def search_chromadb(self, dataset_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """Search ChromaDB collection.

        Args:
            dataset_id: Dataset identifier
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with text and metadata
        """
        if dataset_id not in self.chromadb_collections:
            return []

        config = self.datasets[dataset_id]
        collection = self.chromadb_collections[dataset_id]
        model = self.embedding_models[config.embedding_model]

        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, config.top_k)
        )

        # Format results
        output = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                output.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })

        return output

    def search_tfidf(self, dataset_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """Search using TF-IDF.

        Args:
            dataset_id: Dataset identifier
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with text and scores
        """
        if dataset_id not in self.tfidf_vectorizers:
            return []

        vectorizer = self.tfidf_vectorizers[dataset_id]
        matrix = self.tfidf_matrices[dataset_id]
        texts = self.tfidf_texts[dataset_id]

        # Vectorize query
        query_vec = vectorizer.transform([query])

        # Compute similarities
        similarities = (matrix * query_vec.T).toarray().flatten()

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Format results
        output = []
        for idx in top_indices:
            if similarities[idx] > 0:
                output.append({
                    'text': texts[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })

        return output

    def get_dataset_config(self, dataset_id: str) -> DatasetConfig:
        """Get configuration for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetConfig object
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        return self.datasets[dataset_id]

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs.

        Returns:
            List of dataset identifiers
        """
        return list(self.datasets.keys())

    def get_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get statistics for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'dataset_id': dataset_id,
            'chromadb_enabled': False,
            'tfidf_enabled': False,
            'chromadb_documents': 0,
            'tfidf_documents': 0
        }

        if dataset_id in self.datasets:
            config = self.datasets[dataset_id]
            stats['chromadb_enabled'] = config.chromadb_enabled
            stats['tfidf_enabled'] = config.tfidf_enabled

            if dataset_id in self.chromadb_collections:
                collection = self.chromadb_collections[dataset_id]
                stats['chromadb_documents'] = collection.count()

            if dataset_id in self.tfidf_texts:
                stats['tfidf_documents'] = len(self.tfidf_texts[dataset_id])

        return stats