#!/usr/bin/env python
"""CLI tool for managing imputation datasets and indices.

This script provides commands to:
- List available datasets
- Build ChromaDB and TF-IDF indices
- Validate label mappings
- Test retrieval performance
- Export/import indices
"""

import os
import sys
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from tabulate import tabulate
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.imputation.dataset_manager import DatasetManager
from oasis.imputation.label_mapper import LabelMapper
from oasis.imputation.multi_method_retriever import MultiMethodRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCLI:
    """Command-line interface for dataset management."""

    def __init__(self, registry_path: str = "configs/imputation/dataset_registry.yaml"):
        """Initialize the CLI.

        Args:
            registry_path: Path to dataset registry
        """
        self.registry_path = registry_path
        self.dataset_manager = DatasetManager(registry_path)
        self.label_mapper = LabelMapper()

    def list_datasets(self) -> None:
        """List all registered datasets."""
        datasets = []
        for dataset_id in self.dataset_manager.list_datasets():
            config = self.dataset_manager.get_dataset_config(dataset_id)
            stats = self.dataset_manager.get_statistics(dataset_id)

            datasets.append([
                dataset_id,
                config.name,
                "✓" if config.chromadb_enabled else "✗",
                "✓" if config.tfidf_enabled else "✗",
                stats.get('chromadb_documents', 0),
                stats.get('tfidf_documents', 0)
            ])

        headers = ["Dataset ID", "Name", "ChromaDB", "TF-IDF", "ChromaDB Docs", "TF-IDF Docs"]
        print("\n=== Registered Datasets ===")
        print(tabulate(datasets, headers=headers, tablefmt="grid"))

    async def build_indices(self, dataset_id: str, rebuild: bool = False,
                           methods: Optional[List[str]] = None) -> None:
        """Build indices for a dataset.

        Args:
            dataset_id: Dataset identifier
            rebuild: Whether to rebuild existing indices
            methods: List of methods to build ("chromadb", "tfidf")
        """
        if methods is None:
            methods = ["chromadb", "tfidf"]

        print(f"\n=== Building indices for {dataset_id} ===")

        # Load dataset configuration
        config = self.dataset_manager.get_dataset_config(dataset_id)
        print(f"Dataset: {config.name}")
        print(f"Source: {config.source_path}")

        # Load data
        print("\n📖 Loading dataset...")
        try:
            texts, labels = self.dataset_manager.load_dataset_data(dataset_id)
            print(f"✓ Loaded {len(texts)} documents")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return

        # Build ChromaDB index
        if "chromadb" in methods and config.chromadb_enabled:
            print("\n🔮 Building ChromaDB index...")
            start_time = time.time()
            try:
                self.dataset_manager.initialize_chromadb(dataset_id, rebuild=rebuild)

                # Add documents in batches
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_metadata = []

                    # Add labels as metadata if available
                    if labels:
                        for j in range(i, min(i + batch_size, len(labels))):
                            if j < len(labels):
                                batch_metadata.append({"labels": str(labels[j])})
                    else:
                        batch_metadata = None

                    self.dataset_manager.add_to_chromadb(
                        dataset_id, batch_texts, batch_metadata
                    )
                    print(f"  Added batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

                elapsed = time.time() - start_time
                print(f"✓ ChromaDB index built in {elapsed:.2f} seconds")

            except Exception as e:
                print(f"✗ Failed to build ChromaDB index: {e}")

        # Build TF-IDF index
        if "tfidf" in methods and config.tfidf_enabled:
            print("\n📊 Building TF-IDF index...")
            start_time = time.time()
            try:
                self.dataset_manager.build_tfidf_index(dataset_id, texts)
                elapsed = time.time() - start_time
                print(f"✓ TF-IDF index built in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"✗ Failed to build TF-IDF index: {e}")

        print("\n✅ Index building complete!")

    def validate_mappings(self, dataset_id: str) -> None:
        """Validate label mappings for a dataset.

        Args:
            dataset_id: Dataset identifier
        """
        print(f"\n=== Validating mappings for {dataset_id} ===")

        config = self.dataset_manager.get_dataset_config(dataset_id)

        # Load mapping file
        if not config.label_mapping_file:
            print("⚠️  No label mapping file specified")
            return

        if not os.path.exists(config.label_mapping_file):
            print(f"✗ Mapping file not found: {config.label_mapping_file}")
            return

        # Load mappings
        self.label_mapper.load_mapping_file(dataset_id, config.label_mapping_file)
        stats = self.label_mapper.get_mapping_statistics(dataset_id)

        print(f"\n📋 Mapping Statistics:")
        print(f"  - Label mappings: {stats['num_label_mappings']}")
        print(f"  - Combination rules: {stats['num_combination_rules']}")
        print(f"  - Archetypes covered: {', '.join(stats['archetypes_covered'])}")
        print(f"  - Unique tokens used: {len(stats['tokens_used'])}")

        # Load data to check label coverage
        print(f"\n🔍 Checking label coverage...")
        try:
            texts, labels = self.dataset_manager.load_dataset_data(dataset_id)

            if labels:
                # Count unique labels in dataset
                if config.source_format == "csv" and config.label_columns:
                    # Multi-label format
                    unique_labels = set()
                    for label_row in labels:
                        for i, col in enumerate(config.label_columns):
                            if label_row[i]:  # Check if label is present
                                unique_labels.add(col)
                else:
                    # Single label format
                    unique_labels = set(labels)

                print(f"  Dataset has {len(unique_labels)} unique labels")

                # Check which labels have mappings
                unmapped = []
                for label in unique_labels:
                    archetype, tokens = self.label_mapper.map_label(dataset_id, str(label))
                    if archetype == "benign" and str(label) != "benign":
                        unmapped.append(str(label))

                if unmapped:
                    print(f"  ⚠️  Unmapped labels: {', '.join(unmapped)}")
                else:
                    print(f"  ✓ All labels have mappings")

        except Exception as e:
            print(f"  ✗ Could not validate: {e}")

        # Validate tokens
        print(f"\n🏷️  Validating tokens...")
        invalid_tokens = []
        for token in stats['tokens_used']:
            validated = self.label_mapper.validate_tokens([token])
            if not validated:
                invalid_tokens.append(token)

        if invalid_tokens:
            print(f"  ⚠️  Invalid tokens found: {', '.join(invalid_tokens)}")
        else:
            print(f"  ✓ All tokens are valid")

    async def test_retrieval(self, dataset_id: str, query: str,
                            methods: Optional[List[str]] = None) -> None:
        """Test retrieval on a dataset.

        Args:
            dataset_id: Dataset identifier
            query: Test query
            methods: Methods to test
        """
        print(f"\n=== Testing retrieval for {dataset_id} ===")
        print(f"Query: {query}")

        if methods is None:
            methods = ["chromadb", "tfidf"]

        # Initialize retriever
        retriever = MultiMethodRetriever(
            self.dataset_manager,
            self.label_mapper
        )

        results = {}

        # Test ChromaDB
        if "chromadb" in methods:
            print("\n🔮 Testing ChromaDB retrieval...")
            start_time = time.time()
            chromadb_results = self.dataset_manager.search_chromadb(
                dataset_id, query, top_k=5
            )
            elapsed = time.time() - start_time

            if chromadb_results:
                print(f"  Found {len(chromadb_results)} results in {elapsed:.3f}s")
                for i, result in enumerate(chromadb_results[:3], 1):
                    text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                    distance = result.get('distance', 0)
                    print(f"  {i}. [dist={distance:.3f}] {text}")
            else:
                print(f"  No results found")

            results["chromadb"] = chromadb_results

        # Test TF-IDF
        if "tfidf" in methods:
            print("\n📊 Testing TF-IDF retrieval...")
            start_time = time.time()
            tfidf_results = self.dataset_manager.search_tfidf(
                dataset_id, query, top_k=5
            )
            elapsed = time.time() - start_time

            if tfidf_results:
                print(f"  Found {len(tfidf_results)} results in {elapsed:.3f}s")
                for i, result in enumerate(tfidf_results[:3], 1):
                    text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                    score = result.get('score', 0)
                    print(f"  {i}. [score={score:.3f}] {text}")
            else:
                print(f"  No results found")

            results["tfidf"] = tfidf_results

        # Test combined retrieval
        print("\n🔄 Testing combined retrieval with fallback...")
        combined_result = await retriever.retrieve(
            query=query,
            dataset_ids=[dataset_id],
            top_k=5
        )

        print(f"  Method used: {combined_result.method.value}")
        print(f"  Success: {combined_result.success}")
        print(f"  Latency: {combined_result.latency_ms:.2f}ms")

        if combined_result.results:
            best_text = combined_result.get_best_text()
            if best_text:
                preview = best_text[:150] + "..." if len(best_text) > 150 else best_text
                print(f"  Best match: {preview}")

    def export_statistics(self, output_file: str) -> None:
        """Export statistics for all datasets.

        Args:
            output_file: Output JSON file
        """
        print(f"\n=== Exporting statistics to {output_file} ===")

        all_stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "registry_path": self.registry_path,
            "datasets": {}
        }

        for dataset_id in self.dataset_manager.list_datasets():
            config = self.dataset_manager.get_dataset_config(dataset_id)
            stats = self.dataset_manager.get_statistics(dataset_id)

            # Add label mapping stats if available
            try:
                self.label_mapper.load_mapping_file(dataset_id, config.label_mapping_file)
                mapping_stats = self.label_mapper.get_mapping_statistics(dataset_id)
            except:
                mapping_stats = {}

            all_stats["datasets"][dataset_id] = {
                "config": {
                    "name": config.name,
                    "description": config.description,
                    "source_path": config.source_path,
                    "chromadb_enabled": config.chromadb_enabled,
                    "tfidf_enabled": config.tfidf_enabled
                },
                "index_stats": stats,
                "mapping_stats": mapping_stats
            }

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        print(f"✓ Statistics exported to {output_file}")

    async def benchmark_performance(self, dataset_id: str, num_queries: int = 100) -> None:
        """Benchmark retrieval performance.

        Args:
            dataset_id: Dataset identifier
            num_queries: Number of test queries
        """
        print(f"\n=== Benchmarking {dataset_id} ===")
        print(f"Running {num_queries} test queries...")

        # Load some sample texts as queries
        texts, _ = self.dataset_manager.load_dataset_data(dataset_id)
        if len(texts) < num_queries:
            print(f"⚠️  Dataset has only {len(texts)} texts, using all")
            num_queries = len(texts)

        # Sample queries
        import random
        queries = random.sample(texts, min(num_queries, len(texts)))

        # Initialize retriever
        retriever = MultiMethodRetriever(
            self.dataset_manager,
            self.label_mapper
        )

        # Run benchmark
        results = {
            "chromadb": [],
            "tfidf": [],
            "combined": []
        }

        for i, query in enumerate(queries, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_queries}")

            # Truncate query for testing
            test_query = query[:200] if len(query) > 200 else query

            # Test ChromaDB
            start = time.time()
            chromadb_res = self.dataset_manager.search_chromadb(dataset_id, test_query, top_k=5)
            results["chromadb"].append(time.time() - start)

            # Test TF-IDF
            start = time.time()
            tfidf_res = self.dataset_manager.search_tfidf(dataset_id, test_query, top_k=5)
            results["tfidf"].append(time.time() - start)

            # Test combined
            start = time.time()
            combined_res = await retriever.retrieve(
                query=test_query,
                dataset_ids=[dataset_id],
                top_k=5
            )
            results["combined"].append(time.time() - start)

        # Calculate statistics
        print("\n📊 Performance Results (seconds):")
        print("=" * 60)

        for method, times in results.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                p50 = sorted(times)[len(times) // 2]
                p95 = sorted(times)[int(len(times) * 0.95)]

                print(f"\n{method.upper()}:")
                print(f"  Average: {avg_time:.4f}s")
                print(f"  Min:     {min_time:.4f}s")
                print(f"  Max:     {max_time:.4f}s")
                print(f"  P50:     {p50:.4f}s")
                print(f"  P95:     {p95:.4f}s")

        # Get retriever metrics
        metrics = retriever.get_metrics()
        if metrics:
            print("\n📈 Retriever Metrics:")
            for method, stats in metrics.items():
                print(f"\n{method}:")
                print(f"  Calls:        {stats['calls']}")
                print(f"  Success Rate: {stats['success_rate']:.2%}")
                print(f"  Avg Latency:  {stats['avg_latency_ms']:.2f}ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage imputation datasets and indices"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all datasets")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build indices for a dataset")
    build_parser.add_argument("dataset_id", help="Dataset identifier")
    build_parser.add_argument("--rebuild", action="store_true", help="Rebuild existing indices")
    build_parser.add_argument("--methods", nargs="+", choices=["chromadb", "tfidf"],
                             help="Methods to build (default: all)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate label mappings")
    validate_parser.add_argument("dataset_id", help="Dataset identifier")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test retrieval")
    test_parser.add_argument("dataset_id", help="Dataset identifier")
    test_parser.add_argument("query", help="Test query")
    test_parser.add_argument("--methods", nargs="+", choices=["chromadb", "tfidf"],
                            help="Methods to test (default: all)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export statistics")
    export_parser.add_argument("output_file", help="Output JSON file")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark performance")
    bench_parser.add_argument("dataset_id", help="Dataset identifier")
    bench_parser.add_argument("--num-queries", type=int, default=100,
                             help="Number of test queries (default: 100)")

    # Parse arguments
    args = parser.parse_args()

    # Initialize CLI
    cli = DatasetCLI()

    # Execute command
    if args.command == "list":
        cli.list_datasets()

    elif args.command == "build":
        asyncio.run(cli.build_indices(
            args.dataset_id,
            rebuild=args.rebuild,
            methods=args.methods
        ))

    elif args.command == "validate":
        cli.validate_mappings(args.dataset_id)

    elif args.command == "test":
        asyncio.run(cli.test_retrieval(
            args.dataset_id,
            args.query,
            methods=args.methods
        ))

    elif args.command == "export":
        cli.export_statistics(args.output_file)

    elif args.command == "benchmark":
        asyncio.run(cli.benchmark_performance(
            args.dataset_id,
            num_queries=args.num_queries
        ))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()