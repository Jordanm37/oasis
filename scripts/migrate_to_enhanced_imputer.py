#!/usr/bin/env python
"""Migration script to transition from old RagImputer to EnhancedRagImputer.

This script helps migrate existing imputation data and indices to the new
multi-dataset system.
"""

import os
import sys
import argparse
import shutil
import pickle
import json
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.imputation.dataset_manager import DatasetManager
from oasis.imputation.label_mapper import LabelMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImpuationMigrator:
    """Handles migration from old to new imputation system."""

    def __init__(self, dry_run: bool = False):
        """Initialize the migrator.

        Args:
            dry_run: If True, don't actually modify files
        """
        self.dry_run = dry_run
        self.migration_log = []

    def migrate_tfidf_index(
        self,
        old_path: str,
        dataset_id: str,
        dataset_name: str,
        output_registry: str = "configs/imputation/dataset_registry.yaml"
    ) -> bool:
        """Migrate existing TF-IDF index to new format.

        Args:
            old_path: Path to old TF-IDF pickle file
            dataset_id: New dataset identifier
            dataset_name: Human-readable dataset name
            output_registry: Path to output registry file

        Returns:
            True if successful
        """
        logger.info(f"Migrating TF-IDF index from {old_path}")

        if not os.path.exists(old_path):
            logger.error(f"Old TF-IDF index not found: {old_path}")
            return False

        try:
            # Load old index
            with open(old_path, 'rb') as f:
                old_data = pickle.load(f)

            # Extract components
            if isinstance(old_data, dict):
                vectorizer = old_data.get('vectorizer')
                matrix = old_data.get('matrix')
                texts = old_data.get('texts', [])
            else:
                logger.error("Unexpected TF-IDF index format")
                return False

            # Create new dataset entry
            new_dataset_config = {
                "name": dataset_name,
                "description": f"Migrated from {old_path}",
                "source": {
                    "path": f"data/migrated/{dataset_id}_texts.jsonl",
                    "format": "jsonl",
                    "text_field": "text"
                },
                "label_mapping_file": f"configs/imputation/mappings/{dataset_id}_to_oasis.yaml",
                "indices": {
                    "chromadb": {
                        "enabled": False,  # Will need to be built separately
                        "path": f"data/imputation/chromadb/{dataset_id}",
                        "collection_name": dataset_id,
                        "embedding_model": "all-MiniLM-L6-v2"
                    },
                    "tfidf": {
                        "enabled": True,
                        "path": f"data/imputation/tfidf/{dataset_id}.pkl",
                        "max_features": vectorizer.max_features if hasattr(vectorizer, 'max_features') else 10000,
                        "ngram_range": list(vectorizer.ngram_range) if hasattr(vectorizer, 'ngram_range') else [1, 3]
                    }
                },
                "retrieval": {
                    "top_k": 10,
                    "min_similarity": 0.3
                }
            }

            # Save texts to JSONL
            if not self.dry_run:
                output_text_path = Path(f"data/migrated/{dataset_id}_texts.jsonl")
                output_text_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_text_path, 'w') as f:
                    for i, text in enumerate(texts):
                        json.dump({"id": i, "text": text}, f)
                        f.write('\n')

                logger.info(f"Saved {len(texts)} texts to {output_text_path}")

                # Copy TF-IDF index to new location
                new_tfidf_path = Path(f"data/imputation/tfidf/{dataset_id}.pkl")
                new_tfidf_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_path, new_tfidf_path)
                logger.info(f"Copied TF-IDF index to {new_tfidf_path}")

                # Update registry
                self._update_registry(output_registry, dataset_id, new_dataset_config)

            self.migration_log.append({
                "type": "tfidf_index",
                "dataset_id": dataset_id,
                "old_path": old_path,
                "num_texts": len(texts),
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"Failed to migrate TF-IDF index: {e}")
            self.migration_log.append({
                "type": "tfidf_index",
                "dataset_id": dataset_id,
                "old_path": old_path,
                "error": str(e),
                "success": False
            })
            return False

    def migrate_static_bank(
        self,
        old_yaml_path: str,
        output_path: str = "configs/imputation/static_bank.yaml"
    ) -> bool:
        """Migrate static phrase bank to new format.

        Args:
            old_yaml_path: Path to old static bank YAML
            output_path: Output path for new static bank

        Returns:
            True if successful
        """
        logger.info(f"Migrating static bank from {old_yaml_path}")

        if not os.path.exists(old_yaml_path):
            logger.error(f"Old static bank not found: {old_yaml_path}")
            return False

        try:
            # Load old static bank
            with open(old_yaml_path, 'r') as f:
                old_bank = yaml.safe_load(f)

            # Transform to new format (map old labels to archetypes)
            label_to_archetype = {
                "toxic": "hate_speech",
                "severe_toxic": "extremist",
                "obscene": "bullying",
                "threat": "extremist",
                "insult": "bullying",
                "identity_hate": "hate_speech",
                "neutral": "benign",
                "supportive": "recovery_support"
            }

            new_bank = {}
            for old_label, phrases in old_bank.items():
                # Map to new archetype
                new_archetype = label_to_archetype.get(old_label, old_label)

                if new_archetype not in new_bank:
                    new_bank[new_archetype] = []

                # Add phrases
                if isinstance(phrases, list):
                    new_bank[new_archetype].extend(phrases)
                elif isinstance(phrases, str):
                    new_bank[new_archetype].append(phrases)

            if not self.dry_run:
                # Save new static bank
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    yaml.dump(new_bank, f, default_flow_style=False)

                logger.info(f"Saved migrated static bank to {output_path}")

            self.migration_log.append({
                "type": "static_bank",
                "old_path": old_yaml_path,
                "output_path": str(output_path),
                "num_archetypes": len(new_bank),
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"Failed to migrate static bank: {e}")
            self.migration_log.append({
                "type": "static_bank",
                "old_path": old_yaml_path,
                "error": str(e),
                "success": False
            })
            return False

    def create_default_mapping(
        self,
        dataset_id: str,
        output_path: Optional[str] = None
    ) -> bool:
        """Create a default label mapping file for migrated dataset.

        Args:
            dataset_id: Dataset identifier
            output_path: Output path for mapping file

        Returns:
            True if successful
        """
        if output_path is None:
            output_path = f"configs/imputation/mappings/{dataset_id}_to_oasis.yaml"

        logger.info(f"Creating default mapping for {dataset_id}")

        # Default mapping template
        default_mapping = {
            "label_mappings": {
                "default": {
                    "archetypes": [
                        {
                            "archetype": "benign",
                            "weight": 1.0,
                            "tokens": ["LBL:FRIENDLY", "LBL:SUPPORTIVE"]
                        }
                    ]
                }
            },
            "combination_rules": [],
            "fallback_mappings": {
                "default_archetype": "benign",
                "default_tokens": ["LBL:FRIENDLY"],
                "low_confidence_archetype": "benign",
                "low_confidence_tokens": ["LBL:SUPPORTIVE"]
            },
            "metadata": {
                "version": "1.0.0",
                "created": "migration",
                "description": f"Default mapping for migrated dataset {dataset_id}"
            }
        }

        if not self.dry_run:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                yaml.dump(default_mapping, f, default_flow_style=False)

            logger.info(f"Created default mapping at {output_path}")

        return True

    def _update_registry(
        self,
        registry_path: str,
        dataset_id: str,
        dataset_config: Dict[str, Any]
    ) -> None:
        """Update dataset registry with new dataset.

        Args:
            registry_path: Path to registry file
            dataset_id: Dataset identifier
            dataset_config: Dataset configuration
        """
        # Load existing registry or create new
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f) or {}
        else:
            registry = {"datasets": {}, "global_settings": {}}

        # Add dataset
        if "datasets" not in registry:
            registry["datasets"] = {}

        registry["datasets"][dataset_id] = dataset_config

        # Add default global settings if not present
        if "global_settings" not in registry:
            registry["global_settings"] = {
                "default_embedding_model": "all-MiniLM-L6-v2",
                "cache": {
                    "enabled": True,
                    "ttl_seconds": 3600,
                    "max_size_mb": 1024
                },
                "fallback_chain": [
                    {"method": "chromadb", "timeout_seconds": 5},
                    {"method": "tfidf", "timeout_seconds": 3},
                    {"method": "static_bank", "timeout_seconds": 1}
                ]
            }

        # Save updated registry
        Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False)

        logger.info(f"Updated registry at {registry_path}")

    def validate_migration(self) -> bool:
        """Validate that migration was successful.

        Returns:
            True if all migrations successful
        """
        success_count = sum(1 for log in self.migration_log if log.get("success", False))
        total_count = len(self.migration_log)

        logger.info(f"\n=== Migration Summary ===")
        logger.info(f"Total operations: {total_count}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {total_count - success_count}")

        for log in self.migration_log:
            if log["success"]:
                logger.info(f"✓ {log['type']}: {log.get('dataset_id', 'N/A')}")
            else:
                logger.error(f"✗ {log['type']}: {log.get('error', 'Unknown error')}")

        return success_count == total_count

    def save_migration_log(self, output_path: str = "migration_log.json") -> None:
        """Save migration log to file.

        Args:
            output_path: Output path for log file
        """
        with open(output_path, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        logger.info(f"Saved migration log to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate from old RagImputer to EnhancedRagImputer"
    )

    parser.add_argument(
        "--tfidf-path",
        help="Path to old TF-IDF index pickle file"
    )
    parser.add_argument(
        "--static-bank-path",
        help="Path to old static phrase bank YAML"
    )
    parser.add_argument(
        "--dataset-id",
        help="ID for the migrated dataset"
    )
    parser.add_argument(
        "--dataset-name",
        help="Human-readable name for the dataset"
    )
    parser.add_argument(
        "--output-registry",
        default="configs/imputation/dataset_registry.yaml",
        help="Output registry file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making changes"
    )
    parser.add_argument(
        "--log-file",
        default="migration_log.json",
        help="Output file for migration log"
    )

    args = parser.parse_args()

    # Initialize migrator
    migrator = ImpuationMigrator(dry_run=args.dry_run)

    # Run migrations
    if args.tfidf_path:
        if not args.dataset_id or not args.dataset_name:
            parser.error("--dataset-id and --dataset-name required for TF-IDF migration")

        success = migrator.migrate_tfidf_index(
            args.tfidf_path,
            args.dataset_id,
            args.dataset_name,
            args.output_registry
        )

        if success:
            # Create default mapping
            migrator.create_default_mapping(args.dataset_id)

    if args.static_bank_path:
        migrator.migrate_static_bank(args.static_bank_path)

    # Validate and save log
    if migrator.migration_log:
        migrator.validate_migration()
        migrator.save_migration_log(args.log_file)


if __name__ == "__main__":
    main()