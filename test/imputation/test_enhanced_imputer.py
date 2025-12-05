"""Test suite for the enhanced RAG imputation system."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
import yaml
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oasis.imputation.dataset_manager import DatasetManager, DatasetConfig
from oasis.imputation.label_mapper import LabelMapper, LabelMapping
from oasis.imputation.multi_method_retriever import (
    MultiMethodRetriever,
    RetrievalMethod,
    RetrievalResult
)
from oasis.imputation.enhanced_rag_imputer import EnhancedRagImputer, ImputationRecord


class TestDatasetManager(unittest.TestCase):
    """Test cases for DatasetManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "test_registry.yaml")

        # Create test registry
        test_registry = {
            "datasets": {
                "test_dataset": {
                    "name": "Test Dataset",
                    "description": "Test dataset for unit tests",
                    "source": {
                        "path": os.path.join(self.temp_dir, "test_data.jsonl"),
                        "format": "jsonl",
                        "text_field": "text",
                        "label_field": "label"
                    },
                    "label_mapping_file": os.path.join(self.temp_dir, "test_mapping.yaml"),
                    "indices": {
                        "chromadb": {
                            "enabled": True,
                            "path": os.path.join(self.temp_dir, "chromadb"),
                            "collection_name": "test_collection",
                            "embedding_model": "all-MiniLM-L6-v2"
                        },
                        "tfidf": {
                            "enabled": True,
                            "path": os.path.join(self.temp_dir, "tfidf.pkl"),
                            "max_features": 100,
                            "ngram_range": [1, 2]
                        }
                    },
                    "retrieval": {
                        "top_k": 5,
                        "min_similarity": 0.1
                    }
                }
            },
            "global_settings": {
                "default_embedding_model": "all-MiniLM-L6-v2"
            }
        }

        with open(self.registry_path, 'w') as f:
            yaml.dump(test_registry, f)

        # Create test data
        test_data = [
            {"text": "This is a test document about hate speech", "label": "hate_speech"},
            {"text": "This is supportive and friendly content", "label": "benign"},
            {"text": "This contains bullying and harassment", "label": "bullying"}
        ]

        data_path = os.path.join(self.temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')

        self.dataset_manager = DatasetManager(self.registry_path)

    def test_load_registry(self):
        """Test loading dataset registry."""
        self.assertIn("test_dataset", self.dataset_manager.datasets)
        config = self.dataset_manager.get_dataset_config("test_dataset")
        self.assertEqual(config.name, "Test Dataset")
        self.assertTrue(config.chromadb_enabled)
        self.assertTrue(config.tfidf_enabled)

    def test_load_dataset_data(self):
        """Test loading dataset data."""
        texts, labels = self.dataset_manager.load_dataset_data("test_dataset")
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(labels), 3)
        self.assertIn("hate speech", texts[0])
        self.assertEqual(labels[0], "hate_speech")

    def test_build_tfidf_index(self):
        """Test building TF-IDF index."""
        texts, _ = self.dataset_manager.load_dataset_data("test_dataset")
        self.dataset_manager.build_tfidf_index("test_dataset", texts)

        # Check that index was built
        self.assertIn("test_dataset", self.dataset_manager.tfidf_vectorizers)
        self.assertIn("test_dataset", self.dataset_manager.tfidf_matrices)

        # Test search
        results = self.dataset_manager.search_tfidf("test_dataset", "hate", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("hate speech", results[0]['text'])

    def test_dataset_statistics(self):
        """Test getting dataset statistics."""
        stats = self.dataset_manager.get_statistics("test_dataset")
        self.assertEqual(stats['dataset_id'], "test_dataset")
        self.assertTrue(stats['chromadb_enabled'])
        self.assertTrue(stats['tfidf_enabled'])


class TestLabelMapper(unittest.TestCase):
    """Test cases for LabelMapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.label_mapper = LabelMapper()

        # Create test mapping file
        test_mapping = {
            "label_mappings": {
                "toxic": {
                    "archetypes": [
                        {
                            "archetype": "hate_speech",
                            "weight": 0.6,
                            "tokens": ["LBL:HATE_SLUR", "LBL:DEHUMANIZATION"]
                        },
                        {
                            "archetype": "bullying",
                            "weight": 0.4,
                            "tokens": ["LBL:PERSONAL_ATTACK"]
                        }
                    ]
                },
                "supportive": {
                    "archetypes": [
                        {
                            "archetype": "benign",
                            "weight": 1.0,
                            "tokens": ["LBL:SUPPORTIVE", "LBL:FRIENDLY"]
                        }
                    ]
                }
            },
            "combination_rules": [
                {
                    "condition": {"all_of": ["toxic", "threat"]},
                    "override_archetype": "extremist",
                    "priority_tokens": ["LBL:VIOLENT_THREAT"]
                }
            ],
            "fallback_mappings": {
                "default_archetype": "benign",
                "default_tokens": ["LBL:FRIENDLY"]
            }
        }

        self.mapping_path = os.path.join(self.temp_dir, "test_mapping.yaml")
        with open(self.mapping_path, 'w') as f:
            yaml.dump(test_mapping, f)

    def test_load_mapping_file(self):
        """Test loading label mapping file."""
        self.label_mapper.load_mapping_file("test", self.mapping_path)
        self.assertIn("test", self.label_mapper.label_mappings)

        mappings = self.label_mapper.label_mappings["test"]
        self.assertIn("toxic", mappings)
        self.assertIn("supportive", mappings)

    def test_map_single_label(self):
        """Test mapping single label."""
        self.label_mapper.load_mapping_file("test", self.mapping_path)

        # Test toxic mapping
        archetype, tokens = self.label_mapper.map_label("test", "toxic")
        self.assertIn(archetype, ["hate_speech", "bullying"])
        self.assertTrue(len(tokens) > 0)

        # Test supportive mapping
        archetype, tokens = self.label_mapper.map_label("test", "supportive")
        self.assertEqual(archetype, "benign")
        self.assertIn("LBL:SUPPORTIVE", tokens)

    def test_map_multiple_labels(self):
        """Test mapping multiple labels with combination rules."""
        self.label_mapper.load_mapping_file("test", self.mapping_path)

        # Test combination rule
        archetype, tokens = self.label_mapper.map_multiple_labels(
            "test", ["toxic", "threat"]
        )
        self.assertEqual(archetype, "extremist")
        self.assertIn("LBL:VIOLENT_THREAT", tokens)

    def test_fallback_mapping(self):
        """Test fallback mapping for unknown labels."""
        self.label_mapper.load_mapping_file("test", self.mapping_path)

        archetype, tokens = self.label_mapper.map_label("test", "unknown_label")
        self.assertEqual(archetype, "benign")
        self.assertIn("LBL:FRIENDLY", tokens)

    def test_validate_tokens(self):
        """Test token validation."""
        valid_tokens = ["LBL:HATE_SLUR", "LBL:SUPPORTIVE"]
        validated = self.label_mapper.validate_tokens(valid_tokens)
        self.assertEqual(len(validated), 2)

        # Test invalid token
        invalid_tokens = ["LBL:INVALID_TOKEN"]
        validated = self.label_mapper.validate_tokens(invalid_tokens)
        self.assertEqual(len(validated), 0)


class TestMultiMethodRetriever(unittest.TestCase):
    """Test cases for MultiMethodRetriever."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create minimal dataset manager
        registry_path = os.path.join(self.temp_dir, "test_registry.yaml")
        test_registry = {
            "datasets": {
                "test": {
                    "name": "Test",
                    "description": "Test",
                    "source": {"path": "test.jsonl", "format": "jsonl"},
                    "indices": {
                        "chromadb": {"enabled": False},
                        "tfidf": {"enabled": True}
                    }
                }
            }
        }

        with open(registry_path, 'w') as f:
            yaml.dump(test_registry, f)

        self.dataset_manager = DatasetManager(registry_path)
        self.label_mapper = LabelMapper()

        # Create static bank
        self.static_bank = {
            "benign": ["This is friendly content", "Thanks for sharing!"],
            "hate_speech": ["[harmful content]"]
        }

        self.retriever = MultiMethodRetriever(
            self.dataset_manager,
            self.label_mapper,
            static_bank=self.static_bank
        )

    async def test_static_bank_retrieval(self):
        """Test retrieval from static bank."""
        result = await self.retriever.retrieve(
            query="test query",
            archetype="benign",
            methods=[RetrievalMethod.STATIC_BANK]
        )

        self.assertEqual(result.method, RetrievalMethod.STATIC_BANK)
        self.assertTrue(result.success)
        self.assertGreater(len(result.results), 0)

    async def test_fallback_chain(self):
        """Test fallback chain when methods fail."""
        # Force all methods except static bank to fail
        result = await self.retriever.retrieve(
            query="test query",
            dataset_ids=["nonexistent"],
            archetype="benign",
            methods=[
                RetrievalMethod.CHROMADB,
                RetrievalMethod.TFIDF,
                RetrievalMethod.STATIC_BANK
            ]
        )

        # Should fall back to static bank
        self.assertEqual(result.method, RetrievalMethod.STATIC_BANK)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = self.retriever._get_cache_key("query", ["ds1"], "benign")
        key2 = self.retriever._get_cache_key("query", ["ds1"], "benign")
        key3 = self.retriever._get_cache_key("different", ["ds1"], "benign")

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_metrics_tracking(self):
        """Test performance metrics tracking."""
        # Run some retrievals to generate metrics
        asyncio.run(self.retriever.retrieve(
            query="test",
            methods=[RetrievalMethod.STATIC_BANK]
        ))

        metrics = self.retriever.get_metrics()
        self.assertIn("static_bank", metrics)
        self.assertEqual(metrics["static_bank"]["calls"], 1)


class TestEnhancedRagImputer(unittest.IsolatedAsyncioTestCase):
    """Test cases for EnhancedRagImputer."""

    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test registry and data
        registry_path = os.path.join(self.temp_dir, "test_registry.yaml")
        test_registry = {
            "datasets": {},
            "global_settings": {}
        }

        with open(registry_path, 'w') as f:
            yaml.dump(test_registry, f)

        # Create static bank
        static_bank_path = os.path.join(self.temp_dir, "static_bank.yaml")
        static_bank = {
            "benign": ["Friendly message", "Supportive content"],
            "hate_speech": ["[harmful content]"]
        }
        with open(static_bank_path, 'w') as f:
            yaml.dump(static_bank, f)

        self.imputer = EnhancedRagImputer(
            registry_path=registry_path,
            static_bank_path=static_bank_path,
            enable_obfuscation=True,
            cache_dir=self.temp_dir
        )

    async def test_impute_with_static_fallback(self):
        """Test imputation with static bank fallback."""
        record = await self.imputer.impute(
            persona_id="test_persona",
            thread_id="thread_1",
            step_idx=0,
            label_tokens=["LBL:FRIENDLY"],
            archetype="benign",
            force_method=RetrievalMethod.STATIC_BANK
        )

        self.assertIsNotNone(record.imputed_text)
        self.assertEqual(record.retrieval_method, "static_bank")
        self.assertEqual(record.archetype, "benign")

    async def test_impute_with_cache(self):
        """Test imputation caching."""
        # First call
        record1 = await self.imputer.impute(
            persona_id="test",
            thread_id="t1",
            step_idx=0,
            label_tokens=["LBL:FRIENDLY"],
            archetype="benign",
            force_method=RetrievalMethod.STATIC_BANK
        )

        # Second call (should hit cache)
        record2 = await self.imputer.impute(
            persona_id="test",
            thread_id="t1",
            step_idx=0,
            label_tokens=["LBL:FRIENDLY"],
            archetype="benign"
        )

        self.assertEqual(record2.retrieval_method, "cache")
        self.assertEqual(record1.imputed_text, record2.imputed_text)

    async def test_batch_impute(self):
        """Test batch imputation."""
        requests = [
            {
                "persona_id": f"persona_{i}",
                "thread_id": f"thread_{i}",
                "step_idx": 0,
                "label_tokens": ["LBL:FRIENDLY"],
                "archetype": "benign"
            }
            for i in range(3)
        ]

        records = await self.imputer.batch_impute(requests, batch_size=2)

        self.assertEqual(len(records), 3)
        for record in records:
            self.assertIsNotNone(record.imputed_text)

    def test_build_retrieval_query(self):
        """Test query building."""
        query = self.imputer._build_retrieval_query(
            archetype="hate_speech",
            label_tokens=["LBL:HATE_SLUR", "LBL:DEHUMANIZATION"],
            context="discussing online toxicity"
        )

        self.assertIn("discussing online toxicity", query)
        self.assertIn("hateful", query.lower())
        self.assertIn("hate slur", query.lower())

    def test_obfuscation(self):
        """Test text obfuscation."""
        # Test that harmful archetypes trigger obfuscation
        should_obfuscate = self.imputer._should_obfuscate("hate_speech")
        self.assertTrue(should_obfuscate)

        should_not_obfuscate = self.imputer._should_obfuscate("benign")
        self.assertFalse(should_not_obfuscate)

    async def test_apply_obfuscation(self):
        """Test obfuscation application."""
        original = "This contains harmful content with hate"
        obfuscated = await self.imputer._apply_obfuscation(original, "hate_speech")

        # Obfuscation should modify text (probabilistic, so we can't guarantee specific changes)
        self.assertIsNotNone(obfuscated)
        self.assertTrue(len(obfuscated) > 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetManager))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiMethodRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedRagImputer))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)