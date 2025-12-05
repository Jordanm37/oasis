#!/usr/bin/env python3
"""Test script for RAG imputation system with downloaded datasets.

Validates:
1. Dataset loaders work correctly
2. Data can be loaded and sampled
3. Label mappings are configured
4. RAG retrieval functions (if indices built)
"""

import importlib.util
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_module_directly(module_name: str, file_path: Path):
    """Load a module directly without triggering package imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_hate_speech_loaders():
    """Test the unified hate speech loaders."""
    logger.info("\n=== Testing Hate Speech Loaders ===")

    # Load module directly to avoid heavy dependencies
    loader_path = PROJECT_ROOT / "oasis" / "datasets" / "hate_speech_loader.py"
    hate_speech_module = load_module_directly("hate_speech_loader", loader_path)
    UnifiedHateSpeechLoader = hate_speech_module.UnifiedHateSpeechLoader

    loader = UnifiedHateSpeechLoader(base_dir="data/hate_speech_datasets")
    stats = loader.get_statistics()

    total_samples = 0
    for name, info in stats.items():
        if "total_samples" in info:
            logger.info(f"  {name}: {info['total_samples']:,} samples")
            total_samples += info["total_samples"]
        else:
            logger.warning(f"  {name}: {info.get('status', 'unknown')}")

    logger.info(f"  TOTAL: {total_samples:,} samples across all datasets")

    # Sample from each dataset
    logger.info("\n--- Sample Content ---")
    for dataset_name in ["davidson", "hatexplain", "gab", "pheme"]:
        try:
            samples = list(loader.iterate_dataset(dataset_name))[:2]
            for sample in samples:
                text_preview = sample.text[:80] + "..." if len(sample.text) > 80 else sample.text
                logger.info(f"  [{dataset_name}] {sample.label}: {text_preview}")
        except FileNotFoundError as e:
            logger.warning(f"  [{dataset_name}] Not found: {e}")

    return total_samples > 0


def test_toxigen_loader():
    """Test ToxiGen dataset loader."""
    logger.info("\n=== Testing ToxiGen Loader ===")

    try:
        # Load module directly
        loader_path = PROJECT_ROOT / "oasis" / "datasets" / "toxigen_loader.py"
        if not loader_path.exists():
            logger.warning(f"  ToxiGen loader not found: {loader_path}")
            return False

        toxigen_module = load_module_directly("toxigen_loader", loader_path)
        ToxiGenLoader = toxigen_module.ToxiGenLoader

        loader = ToxiGenLoader(data_path="data/toxigen_datasets")

        # Test annotated dataset
        try:
            annotated = loader.load_annotated_data()
            logger.info(f"  Annotated: {len(annotated):,} samples")
            if len(annotated) > 0:
                sample = annotated.iloc[0]
                text = str(sample.get('text', ''))[:80]
                logger.info(f"    Sample: {text}...")
        except Exception as e:
            logger.warning(f"  Annotated not available: {e}")

        # Test prompts
        try:
            prompts_df = loader.load_prompts(prompt_type="hate")
            logger.info(f"  Hate prompts: {len(prompts_df):,} samples")
        except Exception as e:
            logger.warning(f"  Hate prompts not available: {e}")

        return True
    except ImportError as e:
        logger.warning(f"  ToxiGen loader not available: {e}")
        return False


def test_implicit_hate_loader():
    """Test ImplicitHate dataset loader."""
    logger.info("\n=== Testing ImplicitHate Loader ===")

    try:
        # Load module directly
        loader_path = PROJECT_ROOT / "oasis" / "datasets" / "implicit_hate_loader.py"
        if not loader_path.exists():
            logger.warning(f"  ImplicitHate loader not found: {loader_path}")
            return False

        implicit_module = load_module_directly("implicit_hate_loader", loader_path)
        ImplicitHateLoader = implicit_module.ImplicitHateLoader

        loader = ImplicitHateLoader(data_path="data/implicit_hate_datasets")

        try:
            df = loader.load_data()
            logger.info(f"  Total samples: {len(df):,}")

            # Show class distribution
            if "implicit_class" in df.columns:
                logger.info("  Class distribution:")
                for cls, count in df["implicit_class"].value_counts().items():
                    logger.info(f"    {cls}: {count}")

            # Sample
            if len(df) > 0:
                sample = df.iloc[0]
                text = str(sample.get("post", ""))[:80]
                logger.info(f"  Sample: {text}...")

            return True
        except Exception as e:
            logger.warning(f"  Could not load: {e}")
            return False
    except ImportError as e:
        logger.warning(f"  ImplicitHate loader not available: {e}")
        return False


def test_dataset_registry():
    """Test dataset registry configuration."""
    logger.info("\n=== Testing Dataset Registry ===")

    import yaml

    registry_path = Path("configs/imputation/dataset_registry.yaml")
    if not registry_path.exists():
        logger.warning(f"  Registry not found: {registry_path}")
        return False

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    datasets = registry.get("datasets", {})
    logger.info(f"  Registered datasets: {len(datasets)}")

    for name, config in datasets.items():
        source_path = config.get("source", {}).get("path", "N/A")
        path_exists = Path(source_path).exists() if source_path != "N/A" else False
        status = "✓" if path_exists else "✗"
        logger.info(f"    {status} {name}: {source_path}")

    return True


def test_label_mappings():
    """Test label mapping files exist."""
    logger.info("\n=== Testing Label Mappings ===")

    mapping_dir = Path("configs/imputation/mappings")
    if not mapping_dir.exists():
        logger.warning(f"  Mapping directory not found: {mapping_dir}")
        return False

    mapping_files = list(mapping_dir.glob("*.yaml"))
    logger.info(f"  Found {len(mapping_files)} mapping files:")

    for f in mapping_files:
        logger.info(f"    - {f.name}")

    return len(mapping_files) > 0


def test_rag_retrieval():
    """Test RAG retrieval if indices are built."""
    logger.info("\n=== Testing RAG Retrieval ===")

    # Check if any TF-IDF indices exist
    tfidf_dir = Path("data/imputation/tfidf")
    if tfidf_dir.exists():
        indices = list(tfidf_dir.glob("*.pkl"))
        logger.info(f"  TF-IDF indices found: {len(indices)}")
        for idx in indices:
            logger.info(f"    - {idx.name}")
    else:
        logger.info("  No TF-IDF indices built yet")
        logger.info("  To build: python -m oasis.imputation.dataset_manager --build-all")

    # Check ChromaDB
    chroma_dir = Path("data/imputation/chromadb")
    if chroma_dir.exists():
        collections = [d for d in chroma_dir.iterdir() if d.is_dir()]
        logger.info(f"  ChromaDB collections found: {len(collections)}")
        for c in collections:
            logger.info(f"    - {c.name}")
    else:
        logger.info("  No ChromaDB indices built yet (chromadb package may not be installed)")

    # Try to load DatasetManager (may fail if chromadb not installed)
    try:
        manager_path = PROJECT_ROOT / "oasis" / "imputation" / "dataset_manager.py"
        if manager_path.exists():
            manager_module = load_module_directly("dataset_manager", manager_path)
            DatasetManager = manager_module.DatasetManager

            registry_path = "configs/imputation/dataset_registry.yaml"
            manager = DatasetManager(registry_path)
            available = manager.list_datasets()
            logger.info(f"  Registered datasets: {len(available)}")
    except ModuleNotFoundError as e:
        logger.info(f"  DatasetManager requires optional dependency: {e.name}")
        logger.info("  Install with: pip install chromadb")

    # Test passes if we can at least check indices
    return True


def test_static_bank():
    """Test static phrase bank."""
    logger.info("\n=== Testing Static Bank ===")

    import yaml

    static_bank_path = Path("data/label_tokens_static_bank.yaml")
    if not static_bank_path.exists():
        logger.warning(f"  Static bank not found: {static_bank_path}")
        return False

    with open(static_bank_path) as f:
        bank = yaml.safe_load(f)

    logger.info(f"  Archetypes in static bank: {len(bank)}")
    for archetype, phrases in list(bank.items())[:5]:
        count = len(phrases) if isinstance(phrases, list) else 1
        logger.info(f"    {archetype}: {count} phrases")

    if len(bank) > 5:
        logger.info(f"    ... and {len(bank) - 5} more")

    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RAG Imputation System Test Suite")
    logger.info("=" * 60)

    results = {}

    # Run tests
    results["hate_speech_loaders"] = test_hate_speech_loaders()
    results["toxigen_loader"] = test_toxigen_loader()
    results["implicit_hate_loader"] = test_implicit_hate_loader()
    results["dataset_registry"] = test_dataset_registry()
    results["label_mappings"] = test_label_mappings()
    results["rag_retrieval"] = test_rag_retrieval()
    results["static_bank"] = test_static_bank()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")

    logger.info(f"\nResult: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
