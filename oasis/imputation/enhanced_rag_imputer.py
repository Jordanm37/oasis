"""Enhanced RAG-based Imputation System with Multi-Dataset Support.

This module extends the original RagImputer with multi-dataset vector database support,
combining ChromaDB and TF-IDF retrieval methods with automatic label mapping.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from .dataset_manager import DatasetManager
from .label_mapper import LabelMapper
from .multi_method_retriever import MultiMethodRetriever, RetrievalMethod

logger = logging.getLogger(__name__)


@dataclass
class ImputationRecord:
    """Record of an imputation request and result."""

    persona_id: str
    thread_id: str
    step_idx: int
    label_tokens: List[str]
    archetype: str
    query_context: Optional[str] = None
    retrieved_text: Optional[str] = None
    imputed_text: Optional[str] = None
    retrieval_method: Optional[str] = None
    dataset_id: Optional[str] = None
    confidence: float = 1.0
    obfuscated: bool = False


class EnhancedRagImputer:
    """Enhanced RAG-based imputer with multi-dataset support."""

    def __init__(
        self,
        registry_path: str = "configs/imputation/dataset_registry.yaml",
        llm_client: Optional[Any] = None,
        static_bank_path: Optional[str] = None,
        enable_obfuscation: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the enhanced RAG imputer.

        Args:
            registry_path: Path to dataset registry YAML
            llm_client: Optional LLM client for generation fallback
            static_bank_path: Optional path to static phrase bank YAML
            enable_obfuscation: Whether to apply post-imputation obfuscation
            cache_dir: Directory for caching imputed content
        """
        # Initialize core components
        self.dataset_manager = DatasetManager(registry_path)
        self.label_mapper = LabelMapper()
        self.llm_client = llm_client
        self.enable_obfuscation = enable_obfuscation

        # Load static bank if provided
        self.static_bank = {}
        if static_bank_path and os.path.exists(static_bank_path):
            with open(static_bank_path, 'r') as f:
                self.static_bank = yaml.safe_load(f)
            logger.info(f"Loaded static bank from {static_bank_path}")

        # Initialize multi-method retriever
        self.retriever = MultiMethodRetriever(
            dataset_manager=self.dataset_manager,
            label_mapper=self.label_mapper,
            llm_client=llm_client,
            static_bank=self.static_bank
        )

        # Setup caching
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = self._load_cache()
        else:
            self.cache = {}

        # Load all dataset mappings
        self._load_all_mappings()

        # Obfuscation patterns (from original implementation)
        self.obfuscation_patterns = {
            "leetspeak": {
                "a": ["4", "@"],
                "e": ["3"],
                "i": ["1", "!"],
                "o": ["0"],
                "s": ["5", "$"],
            },
            "spacing": ["", " ", "-", "_", "."],
            "euphemisms": {
                "kill": ["unalive", "kms", "end it"],
                "suicide": ["sewerslide", "s*icide", "s-word"],
                "eating disorder": ["3d", "3ating d1sorder"],
                "anorexia": ["4n4", "@na"],
                "depression": ["big sad", "depresso"],
            }
        }

        logger.info("Initialized EnhancedRagImputer")

    def _load_all_mappings(self) -> None:
        """Load label mappings for all datasets."""
        for dataset_id in self.dataset_manager.list_datasets():
            config = self.dataset_manager.get_dataset_config(dataset_id)
            if config.label_mapping_file and os.path.exists(config.label_mapping_file):
                self.label_mapper.load_mapping_file(dataset_id, config.label_mapping_file)
                logger.info(f"Loaded mappings for {dataset_id}")

    def _load_cache(self) -> Dict[str, str]:
        """Load cache from disk.

        Returns:
            Cache dictionary
        """
        cache_file = Path(self.cache_dir) / "imputation_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / "imputation_cache.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

    async def impute(
        self,
        persona_id: str,
        thread_id: str,
        step_idx: int,
        label_tokens: List[str],
        archetype: str,
        context: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        force_method: Optional[RetrievalMethod] = None,
    ) -> ImputationRecord:
        """Impute content for given persona and context.

        Args:
            persona_id: Persona identifier
            thread_id: Thread identifier
            step_idx: Step index in conversation
            label_tokens: Label tokens to include
            archetype: Target archetype
            context: Optional context for retrieval
            dataset_ids: Optional list of dataset IDs to search
            force_method: Optional method to force (bypasses fallback chain)

        Returns:
            ImputationRecord with imputed content
        """
        # Create record
        record = ImputationRecord(
            persona_id=persona_id,
            thread_id=thread_id,
            step_idx=step_idx,
            label_tokens=label_tokens,
            archetype=archetype,
            query_context=context
        )

        # Check cache
        cache_key = self._get_cache_key(persona_id, thread_id, step_idx, label_tokens)
        if cache_key in self.cache:
            cached_text = self.cache[cache_key]
            record.imputed_text = cached_text
            record.retrieval_method = "cache"
            logger.debug(f"Cache hit for {cache_key}")
            return record

        # Build retrieval query
        query = self._build_retrieval_query(archetype, label_tokens, context)
        record.query_context = query

        # Retrieve content
        methods = [force_method] if force_method else None
        result = await self.retriever.retrieve(
            query=query,
            dataset_ids=dataset_ids,
            archetype=archetype,
            tokens=label_tokens,
            methods=methods,
            top_k=10
        )

        # Extract best text
        retrieved_text = result.get_best_text()
        if retrieved_text:
            record.retrieved_text = retrieved_text
            record.retrieval_method = result.method.value
            record.dataset_id = result.dataset_id
            record.confidence = 1.0 - result.results[0].get('distance', 0) \
                if result.method == RetrievalMethod.CHROMADB else \
                result.results[0].get('score', 1.0)

            # Process retrieved text
            imputed_text = await self._process_retrieved_text(
                retrieved_text, label_tokens, archetype, record
            )
            record.imputed_text = imputed_text

            # Apply obfuscation if enabled
            if self.enable_obfuscation and self._should_obfuscate(archetype):
                obfuscated = await self._apply_obfuscation(imputed_text, archetype)
                record.imputed_text = obfuscated
                record.obfuscated = True

            # Cache the result
            self.cache[cache_key] = record.imputed_text
            self._save_cache()

        else:
            # Fallback to simple generation
            record.imputed_text = self._generate_fallback(archetype, label_tokens)
            record.retrieval_method = "fallback"
            record.confidence = 0.5

        return record

    async def batch_impute(
        self,
        requests: List[Dict[str, Any]],
        batch_size: int = 32,
        dataset_ids: Optional[List[str]] = None,
    ) -> List[ImputationRecord]:
        """Batch impute multiple requests.

        Args:
            requests: List of imputation requests
            batch_size: Batch size for processing
            dataset_ids: Optional dataset IDs to search

        Returns:
            List of ImputationRecord objects
        """
        results = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # Process batch concurrently
            tasks = []
            for req in batch:
                task = self.impute(
                    persona_id=req["persona_id"],
                    thread_id=req["thread_id"],
                    step_idx=req["step_idx"],
                    label_tokens=req["label_tokens"],
                    archetype=req["archetype"],
                    context=req.get("context"),
                    dataset_ids=dataset_ids
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            logger.info(f"Processed batch {i // batch_size + 1}/{(len(requests) - 1) // batch_size + 1}")

        return results

    def _build_retrieval_query(
        self,
        archetype: str,
        label_tokens: List[str],
        context: Optional[str] = None
    ) -> str:
        """Build retrieval query from archetype and tokens.

        Args:
            archetype: Target archetype
            label_tokens: Label tokens
            context: Optional context

        Returns:
            Query string
        """
        # Start with context if provided
        query_parts = []
        if context:
            query_parts.append(context)

        # Add archetype description
        archetype_descriptions = {
            "hate_speech": "hateful offensive content targeting groups",
            "bullying": "personal attacks harassment bullying",
            "extremist": "violent extremist threats",
            "incel_misogyny": "incel misogynistic anti-women",
            "alpha": "alpha male redpill manosphere",
            "pro_ana": "pro-anorexia eating disorder promotion",
            "ed_risk": "eating disorder self-harm content",
            "misinfo": "misinformation false claims conspiracy",
            "conspiracy": "conspiracy theories deep state",
            "trad": "traditional values culture war",
            "gamergate": "gamergate gaming culture war",
            "recovery_support": "recovery support mental health help",
            "benign": "friendly supportive positive content"
        }

        if archetype in archetype_descriptions:
            query_parts.append(archetype_descriptions[archetype])

        # Add token keywords (remove LBL: prefix)
        token_keywords = []
        for token in label_tokens:
            keyword = token.replace("LBL:", "").replace("_", " ").lower()
            token_keywords.append(keyword)

        if token_keywords:
            query_parts.append(" ".join(token_keywords))

        return " ".join(query_parts)

    async def _process_retrieved_text(
        self,
        text: str,
        label_tokens: List[str],
        archetype: str,
        record: ImputationRecord
    ) -> str:
        """Process retrieved text to ensure it contains required elements.

        Args:
            text: Retrieved text
            label_tokens: Required label tokens
            archetype: Target archetype
            record: Imputation record

        Returns:
            Processed text
        """
        # Check if text already contains label tokens
        has_tokens = any(token in text for token in label_tokens)

        if not has_tokens and label_tokens:
            # Inject tokens naturally
            text = self._inject_label_tokens(text, label_tokens)

        # Clean up text
        text = self._clean_text(text)

        return text

    def _inject_label_tokens(self, text: str, label_tokens: List[str]) -> str:
        """Inject label tokens into text naturally.

        Args:
            text: Original text
            label_tokens: Tokens to inject

        Returns:
            Text with injected tokens
        """
        # For now, append tokens at the end
        # In production, this would be more sophisticated
        if label_tokens:
            token_str = " ".join(label_tokens)
            text = f"{text} {token_str}"

        return text

    def _clean_text(self, text: str) -> str:
        """Clean retrieved text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Truncate if too long
        if len(text) > 500:
            text = text[:497] + "..."

        return text

    def _should_obfuscate(self, archetype: str) -> bool:
        """Check if archetype should be obfuscated.

        Args:
            archetype: Archetype name

        Returns:
            True if should obfuscate
        """
        harmful_archetypes = {
            "hate_speech", "extremist", "bullying",
            "pro_ana", "ed_risk", "incel_misogyny",
            "alpha", "misinfo", "conspiracy"
        }
        return archetype in harmful_archetypes

    async def _apply_obfuscation(self, text: str, archetype: str) -> str:
        """Apply obfuscation to text.

        Args:
            text: Text to obfuscate
            archetype: Archetype for context

        Returns:
            Obfuscated text
        """
        # Apply leetspeak randomly
        if random.random() < 0.3:
            for char, replacements in self.obfuscation_patterns["leetspeak"].items():
                if random.random() < 0.5:
                    replacement = random.choice(replacements)
                    text = text.replace(char, replacement)
                    text = text.replace(char.upper(), replacement)

        # Apply euphemisms
        if random.random() < 0.4:
            for term, euphemisms in self.obfuscation_patterns["euphemisms"].items():
                if term.lower() in text.lower():
                    euphemism = random.choice(euphemisms)
                    text = re.sub(
                        re.escape(term),
                        euphemism,
                        text,
                        flags=re.IGNORECASE
                    )

        # Add spacing/characters
        if random.random() < 0.2:
            words = text.split()
            for i in range(len(words)):
                if random.random() < 0.1:
                    char = random.choice(self.obfuscation_patterns["spacing"])
                    if char:
                        words[i] = char.join(list(words[i]))
            text = " ".join(words)

        return text

    def _generate_fallback(self, archetype: str, label_tokens: List[str]) -> str:
        """Generate fallback text when retrieval fails.

        Args:
            archetype: Target archetype
            label_tokens: Label tokens

        Returns:
            Fallback text
        """
        # Use static bank if available
        if archetype in self.static_bank:
            base_text = random.choice(self.static_bank[archetype])
        else:
            # Simple fallback templates
            fallback_templates = {
                "benign": ["That's interesting!", "Thanks for sharing!", "I appreciate that."],
                "hate_speech": ["[harmful content]", "[offensive statement]"],
                "bullying": ["[personal attack]", "[harassment]"],
                "extremist": ["[extremist content]", "[violent rhetoric]"],
                "recovery_support": ["Stay strong!", "You've got this!", "I'm here for you."],
                "default": ["[generated content]", "[placeholder text]"]
            }

            templates = fallback_templates.get(archetype, fallback_templates["default"])
            base_text = random.choice(templates)

        # Add tokens if needed
        if label_tokens:
            base_text = f"{base_text} {' '.join(label_tokens)}"

        return base_text

    def _get_cache_key(
        self,
        persona_id: str,
        thread_id: str,
        step_idx: int,
        label_tokens: List[str]
    ) -> str:
        """Generate cache key.

        Args:
            persona_id: Persona ID
            thread_id: Thread ID
            step_idx: Step index
            label_tokens: Label tokens

        Returns:
            Cache key
        """
        token_str = "_".join(sorted(label_tokens))
        key_str = f"{persona_id}:{thread_id}:{step_idx}:{token_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def initialize_dataset(
        self,
        dataset_id: str,
        rebuild: bool = False
    ) -> None:
        """Initialize a dataset for use.

        Args:
            dataset_id: Dataset identifier
            rebuild: Whether to rebuild indices
        """
        logger.info(f"Initializing dataset {dataset_id}")

        # Initialize indices
        self.dataset_manager.initialize_chromadb(dataset_id, rebuild)
        self.dataset_manager.initialize_tfidf(dataset_id, rebuild)

        # Load mapping
        config = self.dataset_manager.get_dataset_config(dataset_id)
        if config.label_mapping_file:
            self.label_mapper.load_mapping_file(dataset_id, config.label_mapping_file)

        logger.info(f"Dataset {dataset_id} initialized")

    def get_statistics(self) -> Dict[str, Any]:
        """Get imputer statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "cache_size": len(self.cache),
            "datasets": {},
            "retriever_metrics": self.retriever.get_metrics()
        }

        for dataset_id in self.dataset_manager.list_datasets():
            stats["datasets"][dataset_id] = self.dataset_manager.get_statistics(dataset_id)

        return stats