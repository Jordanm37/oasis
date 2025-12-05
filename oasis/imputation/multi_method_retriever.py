"""Multi-Method Retriever for RAG Imputation System.

This module provides the MultiMethodRetriever class that combines multiple
retrieval methods (ChromaDB, TF-IDF) with fallback chains for robust imputation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Local imports
from .dataset_manager import DatasetManager
from .label_mapper import LabelMapper

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    """Enumeration of available retrieval methods."""

    CHROMADB = "chromadb"
    TFIDF = "tfidf"
    LLM_GENERATION = "llm_generation"
    STATIC_BANK = "static_bank"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    method: RetrievalMethod
    dataset_id: str
    query: str
    results: List[Dict[str, Any]]
    archetype: Optional[str] = None
    tokens: Optional[List[str]] = None
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def get_best_text(self) -> Optional[str]:
        """Get the best matching text from results.

        Returns:
            Best matching text or None
        """
        if not self.results:
            return None

        # For ChromaDB results, use the first document
        if self.method == RetrievalMethod.CHROMADB:
            return self.results[0].get('text', '')

        # For TF-IDF results, use the highest scoring text
        if self.method == RetrievalMethod.TFIDF:
            return self.results[0].get('text', '')

        # For other methods, return the first result's text
        return self.results[0].get('text', '') if self.results else None


class MultiMethodRetriever:
    """Combines multiple retrieval methods with fallback chains."""

    def __init__(self,
                 dataset_manager: DatasetManager,
                 label_mapper: LabelMapper,
                 llm_client: Optional[Any] = None,
                 static_bank: Optional[Dict[str, List[str]]] = None):
        """Initialize the MultiMethodRetriever.

        Args:
            dataset_manager: DatasetManager instance
            label_mapper: LabelMapper instance
            llm_client: Optional LLM client for generation fallback
            static_bank: Optional static phrase bank for final fallback
        """
        self.dataset_manager = dataset_manager
        self.label_mapper = label_mapper
        self.llm_client = llm_client
        self.static_bank = static_bank or {}

        # Cache for recent retrievals
        self.cache: Dict[str, RetrievalResult] = {}
        self.cache_ttl = 3600  # 1 hour TTL

        # Performance tracking
        self.metrics: Dict[str, Dict[str, Any]] = {
            'chromadb': {'calls': 0, 'successes': 0, 'total_latency': 0},
            'tfidf': {'calls': 0, 'successes': 0, 'total_latency': 0},
            'llm_generation': {'calls': 0, 'successes': 0, 'total_latency': 0},
            'static_bank': {'calls': 0, 'successes': 0, 'total_latency': 0},
        }

    async def retrieve(self,
                      query: str,
                      dataset_ids: Optional[List[str]] = None,
                      archetype: Optional[str] = None,
                      tokens: Optional[List[str]] = None,
                      methods: Optional[List[RetrievalMethod]] = None,
                      top_k: int = 10) -> RetrievalResult:
        """Retrieve content using multiple methods with fallback.

        Args:
            query: Search query
            dataset_ids: List of dataset IDs to search (None = all)
            archetype: Optional archetype hint
            tokens: Optional token hints
            methods: Methods to try in order (None = use default chain)
            top_k: Number of results to retrieve

        Returns:
            RetrievalResult with best matches
        """
        # Check cache first
        cache_key = self._get_cache_key(query, dataset_ids, archetype)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached.latency_ms < self.cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Determine datasets to search
        if dataset_ids is None:
            dataset_ids = self.dataset_manager.list_datasets()

        # Determine methods to try
        if methods is None:
            methods = self._get_default_method_chain()

        # Try each method in sequence until success
        for method in methods:
            try:
                result = await self._try_method(
                    method=method,
                    query=query,
                    dataset_ids=dataset_ids,
                    archetype=archetype,
                    tokens=tokens,
                    top_k=top_k
                )

                if result.success and result.results:
                    # Cache successful result
                    self.cache[cache_key] = result
                    return result

            except Exception as e:
                logger.error(f"Error in {method.value}: {e}")
                continue

        # All methods failed, return empty result
        return RetrievalResult(
            method=RetrievalMethod.STATIC_BANK,
            dataset_id="none",
            query=query,
            results=[],
            archetype=archetype,
            tokens=tokens,
            success=False,
            error_message="All retrieval methods failed"
        )

    async def _try_method(self,
                         method: RetrievalMethod,
                         query: str,
                         dataset_ids: List[str],
                         archetype: Optional[str],
                         tokens: Optional[List[str]],
                         top_k: int) -> RetrievalResult:
        """Try a specific retrieval method.

        Args:
            method: Retrieval method to use
            query: Search query
            dataset_ids: List of dataset IDs
            archetype: Optional archetype hint
            tokens: Optional token hints
            top_k: Number of results

        Returns:
            RetrievalResult
        """
        start_time = time.time()
        self.metrics[method.value]['calls'] += 1

        try:
            if method == RetrievalMethod.CHROMADB:
                result = await self._retrieve_chromadb(query, dataset_ids, top_k)

            elif method == RetrievalMethod.TFIDF:
                result = await self._retrieve_tfidf(query, dataset_ids, top_k)

            elif method == RetrievalMethod.LLM_GENERATION:
                result = await self._generate_with_llm(query, archetype, tokens)

            elif method == RetrievalMethod.STATIC_BANK:
                result = await self._retrieve_static(archetype, tokens)

            else:
                raise ValueError(f"Unknown retrieval method: {method}")

            # Track metrics
            latency = (time.time() - start_time) * 1000
            result.latency_ms = latency
            self.metrics[method.value]['total_latency'] += latency

            if result.success:
                self.metrics[method.value]['successes'] += 1

            return result

        except Exception as e:
            logger.error(f"Error in {method.value}: {e}")
            return RetrievalResult(
                method=method,
                dataset_id="error",
                query=query,
                results=[],
                success=False,
                error_message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _retrieve_chromadb(self, query: str, dataset_ids: List[str],
                                 top_k: int) -> RetrievalResult:
        """Retrieve using ChromaDB vector search.

        Args:
            query: Search query
            dataset_ids: Dataset IDs to search
            top_k: Number of results

        Returns:
            RetrievalResult
        """
        all_results = []
        best_dataset = None
        best_score = -1

        for dataset_id in dataset_ids:
            config = self.dataset_manager.get_dataset_config(dataset_id)
            if not config.chromadb_enabled:
                continue

            # Search this dataset
            results = self.dataset_manager.search_chromadb(dataset_id, query, top_k)

            # Track best result
            if results:
                score = 1.0 - results[0].get('distance', 1.0)  # Convert distance to similarity
                if score > best_score:
                    best_score = score
                    best_dataset = dataset_id
                    all_results = results

        return RetrievalResult(
            method=RetrievalMethod.CHROMADB,
            dataset_id=best_dataset or "none",
            query=query,
            results=all_results,
            success=bool(all_results)
        )

    async def _retrieve_tfidf(self, query: str, dataset_ids: List[str],
                              top_k: int) -> RetrievalResult:
        """Retrieve using TF-IDF keyword matching.

        Args:
            query: Search query
            dataset_ids: Dataset IDs to search
            top_k: Number of results

        Returns:
            RetrievalResult
        """
        all_results = []
        best_dataset = None
        best_score = 0

        for dataset_id in dataset_ids:
            config = self.dataset_manager.get_dataset_config(dataset_id)
            if not config.tfidf_enabled:
                continue

            # Search this dataset
            results = self.dataset_manager.search_tfidf(dataset_id, query, top_k)

            # Track best result
            if results:
                score = results[0].get('score', 0)
                if score > best_score:
                    best_score = score
                    best_dataset = dataset_id
                    all_results = results

        return RetrievalResult(
            method=RetrievalMethod.TFIDF,
            dataset_id=best_dataset or "none",
            query=query,
            results=all_results,
            success=bool(all_results)
        )

    async def _generate_with_llm(self, query: str, archetype: Optional[str],
                                 tokens: Optional[List[str]]) -> RetrievalResult:
        """Generate content using LLM.

        Args:
            query: Generation prompt
            archetype: Optional archetype hint
            tokens: Optional token hints

        Returns:
            RetrievalResult
        """
        if not self.llm_client:
            return RetrievalResult(
                method=RetrievalMethod.LLM_GENERATION,
                dataset_id="none",
                query=query,
                results=[],
                success=False,
                error_message="LLM client not configured"
            )

        try:
            # Build prompt with archetype and token hints
            prompt = self._build_generation_prompt(query, archetype, tokens)

            # Generate response (implementation depends on LLM client)
            # This is a placeholder - actual implementation would call the LLM
            generated_text = await self._call_llm(prompt)

            return RetrievalResult(
                method=RetrievalMethod.LLM_GENERATION,
                dataset_id="generated",
                query=query,
                results=[{'text': generated_text, 'generated': True}],
                archetype=archetype,
                tokens=tokens,
                success=bool(generated_text)
            )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return RetrievalResult(
                method=RetrievalMethod.LLM_GENERATION,
                dataset_id="error",
                query=query,
                results=[],
                success=False,
                error_message=str(e)
            )

    async def _retrieve_static(self, archetype: Optional[str],
                              tokens: Optional[List[str]]) -> RetrievalResult:
        """Retrieve from static phrase bank.

        Args:
            archetype: Archetype to retrieve for
            tokens: Token hints

        Returns:
            RetrievalResult
        """
        results = []

        # Try archetype-specific phrases first
        if archetype and archetype in self.static_bank:
            import random
            phrase = random.choice(self.static_bank[archetype])
            results.append({'text': phrase, 'static': True})

        # Fallback to benign phrases
        elif 'benign' in self.static_bank:
            import random
            phrase = random.choice(self.static_bank['benign'])
            results.append({'text': phrase, 'static': True})

        return RetrievalResult(
            method=RetrievalMethod.STATIC_BANK,
            dataset_id="static",
            query="",
            results=results,
            archetype=archetype,
            tokens=tokens,
            success=bool(results)
        )

    def _build_generation_prompt(self, query: str, archetype: Optional[str],
                                 tokens: Optional[List[str]]) -> str:
        """Build prompt for LLM generation.

        Args:
            query: Base query
            archetype: Optional archetype
            tokens: Optional tokens

        Returns:
            Formatted prompt
        """
        prompt_parts = [f"Generate text similar to: {query}"]

        if archetype:
            prompt_parts.append(f"Style: {archetype}")

        if tokens:
            token_str = ", ".join(tokens)
            prompt_parts.append(f"Include themes: {token_str}")

        return "\n".join(prompt_parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for generation (placeholder).

        Args:
            prompt: Generation prompt

        Returns:
            Generated text
        """
        # This is a placeholder - actual implementation would depend on the LLM client
        # For now, return a simple message
        return f"[Generated content for: {prompt[:50]}...]"

    def _get_cache_key(self, query: str, dataset_ids: Optional[List[str]],
                      archetype: Optional[str]) -> str:
        """Generate cache key for a query.

        Args:
            query: Search query
            dataset_ids: Dataset IDs
            archetype: Archetype

        Returns:
            Cache key string
        """
        dataset_str = ",".join(sorted(dataset_ids)) if dataset_ids else "all"
        archetype_str = archetype or "none"
        return f"{query[:100]}:{dataset_str}:{archetype_str}"

    def _get_default_method_chain(self) -> List[RetrievalMethod]:
        """Get default method chain from global settings.

        Returns:
            List of retrieval methods to try in order
        """
        # Get from global settings if available
        global_settings = self.dataset_manager.global_settings
        fallback_chain = global_settings.get('fallback_chain', [])

        if fallback_chain:
            methods = []
            for item in fallback_chain:
                method_name = item.get('method', '').upper()
                if method_name == 'CHROMADB':
                    methods.append(RetrievalMethod.CHROMADB)
                elif method_name == 'TFIDF':
                    methods.append(RetrievalMethod.TFIDF)
                elif method_name == 'LLM_GENERATION':
                    methods.append(RetrievalMethod.LLM_GENERATION)
                elif method_name == 'STATIC_BANK':
                    methods.append(RetrievalMethod.STATIC_BANK)
            return methods

        # Default chain
        return [
            RetrievalMethod.CHROMADB,
            RetrievalMethod.TFIDF,
            RetrievalMethod.LLM_GENERATION,
            RetrievalMethod.STATIC_BANK
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        for method, stats in self.metrics.items():
            if stats['calls'] > 0:
                metrics[method] = {
                    'calls': stats['calls'],
                    'successes': stats['successes'],
                    'success_rate': stats['successes'] / stats['calls'],
                    'avg_latency_ms': stats['total_latency'] / stats['calls']
                }
        return metrics

    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.cache.clear()
        logger.info("Retrieval cache cleared")