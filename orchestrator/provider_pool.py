"""Multi-provider distribution pool for load balancing LLM requests.

Usage:
    from orchestrator.provider_pool import ProviderPool

    pool = ProviderPool()
    provider = pool.next_provider()  # Get next provider based on weights
    async with pool.acquire(provider):
        # Make LLM request
        pass
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from configs.llm_settings import (
    API_ENDPOINTS,
    API_KEY_ENV_VARS,
    DEFAULT_RATE_LIMITS,
    PROVIDER_POOL,
    PROVIDER_POOL_ENABLED,
    PROVIDER_WEIGHTS,
    RATE_LIMITS,
    get_rate_limits,
)


@dataclass
class ProviderStats:
    """Track per-provider request statistics."""
    requests: int = 0
    failures: int = 0
    rate_limited: int = 0
    last_request_time: float = 0.0
    tokens_used: int = 0


@dataclass
class TokenBucket:
    """Simple token bucket rate limiter."""
    capacity: float
    tokens: float = field(init=False)
    fill_rate: float  # tokens per second
    last_update: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = self.capacity

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without blocking."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens, waiting if necessary."""
        while not self.try_acquire(tokens):
            wait_time = (tokens - self.tokens) / self.fill_rate
            await asyncio.sleep(min(wait_time, 0.1))


class ProviderLimiter:
    """Rate limiter for a single provider with RPM, TPM, and RPS limits."""

    def __init__(self, provider: str, model: str):
        limits = get_rate_limits(model)

        rpm = limits.get("rpm", DEFAULT_RATE_LIMITS["rpm"])
        tpm = limits.get("tpm", DEFAULT_RATE_LIMITS["tpm"])
        rps = limits.get("rps", DEFAULT_RATE_LIMITS["rps"])

        # Request bucket: refills at rpm/60 per second
        self._request_bucket = TokenBucket(
            capacity=float(rps * 2),  # Allow small burst
            fill_rate=rpm / 60.0
        )

        # Token bucket: refills at tpm/60 per second
        self._token_bucket = TokenBucket(
            capacity=float(min(tpm, 100_000)),  # Cap burst capacity
            fill_rate=tpm / 60.0
        )

        self.provider = provider
        self.model = model
        self.max_retries = limits.get("retries", 3)

    async def acquire(self, est_tokens: int = 1000) -> None:
        """Acquire permission to make a request."""
        await self._request_bucket.acquire(1.0)
        await self._token_bucket.acquire(float(est_tokens))


class ProviderPool:
    """Pool of LLM providers for load balancing and failover.

    Supports:
    - Weighted random selection
    - Round-robin selection
    - Per-provider rate limiting
    - Automatic failover on errors
    """

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        default_model: str = "grok-2",
    ):
        self.enabled = PROVIDER_POOL_ENABLED
        self.providers = providers or PROVIDER_POOL.copy()
        self.weights = weights or PROVIDER_WEIGHTS.copy()
        self.default_model = default_model

        # Filter to providers with available API keys
        self.providers = [p for p in self.providers if self._has_api_key(p)]
        if not self.providers:
            # Fallback to xai if no keys available
            self.providers = ["xai"]

        # Normalize weights
        total = sum(self.weights.get(p, 1.0) for p in self.providers)
        self.weights = {p: self.weights.get(p, 1.0) / total for p in self.providers}

        # Per-provider limiters
        self._limiters: Dict[str, ProviderLimiter] = {}
        self._stats: Dict[str, ProviderStats] = {p: ProviderStats() for p in self.providers}

        # Round-robin state
        self._rr_index = 0
        self._lock = asyncio.Lock()

    def _has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider."""
        env_var = API_KEY_ENV_VARS.get(provider, "")
        return bool(os.getenv(env_var, ""))

    def _get_limiter(self, provider: str, model: Optional[str] = None) -> ProviderLimiter:
        """Get or create rate limiter for provider."""
        model = model or self.default_model
        key = f"{provider}:{model}"
        if key not in self._limiters:
            self._limiters[key] = ProviderLimiter(provider, model)
        return self._limiters[key]

    def next_provider_weighted(self) -> str:
        """Select next provider using weighted random selection."""
        if not self.providers:
            return "xai"

        r = random.random()
        cumulative = 0.0
        for provider in self.providers:
            cumulative += self.weights.get(provider, 0.0)
            if r <= cumulative:
                return provider
        return self.providers[-1]

    def next_provider_roundrobin(self) -> str:
        """Select next provider using round-robin."""
        if not self.providers:
            return "xai"

        provider = self.providers[self._rr_index % len(self.providers)]
        self._rr_index += 1
        return provider

    def next_provider(self, strategy: str = "weighted") -> str:
        """Get next provider based on strategy.

        Args:
            strategy: "weighted" or "roundrobin"
        """
        if not self.enabled or len(self.providers) <= 1:
            return self.providers[0] if self.providers else "xai"

        if strategy == "roundrobin":
            return self.next_provider_roundrobin()
        return self.next_provider_weighted()

    async def acquire(
        self,
        provider: str,
        model: Optional[str] = None,
        est_tokens: int = 1000
    ) -> None:
        """Acquire rate limit permission for a provider."""
        limiter = self._get_limiter(provider, model)
        await limiter.acquire(est_tokens)

        # Update stats
        stats = self._stats.get(provider)
        if stats:
            stats.requests += 1
            stats.last_request_time = time.time()
            stats.tokens_used += est_tokens

    def record_failure(self, provider: str, is_rate_limit: bool = False) -> None:
        """Record a failure for a provider."""
        stats = self._stats.get(provider)
        if stats:
            stats.failures += 1
            if is_rate_limit:
                stats.rate_limited += 1

    def get_fallback_provider(self, failed_provider: str) -> Optional[str]:
        """Get a fallback provider after failure."""
        available = [p for p in self.providers if p != failed_provider]
        if not available:
            return None

        # Choose provider with fewest recent failures
        return min(available, key=lambda p: self._stats[p].failures)

    def get_stats(self) -> Dict[str, ProviderStats]:
        """Get statistics for all providers."""
        return self._stats.copy()

    def get_endpoint(self, provider: str) -> str:
        """Get API endpoint for provider."""
        return API_ENDPOINTS.get(provider, API_ENDPOINTS["xai"])

    def get_api_key(self, provider: str) -> str:
        """Get API key for provider from environment."""
        env_var = API_KEY_ENV_VARS.get(provider, "")
        return os.getenv(env_var, "")


# Singleton instance for global use
_pool: Optional[ProviderPool] = None


def get_provider_pool() -> ProviderPool:
    """Get or create the global provider pool instance."""
    global _pool
    if _pool is None:
        _pool = ProviderPool()
    return _pool
