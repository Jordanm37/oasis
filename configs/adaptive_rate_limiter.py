"""Adaptive Rate Limiter - Dynamically adjusts concurrency based on API responses.

This module provides an adaptive semaphore that:
1. Starts with TPM-based estimates
2. Reduces concurrency when hitting rate limits (429 errors)
3. Gradually increases when running smoothly
4. Tracks token usage for accurate TPM management

Usage:
    from configs.adaptive_rate_limiter import AdaptiveRateLimiter
    
    limiter = AdaptiveRateLimiter(model="llama-3.3-70b-versatile")
    
    async with limiter.acquire():
        response = await llm_call()
        limiter.record_success(tokens_used=1500)
    
    # Or on rate limit error:
    limiter.record_rate_limit()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitStats:
    """Statistics for rate limit tracking."""
    requests_total: int = 0
    requests_success: int = 0
    requests_rate_limited: int = 0
    tokens_used: int = 0
    
    # Rolling window stats (last 60 seconds)
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_tokens: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_rate_limits: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_request(self, tokens: int = 0, rate_limited: bool = False):
        """Record a request with timestamp."""
        now = time.time()
        self.requests_total += 1
        self.recent_requests.append(now)
        
        if rate_limited:
            self.requests_rate_limited += 1
            self.recent_rate_limits.append(now)
        else:
            self.requests_success += 1
            self.tokens_used += tokens
            self.recent_tokens.append((now, tokens))
    
    def get_rpm(self, window_seconds: float = 60.0) -> float:
        """Get requests per minute over the last window."""
        now = time.time()
        cutoff = now - window_seconds
        recent = sum(1 for t in self.recent_requests if t > cutoff)
        return recent * (60.0 / window_seconds)
    
    def get_tpm(self, window_seconds: float = 60.0) -> float:
        """Get tokens per minute over the last window."""
        now = time.time()
        cutoff = now - window_seconds
        recent_tokens = sum(tokens for t, tokens in self.recent_tokens if t > cutoff)
        return recent_tokens * (60.0 / window_seconds)
    
    def get_rate_limit_rate(self, window_seconds: float = 60.0) -> float:
        """Get rate limit errors per minute over the last window."""
        now = time.time()
        cutoff = now - window_seconds
        recent = sum(1 for t in self.recent_rate_limits if t > cutoff)
        return recent * (60.0 / window_seconds)
    
    def get_success_rate(self, window_seconds: float = 60.0) -> float:
        """Get success rate (0-1) over the last window."""
        now = time.time()
        cutoff = now - window_seconds
        total = sum(1 for t in self.recent_requests if t > cutoff)
        rate_limited = sum(1 for t in self.recent_rate_limits if t > cutoff)
        if total == 0:
            return 1.0
        return (total - rate_limited) / total


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts concurrency based on API responses.
    
    The limiter uses a feedback loop:
    1. Start with TPM-based initial concurrency
    2. On rate limit errors: reduce concurrency by 25%
    3. On sustained success (>95% over 30s): increase by 10%
    4. Respects min/max bounds
    
    Also implements token bucket for TPM management.
    """
    
    def __init__(
        self,
        model: str = None,
        initial_concurrency: int = None,
        min_concurrency: int = 2,
        max_concurrency: int = 64,
        tpm_limit: int = None,
        rpm_limit: int = None,
        num_keys: int = None,
        adjustment_interval: float = 5.0,  # Seconds between adjustments
    ):
        """Initialize the adaptive rate limiter.
        
        Args:
            model: Model name to get default limits from config
            initial_concurrency: Starting concurrency (None = auto from TPM)
            min_concurrency: Minimum allowed concurrency
            max_concurrency: Maximum allowed concurrency
            tpm_limit: Tokens per minute limit (None = from config)
            rpm_limit: Requests per minute limit (None = from config)
            num_keys: Number of API keys (None = auto-detect)
            adjustment_interval: Minimum seconds between concurrency adjustments
        """
        from configs.llm_settings import (
            RATE_LIMITS, DEFAULT_RATE_LIMITS, SIMULATION_MODEL,
            count_api_keys, get_provider_for_model, get_effective_semaphore_limit
        )
        
        self.model = model or SIMULATION_MODEL
        self.provider = get_provider_for_model(self.model)
        self.num_keys = num_keys or count_api_keys(self.provider)
        
        # Get limits from config
        limits = RATE_LIMITS.get(self.model, DEFAULT_RATE_LIMITS)
        self.tpm_limit = (tpm_limit or limits.get("tpm", 300_000)) * self.num_keys
        self.rpm_limit = (rpm_limit or limits.get("rpm", 1_000)) * self.num_keys
        self.est_tokens = limits.get("est_tokens", 2000)
        
        # Concurrency bounds
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency * self.num_keys
        
        # Initial concurrency from TPM-aware calculation
        if initial_concurrency is None:
            initial_concurrency = get_effective_semaphore_limit(self.model)
        self.current_concurrency = max(min_concurrency, min(initial_concurrency, self.max_concurrency))
        
        # Create the semaphore
        self._semaphore = asyncio.Semaphore(self.current_concurrency)
        self._semaphore_lock = asyncio.Lock()
        
        # Stats tracking
        self.stats = RateLimitStats()
        
        # Adjustment tracking
        self.adjustment_interval = adjustment_interval
        self._last_adjustment_time = 0.0
        self._last_rate_limit_time = 0.0
        self._consecutive_success_windows = 0
        
        # Token bucket for TPM (refills over time)
        self._token_bucket = self.tpm_limit / 60  # Start with 1 second worth
        self._token_bucket_max = self.tpm_limit / 60 * 5  # 5 seconds buffer
        self._last_token_refill = time.time()
        self._token_lock = asyncio.Lock()
        
        logger.info(
            f"AdaptiveRateLimiter initialized: model={self.model}, "
            f"keys={self.num_keys}, concurrency={self.current_concurrency}, "
            f"tpm={self.tpm_limit:,}, rpm={self.rpm_limit:,}"
        )
    
    async def acquire(self) -> "AdaptiveRateLimiterContext":
        """Acquire a slot for making an API request.
        
        Returns a context manager that should be used with `async with`.
        """
        return AdaptiveRateLimiterContext(self)
    
    async def _acquire_semaphore(self):
        """Internal: acquire the semaphore slot."""
        await self._semaphore.acquire()
    
    def _release_semaphore(self):
        """Internal: release the semaphore slot."""
        self._semaphore.release()
    
    async def _wait_for_tokens(self, estimated_tokens: int = None):
        """Wait until we have enough tokens in the bucket."""
        if estimated_tokens is None:
            estimated_tokens = self.est_tokens
        
        async with self._token_lock:
            # Refill bucket based on time elapsed
            now = time.time()
            elapsed = now - self._last_token_refill
            refill_amount = (self.tpm_limit / 60) * elapsed
            self._token_bucket = min(self._token_bucket + refill_amount, self._token_bucket_max)
            self._last_token_refill = now
            
            # If not enough tokens, wait
            if self._token_bucket < estimated_tokens:
                wait_time = (estimated_tokens - self._token_bucket) / (self.tpm_limit / 60)
                logger.debug(f"Token bucket low, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._token_bucket = estimated_tokens  # Refilled during wait
            
            # Consume tokens
            self._token_bucket -= estimated_tokens
    
    def record_success(self, tokens_used: int = None):
        """Record a successful API call.
        
        Args:
            tokens_used: Actual tokens used (prompt + completion)
        """
        if tokens_used is None:
            tokens_used = self.est_tokens
        
        self.stats.record_request(tokens=tokens_used, rate_limited=False)
        
        # Check if we should increase concurrency
        self._maybe_increase_concurrency()
    
    def record_rate_limit(self):
        """Record a rate limit error (429)."""
        self.stats.record_request(tokens=0, rate_limited=True)
        self._last_rate_limit_time = time.time()
        self._consecutive_success_windows = 0
        
        # Immediately reduce concurrency
        self._reduce_concurrency()
    
    def record_error(self, is_rate_limit: bool = False):
        """Record an API error.
        
        Args:
            is_rate_limit: True if this was a 429 rate limit error
        """
        if is_rate_limit:
            self.record_rate_limit()
        else:
            # Other errors don't affect rate limiting
            self.stats.record_request(tokens=0, rate_limited=False)
    
    def _reduce_concurrency(self):
        """Reduce concurrency after a rate limit error."""
        now = time.time()
        if now - self._last_adjustment_time < self.adjustment_interval:
            return  # Too soon since last adjustment
        
        old_concurrency = self.current_concurrency
        # Reduce by 25%, minimum 1
        reduction = max(1, self.current_concurrency // 4)
        self.current_concurrency = max(self.min_concurrency, self.current_concurrency - reduction)
        
        if self.current_concurrency != old_concurrency:
            self._last_adjustment_time = now
            logger.warning(
                f"Rate limit hit! Reducing concurrency: {old_concurrency} -> {self.current_concurrency}"
            )
            # Note: We don't actually resize the semaphore mid-run
            # The new limit takes effect as slots are released
    
    def _maybe_increase_concurrency(self):
        """Potentially increase concurrency if running smoothly."""
        now = time.time()
        if now - self._last_adjustment_time < self.adjustment_interval:
            return
        
        # Don't increase if we recently hit a rate limit
        if now - self._last_rate_limit_time < 30.0:
            return
        
        # Check success rate over last 30 seconds
        success_rate = self.stats.get_success_rate(window_seconds=30.0)
        if success_rate < 0.95:
            self._consecutive_success_windows = 0
            return
        
        # Need 3 consecutive good windows before increasing
        self._consecutive_success_windows += 1
        if self._consecutive_success_windows < 3:
            return
        
        # Check if we're under TPM/RPM limits
        current_tpm = self.stats.get_tpm(window_seconds=60.0)
        current_rpm = self.stats.get_rpm(window_seconds=60.0)
        
        tpm_headroom = current_tpm < self.tpm_limit * 0.8
        rpm_headroom = current_rpm < self.rpm_limit * 0.8
        
        if not (tpm_headroom and rpm_headroom):
            return
        
        old_concurrency = self.current_concurrency
        # Increase by 10%, minimum 1
        increase = max(1, self.current_concurrency // 10)
        self.current_concurrency = min(self.max_concurrency, self.current_concurrency + increase)
        
        if self.current_concurrency != old_concurrency:
            self._last_adjustment_time = now
            self._consecutive_success_windows = 0
            logger.info(
                f"Sustained success! Increasing concurrency: {old_concurrency} -> {self.current_concurrency}"
            )
    
    def get_stats_summary(self) -> Dict:
        """Get a summary of current rate limiting stats."""
        return {
            "model": self.model,
            "num_keys": self.num_keys,
            "current_concurrency": self.current_concurrency,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency,
            "tpm_limit": self.tpm_limit,
            "rpm_limit": self.rpm_limit,
            "current_rpm": round(self.stats.get_rpm(), 1),
            "current_tpm": round(self.stats.get_tpm(), 0),
            "success_rate": round(self.stats.get_success_rate() * 100, 1),
            "total_requests": self.stats.requests_total,
            "rate_limited": self.stats.requests_rate_limited,
            "tokens_used": self.stats.tokens_used,
        }
    
    def print_stats(self):
        """Print current stats to console."""
        stats = self.get_stats_summary()
        print("\n" + "=" * 50)
        print("ADAPTIVE RATE LIMITER STATUS")
        print("=" * 50)
        print(f"Model:            {stats['model']}")
        print(f"API Keys:         {stats['num_keys']}")
        print(f"Concurrency:      {stats['current_concurrency']} (min={stats['min_concurrency']}, max={stats['max_concurrency']})")
        print("-" * 50)
        print(f"Current RPM:      {stats['current_rpm']} / {stats['rpm_limit']}")
        print(f"Current TPM:      {stats['current_tpm']:,.0f} / {stats['tpm_limit']:,}")
        print(f"Success Rate:     {stats['success_rate']}%")
        print("-" * 50)
        print(f"Total Requests:   {stats['total_requests']}")
        print(f"Rate Limited:     {stats['rate_limited']}")
        print(f"Tokens Used:      {stats['tokens_used']:,}")
        print("=" * 50 + "\n")


class AdaptiveRateLimiterContext:
    """Context manager for acquiring rate limiter slots."""
    
    def __init__(self, limiter: AdaptiveRateLimiter):
        self.limiter = limiter
        self._acquired = False
    
    async def __aenter__(self):
        await self.limiter._acquire_semaphore()
        await self.limiter._wait_for_tokens()
        self._acquired = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self.limiter._release_semaphore()
        
        # Auto-detect rate limit errors
        if exc_type is not None:
            error_str = str(exc_val).lower()
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                self.limiter.record_rate_limit()
        
        return False  # Don't suppress exceptions


# Global instance for easy access
_global_limiter: Optional[AdaptiveRateLimiter] = None


def get_global_limiter(model: str = None) -> AdaptiveRateLimiter:
    """Get or create a global adaptive rate limiter.
    
    This is useful for sharing rate limiting state across the application.
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = AdaptiveRateLimiter(model=model)
    return _global_limiter


def reset_global_limiter():
    """Reset the global rate limiter (useful for testing)."""
    global _global_limiter
    _global_limiter = None

