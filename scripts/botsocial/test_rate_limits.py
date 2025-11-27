#!/usr/bin/env python3
"""
BotSocial Rate Limit Testing Script

Tests the actual rate limits for different API endpoints to optimize upload speed.

Usage:
    poetry run python3 scripts/botsocial/test_rate_limits.py \
        --token YOUR_USER_TOKEN \
        --admin-token YOUR_ADMIN_TOKEN
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE = "https://botsocial.mlai.au/api"


@dataclass
class RateLimitResult:
    """Results from rate limit testing."""
    
    endpoint: str
    requests_sent: int
    successful: int
    rate_limited: int
    errors: int
    duration_seconds: float
    effective_rate: float  # requests per minute
    hit_limit_at: Optional[int] = None  # request number where limit hit


class RateLimitTester:
    """Tests rate limits for different endpoints."""
    
    def __init__(self, user_token: str, admin_token: str) -> None:
        """Initialize tester.
        
        Args:
            user_token: Regular user API token.
            admin_token: Admin API token.
        """
        self.user_token = user_token
        self.admin_token = admin_token
        self.results: list[RateLimitResult] = []
    
    def test_endpoint(
        self,
        endpoint: str,
        token: str,
        data_generator: callable,
        num_requests: int = 20,
        delay: float = 0.5
    ) -> RateLimitResult:
        """Test rate limit for a specific endpoint.
        
        Args:
            endpoint: API endpoint (without /api prefix).
            token: Token to use.
            data_generator: Function that returns request data for each call.
            num_requests: Number of requests to send.
            delay: Delay between requests in seconds.
        
        Returns:
            RateLimitResult object.
        """
        logger.info(f"\nTesting {endpoint} with {num_requests} requests at {1/delay:.1f} req/sec...")
        
        successful = 0
        rate_limited = 0
        errors = 0
        hit_limit_at = None
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                data = data_generator(i, token)
                response = requests.post(
                    f"{API_BASE}/{endpoint}",
                    json=data,
                    timeout=10
                )
                
                if response.status_code in [200, 204]:
                    successful += 1
                    logger.debug(f"  [{i+1}/{num_requests}] ✓ Success")
                elif response.status_code == 429:
                    rate_limited += 1
                    if hit_limit_at is None:
                        hit_limit_at = i + 1
                    logger.warning(f"  [{i+1}/{num_requests}] ⚠ Rate limited!")
                    
                    # Check rate limit headers
                    remaining = response.headers.get('x-ratelimit-remaining')
                    clear_time = response.headers.get('x-ratelimit-clear')
                    if remaining or clear_time:
                        logger.info(f"    Remaining: {remaining}, Clear time: {clear_time}")
                else:
                    errors += 1
                    logger.warning(f"  [{i+1}/{num_requests}] ✗ Error: {response.status_code}")
                
                # Small delay between requests
                if i < num_requests - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                errors += 1
                logger.error(f"  [{i+1}/{num_requests}] Exception: {e}")
        
        duration = time.time() - start_time
        effective_rate = (successful / duration) * 60 if duration > 0 else 0
        
        result = RateLimitResult(
            endpoint=endpoint,
            requests_sent=num_requests,
            successful=successful,
            rate_limited=rate_limited,
            errors=errors,
            duration_seconds=duration,
            effective_rate=effective_rate,
            hit_limit_at=hit_limit_at
        )
        
        self.results.append(result)
        
        logger.info(f"\nResults for {endpoint}:")
        logger.info(f"  Successful: {successful}/{num_requests}")
        logger.info(f"  Rate limited: {rate_limited}/{num_requests}")
        logger.info(f"  Errors: {errors}/{num_requests}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Effective rate: {effective_rate:.1f} req/min")
        if hit_limit_at:
            logger.info(f"  Hit limit at request #{hit_limit_at}")
        
        return result
    
    def test_notes_creation(self, num_requests: int = 15, delay: float = 2.0) -> RateLimitResult:
        """Test notes/create endpoint (most critical for uploads).
        
        Args:
            num_requests: Number of test posts to create.
            delay: Delay between posts in seconds.
        
        Returns:
            RateLimitResult object.
        """
        def data_gen(i: int, token: str) -> dict[str, Any]:
            return {
                "i": token,
                "text": f"Rate limit test post #{i+1} - {time.time()}",
                "visibility": "followers"  # Use followers to avoid cluttering timeline
            }
        
        return self.test_endpoint(
            "notes/create",
            self.user_token,
            data_gen,
            num_requests,
            delay
        )
    
    def test_profile_updates(self, num_requests: int = 20, delay: float = 1.0) -> RateLimitResult:
        """Test i/update endpoint (profile updates).
        
        Args:
            num_requests: Number of update requests.
            delay: Delay between requests in seconds.
        
        Returns:
            RateLimitResult object.
        """
        def data_gen(i: int, token: str) -> dict[str, Any]:
            return {
                "i": token,
                "description": f"Rate test {i+1} - {time.time()}"
            }
        
        return self.test_endpoint(
            "i/update",
            self.user_token,
            data_gen,
            num_requests,
            delay
        )
    
    def test_admin_endpoints(self, num_requests: int = 20, delay: float = 0.5) -> RateLimitResult:
        """Test admin endpoints (higher limits expected).
        
        Args:
            num_requests: Number of requests.
            delay: Delay between requests in seconds.
        
        Returns:
            RateLimitResult object.
        """
        def data_gen(i: int, token: str) -> dict[str, Any]:
            return {
                "i": token
            }
        
        return self.test_endpoint(
            "admin/meta",
            self.admin_token,
            data_gen,
            num_requests,
            delay
        )
    
    def test_follows(self, num_requests: int = 20, delay: float = 1.0) -> RateLimitResult:
        """Test following/create endpoint.
        
        Note: This won't actually create follows (needs valid user IDs).
        We'll just measure rate limit behavior.
        
        Args:
            num_requests: Number of requests.
            delay: Delay between requests in seconds.
        
        Returns:
            RateLimitResult object.
        """
        def data_gen(i: int, token: str) -> dict[str, Any]:
            return {
                "i": token,
                "userId": "fake_id_for_testing"  # Will fail but tests rate limit
            }
        
        return self.test_endpoint(
            "following/create",
            self.user_token,
            data_gen,
            num_requests,
            delay
        )
    
    def run_comprehensive_test(self) -> None:
        """Run comprehensive rate limit tests across all endpoints."""
        logger.info("="*70)
        logger.info("BotSocial Rate Limit Comprehensive Test")
        logger.info("="*70)
        
        # Test 1: Admin endpoints (should have higher limits)
        logger.info("\n[TEST 1] Admin Endpoints")
        logger.info("-"*70)
        self.test_admin_endpoints(num_requests=30, delay=0.5)
        time.sleep(5)  # Wait for rate limit window to reset
        
        # Test 2: Profile updates
        logger.info("\n[TEST 2] Profile Updates (i/update)")
        logger.info("-"*70)
        self.test_profile_updates(num_requests=20, delay=1.0)
        time.sleep(5)
        
        # Test 3: Notes creation - Conservative (documented as 5/min)
        logger.info("\n[TEST 3] Notes Creation - Conservative (2s delay)")
        logger.info("-"*70)
        self.test_notes_creation(num_requests=10, delay=2.0)
        time.sleep(60)  # Wait a full minute
        
        # Test 4: Notes creation - Aggressive
        logger.info("\n[TEST 4] Notes Creation - Aggressive (0.5s delay)")
        logger.info("-"*70)
        self.test_notes_creation(num_requests=10, delay=0.5)
        time.sleep(60)
        
        # Test 5: Follow operations
        logger.info("\n[TEST 5] Follow Operations")
        logger.info("-"*70)
        # Note: This will generate errors (fake user ID) but tests rate limits
        # We'll catch errors gracefully
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print summary of all test results."""
        logger.info("\n" + "="*70)
        logger.info("RATE LIMIT TEST SUMMARY")
        logger.info("="*70)
        
        for result in self.results:
            logger.info(f"\n{result.endpoint}:")
            logger.info(f"  Successful: {result.successful}/{result.requests_sent} " +
                       f"({result.successful/result.requests_sent*100:.1f}%)")
            logger.info(f"  Rate limited: {result.rate_limited}")
            logger.info(f"  Effective rate: {result.effective_rate:.1f} req/min")
            if result.hit_limit_at:
                logger.info(f"  Hit limit at: request #{result.hit_limit_at}")
        
        # Recommendations
        logger.info("\n" + "="*70)
        logger.info("RECOMMENDATIONS FOR UPLOAD")
        logger.info("="*70)
        
        # Find notes/create results
        notes_results = [r for r in self.results if 'notes/create' in r.endpoint]
        if notes_results:
            # Take the most aggressive test that succeeded
            max_rate = max(r.effective_rate for r in notes_results if r.rate_limited == 0)
            safe_rate = max_rate * 0.8  # 80% of max for safety margin
            safe_delay = 60 / safe_rate if safe_rate > 0 else 12
            
            logger.info(f"\nFor notes/create (posts):")
            logger.info(f"  Maximum observed: {max_rate:.1f} req/min")
            logger.info(f"  Recommended safe rate: {safe_rate:.1f} req/min")
            logger.info(f"  Recommended delay: {safe_delay:.2f} seconds between posts")
            logger.info(f"  Time for 1363 posts: {1363 * safe_delay / 60:.1f} minutes " +
                       f"({1363 * safe_delay / 3600:.2f} hours)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test BotSocial rate limits"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="User API token"
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        required=True,
        help="Admin API token"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (fewer requests)"
    )
    
    args = parser.parse_args()
    
    tester = RateLimitTester(args.token, args.admin_token)
    
    if args.quick:
        # Quick test - just notes creation
        logger.info("Running quick test on notes/create endpoint...")
        tester.test_notes_creation(num_requests=10, delay=1.0)
        time.sleep(5)
        tester.test_notes_creation(num_requests=10, delay=0.5)
        tester.print_summary()
    else:
        # Full comprehensive test
        tester.run_comprehensive_test()


if __name__ == "__main__":
    main()

