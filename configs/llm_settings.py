"""Centralized LLM configuration - import variables where needed.

Usage:
    from configs.llm_settings import PERSONA_MODEL, SIMULATION_MODEL, RATE_LIMITS
"""
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PROVIDER & MODEL SELECTION
# =============================================================================
# Options: "xai", "openai", "gemini", "groq"
PERSONA_PROVIDER = "groq"
SIMULATION_PROVIDER = "groq"

# Model names (string, used in conditionals)
PERSONA_MODEL = "llama-3.3-70b-versatile"
SIMULATION_MODEL = "llama-3.3-70b-versatile"

# =============================================================================
# FALLBACK MODELS (used when primary model hits rate limits)
# Order matters - will try each in sequence on rate limit errors
# 
# NOTE: Llama-4 models output XML-style tool calls which are automatically
# converted to JSON by the Llama4ToolAdapter (oasis/llama4_tool_adapter.py).
# This adapter wraps Llama-4 backends transparently in model_provider.py.
# =============================================================================
FALLBACK_MODELS = [
    # Primary model is tried first (from SIMULATION_MODEL)
    # Then these fallbacks in order (fastest first for throughput):
    "meta-llama/llama-4-scout-17b-16e-instruct",       # 750 t/s - fastest Llama-4
    "meta-llama/llama-4-maverick-17b-128e-instruct",   # 600 t/s - good Llama-4
    "llama-3.1-8b-instant",                            # 560 t/s - fast, high max tokens
    "qwen/qwen3-32b",                                  # 400 t/s - good balance
    "llama-3.3-70b-versatile",                         # Best quality (slowest)
]

# Enable/disable fallback behavior
FALLBACK_ENABLED = True
FALLBACK_MAX_RETRIES = 3  # Retries per model before moving to next

# =============================================================================
# TEMPERATURE & TOKEN LIMITS
# =============================================================================
PERSONA_TEMPERATURE = 0.3
# NOTE: Groq recommends lower temperature (0.2-0.5) for tool calls to reduce
# hallucinations and malformed JSON. Higher temps cause tool call format errors.
SIMULATION_TEMPERATURE = 0.3  # Lowered from 0.7 for Groq tool call reliability

# Max tokens - kept low for Groq model compatibility (some have 8192 limit)
PERSONA_MAX_TOKENS = 800
SIMULATION_MAX_TOKENS = 4096

# Per-model max completion tokens (from Groq docs)
# Context window is 131,072 for all models
MODEL_MAX_TOKENS = {
    "llama-3.3-70b-versatile": 32_768,      # Max completion: 32,768
    "llama-3.1-8b-instant": 131_072,         # Max completion: 131,072 (same as context)
    "qwen/qwen3-32b": 40_960,                # Max completion: 40,960
    "meta-llama/llama-4-maverick-17b-128e-instruct": 8_192,  # Max completion: 8,192
    "meta-llama/llama-4-scout-17b-16e-instruct": 8_192,      # Max completion: 8,192
}

# Model context windows (all 131,072 tokens)
MODEL_CONTEXT_WINDOWS = {
    "llama-3.3-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
    "qwen/qwen3-32b": 131_072,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 131_072,
    "meta-llama/llama-4-scout-17b-16e-instruct": 131_072,
}

# Model speeds (tokens/sec) for reference
MODEL_SPEEDS = {
    "llama-3.1-8b-instant": 560,             # Fastest
    "meta-llama/llama-4-scout-17b-16e-instruct": 750,  # Very fast
    "meta-llama/llama-4-maverick-17b-128e-instruct": 600,
    "qwen/qwen3-32b": 400,
    "llama-3.3-70b-versatile": 280,          # Slowest but highest quality
}

# =============================================================================
# IMPUTATION SETTINGS (RAG LLM imputer)
# =============================================================================
IMPUTATION_PROVIDER = "groq"
IMPUTATION_MODEL = "llama-3.3-70b-versatile"
IMPUTATION_TEMPERATURE = 0.35
IMPUTATION_MAX_TOKENS = 512

# RAG imputer runtime knobs - tuned for Groq rate limits
RAG_IMPUTER_MODE = "background"  # "off" | "background" | "sync"
RAG_IMPUTER_MAX_WORKERS = 4      # Can handle more with 100 RPM
RAG_IMPUTER_BATCH_SIZE = 16
RAG_IMPUTER_STATIC_BANK = "data/label_tokens_static_bank.yaml"
RAG_IMPUTER_RETRIEVAL_TOP_K = 4  # Number of few-shot examples from vector DB

# =============================================================================
# STRUCTURED OUTPUT (per-model)
# Models that support response_format: {"type": "json_object"}
# =============================================================================
STRUCTURED_OUTPUT_MODELS = {
    "grok-2",
    "grok-4-fast-non-reasoning",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-nano-2025-08-07",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

# =============================================================================
# RATE LIMITS (per-model) - Groq Developer Plan (Updated Nov 2025)
# rpm = requests per minute
# rpd = requests per day
# tpm = tokens per minute (CRITICAL - often the real bottleneck!)
# rps = requests per second (effective ceiling = rpm/60)
# retries = max retry attempts on rate limit
# est_tokens = estimated tokens per request (for TPM budgeting)
# =============================================================================
RATE_LIMITS = {
    # xAI models (high throughput)
    "grok-2": {"rpm": 480, "rpd": 1_000_000, "tpm": 4_000_000, "rps": 8, "retries": 5, "est_tokens": 2000},
    "grok-4-fast-non-reasoning": {"rpm": 480, "rpd": 1_000_000, "tpm": 4_000_000, "rps": 8, "retries": 5, "est_tokens": 2000},
    # OpenAI models
    "gpt-4o": {"rpm": 500, "rpd": 500_000, "tpm": 150_000, "rps": 10, "retries": 5, "est_tokens": 2000},
    "gpt-4o-mini": {"rpm": 1000, "rpd": 1_000_000, "tpm": 200_000, "rps": 20, "retries": 5, "est_tokens": 1500},
    # Gemini models (high throughput)
    "gemini-2.5-flash": {"rpm": 1000, "rpd": 1_000_000, "tpm": 4_000_000, "rps": 15, "retries": 5, "est_tokens": 2000},
    "gemini-2.5-flash-lite": {"rpm": 2000, "rpd": 2_000_000, "tpm": 4_000_000, "rps": 30, "retries": 5, "est_tokens": 1500},
    
    # ==========================================================================
    # GROQ MODELS - Developer Plan (from official docs Nov 2025)
    # Key insight: TPM is often the bottleneck, not RPM!
    # With 300K TPM and ~2000 tokens/request, effective limit is ~150 req/min
    # ==========================================================================
    
    # llama-3.3-70b: 1K RPM, 500K RPD, 300K TPM (best quality, slowest inference)
    # TPM-limited: 300K / ~2000 tokens = ~150 effective RPM
    "llama-3.3-70b-versatile": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 300_000, 
        "rps": 16, "retries": 5, "est_tokens": 2000,
        "effective_rpm": 150,  # TPM-limited!
    },
    
    # llama-3.1-8b: 1K RPM, 500K RPD, 250K TPM (fast inference, good for volume)
    # TPM-limited: 250K / ~1500 tokens = ~166 effective RPM
    "llama-3.1-8b-instant": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 250_000, 
        "rps": 16, "retries": 5, "est_tokens": 1500,
        "effective_rpm": 166,  # TPM-limited!
    },
    
    # qwen3-32b: Assume similar limits to llama models
    "qwen/qwen3-32b": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 300_000, 
        "rps": 16, "retries": 5, "est_tokens": 2000,
        "effective_rpm": 150,
    },
    
    # llama-4-maverick: 1K RPM, 500K RPD, 300K TPM
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 300_000, 
        "rps": 16, "retries": 5, "est_tokens": 2000,
        "effective_rpm": 150,
    },
    
    # llama-4-scout: 1K RPM, 500K RPD, 300K TPM (fastest llama-4)
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 300_000, 
        "rps": 16, "retries": 5, "est_tokens": 2000,
        "effective_rpm": 150,
    },
    
    # Kimi K2 models (if using)
    "moonshotai/kimi-k2-instruct": {
        "rpm": 1_000, "rpd": 500_000, "tpm": 250_000,
        "rps": 16, "retries": 5, "est_tokens": 2000,
        "effective_rpm": 125,
    },
}

# Default rate limits for unknown models (conservative)
DEFAULT_RATE_LIMITS = {"rpm": 30, "rpd": 10_000, "tpm": 6_000, "rps": 0.5, "retries": 8, "est_tokens": 2000}

# =============================================================================
# CONCURRENCY - TPM-Aware Settings
# =============================================================================
# Key insight: TPM (tokens per minute) is often the real bottleneck!
# 
# With Groq's 300K TPM and ~2000 tokens/request:
#   - Effective RPM = 300K / 2000 = 150 requests/minute = 2.5 RPS
#   - With 3 API keys: 450 effective RPM = 7.5 RPS
#
# Semaphore should be set to allow enough concurrent requests to saturate
# the API without overwhelming it. Rule of thumb:
#   - Semaphore = effective_RPS * avg_latency_seconds
#   - With 2.5 RPS and ~2s latency: semaphore = 5 per key
#   - Adding buffer for variance: ~8-16 per key is safe

# Base limits per API key (TPM-aware, conservative)
SEMAPHORE_LIMIT_PER_KEY = 2   # Reduced to 2 (ultra-conservative) to prevent TPM limits
BATCH_SIZE_PER_KEY = 2        # Match semaphore

# Estimated tokens per request (used for TPM budgeting)
EST_TOKENS_PER_REQUEST = 2000  # Prompt + completion average

def get_effective_rps(model: str, num_keys: int = 1) -> float:
    """Calculate effective requests per second based on TPM limits.
    
    Args:
        model: Model name to get rate limits for
        num_keys: Number of API keys available
        
    Returns:
        Effective RPS considering TPM as the bottleneck
    """
    limits = RATE_LIMITS.get(model, DEFAULT_RATE_LIMITS)
    tpm = limits.get("tpm", 6_000)
    est_tokens = limits.get("est_tokens", EST_TOKENS_PER_REQUEST)
    
    # Effective RPM based on TPM
    effective_rpm = tpm / est_tokens
    
    # Cap at actual RPM limit
    rpm_limit = limits.get("rpm", 30)
    effective_rpm = min(effective_rpm, rpm_limit)
    
    # Scale by number of keys and convert to RPS
    return (effective_rpm * num_keys) / 60


def get_effective_semaphore_limit(model: str = None) -> int:
    """Get semaphore limit based on TPM-aware effective RPS.
    
    The semaphore should be large enough to keep the API saturated
    but not so large that we queue up requests that will be rate-limited.
    
    Formula: semaphore = effective_RPS * avg_latency * buffer
    """
    if model is None:
        model = SIMULATION_MODEL
    
    num_keys = count_api_keys(SIMULATION_PROVIDER) if MULTI_KEY_ENABLED else 1
    effective_rps = get_effective_rps(model, num_keys)
    
    # Assume ~2-3 second average latency for Groq
    avg_latency = 2.5
    buffer = 1.5  # Allow some headroom
    
    calculated = int(effective_rps * avg_latency * buffer)
    
    # Clamp to reasonable bounds
    min_limit = 4
    max_limit = SEMAPHORE_LIMIT_PER_KEY * num_keys
    
    return max(min_limit, min(calculated, max_limit))


def get_effective_batch_size(model: str = None) -> int:
    """Get batch size matched to semaphore for smooth flow."""
    return get_effective_semaphore_limit(model)


def get_optimal_delay_between_requests(model: str = None) -> float:
    """Calculate optimal delay between requests to avoid rate limits.
    
    Returns:
        Delay in seconds between requests
    """
    if model is None:
        model = SIMULATION_MODEL
    
    num_keys = count_api_keys(SIMULATION_PROVIDER) if MULTI_KEY_ENABLED else 1
    effective_rps = get_effective_rps(model, num_keys)
    
    if effective_rps <= 0:
        return 1.0  # Conservative fallback
    
    return 1.0 / effective_rps


# Legacy aliases for backwards compatibility
SEMAPHORE_LIMIT = SEMAPHORE_LIMIT_PER_KEY
BATCH_SIZE = BATCH_SIZE_PER_KEY

# =============================================================================
# PROVIDER POOL (for distributed persona generation)
# Distributes requests across multiple providers for load balancing
# =============================================================================
PROVIDER_POOL_ENABLED = False
PROVIDER_POOL = ["xai", "openai", "gemini", "groq"]
PROVIDER_WEIGHTS = {"xai": 0.4, "openai": 0.3, "gemini": 0.2, "groq": 0.1}

# =============================================================================
# REPORTING
# =============================================================================
AUTO_REPORT = False  # Disable auto-report for faster iteration

# =============================================================================
# API ENDPOINTS
# =============================================================================
API_ENDPOINTS = {
    "xai": "https://api.x.ai/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "groq": "https://api.groq.com/openai/v1",
}

# =============================================================================
# API KEY ENVIRONMENT VARIABLES
# =============================================================================
# Primary API key env var for each provider
API_KEY_ENV_VARS = {
    "xai": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

# =============================================================================
# MULTI-KEY SUPPORT
# =============================================================================
# Enable using multiple API keys per provider for increased throughput
# Keys are auto-discovered: {PROVIDER}_API_KEY, {PROVIDER}_API_KEY_2, etc.
MULTI_KEY_ENABLED = True
MULTI_KEY_MAX_KEYS = 12  # Max keys to check per provider (up to 11 for Groq)

# Explicit key patterns per provider (auto-generated if not specified)
# Format: list of env var names to check in order
API_KEY_PATTERNS = {
    "groq": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5", "GROQ_API_KEY_6", "GROQ_API_KEY_7", "GROQ_API_KEY_8", "GROQ_API_KEY_9", "GROQ_API_KEY_10", "GROQ_API_KEY_11", "GROQ_API_KEY_12", "GROQ_API_KEY_13", "GROQ_API_KEY_14", "GROQ_API_KEY_15", "GROQ_API_KEY_16", "GROQ_API_KEY_17", "GROQ_API_KEY_18", "GROQ_API_KEY_19", "GROQ_API_KEY_20", "GROQ_API_KEY_21", "GROQ_API_KEY_22", "GROQ_API_KEY_23", "GROQ_API_KEY_24", "GROQ_API_KEY_25", "GROQ_API_KEY_26", "GROQ_API_KEY_27", "GROQ_API_KEY_28", "GROQ_API_KEY_29", "GROQ_API_KEY_30", "GROQ_API_KEY_31", "GROQ_API_KEY_32", "GROQ_API_KEY_33", "GROQ_API_KEY_34", "GROQ_API_KEY_35", "GROQ_API_KEY_36", "GROQ_API_KEY_37", "GROQ_API_KEY_38", "GROQ_API_KEY_39", "GROQ_API_KEY_40", "GROQ_API_KEY_41", "GROQ_API_KEY_42", "GROQ_API_KEY_43", "GROQ_API_KEY_44"],
    "openai": ["OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"],
    "xai": ["XAI_API_KEY", "XAI_API_KEY_2", "XAI_API_KEY_3", "XAI_API_KEY_4", "XAI_API_KEY_5"],
    "gemini": ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5", "GEMINI_API_KEY_6", "GEMINI_API_KEY_7", "GEMINI_API_KEY_8"],
}

# =============================================================================
# MULTI-MODEL PARALLEL DISTRIBUTION
# =============================================================================
# Instead of using one model with fallback, distribute load across multiple
# models AND providers in parallel. Each provider has separate rate limits!
#
# With 6 API keys across 4 providers, we multiply our total capacity:
# - Groq (3 keys): 3 × 300K TPM = 900K TPM
# - xAI (1 key): 4M TPM  
# - Gemini (1 key): 4M TPM
# - OpenAI (1 key): 200K TPM
# Total: ~9M TPM vs 300K with single provider!
MULTI_MODEL_ENABLED = True
MULTI_PROVIDER_ENABLED = True  # Use multiple providers (xAI, Gemini, OpenAI, Groq)

# Models to use in parallel (for persona generation - no tool calls needed)
# Format: (model_name, provider) - provider determines API endpoint
# 
# Available keys:
# - Groq: 5 keys (GROQ_API_KEY through GROQ_API_KEY_5)
# - xAI: 1 key
# - Gemini: 1 key  
# - OpenAI: 1 key (persona gen only - expensive for simulation)
PARALLEL_MODELS_PERSONA_MULTI = [
    # Groq models (5 keys available) - primary workhorses
    ("llama-3.3-70b-versatile", "groq"),
    ("llama-3.1-8b-instant", "groq"),
    ("qwen/qwen3-32b", "groq"),
    # xAI Grok-4 (1 key available) - very high throughput
    ("grok-4-fast-non-reasoning", "xai"),
    # Gemini 2.5 Flash Lite (1 key available) - very high throughput  
    ("gemini-2.5-flash-lite", "gemini"),
    # OpenAI GPT-5-nano (1 key available) - persona gen only
    ("gpt-5-nano-2025-08-07", "openai"),
]

# Legacy single-provider list (Groq only)
PARALLEL_MODELS_PERSONA = [
    "llama-3.3-70b-versatile",  # Best quality
    "llama-3.1-8b-instant",      # Fastest
    "qwen/qwen3-32b",            # Good balance
]

# Models for simulation (need reliable tool calling)
# IMPORTANT: Only models with reliable JSON tool calling should be used here.
# Llama-4 models use XML-style tool calls which often fail even with the adapter
# because they use placeholder strings ("post_id_here") or hallucinate tools.
# OpenAI disabled - too expensive and causes 400 errors with tool schemas.
PARALLEL_MODELS_SIMULATION = [
    ("llama-3.3-70b-versatile", "groq"),                         # Most reliable for tool calls
    ("llama-3.1-8b-instant", "groq"),                            # Fast, good tool support
    ("qwen/qwen3-32b", "groq"),                                  # Good tool support
    ("meta-llama/llama-4-maverick-17b-128e-instruct", "groq"),   # Re-enabled: Llama4ToolAdapter handles XML
    ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),       # Re-enabled: fastest Llama-4
    ("gemini-2.5-flash-lite", "gemini"),                         # Flash-lite: faster, higher throughput
    ("grok-4-fast-non-reasoning", "xai"),                        # xAI Grok-4: 4M TPM, very high throughput
    # DISABLED models:
    # ("gpt-5-nano-2025-08-07", "openai"),   # 400 errors with tool schemas
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_rate_limits(model: str) -> dict:
    """Get rate limits for a model, with fallback to defaults."""
    return RATE_LIMITS.get(model, DEFAULT_RATE_LIMITS)


def supports_structured_output(model: str) -> bool:
    """Check if model supports JSON response format."""
    return model in STRUCTURED_OUTPUT_MODELS


def get_provider_for_model(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()
    if "grok" in model_lower:
        return "xai"
    if "gpt" in model_lower:
        return "openai"
    if "gemini" in model_lower:
        return "gemini"
    if "llama" in model_lower or "mixtral" in model_lower or "qwen" in model_lower:
        return "groq"
    return "xai"  # Default


def get_fallback_chain(primary_model: str) -> list:
    """Get the full fallback chain starting with the primary model."""
    if not FALLBACK_ENABLED:
        return [primary_model]
    chain = [primary_model]
    for model in FALLBACK_MODELS:
        if model != primary_model and model not in chain:
            chain.append(model)
    return chain


def get_api_keys_for_provider(provider: str) -> list[str]:
    """Get all available API keys for a provider.
    
    Checks environment variables in order:
    1. Explicit patterns from API_KEY_PATTERNS if defined
    2. Auto-generated patterns: {PROVIDER}_API_KEY, {PROVIDER}_API_KEY_2, etc.
    
    Returns:
        List of valid API keys (non-empty strings found in env)
    """
    keys = []
    provider_lower = provider.lower()
    
    # Get patterns to check
    if provider_lower in API_KEY_PATTERNS:
        patterns = API_KEY_PATTERNS[provider_lower]
    else:
        # Auto-generate patterns
        base_var = API_KEY_ENV_VARS.get(provider_lower, f"{provider.upper()}_API_KEY")
        patterns = [base_var]
        for i in range(2, MULTI_KEY_MAX_KEYS + 1):
            patterns.append(f"{base_var}_{i}")
    
    # Check each pattern
    for pattern in patterns:
        key = os.getenv(pattern, "").strip()
        if key:
            keys.append(key)
    
    return keys


def get_model_key_pairs(for_simulation: bool = False) -> list[tuple[str, str, str]]:
    """Get (model, api_key, provider) tuples for parallel multi-model/provider execution.
    
    When MULTI_PROVIDER_ENABLED, uses all available providers (Groq, xAI, Gemini, OpenAI).
    Otherwise falls back to Groq-only models.
    
    Args:
        for_simulation: If True, use PARALLEL_MODELS_SIMULATION (tool-call safe, Groq only)
                       If False, use PARALLEL_MODELS_PERSONA_MULTI (all providers)
    
    Returns:
        List of (model_name, api_key, provider) tuples for parallel execution
    """
    pairs: list[tuple[str, str, str]] = []
    
    if not MULTI_MODEL_ENABLED:
        # Fall back to single model with all keys
        provider = get_provider_for_model(SIMULATION_MODEL if for_simulation else PERSONA_MODEL)
        keys = get_api_keys_for_provider(provider)
        model = SIMULATION_MODEL if for_simulation else PERSONA_MODEL
        return [(model, key, provider) for key in keys]
    
    # For simulation, use multi-provider models (Groq + Gemini + Llama-4)
    if for_simulation:
        # PARALLEL_MODELS_SIMULATION is now list of (model, provider) tuples
        for model, provider in PARALLEL_MODELS_SIMULATION:
            keys = get_api_keys_for_provider(provider)
            for key in keys:
                pairs.append((model, key, provider))
        return pairs
    
    # For persona generation, use all providers if enabled
    if MULTI_PROVIDER_ENABLED:
        # Collect all available (model, provider) pairs with their keys
        for model, provider in PARALLEL_MODELS_PERSONA_MULTI:
            keys = get_api_keys_for_provider(provider)
            for key in keys:
                pairs.append((model, key, provider))
        return pairs
    
    # Fallback: Groq only
    models = PARALLEL_MODELS_PERSONA
    provider = "groq"
    keys = get_api_keys_for_provider(provider)
    for i, key in enumerate(keys):
        model = models[i % len(models)]
        pairs.append((model, key, provider))
    
    return pairs


def get_parallel_model_summary() -> str:
    """Get a summary string of parallel model configuration."""
    pairs = get_model_key_pairs(for_simulation=False)
    if not pairs:
        return "No API keys available"
    
    # Count by provider and model
    provider_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    for model, _, provider in pairs:
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        short_model = model.split('/')[-1]
        model_counts[short_model] = model_counts.get(short_model, 0) + 1
    
    provider_parts = [f"{p}×{c}" for p, c in sorted(provider_counts.items())]
    model_parts = [f"{m}×{c}" for m, c in sorted(model_counts.items())]
    
    return f"{len(pairs)} keys: providers=[{', '.join(provider_parts)}] models=[{', '.join(model_parts)}]"


def get_primary_api_key(provider: str) -> str:
    """Get the primary (first) API key for a provider."""
    keys = get_api_keys_for_provider(provider)
    if not keys:
        raise RuntimeError(f"No API key found for provider: {provider}")
    return keys[0]


def count_api_keys(provider: str) -> int:
    """Count how many API keys are available for a provider."""
    return len(get_api_keys_for_provider(provider))


def print_rate_limit_summary(model: str = None) -> None:
    """Print a summary of rate limits and effective concurrency settings.
    
    Useful for debugging and understanding current configuration.
    """
    if model is None:
        model = SIMULATION_MODEL
    
    provider = get_provider_for_model(model)
    num_keys = count_api_keys(provider)
    limits = RATE_LIMITS.get(model, DEFAULT_RATE_LIMITS)
    
    print("\n" + "=" * 60)
    print("RATE LIMIT CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model:           {model}")
    print(f"Provider:        {provider}")
    print(f"API Keys:        {num_keys}")
    print("-" * 60)
    print("Per-Key Limits:")
    print(f"  RPM:           {limits.get('rpm', 'N/A'):,}")
    print(f"  RPD:           {limits.get('rpd', 'N/A'):,}")
    print(f"  TPM:           {limits.get('tpm', 'N/A'):,}")
    print(f"  Est Tokens:    {limits.get('est_tokens', EST_TOKENS_PER_REQUEST):,}")
    print("-" * 60)
    print("Effective Limits (with all keys):")
    effective_rps = get_effective_rps(model, num_keys)
    print(f"  Effective RPS: {effective_rps:.2f}")
    print(f"  Effective RPM: {effective_rps * 60:.0f}")
    print(f"  Semaphore:     {get_effective_semaphore_limit(model)}")
    print(f"  Batch Size:    {get_effective_batch_size(model)}")
    print(f"  Delay/Request: {get_optimal_delay_between_requests(model):.3f}s")
    print("-" * 60)
    print("TPM Analysis:")
    tpm = limits.get('tpm', 6000)
    est_tokens = limits.get('est_tokens', EST_TOKENS_PER_REQUEST)
    tpm_limited_rpm = tpm / est_tokens
    rpm = limits.get('rpm', 30)
    if tpm_limited_rpm < rpm:
        print(f"  ⚠️  TPM-LIMITED: {tpm_limited_rpm:.0f} effective RPM < {rpm} RPM limit")
        print(f"  Bottleneck: Token consumption, not request count")
    else:
        print(f"  ✓  RPM-LIMITED: {rpm} RPM < {tpm_limited_rpm:.0f} TPM-based limit")
        print(f"  Bottleneck: Request count, not token consumption")
    print("=" * 60 + "\n")


# =============================================================================
# TRAJECTORY STAGE CONFIGURATION
# =============================================================================
# Each persona is assigned a static intensity stage at generation time.
# This affects language intensity, slang usage, and label token frequency.
#
# Stages:
#   - curious: New to community, testing waters, mild language
#   - active: Engaged member, uses slang, moderate intensity
#   - entrenched: Hardcore, uses extreme slang, high label density

from typing import Any

# Distribution of stages per archetype (must sum to 1.0)
# SKEWED TOWARDS ENTRENCHED for more intense, realistic toxic content
TRAJECTORY_STAGE_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    # Benign archetypes - mostly mild, some active
    "benign": {"curious": 0.7, "active": 0.3, "entrenched": 0.0},
    "recovery_support": {"curious": 0.4, "active": 0.5, "entrenched": 0.1},
    
    # ED cluster - heavy entrenched for realistic pro-ana content
    "ed_risk": {"curious": 0.15, "active": 0.35, "entrenched": 0.5},
    "pro_ana": {"curious": 0.1, "active": 0.3, "entrenched": 0.6},
    
    # Incel/Manosphere - heavily entrenched
    "incel_misogyny": {"curious": 0.1, "active": 0.3, "entrenched": 0.6},
    "alpha": {"curious": 0.15, "active": 0.35, "entrenched": 0.5},
    
    # Misinfo/Conspiracy - skews entrenched
    "misinfo": {"curious": 0.15, "active": 0.35, "entrenched": 0.5},
    "conspiracy": {"curious": 0.1, "active": 0.3, "entrenched": 0.6},
    
    # Culture war - mostly entrenched
    "trad": {"curious": 0.15, "active": 0.35, "entrenched": 0.5},
    "gamergate": {"curious": 0.1, "active": 0.35, "entrenched": 0.55},
    
    # Extreme harm - heavily entrenched
    "extremist": {"curious": 0.05, "active": 0.2, "entrenched": 0.75},
    "hate_speech": {"curious": 0.05, "active": 0.2, "entrenched": 0.75},
    "bullying": {"curious": 0.1, "active": 0.3, "entrenched": 0.6},
}

# Behavioral modifiers per stage
TRAJECTORY_STAGE_MODIFIERS: Dict[str, Dict[str, Any]] = {
    "curious": {
        "label_frequency_multiplier": 0.5,   # Half the normal label tokens
        "aggression_level": 0.25,            # Low aggression (0.0-1.0)
        "slang_fluency": "outsider",         # Uses mainstream language
        "in_group_language_prob": 0.15,      # Rarely uses in-group terms
        "reply_aggression_cap": 0.4,         # Max aggression even in heated threads
        "prompt_hint": "You're new to this community and still learning the language. You observe more than participate. Your posts are tentative and use mainstream terms.",
    },
    "active": {
        "label_frequency_multiplier": 1.0,   # Normal label frequency
        "aggression_level": 0.55,            # Moderate aggression
        "slang_fluency": "fluent",           # Comfortable with slang
        "in_group_language_prob": 0.6,       # Often uses in-group terms
        "reply_aggression_cap": 0.8,         # Can get quite aggressive
        "prompt_hint": "You're an active member of this community. You know the slang and participate regularly. You're comfortable expressing your views.",
    },
    "entrenched": {
        "label_frequency_multiplier": 1.5,   # 50% more label tokens
        "aggression_level": 0.85,            # High aggression
        "slang_fluency": "native",           # Uses hardcore slang naturally
        "in_group_language_prob": 0.9,       # Almost always uses in-group terms
        "reply_aggression_cap": 1.0,         # No cap on aggression
        "prompt_hint": "You're a veteran of this community. You use hardcore language naturally and lead discussions. You don't hold back.",
    },
}

# Slang tier access per fluency level
SLANG_FLUENCY_TIERS: Dict[str, List[str]] = {
    "outsider": ["mainstream"],              # Only mainstream terms
    "learning": ["mainstream", "mild"],      # Starting to pick up slang
    "fluent": ["mainstream", "mild", "moderate"],  # Comfortable with most slang
    "native": ["mainstream", "mild", "moderate", "severe"],  # Uses all slang
}
