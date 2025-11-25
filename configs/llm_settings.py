"""Centralized LLM configuration - import variables where needed.

Usage:
    from configs.llm_settings import PERSONA_MODEL, SIMULATION_MODEL, RATE_LIMITS
"""
import os
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
# Ordered by speed (t/s): scout(750) > maverick(600) > 8b(560) > qwen(400) > 70b(280)
# =============================================================================
FALLBACK_MODELS = [
    # Primary model is tried first (from SIMULATION_MODEL)
    # Then these fallbacks in order (fastest first for throughput):
    "meta-llama/llama-4-scout-17b-16e-instruct",      # 750 t/s - fastest
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # 600 t/s - fast, good quality
    "llama-3.1-8b-instant",                            # 560 t/s - fast, high max tokens
    "qwen/qwen3-32b",                                  # 400 t/s - good balance
]

# Enable/disable fallback behavior
FALLBACK_ENABLED = True
FALLBACK_MAX_RETRIES = 3  # Retries per model before moving to next

# =============================================================================
# TEMPERATURE & TOKEN LIMITS
# =============================================================================
PERSONA_TEMPERATURE = 0.2
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

# =============================================================================
# STRUCTURED OUTPUT (per-model)
# Models that support response_format: {"type": "json_object"}
# =============================================================================
STRUCTURED_OUTPUT_MODELS = {
    "grok-2",
    "grok-4-fast-non-reasoning",
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

# =============================================================================
# RATE LIMITS (per-model) - Groq Developer Plan
# rpm = requests per minute
# tpm = tokens per minute
# rps = requests per second (effective ceiling = rpm/60)
# retries = max retry attempts on rate limit
# =============================================================================
RATE_LIMITS = {
    # xAI models (high throughput)
    "grok-2": {"rpm": 480, "tpm": 4_000_000, "rps": 8, "retries": 5},
    "grok-4-fast-non-reasoning": {"rpm": 480, "tpm": 4_000_000, "rps": 8, "retries": 5},
    # OpenAI models
    "gpt-4o": {"rpm": 500, "tpm": 150_000, "rps": 10, "retries": 5},
    "gpt-4o-mini": {"rpm": 1000, "tpm": 200_000, "rps": 20, "retries": 5},
    # Gemini models (high throughput)
    "gemini-2.5-flash": {"rpm": 1000, "tpm": 4_000_000, "rps": 15, "retries": 5},
    "gemini-2.5-flash-lite": {"rpm": 2000, "tpm": 4_000_000, "rps": 30, "retries": 5},
    # Groq models (Developer Plan limits from docs)
    # llama-3.3-70b: 1K RPM, 300K TPM (best quality, slowest)
    "llama-3.3-70b-versatile": {"rpm": 1_000, "tpm": 300_000, "rps": 16, "retries": 5},
    # llama-3.1-8b: 1K RPM, 250K TPM (fastest, good for high volume)
    "llama-3.1-8b-instant": {"rpm": 1_000, "tpm": 250_000, "rps": 16, "retries": 5},
    # qwen3-32b: 1K RPM, 300K TPM (good balance)
    "qwen/qwen3-32b": {"rpm": 1_000, "tpm": 300_000, "rps": 16, "retries": 5},
    # llama-4-maverick: 1K RPM, 300K TPM (fast, good quality)
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"rpm": 1_000, "tpm": 300_000, "rps": 16, "retries": 5},
    # llama-4-scout: 1K RPM, 300K TPM (fastest llama-4)
    "meta-llama/llama-4-scout-17b-16e-instruct": {"rpm": 1_000, "tpm": 300_000, "rps": 16, "retries": 5},
}

# Default rate limits for unknown models
DEFAULT_RATE_LIMITS = {"rpm": 30, "tpm": 6_000, "rps": 0.5, "retries": 8}

# =============================================================================
# CONCURRENCY
# =============================================================================
# SEMAPHORE_LIMIT controls max concurrent LLM requests in flight.
# With 1000 RPM (~16 RPS) per key, we can scale with multiple keys.
# Base: 64 concurrent requests per API key
SEMAPHORE_LIMIT_PER_KEY = 64
BATCH_SIZE_PER_KEY = 64

# Dynamic limits based on available keys (computed at runtime)
def get_effective_semaphore_limit() -> int:
    """Get semaphore limit scaled by number of API keys."""
    if not MULTI_KEY_ENABLED:
        return SEMAPHORE_LIMIT_PER_KEY
    # Scale by number of keys for the primary provider
    num_keys = count_api_keys(SIMULATION_PROVIDER)
    return SEMAPHORE_LIMIT_PER_KEY * max(1, num_keys)

def get_effective_batch_size() -> int:
    """Get batch size scaled by number of API keys."""
    if not MULTI_KEY_ENABLED:
        return BATCH_SIZE_PER_KEY
    num_keys = count_api_keys(SIMULATION_PROVIDER)
    return BATCH_SIZE_PER_KEY * max(1, num_keys)

# Legacy aliases for backwards compatibility
SEMAPHORE_LIMIT = SEMAPHORE_LIMIT_PER_KEY  # Will be overridden at runtime
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
MULTI_KEY_MAX_KEYS = 5  # Max keys to check per provider (1, 2, 3, 4, 5)

# Explicit key patterns per provider (auto-generated if not specified)
# Format: list of env var names to check in order
API_KEY_PATTERNS = {
    "groq": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"],
    "openai": ["OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"],
    "xai": ["XAI_API_KEY", "XAI_API_KEY_2", "XAI_API_KEY_3"],
    "gemini": ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"],
}

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


def get_primary_api_key(provider: str) -> str:
    """Get the primary (first) API key for a provider."""
    keys = get_api_keys_for_provider(provider)
    if not keys:
        raise RuntimeError(f"No API key found for provider: {provider}")
    return keys[0]


def count_api_keys(provider: str) -> int:
    """Count how many API keys are available for a provider."""
    return len(get_api_keys_for_provider(provider))
