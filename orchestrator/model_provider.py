from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from camel.models import BaseModelBackend, ModelFactory, ModelManager
from camel.types import ModelPlatformType, ModelType

from oasis.llama4_tool_adapter import wrap_llama4_backend
from orchestrator.llm_config import LLM_CONFIG

logger = logging.getLogger(__name__)

try:
    # Reuse the centralized xAI limiter used by ExtendedSocialAgent to avoid duplicate logic
    from generation.extended_agent import _XAI_LIMITER as _GLOBAL_XAI_LIMITER  # type: ignore
except Exception:
    _GLOBAL_XAI_LIMITER = None  # Fallback if import path changes; wrapper will no-op

ProviderName = Literal["openai", "xai", "gemini", "groq"]

# Rate limit error patterns to detect
RATE_LIMIT_PATTERNS = [
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "quota exceeded",
    "throttl",
]


@dataclass(frozen=True)
class LLMProviderSettings:
    r"""Settings for building a CAMEL BaseModelBackend."""
    provider: ProviderName
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = 3600.0


def _attach_rate_limiters(backend: BaseModelBackend) -> BaseModelBackend:
    """Monkey-patch backend.run/arun to acquire the global xAI limiter per request."""
    limiter = _GLOBAL_XAI_LIMITER
    if limiter is None:
        return backend
    import types

    if hasattr(backend, "arun"):
        orig_arun = backend.arun

        async def limited_arun(self, messages, response_format=None, tools=None):
            try:
                await limiter.acquire(limiter.estimate_tokens())
            except Exception:
                pass
            return await orig_arun(messages, response_format, tools)

        backend.arun = types.MethodType(limited_arun, backend)  # type: ignore[assignment]

    if hasattr(backend, "run"):
        orig_run = backend.run

        def limited_run(self, messages, response_format=None, tools=None):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(limiter.acquire(limiter.estimate_tokens()))
            except Exception:
                pass
            return orig_run(messages, response_format, tools)

        backend.run = types.MethodType(limited_run, backend)  # type: ignore[assignment]

    return backend


def create_model_backend(settings: LLMProviderSettings) -> BaseModelBackend:
    r"""Create a CAMEL model backend for the given provider settings."""
    provider = settings.provider.lower()
    cfg = LLM_CONFIG
    # NOTE:
    # - OpenAI-compatible (xAI) and OpenAI SDKs expect `max_tokens` only.
    # - Gemini SDK expects `max_output_tokens`.
    # Build per-provider defaults to avoid passing unsupported kwargs.
    if provider == "xai":
        api_key = settings.api_key or os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("xAI selected but no API key provided.")
        base_url = settings.base_url or "https://api.x.ai/v1"
        default_model_cfg = {"max_tokens": int(cfg.xai_max_tokens)}
        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=settings.model_name,
            api_key=api_key,
            url=base_url,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
        # Attach request-level rate limiter to prevent 429 bursts
        return _attach_rate_limiters(backend)
    if provider == "groq":
        api_key = settings.api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("Groq selected but no API key provided.")
        base_url = settings.base_url or "https://api.groq.com/openai/v1"
        # Groq recommends lower temperature (0.2-0.5) for tool calls to reduce
        # hallucinations and malformed JSON in function call generation
        from configs.llm_settings import SIMULATION_TEMPERATURE
        default_model_cfg = {
            "max_tokens": int(cfg.openai_max_tokens),
            "temperature": SIMULATION_TEMPERATURE,  # Low temp for tool call reliability
        }
        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=settings.model_name,
            api_key=api_key,
            url=base_url,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
        # Wrap Llama-4 models with XML-to-JSON tool call adapter
        return wrap_llama4_backend(backend, settings.model_name)
    if provider == "gemini":
        api_key = settings.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Gemini selected but no API key provided.")
        
        # Use native Gemini adapter with safety settings disabled
        # This bypasses the OpenAI-compatible endpoint which doesn't support safety_settings
        from oasis.gemini_native_adapter import create_gemini_native_backend
        from configs.llm_settings import SIMULATION_TEMPERATURE
        
        return create_gemini_native_backend(
            model_name=settings.model_name,
            api_key=api_key,
            model_config_dict={
                "temperature": SIMULATION_TEMPERATURE,
                "max_output_tokens": int(cfg.gemini_max_output_tokens) if hasattr(cfg, 'gemini_max_output_tokens') else 4096,
            },
            timeout=settings.timeout_seconds,
        )
    if provider == "openai":
        # Use standard OpenAI platform; model_name can be a ModelType or string
        model_type = getattr(ModelType, settings.model_name, settings.model_name)  # type: ignore[arg-type]
        api_key = settings.api_key or os.getenv("OPENAI_API_KEY", "")
        default_model_cfg = {"max_tokens": int(cfg.openai_max_tokens)}
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            api_key=api_key,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
    raise ValueError(f"Unknown provider: {settings.provider}")



def create_fallback_backend(
    settings: LLMProviderSettings,
    fallback_models: Optional[List[str]] = None,
    use_multi_key: Optional[bool] = None,
) -> ModelManager:
    """Create a model backend with automatic fallback support using CAMEL's ModelManager.
    
    Supports multiple API keys per provider for increased throughput. When multi-key
    is enabled, each model in the fallback chain is duplicated for each available
    API key, effectively multiplying throughput.
    
    Args:
        settings: Primary model settings
        fallback_models: Optional list of fallback model names
        use_multi_key: Override MULTI_KEY_ENABLED setting (None = use config)
        
    Returns:
        ModelManager that handles multiple models with round-robin fallback
    """
    from configs.llm_settings import (
        FALLBACK_ENABLED,
        FALLBACK_MODELS,
        MULTI_KEY_ENABLED,
        get_api_keys_for_provider,
        get_provider_for_model,
    )
    
    # Determine if multi-key is enabled
    multi_key = use_multi_key if use_multi_key is not None else MULTI_KEY_ENABLED
    
    # Build the model chain (primary + fallbacks)
    model_chain = [settings.model_name]
    if FALLBACK_ENABLED:
        fallback_list = fallback_models or FALLBACK_MODELS
        for model_name in fallback_list:
            if model_name not in model_chain:
                model_chain.append(model_name)
    
    # Build list of backends
    backends: List[BaseModelBackend] = []
    
    for model_name in model_chain:
        provider = get_provider_for_model(model_name)
        
        # Get API keys for this provider
        if multi_key:
            api_keys = get_api_keys_for_provider(provider)
        else:
            # Single key mode - use provided key or primary from env
            single_key = settings.api_key if model_name == settings.model_name else None
            if not single_key:
                keys = get_api_keys_for_provider(provider)
                single_key = keys[0] if keys else None
            api_keys = [single_key] if single_key else []
        
        if not api_keys:
            logger.warning(f"No API keys found for provider {provider}, skipping {model_name}")
            continue
        
        # Create a backend for each API key
        for key_idx, api_key in enumerate(api_keys):
            try:
                model_settings = LLMProviderSettings(
                    provider=provider,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=settings.base_url,
                    timeout_seconds=settings.timeout_seconds,
                )
                backend = create_model_backend(model_settings)
                backends.append(backend)
                
                key_label = f"key_{key_idx + 1}" if len(api_keys) > 1 else "primary"
                logger.info(f"Added model: {model_name} ({key_label})")
            except Exception as e:
                logger.warning(f"Failed to create backend for {model_name} (key {key_idx + 1}): {e}")
    
    if not backends:
        raise RuntimeError("No backends could be created - check API keys")
    
    num_keys = len(get_api_keys_for_provider(settings.provider)) if multi_key else 1
    logger.info(
        f"Created ModelManager with {len(backends)} backends "
        f"({len(model_chain)} models × {num_keys} keys)"
    )
    
    # Use round_robin strategy - CAMEL will cycle through backends
    return ModelManager(backends, scheduling_strategy="round_robin")
