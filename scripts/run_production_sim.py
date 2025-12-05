#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LOGGING SETUP - Comprehensive error tracking
# =============================================================================
def setup_logging(log_level: str = "DEBUG") -> logging.Logger:
    """Set up comprehensive logging for debugging."""
    # Create logger
    logger = logging.getLogger("production_sim")
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with detailed format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"production_sim_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()


# =============================================================================
# TIMING UTILITIES
# =============================================================================
class Timer:
    """Context manager for timing code blocks with logging."""
    
    def __init__(self, name: str, log_level: str = "info"):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.debug(f"⏱️  START: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if exc_type is not None:
            logger.error(f"⏱️  FAILED: {self.name} after {self.elapsed:.3f}s - {exc_type.__name__}: {exc_val}")
        else:
            log_func = getattr(logger, self.log_level, logger.info)
            log_func(f"⏱️  DONE: {self.name} in {self.elapsed:.3f}s")
        
        return False  # Don't suppress exceptions


class TimingStats:
    """Track timing statistics across the simulation."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_time = time.perf_counter()
        # Agent-level timing aggregation
        self.agent_timings: Dict[str, List[float]] = {}
    
    def record(self, category: str, elapsed: float):
        """Record a timing for a category."""
        if category not in self.timings:
            self.timings[category] = []
        self.timings[category].append(elapsed)
    
    def record_agent_timing(self, timing_dict: Dict[str, float]):
        """Record timing from an agent's _last_timing dict."""
        for key, value in timing_dict.items():
            if key not in self.agent_timings:
                self.agent_timings[key] = []
            self.agent_timings[key].append(value)
    
    def get_stats(self, category: str) -> Dict[str, float]:
        """Get statistics for a category."""
        if category not in self.timings or not self.timings[category]:
            return {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}
        
        times = self.timings[category]
        return {
            "count": len(times),
            "total": sum(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def get_agent_stats(self, category: str) -> Dict[str, float]:
        """Get statistics for agent-level timing category."""
        if category not in self.agent_timings or not self.agent_timings[category]:
            return {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}
        
        times = self.agent_timings[category]
        return {
            "count": len(times),
            "total": sum(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def log_summary(self):
        """Log a summary of all timing statistics."""
        total_elapsed = time.perf_counter() - self.start_time
        
        logger.info("=" * 70)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total elapsed time: {total_elapsed:.2f}s")
        logger.info("-" * 70)
        
        # Step-level timings
        logger.info("STEP-LEVEL TIMINGS:")
        for category in sorted(self.timings.keys()):
            stats = self.get_stats(category)
            logger.info(
                f"  {category:28s} | count={stats['count']:4d} | "
                f"total={stats['total']:8.2f}s | avg={stats['avg']:6.3f}s | "
                f"min={stats['min']:6.3f}s | max={stats['max']:6.3f}s"
            )
        
        # Agent-level timings (if collected)
        if self.agent_timings:
            logger.info("-" * 70)
            logger.info("AGENT-LEVEL TIMINGS (per-agent averages):")
            for category in sorted(self.agent_timings.keys()):
                stats = self.get_agent_stats(category)
                logger.info(
                    f"  agent.{category:24s} | count={stats['count']:4d} | "
                    f"total={stats['total']:8.2f}s | avg={stats['avg']:6.4f}s | "
                    f"min={stats['min']:6.4f}s | max={stats['max']:6.4f}s"
                )
        
        logger.info("=" * 70)


# Global timing stats
timing_stats = TimingStats()


from camel.models import BaseModelBackend
from camel.types import ModelType

import oasis
from configs.llm_settings import (API_ENDPOINTS, API_KEY_ENV_VARS, AUTO_REPORT,
                                  IMPUTATION_MAX_TOKENS, IMPUTATION_MODEL,
                                  IMPUTATION_PROVIDER, IMPUTATION_TEMPERATURE,
                                  RAG_IMPUTER_BATCH_SIZE,
                                  RAG_IMPUTER_MAX_WORKERS, RAG_IMPUTER_MODE,
                                  RAG_IMPUTER_STATIC_BANK,
                                  SIMULATION_MAX_TOKENS, SIMULATION_MODEL,
                                  SIMULATION_PROVIDER, SIMULATION_TEMPERATURE)
from generation.emission_policy import EmissionPolicy
from oasis.clock.clock import Clock
from oasis.imputation.rag_llm_imputer import RagImputer, RagImputerConfig
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import RecsysType
from orchestrator.agent_factory import build_agent_graph_from_csv
from orchestrator.expect_registry import ExpectRegistry
from orchestrator.interceptor_channel import InterceptorChannel
from orchestrator.manifest_loader import load_label_lexicons, load_manifest
from orchestrator.model_provider import (LLMProviderSettings,
                                         create_fallback_backend,
                                         create_model_backend)
from orchestrator.scheduler import MultiLabelScheduler, MultiLabelTargets
from orchestrator.sidecar_logger import SidecarLogger

# Enhanced simulation features (optional)
try:
    from orchestrator.simulation_coordinator import (
        SimulationCoordinator, SimulationCoordinatorConfig)
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    SimulationCoordinator = None
    SimulationCoordinatorConfig = None

# Event Manager for triggered campaigns
try:
    from orchestrator.event_manager import EventManager
    EVENT_MANAGER_AVAILABLE = True
except ImportError:
    EVENT_MANAGER_AVAILABLE = False
    EventManager = None


#
# LLM selection - uses centralized config from configs/llm_settings.py
# To change the model/provider, edit configs/llm_settings.py
#
def _build_llm_settings() -> LLMProviderSettings:
    """Build LLM settings from centralized config."""
    api_key_env = API_KEY_ENV_VARS.get(SIMULATION_PROVIDER, "XAI_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()
    base_url = API_ENDPOINTS.get(SIMULATION_PROVIDER)

    return LLMProviderSettings(
        provider=SIMULATION_PROVIDER,
        model_name=SIMULATION_MODEL,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=3600.0,
    )


def _validate_api_key(api_key: str, provider: str) -> bool:
    """Validate an API key by making a lightweight API call.
    
    Returns True if the key is valid, False otherwise.
    """
    import httpx
    
    # Provider-specific validation endpoints
    endpoints = {
        "groq": "https://api.groq.com/openai/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "openai": "https://api.openai.com/v1/models",
        "xai": "https://api.x.ai/v1/models",
        "gemini": None,  # Skip validation for Gemini (different auth)
    }
    
    endpoint = endpoints.get(provider)
    if endpoint is None:
        return True  # Can't validate, assume valid
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return response.status_code == 200
    except Exception:
        return False  # Network error, assume invalid


def _build_multi_provider_model(validate_keys: bool = True):
    """Build a ModelManager with multiple providers and keys for simulation.
    
    Uses all available keys across Groq, Gemini, and other providers for maximum throughput.
    Llama-4 models are wrapped with Llama4ToolAdapter for XML-to-JSON conversion.
    Falls back to single-provider if multi-key is disabled.
    
    Args:
        validate_keys: If True, validate each API key before use (slower startup, fewer runtime errors).
    """
    from configs.llm_settings import (
        MULTI_KEY_ENABLED,
        MULTI_MODEL_ENABLED,
        PARALLEL_MODELS_SIMULATION,
        get_api_keys_for_provider,
    )
    
    if not MULTI_KEY_ENABLED and not MULTI_MODEL_ENABLED:
        # Single model, single key
        return create_fallback_backend(_build_llm_settings())
    
    # Build backends for each (model, provider) × keys combination
    from camel.models import BaseModelBackend, ModelManager
    backends: list[BaseModelBackend] = []
    provider_counts: dict[str, int] = {}
    skipped_keys: dict[str, int] = {}
    
    # Validate keys once per provider (they're shared across models)
    validated_keys: dict[str, dict[str, bool]] = {}  # provider -> {key: is_valid}
    
    # PARALLEL_MODELS_SIMULATION is now list of (model, provider) tuples
    for model_name, provider in PARALLEL_MODELS_SIMULATION:
        keys = get_api_keys_for_provider(provider)
        if not keys:
            print(f"[production] WARNING: No {provider} API keys found, skipping {model_name}")
            continue
        
        # Validate keys for this provider (once per provider)
        if validate_keys and provider not in validated_keys:
            print(f"[production] Validating {len(keys)} {provider} API keys...")
            validated_keys[provider] = {}
            for key_idx, api_key in enumerate(keys):
                is_valid = _validate_api_key(api_key, provider)
                validated_keys[provider][api_key] = is_valid
                if not is_valid:
                    print(f"[production] ⚠️  {provider} key {key_idx + 1} is INVALID - skipping")
                    skipped_keys[provider] = skipped_keys.get(provider, 0) + 1
        
        for key_idx, api_key in enumerate(keys):
            # Skip invalid keys
            if validate_keys and provider in validated_keys:
                if not validated_keys[provider].get(api_key, True):
                    continue
            
            try:
                settings = LLMProviderSettings(
                    provider=provider,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=API_ENDPOINTS.get(provider),
                    timeout_seconds=3600.0,
                )
                backend = create_model_backend(settings)
                backends.append(backend)
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            except Exception as e:
                print(f"[production] Failed to create backend for {model_name} ({provider} key {key_idx + 1}): {e}")
    
    if not backends:
        print("[production] WARNING: No backends created, falling back to single model")
        return create_fallback_backend(_build_llm_settings())
    
    # Log provider distribution
    provider_summary = ", ".join(f"{p}×{c}" for p, c in sorted(provider_counts.items()))
    skipped_summary = ", ".join(f"{p}×{c} skipped" for p, c in sorted(skipped_keys.items())) if skipped_keys else "none skipped"
    print(f"[production] Created ModelManager with {len(backends)} backends ({provider_summary}) [{skipped_summary}]")
    
    return ModelManager(backends, scheduling_strategy="round_robin")


LLM_SETTINGS: LLMProviderSettings = _build_llm_settings()


def _build_imputer_llm_settings() -> LLMProviderSettings:
    """Build provider settings for the background imputer."""
    provider = IMPUTATION_PROVIDER or SIMULATION_PROVIDER
    api_key_env = API_KEY_ENV_VARS.get(provider, "XAI_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()
    base_url = API_ENDPOINTS.get(provider)
    return LLMProviderSettings(
        provider=provider,
        model_name=IMPUTATION_MODEL,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=3600.0,
    )


def _validate_personas_and_edges(manifest, personas_csv: Path,
                                 edges_path: Path | None) -> None:
    r"""Validate that personas and edges are consistent with the manifest.

    This guards against running a production simulation with mismatched
    persona counts or an out-of-date follow graph.
    """
    if not personas_csv.exists():
        raise SystemExit(
            f"[production] Personas CSV not found: {personas_csv}. "
            "Generate personas before running the simulation."
        )
    # Count persona rows (excluding header).
    with personas_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        persona_rows = sum(1 for _ in reader)
    if persona_rows <= 0:
        raise SystemExit(
            f"[production] Personas CSV {personas_csv} contains no rows.")

    # If manifest declares a population (production manifest) or personas
    # counts (MVP master config), ensure totals match the personas CSV.
    manifest_data = getattr(manifest, "data", {}) or {}
    expected_total: int | None = None

    # Primary source of truth: explicit population block.
    population_cfg = manifest_data.get("population", {})
    if isinstance(population_cfg, dict) and population_cfg:
        try:
            expected_total = sum(int(v) for v in population_cfg.values())
        except Exception:
            expected_total = None

    # Fallback: personas counts in configs like configs/mvp_master.yaml.
    if expected_total is None:
        personas_cfg = manifest_data.get("personas", {})
        if isinstance(personas_cfg, dict) and personas_cfg:
            # Only sum known count keys, not seed/paths/etc.
            count_keys = [
                key for key in personas_cfg.keys()
                if key in ("incel", "misinfo", "benign")
            ]
            if count_keys:
                try:
                    expected_total = sum(
                        int(personas_cfg[key]) for key in count_keys)
                except Exception:
                    expected_total = None

    if expected_total and expected_total > 0 and persona_rows != expected_total:
        raise SystemExit(
            "[production] Personas CSV row count does not match manifest population/persona counts. "
            f"CSV rows={persona_rows}, manifest total agents={expected_total}. "
            "Regenerate personas for this manifest."
        )

    if edges_path is None:
        # Caller chose to run without a seeded follow graph.
        return

    if not edges_path.exists():
        raise SystemExit(
            f"[production] Edges CSV not found: {edges_path}. "
            "Generate the follow graph with scripts/build_graph.py or set PROD_EDGES_CSV."
        )

    max_id = -1
    with edges_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                follower = int(r["follower_id"])
                followee = int(r["followee_id"])
            except Exception:
                continue
            if follower > max_id:
                max_id = follower
            if followee > max_id:
                max_id = followee

    if max_id >= persona_rows:
        raise SystemExit(
            "[production] Edges CSV references user_id outside the personas index range. "
            f"Max referenced user_id={max_id}, personas rows={persona_rows}. "
            "Regenerate the graph after updating personas."
        )


def _follow_table_empty(env) -> bool:
    try:
        cur = env.platform.db_cursor
        cur.execute("SELECT COUNT(*) FROM follow")
        cnt = cur.fetchone()[0]
        return int(cnt) == 0
    except Exception:
        return True


def _seed_initial_follows_from_csv(env, edges_csv: Path) -> int:
    """Seed initial follow edges into the SQLite DB and AgentGraph."""
    if not edges_csv.exists():
        return 0
    rows = []
    upd_followings = []
    upd_followers = []
    try:
        current_time = env.platform.sandbox_clock.get_time_step()
    except Exception:
        current_time = 0
    with edges_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                follower = int(r["follower_id"])
                followee = int(r["followee_id"])
            except Exception:
                continue
            if follower == followee:
                continue
            rows.append((follower, followee, current_time))
            upd_followings.append((follower,))
            upd_followers.append((followee,))
    if not rows:
        return 0
    # insert follows
    env.platform.pl_utils._execute_many_db_command(
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
        rows,
        commit=True,
    )
    # update counters
    env.platform.pl_utils._execute_many_db_command(
        "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?",
        upd_followings,
        commit=True,
    )
    env.platform.pl_utils._execute_many_db_command(
        "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?",
        upd_followers,
        commit=True,
    )
    # sync in-memory graph
    for follower, followee, _ in rows:
        try:
            env.agent_graph.add_edge(follower, followee)
        except Exception:
            pass
    return len(rows)


async def run(
    manifest_path: Path,
    personas_csv: Path,
    db_path: Path,
    steps: int,
    edges_csv: Path | None,
    warmup_steps: int,
    rag_mode: str,
    rag_workers: int,
    rag_batch_size: int,
    enable_thread_dynamics: bool = False,
    enable_obfuscation: bool = False,
    enable_vector_retrieval: bool = True,
    tfidf_only: bool = False,
) -> None:
    run_t0 = time.perf_counter()
    
    # Reset timing stats for this run
    global timing_stats
    timing_stats = TimingStats()
    
    logger.info("=" * 60)
    logger.info("PRODUCTION SIMULATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Run started at {datetime.now().isoformat()}")
    logger.info(f"Parameters: steps={steps}, warmup={warmup_steps}, rag_mode={rag_mode}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Personas CSV: {personas_csv}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Thread dynamics: {enable_thread_dynamics}, Obfuscation: {enable_obfuscation}")
    
    print(f"[production] Run started at {datetime.now().isoformat()}")
    
    # Load manifest
    with Timer("load_manifest") as t:
        try:
            manifest = load_manifest(manifest_path)
            run_seed = manifest.run_seed
            logger.info(f"Manifest loaded successfully. Run seed: {run_seed}")
        except Exception as e:
            logger.error(f"FAILED to load manifest: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    # Choose edges path with env override, then validate inputs.
    override = os.getenv("PROD_EDGES_CSV")
    edges_path = Path(override) if override else edges_csv
    logger.debug(f"Edges path: {edges_path}")
    
    with Timer("validate_inputs") as t:
        try:
            _validate_personas_and_edges(manifest, personas_csv, edges_path)
            logger.info("Personas and edges validated successfully")
        except Exception as e:
            logger.error(f"FAILED to validate personas/edges: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    sidecar = SidecarLogger(path=db_path.parent / "sidecar.jsonl")
    logger.debug(f"Sidecar logger initialized: {db_path.parent / 'sidecar.jsonl'}")

    # Load ontology-driven label→lexicon map (best-effort; defaults if missing)
    with Timer("load_lexicons") as t:
        try:
            default_ontology = Path("configs/personas/ontology_unified.yaml")
            label_lexicons = load_label_lexicons(default_ontology) if default_ontology.exists() else {}
            logger.info(f"Loaded {len(label_lexicons)} label lexicons from ontology")
        except Exception as e:
            logger.warning(f"Failed to load label lexicons (using defaults): {e}")
            label_lexicons = {}
    timing_stats.record("initialization", t.elapsed)
    imputer: Optional[RagImputer] = None
    rag_mode_value = (rag_mode or RAG_IMPUTER_MODE).strip().lower()
    if rag_mode_value not in ("off", "background", "sync"):
        rag_mode_value = RAG_IMPUTER_MODE
    rag_workers_effective = rag_workers or RAG_IMPUTER_MAX_WORKERS
    rag_batch_effective = rag_batch_size or RAG_IMPUTER_BATCH_SIZE
    if rag_mode_value != "off":
        try:
            imputer_cfg = RagImputerConfig(
                db_path=db_path,
                llm_settings=_build_imputer_llm_settings(),
                mode=rag_mode_value,
                batch_size=rag_batch_effective,
                max_workers=rag_workers_effective,
                temperature=IMPUTATION_TEMPERATURE,
                max_tokens=IMPUTATION_MAX_TOKENS,
                lexicon_terms=label_lexicons or {},
                static_bank_path=Path(RAG_IMPUTER_STATIC_BANK),
                run_seed=run_seed,
                # Post-imputation obfuscation (targets harmful terms after replacement)
                enable_obfuscation=enable_obfuscation,
                obfuscation_deterministic=False,  # Non-deterministic for realistic variety
                # Vector DB retrieval settings
                enable_vector_retrieval=enable_vector_retrieval,
                use_chromadb=not tfidf_only,
            )
            imputer = RagImputer(imputer_cfg)
            await imputer.start()
            if enable_obfuscation:
                print("[production] Post-imputation obfuscation ENABLED")
            if enable_vector_retrieval:
                retrieval_type = "TF-IDF only" if tfidf_only else "ChromaDB + TF-IDF"
                print(f"[production] Vector retrieval ENABLED ({retrieval_type})")
        except Exception as exc:
            print(f"[production] Failed to start rag imputer: {exc}")
            imputer = None
            rag_mode_value = "off"
    # Create emission policy
    with Timer("create_emission_policy") as t:
        try:
            policy = EmissionPolicy(
                run_seed=run_seed,
                post_label_mode_probs=manifest.post_label_mode_probs,
                label_to_tokens=None,  # defaults cover the 6-parent-family setup
                label_lexicon_terms=label_lexicons or None,
            )
            logger.info(f"EmissionPolicy created with mode probs: {manifest.post_label_mode_probs}")
        except Exception as e:
            logger.error(f"FAILED to create EmissionPolicy: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)
    
    expect_registry = ExpectRegistry()
    targets_cfg = manifest.multi_label_targets
    logger.debug(f"Multi-label targets: {targets_cfg}")
    
    scheduler = MultiLabelScheduler(
        MultiLabelTargets(
            target_rate=targets_cfg["target_rate"],
            min_rate=targets_cfg["min_rate"],
            max_rate=targets_cfg["max_rate"],
        )
    )
    logger.info("Scheduler initialized")

    # Define the model for the agents (LLMAction only; no manual actions)
    # Use multi-provider backend for maximum throughput with round-robin across all keys
    with Timer("build_model") as t:
        try:
            model = _build_multi_provider_model()
            logger.info(f"Model built: {type(model).__name__}")
        except Exception as e:
            logger.error(f"FAILED to build model: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    # Setup platform and channel
    with Timer("create_platform") as t:
        try:
            base_channel = Channel()
            channel = InterceptorChannel(base_channel, expect_registry)
            # RecSys tuning for better thread engagement:
            # - refresh_rec_post_count: More posts shown = more chances to comment
            # - max_rec_post_len: Larger pool = more variety in recommendations
            # - following_post_count: Show more posts from followed users
            platform = Platform(
                db_path=str(db_path),
                channel=channel,
                sandbox_clock=Clock(k=60),
                start_time=None,
                recsys_type=RecsysType.RANDOM,
                refresh_rec_post_count=8,   # Increased from 3 - show more posts per refresh
                max_rec_post_len=10,        # Increased from 5 - larger recommendation pool
                following_post_count=5,     # Increased from 2 - more posts from follows
                show_score=False,
                allow_self_rating=False,
            )
            logger.info(f"Platform created with DB: {db_path}")
        except Exception as e:
            logger.error(f"FAILED to create platform: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    # Restrict available actions for Twitter simulation
    allowed_actions = [
        RecsysType  # placeholder to keep import grouped (no-op)
    ]
    # Define allowed ActionTypes explicitly
    # NOTE: Groq recommends 3-5 tools optimal, max 10-15. Too many tools causes
    # malformed tool call errors. Reduced to essential actions for reliability.
    from oasis.social_platform.typing import ActionType as _AT  # local alias
    allowed_actions = [
        # Core content creation (most important for dataset generation)
        _AT.CREATE_POST,
        _AT.CREATE_COMMENT,
        # Engagement actions
        _AT.LIKE_POST,
        # _AT.REPOST,  # Disabled: reposts don't add commentary, prefer QUOTE_POST for richer data
        # Optional: uncomment for more variety (but may increase tool call errors)
        _AT.QUOTE_POST,
        # _AT.FOLLOW,
        # _AT.SEARCH_POSTS,
        # NOTE: DO_NOTHING and REFRESH removed - their schemas have required 'placeholder'
        # param that LLMs don't fill, causing tool call validation errors.
    ]

    # Build agent graph using ExtendedSocialAgent with persona class metadata
    logger.info("Building agent graph from CSV...")
    logger.debug(f"Personas CSV: {personas_csv}")
    logger.debug(f"Allowed actions: {[a.name for a in allowed_actions if hasattr(a, 'name')]}")
    
    with Timer("build_agent_graph") as t:
        try:
            agent_graph = await build_agent_graph_from_csv(
                personas_csv=personas_csv,
                model=model,
                channel=channel,
                available_actions=allowed_actions,
                emission_policy=policy,
                sidecar_logger=sidecar,
                run_seed=run_seed,
                expect_registry=expect_registry,
                scheduler=scheduler,
                guidance_config=manifest.guidance_config,
            )
            agent_count = len(list(agent_graph.get_agents()))
            logger.info(f"Agent graph built successfully with {agent_count} agents")
        except Exception as e:
            logger.error(f"FAILED to build agent graph: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    # Initialize SimulationCoordinator for thread dynamics (optional)
    coordinator: Optional[SimulationCoordinator] = None
    if enable_thread_dynamics and ENHANCED_FEATURES_AVAILABLE and SimulationCoordinator is not None:
        with Timer("init_coordinator") as t:
            try:
                coordinator = SimulationCoordinator(
                    config=SimulationCoordinatorConfig(
                        enable_thread_dynamics=True,
                        enable_obfuscation=False,  # Obfuscation handled by RagImputer
                    ),
                    seed=run_seed,
                )
                # Register agents with their archetypes
                for agent_id, agent in agent_graph.get_agents():
                    archetype = getattr(agent, "_persona_cfg", None)
                    if archetype:
                        coordinator.register_agent(agent_id, archetype.primary_label)
                print("[production] Thread dynamics ENABLED")
            except Exception as exc:
                print(f"[production] Failed to initialize thread coordinator: {exc}")
                coordinator = None
        timing_stats.record("initialization", t.elapsed)

    # Initialize EventManager for triggered campaigns
    event_manager: Optional[EventManager] = None
    if EVENT_MANAGER_AVAILABLE and EventManager is not None:
        with Timer("init_event_manager") as t:
            try:
                events_config_path = Path("configs/simulation_events.yaml")
                event_manager = EventManager(
                    config_path=events_config_path,
                    seed=run_seed,
                    enabled=True,
                )
                logger.info(f"EventManager ENABLED ({len(event_manager.events)} events loaded)")
                print(f"[production] EventManager ENABLED ({len(event_manager.events)} events loaded)")
            except Exception as exc:
                logger.warning(f"Failed to initialize EventManager: {exc}")
                print(f"[production] Failed to initialize EventManager: {exc}")
                event_manager = None
        timing_stats.record("initialization", t.elapsed)
    else:
        logger.debug("EventManager not available or disabled")

    # Create environment
    with Timer("create_environment") as t:
        try:
            env = oasis.make(
                agent_graph=agent_graph,
                platform=platform,
                database_path=str(db_path),
            )
            logger.info("OASIS environment created successfully")
        except Exception as e:
            logger.error(f"FAILED to create OASIS environment: {e}")
            logger.error(traceback.format_exc())
            raise
    timing_stats.record("initialization", t.elapsed)

    async def _maybe_impute() -> None:
        if imputer is None:
            return
        try:
            await imputer.enqueue_new_rows()
            if rag_mode_value == "sync":
                await imputer.flush()
        except Exception as e:
            logger.error(f"Imputation error: {e}")

    # Reset environment
    with Timer("env_reset") as t:
        try:
            await env.reset()
        except Exception as e:
            logger.error(f"FAILED to reset environment: {e}")
            logger.error(traceback.format_exc())
            raise
    logger.info(f"env.reset() completed in {t.elapsed:.2f}s")
    print(f"[production] env.reset() took {t.elapsed:.2f}s")
    timing_stats.record("initialization", t.elapsed)
    
    await _maybe_impute()

    # Seed initial follow graph if provided and table is empty
    with Timer("seed_follows") as t:
        try:
            if _follow_table_empty(env):
                if edges_path:
                    seeded = _seed_initial_follows_from_csv(env, edges_path)
                    if seeded:
                        logger.info(f"Seeded {seeded} follow edges from {edges_path}")
                        print(f"[production] Seeded {seeded} follow edges from {edges_path}")
            else:
                logger.debug("Follow table non-empty; skipping follow seed.")
                print("[production] Follow table non-empty; skipping follow seed.")
        except Exception as e:
            logger.warning(f"Follow seeding failed: {e}")
            print(f"[production] Follow seeding failed: {e}")
    timing_stats.record("initialization", t.elapsed)

    logger.info("=" * 60)
    logger.info("ENTERING SIMULATION LOOPS")
    logger.info("=" * 60)
    print("[DEBUG] About to enter warmup loop...")
    sys.stdout.flush()

    def _apply_coordination_modifiers(agent, coord: Optional[SimulationCoordinator]) -> None:
        """Apply thread coordination modifiers to an agent before their action."""
        if coord is None:
            return
        try:
            modifiers = coord.get_agent_modifiers(agent.social_agent_id)
            # Convert to dict if it's an object with to_dict method
            if hasattr(modifiers, 'to_dict'):
                modifiers = modifiers.to_dict()
            elif not isinstance(modifiers, dict):
                modifiers = {}
            # Store modifiers on agent for use in perform_action_by_llm
            agent._coordination_modifiers = modifiers
        except Exception as e:
            logger.debug(f"Coordination modifier error for agent {agent.social_agent_id}: {e}")
            agent._coordination_modifiers = {}

    def _apply_event_modifiers(agent, evt_mgr: Optional[EventManager]) -> None:
        """Apply event modifiers to an agent based on active simulation events."""
        if evt_mgr is None:
            return
        try:
            persona_cfg = getattr(agent, "_persona_cfg", None)
            if persona_cfg is None:
                return
            archetype = persona_cfg.primary_label
            modifiers = evt_mgr.get_agent_modifiers(archetype)
            agent._event_modifiers = modifiers.to_dict()
        except Exception as e:
            logger.debug(f"Event modifier error for agent {agent.social_agent_id}: {e}")
            agent._event_modifiers = {}

    # Track current step for event manager
    global_step = 0
    
    # Track errors per step
    step_errors: Dict[int, List[str]] = {}

    # Warmup: run a few LLMAction steps to populate initial content before main loop
    logger.info(f"Starting {warmup_steps} warmup steps...")
    warmup_total_start = time.perf_counter()
    
    for i in range(max(0, int(warmup_steps))):
        step_start = time.perf_counter()
        logger.info(f"WARMUP STEP {i + 1}/{warmup_steps} starting...")
        agents_snapshot = list(env.agent_graph.get_agents())
        agent_count = len(agents_snapshot)
        print(f"[production] Starting warmup {i + 1}/{warmup_steps}, building actions dict for {agent_count} agents...")
        sys.stdout.flush()
        
        # Advance coordinator (if enabled)
        coord_time = 0
        if coordinator:
            t_coord = time.perf_counter()
            try:
                coordinator.step()
                coord_time = time.perf_counter() - t_coord
                logger.debug(f"Coordinator stepped in {coord_time:.3f}s")
            except Exception as e:
                logger.error(f"Coordinator step error: {e}")
        
        # Advance event manager (if enabled)
        event_time = 0
        if event_manager:
            t_event = time.perf_counter()
            try:
                active_events = event_manager.step(global_step)
                event_time = time.perf_counter() - t_event
                if active_events:
                    event_ids = [e.id for e in active_events]
                    logger.info(f"Active events: {event_ids} (took {event_time:.3f}s)")
                    print(f"[production] Active events: {event_ids}")
            except Exception as e:
                logger.error(f"EventManager step error: {e}")
        
        # Apply coordination and event modifiers to each agent
        t_modifiers = time.perf_counter()
        modifier_errors = 0
        for _, agent in agents_snapshot:
            try:
                _apply_coordination_modifiers(agent, coordinator)
                _apply_event_modifiers(agent, event_manager)
            except Exception as e:
                modifier_errors += 1
                logger.debug(f"Modifier application error: {e}")
        modifier_time = time.perf_counter() - t_modifiers
        
        if modifier_errors > 0:
            logger.warning(f"Had {modifier_errors} modifier application errors")
        
        global_step += 1
        
        # Build actions dict
        t_actions = time.perf_counter()
        try:
            actions = {agent: oasis.LLMAction() for _, agent in agents_snapshot}
            actions_time = time.perf_counter() - t_actions
            logger.debug(f"Actions dict built with {len(actions)} agents in {actions_time:.3f}s")
        except Exception as e:
            logger.error(f"FAILED to build actions dict: {e}")
            logger.error(traceback.format_exc())
            raise
        
        print(f"[production] Actions dict built. Calling env.step()...")
        sys.stdout.flush()
        
        # Execute step - this is the main LLM call
        t_step = time.perf_counter()
        try:
            await env.step(actions)
            llm_time = time.perf_counter() - t_step
            timing_stats.record("warmup_step", llm_time)
            timing_stats.record("llm_calls", llm_time)
            logger.info(f"Warmup step {i + 1} LLM calls completed in {llm_time:.2f}s")
            
            # Collect agent-level timing stats
            for _, agent in agents_snapshot:
                agent_timing = getattr(agent, "_last_timing", None)
                if agent_timing:
                    timing_stats.record_agent_timing(agent_timing)
                    
        except Exception as e:
            llm_time = time.perf_counter() - t_step
            logger.error(f"FAILED env.step() in warmup {i + 1} after {llm_time:.2f}s: {e}")
            logger.error(traceback.format_exc())
            step_errors.setdefault(global_step, []).append(str(e))
            timing_stats.record("warmup_step_error", llm_time)
            continue
        
        step_elapsed = time.perf_counter() - step_start
        
        # Detailed warmup statistics
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM post")
            post_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM comment")
            comment_count = cur.fetchone()[0]
            conn.close()
            
            rate = agent_count / llm_time if llm_time > 0 else 0
            logger.info(f"Warmup {i + 1} timing breakdown: total={step_elapsed:.2f}s, llm={llm_time:.2f}s, modifiers={modifier_time:.3f}s, coord={coord_time:.3f}s, event={event_time:.3f}s")
            print(f"[production] Warmup {i + 1}/{warmup_steps}: {step_elapsed:.2f}s ({rate:.1f} agents/s) | "
                  f"Posts: {post_count}, Comments: {comment_count}")
        except Exception as e:
            print(f"[production] warmup {i + 1}/{warmup_steps} took {step_elapsed:.2f}s (stats error: {e})")
        sys.stdout.flush()
        
        # Imputation timing
        t_impute = time.perf_counter()
        await _maybe_impute()
        impute_time = time.perf_counter() - t_impute
        if impute_time > 0.1:
            timing_stats.record("imputation", impute_time)
    
    warmup_total = time.perf_counter() - warmup_total_start
    if warmup_steps > 0:
        logger.info(f"Warmup completed: {warmup_steps} steps in {warmup_total:.2f}s (avg {warmup_total/warmup_steps:.2f}s/step)")

    # LLMAction for all agents across steps
    logger.info("=" * 60)
    logger.info(f"STARTING MAIN SIMULATION: {steps} steps")
    logger.info("=" * 60)
    
    main_loop_start = time.perf_counter()
    
    for i in range(max(0, int(steps))):
        step_start = time.perf_counter()
        logger.info(f"MAIN STEP {i + 1}/{steps} starting (global_step={global_step})...")
        agents_snapshot = list(env.agent_graph.get_agents())
        agent_count = len(agents_snapshot)
        print(f"[production] Starting step {i + 1}/{steps}, building actions dict for {agent_count} agents...")
        sys.stdout.flush()
        
        # Advance coordinator (if enabled)
        coord_time = 0
        if coordinator:
            t_coord = time.perf_counter()
            try:
                coordinator.step()
                coord_time = time.perf_counter() - t_coord
                logger.debug(f"Coordinator stepped in {coord_time:.3f}s")
            except Exception as e:
                logger.error(f"Coordinator step error: {e}")
        
        # Advance event manager (if enabled)
        event_time = 0
        if event_manager:
            t_event = time.perf_counter()
            try:
                active_events = event_manager.step(global_step)
                event_time = time.perf_counter() - t_event
                if active_events:
                    event_ids = [e.id for e in active_events]
                    logger.info(f"Active events: {event_ids} (took {event_time:.3f}s)")
                    print(f"[production] Active events: {event_ids}")
            except Exception as e:
                logger.error(f"EventManager step error: {e}")
        
        # Apply coordination and event modifiers to each agent
        t_modifiers = time.perf_counter()
        modifier_errors = 0
        for _, agent in agents_snapshot:
            try:
                _apply_coordination_modifiers(agent, coordinator)
                _apply_event_modifiers(agent, event_manager)
            except Exception as e:
                modifier_errors += 1
                logger.debug(f"Modifier error: {e}")
        modifier_time = time.perf_counter() - t_modifiers
        
        if modifier_errors > 0:
            logger.warning(f"Had {modifier_errors} modifier errors in step {i + 1}")
        
        global_step += 1
        
        # Build actions dict
        t_actions = time.perf_counter()
        try:
            actions = {agent: oasis.LLMAction() for _, agent in agents_snapshot}
            actions_time = time.perf_counter() - t_actions
            logger.debug(f"Actions dict built with {len(actions)} agents in {actions_time:.3f}s")
        except Exception as e:
            logger.error(f"FAILED to build actions dict in step {i + 1}: {e}")
            logger.error(traceback.format_exc())
            step_errors.setdefault(global_step, []).append(f"actions_build: {e}")
            continue
        
        print(f"[production] Actions dict built. Calling env.step()...")
        sys.stdout.flush()
        
        # Execute step - this is the main LLM call
        t_step = time.perf_counter()
        try:
            await env.step(actions)
            llm_time = time.perf_counter() - t_step
            timing_stats.record("main_step", llm_time)
            timing_stats.record("llm_calls", llm_time)
            logger.info(f"Main step {i + 1} LLM calls completed in {llm_time:.2f}s")
            
            # Collect agent-level timing stats
            for _, agent in agents_snapshot:
                agent_timing = getattr(agent, "_last_timing", None)
                if agent_timing:
                    timing_stats.record_agent_timing(agent_timing)
                    
        except Exception as e:
            llm_time = time.perf_counter() - t_step
            logger.error(f"FAILED env.step() in main step {i + 1} after {llm_time:.2f}s: {e}")
            logger.error(traceback.format_exc())
            step_errors.setdefault(global_step, []).append(f"env_step: {e}")
            timing_stats.record("main_step_error", llm_time)
            print(f"[production] Step {i + 1} FAILED after {llm_time:.2f}s - continuing...")
            continue
        
        step_elapsed = time.perf_counter() - step_start
        
        # Detailed step statistics
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM post")
            post_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM comment")
            comment_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM trace WHERE step = ?", (global_step,))
            actions_this_step = cur.fetchone()[0]
            conn.close()
            
            rate = agent_count / llm_time if llm_time > 0 else 0
            logger.info(f"Step {i + 1} timing breakdown: total={step_elapsed:.2f}s, llm={llm_time:.2f}s, modifiers={modifier_time:.3f}s, coord={coord_time:.3f}s, event={event_time:.3f}s")
            print(f"[production] Step {i + 1}/{steps}: {step_elapsed:.2f}s ({rate:.1f} agents/s) | "
                  f"Posts: {post_count}, Comments: {comment_count}, Actions this step: {actions_this_step}")
        except Exception as e:
            logger.warning(f"Stats collection error: {e}")
            print(f"[production] step {i + 1}/{steps} took {step_elapsed:.2f}s (stats error: {e})")
        sys.stdout.flush()

        # Imputation timing
        t_impute = time.perf_counter()
        try:
            await _maybe_impute()
            impute_time = time.perf_counter() - t_impute
            if impute_time > 0.1:
                timing_stats.record("imputation", impute_time)
        except Exception as e:
            logger.error(f"Imputation error in step {i + 1}: {e}")

    # Record main loop total
    main_loop_total = time.perf_counter() - main_loop_start
    if steps > 0:
        logger.info(f"Main loop completed: {steps} steps in {main_loop_total:.2f}s (avg {main_loop_total/steps:.2f}s/step)")

    # Final cleanup
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE - CLEANUP")
    logger.info("=" * 60)
    
    # Log error summary
    if step_errors:
        logger.warning(f"Encountered errors in {len(step_errors)} steps:")
        for step_num, errors in step_errors.items():
            for err in errors:
                logger.warning(f"  Step {step_num}: {err}")
    else:
        logger.info("No step errors encountered")
    
    # Final imputation
    if imputer is not None:
        with Timer("final_imputation") as t:
            try:
                await imputer.enqueue_new_rows()
                await imputer.flush()
                logger.info("Final imputation complete")
            except Exception as e:
                logger.error(f"Final imputation error: {e}")
        timing_stats.record("cleanup", t.elapsed)
    
    # Close environment
    with Timer("close_environment") as t:
        try:
            await env.close()
            logger.info("Environment closed")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    timing_stats.record("cleanup", t.elapsed)
    
    # Shutdown imputer
    if imputer is not None:
        with Timer("shutdown_imputer") as t:
            try:
                await imputer.shutdown()
                logger.info("Imputer shutdown complete")
            except Exception as e:
                logger.error(f"Imputer shutdown error: {e}")
        timing_stats.record("cleanup", t.elapsed)
    
    total_time = time.perf_counter() - run_t0
    logger.info(f"Total run time: {total_time:.2f}s")
    print(f"[production] Total run time: {total_time:.2f}s")
    
    # Log timing summary
    timing_stats.log_summary()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Steps completed: {steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Total time: {total_time:.2f}s")
    if steps > 0:
        logger.info(f"Average step time: {main_loop_total/steps:.2f}s")
    logger.info(f"Errors: {len(step_errors)} steps had errors")
    logger.info(f"Log file: log/production_sim_*.log")
    
    # Print timing summary to console
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print("STEP-LEVEL TIMINGS:")
    for category in sorted(timing_stats.timings.keys()):
        stats = timing_stats.get_stats(category)
        print(f"  {category:23s} | count={stats['count']:4d} | total={stats['total']:7.2f}s | avg={stats['avg']:6.3f}s")
    
    if timing_stats.agent_timings:
        print("-" * 70)
        print("AGENT-LEVEL TIMINGS (per-agent breakdown):")
        for category in sorted(timing_stats.agent_timings.keys()):
            stats = timing_stats.get_agent_stats(category)
            print(f"  agent.{category:19s} | count={stats['count']:4d} | total={stats['total']:7.2f}s | avg={stats['avg']:6.4f}s")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production simulation with ExtendedSocialAgent (LLMAction only)."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest YAML.")
    parser.add_argument("--personas-csv", type=str, required=True, help="Path to personas CSV with class columns.")
    parser.add_argument("--db", type=str, required=True, help="Path to output sqlite DB.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    parser.add_argument("--edges-csv", type=str, default="", help="Path to edges CSV (follower_id,followee_id). Env PROD_EDGES_CSV overrides.")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup LLMAction steps before main loop.")
    parser.add_argument(
        "--rag-imputer",
        type=str,
        choices=["off", "background", "sync"],
        default=None,
        help="Enable LLM-based placeholder replacement (default derives from config).",
    )
    parser.add_argument("--rag-workers", type=int, default=0, help="Override RAG imputer worker count.")
    parser.add_argument("--rag-batch-size", type=int, default=0, help="Override queue size per batch.")
    parser.add_argument(
        "--enable-vector-retrieval",
        action="store_true",
        default=True,
        help="Enable vector DB retrieval for few-shot examples (default: True).",
    )
    parser.add_argument(
        "--disable-vector-retrieval",
        action="store_true",
        help="Disable vector DB retrieval (use LLM only).",
    )
    parser.add_argument(
        "--tfidf-only",
        action="store_true",
        help="Use TF-IDF only (no ChromaDB) for vector retrieval.",
    )
    parser.add_argument(
        "--fresh-db",
        action="store_true",
        help="If set, delete the existing DB file before running.",
    )
    parser.add_argument(
        "--unique-db",
        action="store_true",
        help="If set, append a timestamp to the DB filename to create a new DB per run.",
    )
    # Optional: Auto-generate JSONL and visualizations after run
    parser.add_argument(
        "--report",
        action="store_true",
        help="If set, generate JSONL export and visualizations after the run.",
    )
    parser.add_argument(
        "--report-out-dir",
        type=str,
        default="",
        help="Output directory for reports (defaults to <db_dir>/production).",
    )
    parser.add_argument(
        "--report-export-jsonl",
        type=str,
        default="",
        help="Path to write JSONL export (defaults to <out_dir>/production_export.jsonl).",
    )
    parser.add_argument(
        "--report-export-actions",
        type=str,
        default="",
        help="Path to write actions JSONL (defaults to <out_dir>/production_actions.jsonl).",
    )
    parser.add_argument(
        "--report-threads-html",
        type=str,
        default="",
        help="Path to write threaded HTML (defaults to <out_dir>/production_threads.html).",
    )
    # Enhanced simulation features
    parser.add_argument(
        "--enable-thread-dynamics",
        action="store_true",
        help="Enable thread dynamics (pile-ons, echo chambers, debates).",
    )
    parser.add_argument(
        "--enable-obfuscation",
        action="store_true",
        help="Enable post-imputation obfuscation of harmful terms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    personas_csv = Path(os.path.abspath(args.personas_csv))
    db_path = Path(os.path.abspath(args.db))
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Choose DB handling strategy
    if args.unique_db:
        # Append timestamp before suffix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = db_path.with_name(f"{db_path.stem}_{ts}{db_path.suffix}")
    elif args.fresh_db and db_path.exists():
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"[production] Failed to remove existing DB: {e}")
    edges_csv = Path(os.path.abspath(args.edges_csv)) if args.edges_csv else None
    # Determine vector retrieval settings
    enable_vector = args.enable_vector_retrieval and not args.disable_vector_retrieval
    asyncio.run(
        run(
            manifest_path,
            personas_csv,
            db_path,
            args.steps,
            edges_csv,
            args.warmup_steps,
            args.rag_imputer or "",
            args.rag_workers,
            args.rag_batch_size,
            enable_thread_dynamics=args.enable_thread_dynamics,
            enable_obfuscation=args.enable_obfuscation,
            enable_vector_retrieval=enable_vector,
            tfidf_only=args.tfidf_only,
        )
    )

    # Optional post-run report generation (uses AUTO_REPORT from config as default)
    should_report = args.report or AUTO_REPORT
    if should_report:
        try:
            report_script = Path(__file__).resolve().parent / "report_production.py"
            if not report_script.exists():
                print(f"[production] Report script not found at {report_script}")
                return
            db_dir = db_path.parent
            out_dir = Path(args.report_out_dir) if args.report_out_dir else (db_dir / "production")
            export_jsonl = Path(args.report_export_jsonl) if args.report_export_jsonl else (out_dir / "production_export.jsonl")
            export_actions = Path(args.report_export_actions) if args.report_export_actions else (out_dir / "production_actions.jsonl")
            threads_html = Path(args.report_threads_html) if args.report_threads_html else (out_dir / "production_threads.html")
            sidecar = db_dir / "sidecar.jsonl"
            cmd = [
                sys.executable,
                str(report_script),
                "--db",
                str(db_path),
                "--out-dir",
                str(out_dir),
                "--export-jsonl",
                str(export_jsonl),
                "--export-actions",
                str(export_actions),
                "--threads-html",
                str(threads_html),
            ]
            if sidecar.exists():
                cmd.extend(["--sidecar", str(sidecar)])
            print(f"[production] Generating report via: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[production] Report generation failed: {e}")


if __name__ == "__main__":
    main()


