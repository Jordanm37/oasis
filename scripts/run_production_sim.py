#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

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
        SimulationCoordinator,
        SimulationCoordinatorConfig,
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    SimulationCoordinator = None
    SimulationCoordinatorConfig = None


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
) -> None:
    run_t0 = time.perf_counter()
    print(f"[production] Run started at {datetime.now().isoformat()}")
    manifest = load_manifest(manifest_path)
    run_seed = manifest.run_seed

    # Choose edges path with env override, then validate inputs.
    override = os.getenv("PROD_EDGES_CSV")
    edges_path = Path(override) if override else edges_csv
    _validate_personas_and_edges(manifest, personas_csv, edges_path)

    sidecar = SidecarLogger(path=db_path.parent / "sidecar.jsonl")

    # Load ontology-driven label→lexicon map (best-effort; defaults if missing)
    try:
        default_ontology = Path("configs/personas/ontology_unified.yaml")
        label_lexicons = load_label_lexicons(default_ontology) if default_ontology.exists() else {}
    except Exception:
        label_lexicons = {}
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
            )
            imputer = RagImputer(imputer_cfg)
            await imputer.start()
            if enable_obfuscation:
                print("[production] Post-imputation obfuscation ENABLED")
        except Exception as exc:
            print(f"[production] Failed to start rag imputer: {exc}")
            imputer = None
            rag_mode_value = "off"
    policy = EmissionPolicy(
        run_seed=run_seed,
        post_label_mode_probs=manifest.post_label_mode_probs,
        label_to_tokens=None,  # defaults cover the 6-parent-family setup
        label_lexicon_terms=label_lexicons or None,
    )
    expect_registry = ExpectRegistry()
    targets_cfg = manifest.multi_label_targets
    scheduler = MultiLabelScheduler(
        MultiLabelTargets(
            target_rate=targets_cfg["target_rate"],
            min_rate=targets_cfg["min_rate"],
            max_rate=targets_cfg["max_rate"],
        )
    )

    # Define the model for the agents (LLMAction only; no manual actions)
    # Use fallback backend for automatic retry on rate limits
    model = create_fallback_backend(LLM_SETTINGS)

    # Setup platform and channel
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
        _AT.REPOST,
        # Discovery (needed to see content)
        _AT.REFRESH,
        # Neutral action
        _AT.DO_NOTHING,
        # Optional: uncomment for more variety (but may increase tool call errors)
        _AT.QUOTE_POST,
        # _AT.FOLLOW,
        # _AT.SEARCH_POSTS,
    ]

    # Build agent graph using ExtendedSocialAgent with persona class metadata
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

    # Initialize SimulationCoordinator for thread dynamics (optional)
    coordinator: Optional[SimulationCoordinator] = None
    if enable_thread_dynamics and ENHANCED_FEATURES_AVAILABLE and SimulationCoordinator is not None:
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

    # Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
    )

    async def _maybe_impute() -> None:
        if imputer is None:
            return
        await imputer.enqueue_new_rows()
        if rag_mode_value == "sync":
            await imputer.flush()

    t0 = time.perf_counter()
    await env.reset()
    print(f"[production] env.reset() took {time.perf_counter() - t0:.2f}s")
    await _maybe_impute()

    # Seed initial follow graph if provided and table is empty
    try:
        if _follow_table_empty(env):
            if edges_path:
                seeded = _seed_initial_follows_from_csv(env, edges_path)
                if seeded:
                    print(
                        f"[production] Seeded {seeded} follow edges from {edges_path}"
                    )
        else:
            print("[production] Follow table non-empty; skipping follow seed.")
    except Exception as e:
        print(f"[production] Follow seeding failed: {e}")

    print("[DEBUG] About to enter warmup loop...")
    sys.stdout.flush()

    def _apply_coordination_modifiers(agent, coord: Optional[SimulationCoordinator]) -> None:
        """Apply thread coordination modifiers to an agent before their action."""
        if coord is None:
            return
        modifiers = coord.get_agent_modifiers(agent.social_agent_id)
        # Store modifiers on agent for use in perform_action_by_llm
        agent._coordination_modifiers = modifiers

    # Warmup: run a few LLMAction steps to populate initial content before main loop
    for i in range(max(0, int(warmup_steps))):
        print(f"[production] Starting warmup {i + 1}/{warmup_steps}, building actions dict for {len(list(env.agent_graph.get_agents()))} agents...")
        sys.stdout.flush()
        
        # Advance coordinator (if enabled)
        if coordinator:
            coordinator.step()
        
        # Apply coordination modifiers to each agent
        for _, agent in env.agent_graph.get_agents():
            _apply_coordination_modifiers(agent, coordinator)
        
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        print(f"[production] Actions dict built. Calling env.step()...")
        sys.stdout.flush()
        t_step = time.perf_counter()
        await env.step(actions)
        print(f"[production] warmup {i + 1}/{warmup_steps} took {time.perf_counter() - t_step:.2f}s")
        sys.stdout.flush()
        await _maybe_impute()

    # LLMAction for all agents across steps
    for i in range(max(0, int(steps))):
        print(f"[production] Starting step {i + 1}/{steps}, building actions dict for {len(list(env.agent_graph.get_agents()))} agents...")
        sys.stdout.flush()
        
        # Advance coordinator (if enabled)
        if coordinator:
            coordinator.step()
        
        # Apply coordination modifiers to each agent
        for _, agent in env.agent_graph.get_agents():
            _apply_coordination_modifiers(agent, coordinator)
        
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        print(f"[production] Actions dict built. Calling env.step()...")
        sys.stdout.flush()
        t_step = time.perf_counter()
        await env.step(actions)
        print(f"[production] step {i + 1}/{steps} took {time.perf_counter() - t_step:.2f}s")
        sys.stdout.flush()
        await _maybe_impute()

    if imputer is not None:
        await imputer.enqueue_new_rows()
        await imputer.flush()
    await env.close()
    if imputer is not None:
        await imputer.shutdown()
    print(f"[production] Total run time: {time.perf_counter() - run_t0:.2f}s")


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


