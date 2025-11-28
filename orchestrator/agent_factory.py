from __future__ import annotations

import asyncio
import csv
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from camel.models import BaseModelBackend

from generation.emission_policy import EmissionPolicy, PersonaConfig
from generation.labeler import DEFAULT_TOKEN_TO_CATEGORIES, set_token_category_map
from generation.extended_agent import ExtendedSocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType
from orchestrator.sidecar_logger import SidecarLogger

# Set up logging
logger = logging.getLogger("agent_factory")


def _infer_primary_label(username: str) -> str:
    lowered = username.lower()
    if lowered.startswith("incel_"):
        return "incel"
    if lowered.startswith("misinfo_"):
        return "misinfo"
    if lowered.startswith("benign_"):
        return "benign"
    # default benign
    return "benign"


def _load_json_field(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


async def build_agent_graph_from_csv(
    personas_csv: Path,
    model: BaseModelBackend,
    channel: Channel,
    available_actions: Optional[List[ActionType]] = None,
    emission_policy: Optional[EmissionPolicy] = None,
    sidecar_logger: Optional[SidecarLogger] = None,
    run_seed: int = 314159,
    guidance_config: Optional[Dict[str, Any]] = None,
    message_window_size: Optional[int] = 10,
    **extended_kwargs: Any,
) -> AgentGraph:
    r"""Create an AgentGraph using ExtendedSocialAgent from a simple personas CSV.

    CSV columns expected: username, description, user_char, primary_label,
    secondary_label (optional), allowed_labels (JSON list), label_mode_cap
    """
    import time
    build_start = time.perf_counter()
    
    logger.info(f"Building agent graph from {personas_csv}")
    
    graph = AgentGraph()
    token_category_map: Dict[str, List[str]] = {
        token: cats[:] for token, cats in DEFAULT_TOKEN_TO_CATEGORIES.items()
    }
    
    # Load CSV
    csv_load_start = time.perf_counter()
    logger.debug(f"Loading personas CSV: {personas_csv}")
    try:
        with personas_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        csv_load_time = time.perf_counter() - csv_load_start
        logger.info(f"Loaded {len(rows)} personas from CSV in {csv_load_time:.3f}s")
    except Exception as e:
        logger.error(f"FAILED to load personas CSV: {e}")
        raise
    
    # Track loading stats
    loaded_count = 0
    error_count = 0
    trajectory_stats = {"curious": 0, "active": 0, "entrenched": 0}
    agent_creation_times = []
    
    for idx, row in enumerate(rows):
        username: str = row.get("username") or f"user_{idx}"
        description: str = row.get("description", "")
        user_char: str = row.get("user_char", "")

        # Require explicit persona columns; do not infer
        primary = row.get("primary_label")
        label_mode_cap = row.get("label_mode_cap")
        allowed_raw = row.get("allowed_labels")
        if primary is None or label_mode_cap is None or allowed_raw is None:
            raise ValueError(
                "Persona CSV must include `primary_label`, `label_mode_cap`, and `allowed_labels`."
            )
        try:
            allowed = json.loads(allowed_raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid JSON in `allowed_labels` for username={username}"
            ) from exc

        # Optional per-persona emission configuration
        emission_params = None
        emission_params_raw = row.get("emission_params_json")
        if emission_params_raw:
            try:
                parsed = json.loads(emission_params_raw)
                if isinstance(parsed, dict):
                    emission_params = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                emission_params = None

        # Optional per-persona label pair preferences
        pair_probs = None
        pair_probs_raw = row.get("pair_probs_json")
        if pair_probs_raw:
            try:
                parsed = json.loads(pair_probs_raw)
                if isinstance(parsed, dict):
                    pair_probs = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pair_probs = None

        # Optional harm priors metadata
        harm_priors = None
        harm_priors_raw = row.get("harm_priors_json")
        if harm_priors_raw:
            try:
                parsed = json.loads(harm_priors_raw)
                if isinstance(parsed, dict):
                    harm_priors = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                harm_priors = None

        # Optional new persona metadata
        prompt_metadata = _load_json_field(row.get("prompt_metadata_json"))
        lexicon_samples = _load_json_field(row.get("lexicon_samples_json"))
        style_variation = _load_json_field(row.get("style_variation_json"))
        allowed_label_tokens = None
        tokens_raw = row.get("allowed_label_tokens_json")
        if tokens_raw:
             try:
                parsed_tokens = json.loads(tokens_raw)
                if isinstance(parsed_tokens, dict):
                    allowed_label_tokens = {
                        str(label): [str(tok) for tok in tokens]
                        for label, tokens in parsed_tokens.items()
                        if isinstance(tokens, list)
                    }
             except Exception:
                allowed_label_tokens = None

        # Read trajectory stage from CSV (default to "active" if not present)
        trajectory_stage = row.get("trajectory_stage", "active")
        slang_fluency = row.get("slang_fluency", "fluent")
        
        # Track trajectory stats
        if trajectory_stage in trajectory_stats:
            trajectory_stats[trajectory_stage] += 1

        try:
            persona_cfg = PersonaConfig(
                persona_id=f"{primary}_{idx:04d}",
                primary_label=str(primary),
                allowed_labels=allowed,
                label_mode_cap=str(label_mode_cap),
                trajectory_stage=str(trajectory_stage),
                slang_fluency=str(slang_fluency),
                benign_on_none_prob=0.6,
                max_labels_per_post=2,
                emission_probs=emission_params,
                pair_probs=pair_probs,
                allowed_label_tokens=allowed_label_tokens,
                prompt_metadata=prompt_metadata,
                lexicon_samples=lexicon_samples,
                style_variation=style_variation,
            )
            logger.debug(f"Created PersonaConfig for {username}: {primary}, stage={trajectory_stage}")
        except Exception as e:
            logger.error(f"FAILED to create PersonaConfig for {username}: {e}")
            error_count += 1
            continue

        user_profile_data = {
            "user_profile": user_char,
            "prompt_metadata": prompt_metadata or {},
            "lexicon_samples": lexicon_samples or {},
            "style_variation": style_variation or {},
        }

        user_info = UserInfo(
            name=username,
            description=description,
            profile={"other_info": user_profile_data},
            recsys_type="twitter",
        )
        
        agent_start = time.perf_counter()
        try:
            agent = ExtendedSocialAgent(
                agent_id=idx,
                user_info=user_info,
                channel=channel,
                model=model,
                agent_graph=graph,
                available_actions=available_actions,
                persona_cfg=persona_cfg,
                emission_policy=emission_policy,
                sidecar_logger=sidecar_logger,
                run_seed=run_seed,
                harm_priors=harm_priors,
                guidance_config=guidance_config or {},
                message_window_size=message_window_size,
                **extended_kwargs,
            )
            graph.add_agent(agent)
            agent_time = time.perf_counter() - agent_start
            agent_creation_times.append(agent_time)
            loaded_count += 1
            logger.debug(f"Created agent {idx}: {username} in {agent_time:.3f}s")
        except Exception as e:
            logger.error(f"FAILED to create agent for {username}: {e}")
            logger.error(traceback.format_exc())
            error_count += 1
            continue
        
        # Merge token map
        if allowed_label_tokens:
            for label, tokens in allowed_label_tokens.items():
                for token in tokens:
                    bucket = token_category_map.setdefault(token, [])
                    if label not in bucket:
                        bucket.append(label)

    # Calculate timing stats
    build_total = time.perf_counter() - build_start
    avg_agent_time = sum(agent_creation_times) / len(agent_creation_times) if agent_creation_times else 0
    
    # Log summary
    logger.info(f"Agent graph built: {loaded_count} agents loaded, {error_count} errors")
    logger.info(f"Trajectory distribution: {trajectory_stats}")
    logger.info(f"Build timing: total={build_total:.2f}s, avg_per_agent={avg_agent_time:.4f}s")
    
    if error_count > 0:
        logger.warning(f"Had {error_count} errors loading agents!")
    
    set_token_category_map(token_category_map)
    return graph


