"""Enhanced agent factory with trajectory, fluency, contamination support.

This module extends the base agent_factory to support:
- Enhanced persona configurations (trajectory, fluency, contamination)
- Thread coordination modifiers
- Obfuscation post-processing

The enhanced agents still work with the existing RecSys - they just have
additional behavior modifiers that affect HOW they generate content.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from camel.models import BaseModelBackend

from generation.emission_policy import EmissionPolicy, PersonaConfig
from generation.labeler import DEFAULT_TOKEN_TO_CATEGORIES, set_token_category_map
from generation.extended_agent import ExtendedSocialAgent
from generation.enhanced_persona_generator import EnhancedPersonaConfig
from generation.trajectory_config import get_trajectory_config, get_fluency_config
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType
from orchestrator.sidecar_logger import SidecarLogger


def _load_json_field(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a JSON string field, returning None on failure."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _parse_list_field(raw: Optional[str]) -> List[str]:
    """Parse a comma-separated or JSON list field."""
    if not raw:
        return []
    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Fall back to comma-separated
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_enhanced_persona_config(row: Dict[str, str], idx: int) -> Optional[EnhancedPersonaConfig]:
    """Load enhanced persona config from CSV row if enhancement columns exist."""
    # Check if enhancement columns exist
    if "trajectory_stage" not in row:
        return None
    
    return EnhancedPersonaConfig(
        user_id=idx,
        username=row.get("username", f"user_{idx}"),
        name=row.get("username", f"user_{idx}"),
        primary_label=row.get("primary_label", "benign"),
        bio=row.get("description", ""),
        persona_prompt_post=row.get("user_char", ""),
        persona_prompt_comment=row.get("user_char", ""),
        persona_goal=_load_json_field(row.get("prompt_metadata_json") or "{}").get("persona_goal", ""),
        persona_style_quirks=_load_json_field(row.get("prompt_metadata_json") or "{}").get("style_quirks", ""),
        trajectory_stage=row.get("trajectory_stage", "active"),
        slang_fluency=row.get("slang_fluency", "fluent"),
        secondary_archetype=row.get("secondary_archetype") or None,
        secondary_influence=float(row.get("secondary_influence", 0.0) or 0.0),
        benign_post_rate=float(row.get("benign_post_rate", 0.15) or 0.15),
        token_density_multiplier=float(row.get("token_density_multiplier", 1.0) or 1.0),
        aggression_multiplier=float(row.get("aggression_multiplier", 1.0) or 1.0),
        demographic_profile=row.get("demographic_profile", "default"),
        obfuscation_intensity=float(row.get("obfuscation_intensity", 0.3) or 0.3),
        allowed_label_tokens=_parse_list_field(row.get("allowed_label_tokens")),
        lexicon_required=_parse_list_field(row.get("lexicon_required")),
        lexicon_optional=_parse_list_field(row.get("lexicon_optional")),
    )


async def build_enhanced_agent_graph_from_csv(
    personas_csv: Path,
    model: BaseModelBackend,
    channel: Channel,
    available_actions: Optional[List[ActionType]] = None,
    emission_policy: Optional[EmissionPolicy] = None,
    sidecar_logger: Optional[SidecarLogger] = None,
    run_seed: int = 314159,
    guidance_config: Optional[Dict[str, Any]] = None,
    enable_obfuscation: bool = True,
    **extended_kwargs: Any,
) -> AgentGraph:
    """Create an AgentGraph with enhanced persona support.
    
    This function extends build_agent_graph_from_csv to:
    - Load enhanced persona columns (trajectory, fluency, contamination)
    - Apply trajectory/fluency modifiers to agent behavior
    - Enable obfuscation post-processing
    
    The agents still work with the existing RecSys - they see the same
    timeline, but their response generation is modified by the enhancements.
    
    Args:
        personas_csv: Path to personas CSV (can be base or enhanced)
        model: LLM model backend
        channel: Platform channel
        available_actions: Allowed action types
        emission_policy: Token emission policy
        sidecar_logger: Logging
        run_seed: Deterministic seed
        guidance_config: Guidance configuration
        enable_obfuscation: Whether to apply obfuscation post-processing
        **extended_kwargs: Additional kwargs for ExtendedSocialAgent
    
    Returns:
        AgentGraph with enhanced agents
    """
    graph = AgentGraph()
    token_category_map: Dict[str, List[str]] = {
        token: cats[:] for token, cats in DEFAULT_TOKEN_TO_CATEGORIES.items()
    }
    
    with personas_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Track enhanced configs for agents
    enhanced_configs: Dict[int, EnhancedPersonaConfig] = {}
    
    for idx, row in enumerate(rows):
        username: str = row.get("username") or f"user_{idx}"
        description: str = row.get("description", "")
        user_char: str = row.get("user_char", "")

        # Require explicit persona columns
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

        # Load enhanced persona config if available
        enhanced_cfg = _load_enhanced_persona_config(row, idx)
        if enhanced_cfg:
            enhanced_configs[idx] = enhanced_cfg
            
            # Apply trajectory modifiers to guidance config
            traj_cfg = get_trajectory_config(enhanced_cfg.trajectory_stage)
            fluency_cfg = get_fluency_config(enhanced_cfg.slang_fluency)
            
            # Merge trajectory/fluency into guidance
            enhanced_guidance = dict(guidance_config or {})
            enhanced_guidance["trajectory_stage"] = enhanced_cfg.trajectory_stage
            enhanced_guidance["slang_fluency"] = enhanced_cfg.slang_fluency
            enhanced_guidance["token_density_multiplier"] = traj_cfg.token_density_multiplier
            enhanced_guidance["aggression_multiplier"] = traj_cfg.aggression_multiplier
            enhanced_guidance["benign_post_rate"] = enhanced_cfg.benign_post_rate
            enhanced_guidance["secondary_archetype"] = enhanced_cfg.secondary_archetype
            enhanced_guidance["secondary_influence"] = enhanced_cfg.secondary_influence
            enhanced_guidance["obfuscation_intensity"] = enhanced_cfg.obfuscation_intensity
            enhanced_guidance["enable_obfuscation"] = enable_obfuscation
            agent_guidance = enhanced_guidance
        else:
            agent_guidance = guidance_config

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

        persona_cfg = PersonaConfig(
            persona_id=f"{primary}_{idx:04d}",
            primary_label=str(primary),
            allowed_labels=allowed,
            label_mode_cap=str(label_mode_cap),
            benign_on_none_prob=0.6,
            max_labels_per_post=2,
            emission_probs=emission_params,
            pair_probs=pair_probs,
            allowed_label_tokens=allowed_label_tokens,
            prompt_metadata=prompt_metadata,
            lexicon_samples=lexicon_samples,
            style_variation=style_variation,
        )

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
            guidance_config=agent_guidance or {},
            **extended_kwargs,
        )
        graph.add_agent(agent)
        
        # Merge token map
        if allowed_label_tokens:
            for label, tokens in allowed_label_tokens.items():
                for token in tokens:
                    bucket = token_category_map.setdefault(token, [])
                    if label not in bucket:
                        bucket.append(label)

    set_token_category_map(token_category_map)
    
    # Store enhanced configs on the graph for later use
    graph._enhanced_configs = enhanced_configs
    
    return graph

