#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import aiohttp
import requests
from dotenv import load_dotenv
from yaml import safe_load

load_dotenv()

from configs.llm_settings import (API_ENDPOINTS, API_KEY_ENV_VARS,
                                  FALLBACK_ENABLED, FALLBACK_MAX_RETRIES,
                                  FALLBACK_MODELS, MULTI_MODEL_ENABLED,
                                  PERSONA_MAX_TOKENS, PERSONA_MODEL,
                                  PERSONA_PROVIDER, PERSONA_TEMPERATURE,
                                  PROVIDER_POOL, PROVIDER_POOL_ENABLED,
                                  PROVIDER_WEIGHTS, STRUCTURED_OUTPUT_MODELS,
                                  TRAJECTORY_STAGE_DISTRIBUTION,
                                  TRAJECTORY_STAGE_MODIFIERS,
                                  get_api_keys_for_provider,
                                  get_model_key_pairs,
                                  get_parallel_model_summary,
    get_provider_for_model,
                                  supports_structured_output)
from generation.emission_policy import DEFAULT_LABEL_TO_TOKENS
from oasis.persona import (PersonaGenerator, PersonaSeed,
                           PromptSynthesisResult, build_llm_prompt,
                           build_requests_from_spec, load_ontology,
                           load_persona_seeds, sample_seed)

DEFAULT_ONTOLOGY = Path("configs/personas/ontology_unified.yaml")
FIELDNAMES = [
    "username",
    "description",
    "user_char",
    "primary_label",
    "secondary_label",
    "allowed_labels",
    "allowed_label_tokens_json",
    "label_mode_cap",
    "trajectory_stage",
    "slang_fluency",
    "trajectory_hint",
    "s_json",
    "p_json",
    "c_json",
    "narratives_json",
    "emission_params_json",
    "pair_probs_json",
    "prompt_metadata_json",
    "lexicon_samples_json",
    "style_variation_json",
    "style_indicators_json",
]


# -----------------------
# CLI and config helpers
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV with LLM SPC/narratives and 13-class taxonomy."
    )
    parser.add_argument("--out", type=str, default="./data/personas_mvp_llm.csv")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--ontology",
        type=str,
        default=str(DEFAULT_ONTOLOGY),
        help="Path to persona ontology YAML.",
    )
    parser.add_argument("--plan", type=str, default="", help="Optional counts file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("rag", "llm_only", "legacy"),
        default="rag",
        help="Persona prompt generation mode.",
    )

    # Counts for 13 unified classes
    parser.add_argument("--benign", type=int, default=None)
    parser.add_argument("--recovery", type=int, default=None)
    parser.add_argument("--ed-risk", dest="ed_risk", type=int, default=None)
    parser.add_argument("--pro-ana", dest="pro_ana", type=int, default=None)
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--conspiracy", type=int, default=None)
    parser.add_argument("--trad", type=int, default=None)
    parser.add_argument("--gamergate", type=int, default=None)
    parser.add_argument("--extremist", type=int, default=None)
    parser.add_argument("--hate", dest="hate_speech", type=int, default=None)
    parser.add_argument("--bully", dest="bullying", type=int, default=None)

    # Label mode controls
    parser.add_argument(
        "--single-ratio",
        type=float,
        default=0.7,
        help="Fraction of personas constrained to single-label mode (benign is always single).",
    )

    # LLM controls
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Grok LLM SPC/narrative generation with caching.",
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=str,
        default="./data/spc_cache",
        help="Directory for SPC cache files.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=PERSONA_MODEL,
        help=f"LLM model name (default: {PERSONA_MODEL}).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="",
        help="LLM API key (falls back to env XAI_API_KEY).",
    )
    parser.add_argument(
        "--personality-csv",
        type=str,
        default="data/personality.csv",
        help="Optional personality CSV (column 'Persona') to inject persona-level traits.",
    )
    parser.add_argument(
        "--seed-pool",
        type=str,
        default="",
        help="Explicit path to seed CSV (defaults to --personality-csv).",
    )
    parser.add_argument(
        "--style-indicators",
        type=str,
        default="configs/style_indicators.yaml",
        help="Path to YAML with per-class style indicator pools.",
    )
    return parser.parse_args()


def _load_personas_cfg(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with file.open("r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    personas = data.get("personas", {}) if isinstance(data, Mapping) else {}
    if not isinstance(personas, Mapping):
        raise TypeError("`personas` section must be a mapping")
    return personas


def _load_plan_file(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")
    with file.open("r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Plan file must contain a mapping at the root level")
    return data


def _coalesce_spec(
    base_cfg: Mapping[str, object],
    plan_cfg: Mapping[str, object],
    cli_counts: Dict[str, int | None],
) -> Dict[str, object]:
    spec: Dict[str, object] = {}
    base_mix = base_cfg.get("mix", base_cfg) if isinstance(base_cfg, Mapping) else base_cfg
    for source in (base_mix, plan_cfg):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if key in {"seed", "personas_csv", "ontology", "single_ratio", "double_ratio"}:
                continue
            spec[key] = value
    for key, value in cli_counts.items():
        if value is not None:
            spec[key] = value
    # Remap legacy keys if present.
    if "incel" in spec:
        if "incel_misogyny" not in spec:
            spec["incel_misogyny"] = spec["incel"]
        spec.pop("incel", None)
    if not spec:
        raise ValueError("No persona allocation specified.")
    return spec


def _determine_output_path(args: argparse.Namespace, cfg: Mapping[str, object]) -> Path:
    if args.out:
        return Path(args.out)
    if isinstance(cfg, Mapping) and cfg.get("personas_csv"):
        return Path(str(cfg["personas_csv"]))
    return Path("./data/personas_mvp_llm.csv")


def _gather_cli_counts(args: argparse.Namespace) -> Dict[str, int | None]:
    return {
        "benign": args.benign,
        "recovery_ed": args.recovery,
        "ed_risk": args.ed_risk,
        "pro_ana": args.pro_ana,
        "incel_misogyny": args.incel,
        "alpha": args.alpha,
        "misinfo": args.misinfo,
        "conspiracy": args.conspiracy,
        "trad": args.trad,
        "gamergate": args.gamergate,
        "extremist": args.extremist,
        "hate_speech": args.hate_speech,
        "bullying": args.bullying,
    }


# -----------------------
# LLM helpers with cache
# -----------------------
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_personality_list(path: str) -> List[str]:
    personality: List[str] = []
    p = Path(path)
    if not p.exists():
        return personality
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(p)
        col = "Persona" if "Persona" in df.columns else df.columns[0]
        personality = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    except Exception:
        pass
    return personality


def _persona_hint(username: str, personality_list: List[str]) -> str:
    if not personality_list:
        return ""
    h = int(_sha256(username), 16)
    return personality_list[h % len(personality_list)]


def _resolve_lexicon_path(base_path: Path, entry_path: str) -> Path:
    """Resolve lexicon path relative to workspace root (not ontology parent).
    
    The ontology references paths like 'configs/lexicons/benign.yaml' which
    are relative to the workspace root, not the ontology file location.
    """
    path = Path(entry_path)
    if not path.is_absolute():
        # Resolve relative to workspace root (cwd when running from root)
        # First try cwd, then try relative to base_path's grandparent (configs/)
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        # Fallback: try from base_path's parent (configs/personas -> configs)
        alt_path = base_path.parent.parent / path.name.replace("configs/", "")
        if alt_path.exists():
            return alt_path
        # Last resort: try the lexicons directory directly
        lexicons_path = base_path.parent.parent / "lexicons" / Path(path).name
        if lexicons_path.exists():
            return lexicons_path
        # Return cwd path for error reporting
        return cwd_path
    return path


def _load_lexicon_pools(ontology, ontology_path: Path) -> Dict[str, Dict[str, List[str]]]:
    pools: Dict[str, Dict[str, List[str]]] = {}
    collections = getattr(ontology, "lexicon_collections", {}) or {}
    for name, entry in collections.items():
        file_path = entry.get("file")
        if not file_path:
            continue
        resolved = _resolve_lexicon_path(ontology_path, file_path)
        try:
            with resolved.open("r", encoding="utf-8") as fh:
                data = safe_load(fh) or {}
        except FileNotFoundError:
            print(f"[WARN] Lexicon file missing for pool '{name}': {resolved}")
            continue
        pools[name] = {
            "required": [str(v) for v in data.get("required", [])],
            "optional": [str(v) for v in data.get("optional", [])],
        }
    return pools


def _sample_lexicon_subset(
    refs: List[str],
    pools: Mapping[str, Dict[str, List[str]]],
    rng: random.Random,
    max_required: int = 3,
    max_optional: int = 4,
) -> Dict[str, object]:
    sampled: Dict[str, object] = {"refs": refs, "sampled": {"required": [], "optional": []}}
    req_bucket: List[str] = []
    opt_bucket: List[str] = []
    for ref in refs:
        pool = pools.get(ref)
        if not pool:
            continue
        req_bucket.extend(pool.get("required", []))
        opt_bucket.extend(pool.get("optional", []))
    if req_bucket:
        k = min(max_required, len(req_bucket))
        sampled["sampled"]["required"] = rng.sample(req_bucket, k)
    if opt_bucket:
        k = min(max_optional, len(opt_bucket))
        sampled["sampled"]["optional"] = rng.sample(opt_bucket, k)
    return sampled


def _sample_style_variation(variant_id: str, variant_meta: Mapping[str, Any], rng: random.Random) -> Dict[str, object]:
    rules = variant_meta.get("style_variation_rules") or {}
    variation: Dict[str, object] = {"variant_id": variant_id}
    for key, cfg in rules.items():
        if isinstance(cfg, Mapping) and "min" in cfg and "max" in cfg:
            variation[key] = round(rng.uniform(float(cfg["min"]), float(cfg["max"])), 4)
        elif isinstance(cfg, list) and cfg:
            variation[key] = rng.choice(cfg)
    return variation
def _format_style_indicator_lines(indicators: Mapping[str, str]) -> List[str]:
    if not indicators:
        return []
    lines = ["Style fingerprint (embed directly into [Style]):"]
    for key in sorted(indicators.keys()):
        lines.append(f"  * {key}: {indicators[key]}")
    return lines


def _inject_style_fingerprint(system_prompt: str, indicators: Mapping[str, str]) -> str:
    if not indicators:
        return system_prompt
    fingerprint = "\n".join(_format_style_indicator_lines(indicators))
    marker = "[Style]"
    idx = system_prompt.find(marker)
    if idx == -1:
        return system_prompt + f"\n\n{fingerprint}"
    insert_pos = idx + len(marker)
    next_section = system_prompt.find("\n[", insert_pos)
    if next_section == -1:
        next_section = len(system_prompt)
    before = system_prompt[:insert_pos]
    style_body = system_prompt[insert_pos:next_section]
    after = system_prompt[next_section:]
    if fingerprint in style_body:
        return system_prompt
    augmented = f"{before}\n{fingerprint}\n{style_body}"
    return augmented + after




def _fallback_seed(row: Mapping[str, str], seed_id: int = -1) -> PersonaSeed:
    persona = row.get("description") or row.get("persona_summary") or ""
    return PersonaSeed(seed_id=seed_id, persona=persona, chat_lines=[], keywords=[])


def _resolve_variant_spec(ontology, row: Mapping[str, str]):
    variant_id = _variant_id_from_row(row)
    if variant_id:
        try:
            return ontology.get_variant(variant_id)
        except KeyError:
            pass
    group = str(row.get("persona_group", "")).strip()
    if group and group in ontology.archetypes:
        arche = ontology.archetypes[group]
        for variant in arche.variants.values():
            return variant
    return None


def _sample_style_indicators(
    primary_label: str,
    pools: Mapping[str, Dict[str, List[str]]],
    rng: random.Random,
    sampling_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Sample style indicators with probabilistic category selection.

    Uses tiered sampling where:
    - Core categories (emoji, tone_tag, hashtag_usage) are always sampled
    - Common categories (sentence_variation, emotional_arc, reply_cadence) are usually sampled
    - Occasional categories (abbreviation_style, punctuation_habits) are sometimes sampled
    - Rare categories (typo_style, capitalization_style) are rarely sampled
    - Very rare categories (visual_style) are almost never sampled

    Per-archetype overrides can adjust the rare/occasional probabilities.
    """
    entries = pools.get(primary_label) or {}
    if not entries:
        return {}

    # Default tier configuration
    default_tiers = {
        "core": 1.0,
        "common": 0.7,
        "occasional": 0.4,
        "rare": 0.15,
        "very_rare": 0.05,
    }

    # Default category-to-tier mapping
    category_tiers = {
        "emoji": "core",
        "tone_tag": "core",
        "hashtag_usage": "core",
        "sentence_variation": "common",
        "emotional_arc": "common",
        "reply_cadence": "common",
        "abbreviation_style": "occasional",
        "punctuation_habits": "occasional",
        "typo_style": "rare",
        "capitalization_style": "rare",
        "visual_style": "very_rare",
    }

    # Load config overrides if provided
    if sampling_config:
        config_tiers = sampling_config.get("default_tiers", {})
        default_tiers.update(config_tiers)

        config_category_tiers = sampling_config.get("category_tiers", {})
        category_tiers.update(config_category_tiers)

        # Apply per-archetype overrides
        archetype_overrides = sampling_config.get("archetype_overrides", {})
        if primary_label in archetype_overrides:
            overrides = archetype_overrides[primary_label]
            for tier_name, prob in overrides.items():
                if tier_name in default_tiers:
                    default_tiers[tier_name] = float(prob)

    indicators: Dict[str, str] = {}
    for category in sorted(entries.keys()):
        values = entries.get(category) or []
        if not values:
            continue

        # Determine tier and probability for this category
        tier = category_tiers.get(category, "common")
        probability = default_tiers.get(tier, 0.5)

        # Roll to see if we sample this category
        if rng.random() > probability:
            continue

        # Sample one indicator from the category
        idx = rng.randrange(len(values))
        indicators[category] = values[idx]

    return indicators




def _load_style_indicators(path: Path) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    """Load style indicators and sampling configuration.

    Returns:
        Tuple of (indicator_pools, sampling_config).
        - indicator_pools: Dict mapping archetype -> category -> list of indicators
        - sampling_config: Dict with tier probabilities and category mappings
    """
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}

    # Extract sampling configuration
    sampling_config = data.pop("_sampling_config", {})

    pools: Dict[str, Dict[str, List[str]]] = {}
    for label, entries in data.items():
        if not isinstance(entries, Mapping):
            continue
        pools[str(label)] = {
            key: [str(v) for v in (vals or [])]
            for key, vals in entries.items()
            if isinstance(vals, list)
        }
    return pools, sampling_config


def _mock_llm_generate_factory(seed: PersonaSeed, variant_summary: str):
    def _generate(system_instruction: str, user_text: str, config: Optional[Dict[str, Any]] = None) -> str:
        seed_val = config.get("seed") if config else None
        local_rng = random.Random(seed_val)
        keywords = ", ".join(seed.keywords[:3]) or "lived experience"
        persona_text = seed.persona or "An online user with layered backstory"
        goal = f"Advance {variant_summary.lower()} talking points online."
        quirks = [
            f"references {keywords}",
            "ties anecdotes to personal history",
            "keeps tone consistent with worldview",
        ]
        payload = {
            "system_prompt": f"You are {persona_text}. Blend your history ({keywords}) with the stated worldview and speak in first person.",
            "persona_goal": goal,
            "style_quirks": "; ".join(quirks),
            "lexical_required": keywords,
            "lexical_optional": "rotate slang; keep phrasing fresh",
        }
        return json.dumps(payload, ensure_ascii=False)

    return _generate


def _variant_payload(variant_spec, style_indicators: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    metadata = variant_spec.metadata or {}
    return {
        "display_name": variant_spec.display_name,
        "worldview": metadata.get("worldview", variant_spec.summary),
        "topics": variant_spec.topics,
        "instructional_hints": metadata.get("instructional_hints", []),
        "lexicon_refs": metadata.get("lexicon_refs", []),
        "style_variation_rules": metadata.get("style_variation_rules", {}),
        "style_indicators": style_indicators or {},
    }


def _synthesize_prompt_for_row(
    row: Mapping[str, str],
    variant_spec,
    *,
    mode: str,
    seeds: List[PersonaSeed],
    lexicon_pools: Mapping[str, Dict[str, List[str]]],
    style_indicator_pools: Mapping[str, Dict[str, List[str]]],
    style_sampling_config: Optional[Mapping[str, Any]] = None,
    rng: random.Random,
    llm_generate=None,
) -> Tuple[Optional[PromptSynthesisResult], Dict[str, object], Dict[str, object], Dict[str, object]]:
    if mode == "legacy":
        return None, {}, {}, {}
    seed_obj: Optional[PersonaSeed] = None
    if seeds:
        hints = variant_spec.topics or []
        seed_obj = sample_seed(seeds, keyword_hints=hints, rng=rng)
    if seed_obj is None:
        seed_obj = _fallback_seed(row)
    primary_label = row.get("primary_label") or row.get("persona_group") or variant_spec.archetype
    style_indicator_map = _sample_style_indicators(
        str(primary_label),
        style_indicator_pools,
        rng,
        sampling_config=style_sampling_config,
    )
    variant_meta = _variant_payload(variant_spec, style_indicators=style_indicator_map)
    if llm_generate:
        llm_callable = llm_generate
    else:
        llm_callable = _mock_llm_generate_factory(seed_obj, variant_spec.display_name)
    result = build_llm_prompt(
        seed=seed_obj,
        variant_meta=variant_meta,
        llm_generate=llm_callable,
        rng_seed=rng.randint(0, 1_000_000_000),
    )
    lexicon_sample = _sample_lexicon_subset(
        variant_meta.get("lexicon_refs", []),
        lexicon_pools,
        rng,
    )
    style_variation = _sample_style_variation(variant_spec.id, variant_meta, rng)
    annotated_prompt = _inject_style_fingerprint(result.system_prompt, style_indicator_map)
    result = PromptSynthesisResult(
        system_prompt=annotated_prompt,
        persona_goal=result.persona_goal,
        style_quirks=result.style_quirks,
        lexical_required=result.lexical_required,
        lexical_optional=result.lexical_optional,
    )
    prompt_metadata = {
        "persona_goal": result.persona_goal,
        "style_quirks": result.style_quirks,
        "lexical_required": result.lexical_required,
        "lexical_optional": result.lexical_optional,
        "style_indicators": style_indicator_map,
    }
    return result, prompt_metadata, lexicon_sample, style_variation


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _variant_id_from_row(row: Mapping[str, str]) -> str:
    group = str(row.get("persona_group", "")).strip()
    variant = str(row.get("persona_variant", "")).strip()
    return f"{group}.{variant}" if variant else group


def _style_overrides(row: Mapping[str, str]) -> Dict[str, object]:
    style = {
        "formality": row.get("persona_register", ""),
        "sarcasm_rate": row.get("persona_style_quirks", ""),
        "aggression": row.get("persona_tone", ""),
        "emoji_usage": row.get("persona_style_emoji_usage", ""),
        "typo_rate": row.get("persona_style_typo", ""),
        "pacing": row.get("persona_style_pacing", ""),
        "dialect": row.get("persona_style_dialect", ""),
    }
    return {k: v for k, v in style.items() if v not in (None, "")}


def _build_llm_prompt(row: Mapping[str, str], persona_hint: str) -> Tuple[str, str]:
    """Return (prompt_text, prompt_hash). Includes persona-level hint for diversity."""
    identity = row.get("persona_summary", "") or row.get("description", "")
    topics = row.get("persona_topics", "")
    style = row.get("persona_writing_style", row.get("persona_tone", ""))
    label_primary = row.get("label_primary", "")
    label_secondary = row.get("label_secondary", "")
    variant_id = _variant_id_from_row(row)
    username = row.get("username", "")
    style_over = _style_overrides(row)
    hint_block = f"Persona hint: {persona_hint}\n" if persona_hint else ""
    style_block = f"Style overrides: {style_over}\n" if style_over else ""
    prompt = (
        "You generate structured persona SPC blocks and narratives.\n"
        "Return JSON with keys: S, P, C, narratives.\n"
        "S: {groups:list[str], demographics:{gender_proxy,age_band,region_proxy}, role_in_community:str}\n"
        "P: {traits: dict[str,float], values: dict[str,float], communication_style:{formality,sarcasm_rate,aggression}}\n"
        "C: {stage_in_trajectory:str, offline_stressors:list[str], support_exposure:float}\n"
        "narratives: {c_essay:str, p_intro:str}\n"
        "Keep numbers 0-1 where applicable. No extra text.\n"
        f"Variant: {variant_id}; Username: {username}\n"
        f"Identity: {identity}\n"
        f"Topics: {topics}\n"
        f"Style: {style}\n"
        f"Primary label: {label_primary}; Secondary labels: {label_secondary}\n"
        f"{hint_block}{style_block}"
    )
    return prompt, _sha256(prompt)


def _call_text_completion(
    system_instruction: str,
    user_text: str,
    model: str,
    api_key: str,
    *,
    provider: Optional[str] = None,
    expect_json: bool = False,
    extra_config: Optional[Dict[str, object]] = None,
    max_retries: int = 3,
) -> str:
    """General text/JSON completion helper with retry logic."""
    provider = provider or get_provider_for_model(model)
    base_url = API_ENDPOINTS.get(provider, API_ENDPOINTS["xai"])

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: Dict[str, object] = {
        "model": model,
        "temperature": PERSONA_TEMPERATURE,
        "max_tokens": PERSONA_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ],
    }
    if extra_config:
        payload.update(extra_config)
    if expect_json and supports_structured_output(model):
        payload["response_format"] = {"type": "json_object"}

    url = f"{base_url}/chat/completions"
    
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            last_error = e
            # Check if rate limited or server error - retry with backoff
            if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
                continue
            # For 400 errors, try a different model from fallback list
            if e.response is not None and e.response.status_code == 400 and FALLBACK_ENABLED:
                for fallback_model in FALLBACK_MODELS:
                    if "llama-4" in fallback_model.lower():
                        continue  # Skip Llama-4 for prompt synthesis
                    try:
                        fallback_provider = get_provider_for_model(fallback_model)
                        fallback_keys = get_api_keys_for_provider(fallback_provider)
                        if not fallback_keys:
                            continue
                        fallback_key = fallback_keys[0]
                        fallback_url = f"{API_ENDPOINTS.get(fallback_provider, base_url)}/chat/completions"
                        fallback_headers = {"Authorization": f"Bearer {fallback_key}", "Content-Type": "application/json"}
                        fallback_payload = dict(payload)
                        fallback_payload["model"] = fallback_model
                        if expect_json and not supports_structured_output(fallback_model):
                            fallback_payload.pop("response_format", None)
                        resp = requests.post(fallback_url, headers=fallback_headers, data=json.dumps(fallback_payload), timeout=60)
                        resp.raise_for_status()
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
                    except Exception:
                        continue
            raise
        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = (2 ** attempt) + random.random()
            time.sleep(wait_time)
            continue
    
    if last_error:
        raise last_error
    raise RuntimeError("Max retries exceeded")


async def _call_text_completion_async(
    session: aiohttp.ClientSession,
    system_instruction: str,
    user_text: str,
    model: str,
    api_key: str,
    *,
    provider: Optional[str] = None,
    expect_json: bool = False,
    extra_config: Optional[Dict[str, object]] = None,
) -> str:
    """Async version of text/JSON completion helper."""
    provider = provider or get_provider_for_model(model)
    base_url = API_ENDPOINTS.get(provider, API_ENDPOINTS["xai"])

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: Dict[str, object] = {
        "model": model,
        "temperature": PERSONA_TEMPERATURE,
        "max_tokens": PERSONA_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ],
    }
    if extra_config:
        payload.update(extra_config)
    if expect_json and supports_structured_output(model):
        payload["response_format"] = {"type": "json_object"}

    url = f"{base_url}/chat/completions"
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


def _make_llm_prompt_generator(model: str, api_key: str, provider: Optional[str] = None):
    def _generate(system_instruction: str, user_text: str, config: Optional[Dict[str, object]] = None) -> str:
        extra = {}
        if config:
            for key in ("seed", "temperature", "top_p"):
                if key in config:
                    extra[key] = config[key]
        return _call_text_completion(
            system_instruction=system_instruction,
            user_text=user_text,
            model=model,
            api_key=api_key,
            provider=provider,
            expect_json=True,
            extra_config=extra or None,
        )

    return _generate


def _call_llm(
    prompt: str,
    model: str,
    api_key: str,
    provider: Optional[str] = None,
) -> Dict[str, object]:
    """Call LLM API for persona generation. Supports xAI, OpenAI, and Gemini."""
    content = _call_text_completion(
        system_instruction="You are a concise JSON-only generator for SPC persona profiles.",
        user_text=prompt,
        model=model,
        api_key=api_key,
        provider=provider,
        expect_json=True,
    )
    return json.loads(content)


# Backwards compatibility alias
_call_grok = _call_llm


def _load_cache(cache_file: Path, expected_hash: str, model: str) -> Optional[Dict[str, object]]:
    if not cache_file.exists():
        return None
    with cache_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if (
        data.get("prompt_hash") != expected_hash
        or data.get("model") != model
        or "S" not in data
        or "P" not in data
        or "C" not in data
        or "narratives" not in data
    ):
        return None
    return data


def _save_cache(cache_file: Path, payload: Dict[str, object]) -> None:
    _ensure_cache_dir(cache_file.parent)
    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# -----------------------
# Transform helpers
# -----------------------
def _extract_metadata(row: Mapping[str, str], key: str) -> str:
    for prefix in ("variant", "archetype"):
        value = row.get(f"{prefix}_{key}")
        if value and str(value).strip().lower() not in {"", "none", "null"}:
            return str(value).strip()
    return ""


def _extract_metadata_list(row: Mapping[str, str], key: str) -> List[str]:
    raw = _extract_metadata(row, key)
    if not raw:
        return []
    for parser in (json.loads, lambda x: eval(x, {"__builtins__": {}})):  # noqa: S307
        try:
            loaded = parser(raw)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except Exception:
            continue
    return [item.strip() for item in raw.split(",") if item.strip()]


# Auto-generated secondary label relationships based on archetype clusters
# This enables multi-class personas even when ontology doesn't define secondary labels
ARCHETYPE_SECONDARY_LABELS: Dict[str, List[str]] = {
    # Benign cluster - no secondary (always single-label)
    "benign": [],
    "recovery_support": ["benign"],  # Can also be supportive/benign
    
    # ED cluster - related harmful behaviors
    "ed_risk": ["pro_ana", "bullying"],
    "pro_ana": ["ed_risk", "bullying"],
    
    # Incel/Manosphere cluster - overlapping misogyny
    "incel_misogyny": ["bullying", "hate_speech", "alpha"],
    "alpha": ["incel_misogyny", "misinfo"],
    
    # Misinfo/Conspiracy cluster - overlapping distrust
    "misinfo": ["conspiracy"],
    "conspiracy": ["misinfo", "extremist"],
    
    # Culture war cluster - overlapping reactionary views
    "trad": ["conspiracy", "alpha"],
    "gamergate": ["bullying", "incel_misogyny", "hate_speech"],
    
    # Extreme harm cluster - overlapping violence/hate
    "extremist": ["hate_speech", "conspiracy"],
    "hate_speech": ["extremist", "bullying"],
    "bullying": ["hate_speech", "incel_misogyny"],
}


def _allowed_candidates(row: Mapping[str, str]) -> List[str]:
    """Build list of allowed label candidates for a persona.
    
    Priority:
    1. Explicit override from ontology (allowed_labels_override)
    2. Ontology-defined secondary labels (label_secondary)
    3. Auto-generated secondary labels from ARCHETYPE_SECONDARY_LABELS
    """
    override = _extract_metadata_list(row, "allowed_labels_override")
    if override:
        return override
    
    primary = str(row.get("label_primary", "")).strip().lower()
    
    # Try ontology-defined secondary labels first
    secondary_raw = str(row.get("label_secondary", "")).strip()
    # Filter out "None" string which comes from missing ontology values
    if secondary_raw.lower() in ("none", "null", ""):
        secondary_raw = ""
    secondary = [tok.strip().lower() for tok in secondary_raw.split(";") if tok.strip()]
    
    # If no ontology secondary labels, use auto-generated ones
    if not secondary and primary:
        secondary = ARCHETYPE_SECONDARY_LABELS.get(primary, [])
    
    candidates = [tok for tok in [primary, *secondary] if tok]
    if not candidates:
        candidates = [primary] if primary else []
    return list(dict.fromkeys(candidates))


def _infer_primary_label(row: Mapping[str, str]) -> str:
    group = str(row.get("persona_group", "")).strip().lower()
    if group:
        return group
    token = str(row.get("label_primary", "")).strip().lower()
    return token or "benign"


def _assign_trajectory_stage(rng: random.Random, primary_label: str) -> Tuple[str, str, str]:
    """Assign a trajectory stage and slang fluency based on archetype distribution.
    
    Args:
        rng: Random number generator for deterministic assignment
        primary_label: The persona's primary archetype label
    
    Returns:
        Tuple of (trajectory_stage, slang_fluency, trajectory_hint)
    """
    # Get distribution for this archetype (default to balanced if unknown)
    distribution = TRAJECTORY_STAGE_DISTRIBUTION.get(
        primary_label,
        {"curious": 0.33, "active": 0.34, "entrenched": 0.33}
    )
    
    # Sample stage based on distribution
    stages = list(distribution.keys())
    weights = list(distribution.values())
    stage = rng.choices(stages, weights=weights, k=1)[0]
    
    # Derive slang fluency and hint from stage
    stage_modifiers = TRAJECTORY_STAGE_MODIFIERS.get(stage, {})
    slang_fluency = stage_modifiers.get("slang_fluency", "fluent")
    trajectory_hint = stage_modifiers.get("prompt_hint", "")
    
    return stage, slang_fluency, trajectory_hint


def _allowed_label_tokens_map(labels: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for label in labels:
        tokens = DEFAULT_LABEL_TO_TOKENS.get(label)
        if not tokens:
            tokens = [f"LBL:{label.upper()}"]
        mapping[label] = [str(t) for t in tokens]
    return mapping


def _default_emission_params(primary: str) -> Dict[str, float]:
    defaults: Dict[str, Dict[str, float]] = {
        # Benign / Recovery cluster
        "benign": {"LBL:SUPPORTIVE": 0.02},
        "recovery_ed": {"LBL:RECOVERY_SUPPORT": 0.02, "LBL:SUPPORTIVE": 0.01},
        # ED cluster
        "ed_risk": {"LBL:ED_RISK": 0.03, "LBL:ED_METHOD": 0.02},
        "pro_ana": {"LBL:MEANSPO": 0.03, "LBL:ED_COACHING": 0.02},
        # Incel / Manosphere cluster
        "incel_misogyny": {"LBL:INCEL_MISOGYNY": 0.035},
        "alpha": {"LBL:MISOGYNISTIC_LECTURE": 0.04, "LBL:OBJECTIFICATION": 0.03},
        # Misinfo / Conspiracy cluster
        "misinfo": {"LBL:MISINFO_CLAIM": 0.03},
        "conspiracy": {"LBL:CONSPIRACY": 0.03, "LBL:DEEPSTATE": 0.02},
        # Culture war cluster
        "trad": {"LBL:DOGWHISTLE": 0.025, "LBL:GENDER_ESSENTIALISM": 0.025},
        "gamergate": {"LBL:CULTURE_WAR": 0.03, "LBL:GATEKEEPING": 0.02},
        # Extreme harm cluster
        "extremist": {"LBL:VIOLENT_THREAT": 0.02, "LBL:HATE_SLUR": 0.03, "LBL:ACCELERATIONISM": 0.02},
        "hate_speech": {"LBL:HATE_SLUR": 0.05, "LBL:DEHUMANIZATION": 0.03},
        "bullying": {"LBL:PERSONAL_ATTACK": 0.04, "LBL:DOXXING_THREAT": 0.01},
    }
    return defaults.get(primary, {"LBL:SUPPORTIVE": 0.01})


def _render_preamble(s_block: Dict[str, object], p_block: Dict[str, object], c_block: Dict[str, object]) -> str:
    groups = s_block.get("groups") or []
    role = s_block.get("role_in_community", "")
    demo = s_block.get("demographics", {})
    traits = p_block.get("traits", {})
    stage = c_block.get("stage_in_trajectory", "")
    stressors = c_block.get("offline_stressors", [])
    support = c_block.get("support_exposure", "")
    groups_txt = ", ".join(groups) if groups else "mixed online spaces"
    trait_top = ", ".join(list(traits.keys())[:3]) if traits else ""
    stress_txt = ", ".join(stressors) if stressors else "day-to-day routines"
    return (
        f"[SPC] In {groups_txt} as {role}. "
        f"Traits: {trait_top or 'balanced'}. "
        f"Stage: {stage}; stressors: {stress_txt}; support_exposure={support}."
    )


def _pick_double_mode(rng: random.Random, primary: str, candidates: List[str], single_ratio: float) -> Tuple[bool, List[str], str]:
    # Benign and recovery_ed are always single-label (non-harmful)
    if primary in ("benign", "recovery_ed"):
        return False, [primary], ""
    if len(candidates) < 2:
        return False, [primary], ""
    is_double = rng.random() > single_ratio
    allowed = candidates if is_double else [primary]
    secondary = allowed[1] if is_double and len(allowed) > 1 else ""
    label_cap = "double" if is_double else "single"
    return is_double, allowed, secondary


def _transform_rows(
    raw_rows: List[Dict[str, str]],
    seed: int,
    single_ratio: float,
    profile_lookup: Mapping[str, Dict[str, object]],
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    final_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        primary = _infer_primary_label(row)
        username = row.get("username", f"user_{idx}")
        profile = profile_lookup.get(username) or {}
        prompt_meta = profile.get("prompt_metadata") if isinstance(profile, Mapping) else {}
        lexicon_samples = profile.get("lexicon_samples") if isinstance(profile, Mapping) else {}
        style_variation = profile.get("style_variation") if isinstance(profile, Mapping) else {}
        s_block = profile.get("S", {}) if isinstance(profile, Mapping) else {}
        p_block = profile.get("P", {}) if isinstance(profile, Mapping) else {}
        c_block = profile.get("C", {}) if isinstance(profile, Mapping) else {}
        narratives = profile.get("narratives", {}) if isinstance(profile, Mapping) else {}

        if not s_block:
            s_block = {
                "groups": [row.get("persona_group", "mainstream")],
                "demographics": {
                    "gender_proxy": "unspecified",
                    "age_band": "25-34",
                    "region_proxy": "global_north",
                },
                "role_in_community": "conversationalist",
            }
        if not p_block:
            p_block = {
                "traits": {"grievance_level": 0.2, "institutional_trust": 0.6, "empathy": 0.7},
                "values": {},
                "communication_style": {"formality": "medium", "sarcasm_rate": 0.1, "aggression": 0.1},
            }
        if not c_block:
            c_block = {
                "stage_in_trajectory": "benign",
                "offline_stressors": [],
                "support_exposure": 0.6,
            }
        if not narratives:
            narratives = {
                "c_essay": "Daily life mixes work, small chats, and occasional posting.",
                "p_intro": "Staying active online to swap stories.",
            }

        preamble = _render_preamble(s_block, p_block, c_block)
        base_prompt = (profile.get("system_prompt") or row.get("user_char", "") or "").strip()
        user_char = (base_prompt + "\n\n" + preamble + "\n" + narratives.get("p_intro", "")).strip()

        candidates = _allowed_candidates(row)
        is_double, allowed_labels, secondary_label = _pick_double_mode(rng, primary, candidates, single_ratio)
        label_cap = "double" if is_double else "single"
        label_tokens_map = _allowed_label_tokens_map(allowed_labels)

        # Assign trajectory stage and slang fluency
        trajectory_stage, slang_fluency, trajectory_hint = _assign_trajectory_stage(rng, primary)

        final_rows.append(
            {
                "username": username,
                "description": row.get("description") or row.get("persona_summary", ""),
                "user_char": user_char,
                "primary_label": primary,
                "secondary_label": secondary_label,
                "allowed_labels": json.dumps(allowed_labels, ensure_ascii=False),
                "allowed_label_tokens_json": json.dumps(label_tokens_map, ensure_ascii=False),
                "label_mode_cap": label_cap,
                "trajectory_stage": trajectory_stage,
                "slang_fluency": slang_fluency,
                "trajectory_hint": trajectory_hint,
                "s_json": json.dumps(s_block, ensure_ascii=False),
                "p_json": json.dumps(p_block, ensure_ascii=False),
                "c_json": json.dumps(c_block, ensure_ascii=False),
                "narratives_json": json.dumps(narratives, ensure_ascii=False),
                "emission_params_json": json.dumps(_default_emission_params(primary), ensure_ascii=False),
                "pair_probs_json": json.dumps({}, ensure_ascii=False),
                "prompt_metadata_json": json.dumps(prompt_meta or {}, ensure_ascii=False),
                "lexicon_samples_json": json.dumps(lexicon_samples or {}, ensure_ascii=False),
                "style_variation_json": json.dumps(style_variation or {}, ensure_ascii=False),
                "style_indicators_json": json.dumps((prompt_meta or {}).get("style_indicators", {}), ensure_ascii=False),
            }
        )
    return final_rows


# -----------------------
# Main
# -----------------------
def main() -> None:
    args = parse_args()
    personas_cfg = _load_personas_cfg(args.config)
    plan_cfg = _load_plan_file(args.plan)

    ontology_path = Path(args.ontology or personas_cfg.get("ontology", str(DEFAULT_ONTOLOGY)))
    ontology = load_ontology(ontology_path)
    lexicon_pools = _load_lexicon_pools(ontology, ontology_path)
    style_indicator_pools, style_sampling_config = _load_style_indicators(Path(args.style_indicators))

    seed = int(personas_cfg.get("seed", args.seed))
    single_ratio = float(personas_cfg.get("single_ratio", args.single_ratio))
    generator = PersonaGenerator(ontology=ontology, seed=seed)

    spec = _coalesce_spec(personas_cfg, plan_cfg, _gather_cli_counts(args))
    requests_list = build_requests_from_spec(generator, spec)
    
    # Generate raw rows - this is lightweight (just persona specs, no LLM calls yet)
    print(f"[INFO] Generating {len(requests_list)} persona specs...")
    sys.stdout.flush()
    raw_rows = generator.generate(requests_list)
    print(f"[INFO] Generated {len(raw_rows)} persona specs")
    sys.stdout.flush()

    seed_pool_path = Path(args.seed_pool or args.personality_csv)
    persona_seeds: List[PersonaSeed] = []
    if args.mode in {"rag", "llm_only"} and seed_pool_path.exists():
        persona_seeds = load_persona_seeds(seed_pool_path)
    personality_list = _load_personality_list(args.personality_csv) if args.mode == "legacy" else []

    # Build per-variant profiles (LLM with cache)
    cache_dir = Path(args.llm_cache_dir)
    use_llm = bool(args.use_llm)
    model = args.llm_model
    # Determine API key based on model/provider
    api_key = args.llm_api_key
    if not api_key:
        provider = get_provider_for_model(model)
        env_var = API_KEY_ENV_VARS.get(provider, "XAI_API_KEY")
        api_key = os.environ.get(env_var, "") or os.environ.get("XAI_API_KEY", "")
    prompt_llm_generate = None
    if args.mode in {"rag", "llm_only"} and use_llm and api_key:
        prompt_llm_generate = _make_llm_prompt_generator(model, api_key)
    elif args.mode in {"rag", "llm_only"} and use_llm and not api_key:
        print("[WARN] Prompt synthesis requested LLM mode but no API key provided; using mock prompts.")

    profile_lookup: Dict[str, Dict[str, object]] = {}
    
    # PARALLEL PROMPT SYNTHESIS: Process all rows in parallel batches
    if args.mode != "legacy":
        print(f"[INFO] Building prompt metadata for {len(raw_rows)} personas (parallel)...")
        sys.stdout.flush()
        
        # Prepare tasks for parallel processing
        prompt_tasks: List[Dict[str, Any]] = []
        for idx, row in enumerate(raw_rows):
            username = row.get("username", "")
            variant_spec = _resolve_variant_spec(ontology, row)
            if variant_spec:
                prompt_tasks.append({
                    "idx": idx,
                    "row": row,
                    "username": username,
                    "variant_spec": variant_spec,
                    "rng_seed": seed + idx,  # Deterministic per-row seed
                })
        
        print(f"[INFO] {len(prompt_tasks)} personas need prompt synthesis")
        sys.stdout.flush()
        
        # Get model-key pairs for parallel execution
        rotation_pairs = get_model_key_pairs(for_simulation=False)
        
        PROMPT_BATCH_SIZE = 500
        completed_prompts = 0
        failed_prompts = 0
        prompt_start_time = time.time()
        
        async def _process_prompt_batch(batch_tasks: List[Dict[str, Any]], batch_offset: int):
            """Process a batch of prompt synthesis tasks in parallel."""
            nonlocal completed_prompts, failed_prompts
            semaphore = asyncio.Semaphore(50)
            
            async def _process_one_prompt(task: Dict[str, Any], task_idx: int, session: aiohttp.ClientSession):
                nonlocal completed_prompts, failed_prompts
                username = task["username"]
                row = task["row"]
                variant_spec = task["variant_spec"]
                task_rng = random.Random(task["rng_seed"])
                
                # Select model/key for this task
                global_idx = batch_offset + task_idx
                task_model, task_key, task_provider = rotation_pairs[global_idx % len(rotation_pairs)]
                
                try:
                    async with semaphore:
                        # Build the prompt data without LLM first
                        seed_obj: Optional[PersonaSeed] = None
                        if persona_seeds:
                            hints = variant_spec.topics or []
                            seed_obj = sample_seed(persona_seeds, keyword_hints=hints, rng=task_rng)
                        if seed_obj is None:
                            seed_obj = _fallback_seed(row)
                        
                        primary_label = row.get("primary_label") or row.get("persona_group") or variant_spec.archetype
                        style_indicator_map = _sample_style_indicators(
                            str(primary_label),
                            style_indicator_pools,
                            task_rng,
                            sampling_config=style_sampling_config,
                        )
                        variant_meta = _variant_payload(variant_spec, style_indicators=style_indicator_map)
                        
                        # Use mock LLM for prompt building (the actual LLM call is in SPC enrichment)
                        llm_callable = _mock_llm_generate_factory(seed_obj, variant_spec.display_name)
                        result = build_llm_prompt(
                            seed=seed_obj,
                            variant_meta=variant_meta,
                            llm_generate=llm_callable,
                            rng_seed=task_rng.randint(0, 1_000_000_000),
                        )
                        
                        lexicon_sample = _sample_lexicon_subset(
                            variant_meta.get("lexicon_refs", []),
                            lexicon_pools,
                            task_rng,
                        )
                        style_variation = _sample_style_variation(variant_spec.id, variant_meta, task_rng)
                        annotated_prompt = _inject_style_fingerprint(result.system_prompt, style_indicator_map)
                        
                        final_result = PromptSynthesisResult(
                            system_prompt=annotated_prompt,
                            persona_goal=result.persona_goal,
                            style_quirks=result.style_quirks,
                            lexical_required=result.lexical_required,
                            lexical_optional=result.lexical_optional,
                        )
                        
                        prompt_metadata = {
                            "persona_goal": final_result.persona_goal,
                            "style_quirks": final_result.style_quirks,
                            "lexical_required": final_result.lexical_required,
                            "lexical_optional": final_result.lexical_optional,
                            "style_indicators": style_indicator_map,
                        }
                        
                        # Store results
                        entry = profile_lookup.setdefault(username, {})
                        entry["system_prompt"] = final_result.system_prompt
                        entry["prompt_metadata"] = prompt_metadata
                        entry["lexicon_samples"] = lexicon_sample
                        entry["style_variation"] = style_variation
                        
                        completed_prompts += 1
                except Exception as e:
                    failed_prompts += 1
                    if failed_prompts <= 5:
                        print(f"[WARN] Prompt synthesis failed for {username}: {e}")
                
                # Progress update
                total_done = completed_prompts + failed_prompts
                if total_done % 100 == 0:
                    elapsed = time.time() - prompt_start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (len(prompt_tasks) - total_done) / rate if rate > 0 else 0
                    print(f"[INFO] Prompt synthesis: {total_done}/{len(prompt_tasks)} ({rate:.1f}/s, ETA: {eta:.0f}s)")
                    sys.stdout.flush()
            
            connector = aiohttp.TCPConnector(limit=100)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [_process_one_prompt(task, i, session) for i, task in enumerate(batch_tasks)]
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process in batches
        for batch_start in range(0, len(prompt_tasks), PROMPT_BATCH_SIZE):
            batch_end = min(batch_start + PROMPT_BATCH_SIZE, len(prompt_tasks))
            batch = prompt_tasks[batch_start:batch_end]
            
            print(f"[INFO] Processing prompt batch {batch_start//PROMPT_BATCH_SIZE + 1}/{(len(prompt_tasks) + PROMPT_BATCH_SIZE - 1)//PROMPT_BATCH_SIZE} ({len(batch)} tasks)...")
            sys.stdout.flush()
            
            asyncio.run(_process_prompt_batch(batch, batch_start))
        
        prompt_elapsed = time.time() - prompt_start_time
        print(f"[INFO] Prompt synthesis complete: {completed_prompts} succeeded, {failed_prompts} failed in {prompt_elapsed:.1f}s")
        sys.stdout.flush()
    else:
        print(f"[INFO] Legacy mode - skipping prompt synthesis")
        sys.stdout.flush()

    # Collect tasks that need LLM calls (not cached)
    llm_tasks: List[Dict[str, Any]] = []
    for row in raw_rows:
        username = row.get("username", "")
        entry = profile_lookup.get(username, {})

        if "S" in entry:
            continue

        persona_hint = _persona_hint(username, personality_list)
        prompt, prompt_hash = _build_llm_prompt(row, persona_hint)
        cache_file = cache_dir / f"{username}__{model}__{prompt_hash[:12]}.json"
        cached = _load_cache(cache_file, prompt_hash, model)
        if cached:
            profile_lookup.setdefault(username, {}).update(cached)
            continue
        
        if use_llm:
            llm_tasks.append({
                "row": row,
                "username": username,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "cache_file": cache_file,
            })
    
    # Get model-key pairs for parallel multi-model execution
    model_key_pairs = get_model_key_pairs(for_simulation=False)
    provider = get_provider_for_model(model)
    all_api_keys = get_api_keys_for_provider(provider)
    
    # Get fallback models (excluding Llama-4 for persona gen - they don't need tool calls)
    fallback_models = [m for m in FALLBACK_MODELS if "llama-4" not in m.lower()] if FALLBACK_ENABLED else []
    
    print(f"[INFO] Processing {len(raw_rows)} personas...")
    print(f"[INFO] use_llm={use_llm}, api_key={'set' if api_key else 'NOT SET'}, model={model}")
    print(f"[INFO] LLM tasks to process: {len(llm_tasks)} (use_llm={use_llm}, api_key={'set' if all_api_keys else 'NOT SET'})")
    
    if MULTI_MODEL_ENABLED and model_key_pairs:
        print(f"[INFO] Multi-model parallel: {get_parallel_model_summary()}")
    else:
        print(f"[INFO] API keys available: {len(all_api_keys)}, Fallback models: {len(fallback_models)}")
    sys.stdout.flush()
    
    if llm_tasks and use_llm and (api_key or all_api_keys):
        print(f"[INFO] Running {len(llm_tasks)} LLM calls in parallel (batched for memory efficiency)...")
        sys.stdout.flush()
        
        # Process LLM tasks in batches to limit memory usage
        LLM_BATCH_SIZE = 500  # Process 500 at a time
        
        async def _process_llm_batch(batch_tasks: List[Dict[str, Any]], batch_offset: int):
            """Process a batch of LLM tasks using multi-model parallel execution."""
            # Semaphore per model to respect per-model rate limits
            semaphore = asyncio.Semaphore(50)
            completed = 0
            failed = 0
            start_time = time.time()
            
            # Build list of (model, key, provider) to rotate through
            if MULTI_MODEL_ENABLED and model_key_pairs:
                rotation_pairs = model_key_pairs  # Already (model, key, provider) tuples
            else:
                # Fall back to single model with all keys
                rotation_pairs = [(model, k, provider) for k in all_api_keys] if all_api_keys else [(model, api_key, provider)]
            
            async def _process_one(task: Dict[str, Any], task_idx: int, session: aiohttp.ClientSession) -> None:
                nonlocal completed, failed
                username = task["username"]
                prompt = task["prompt"]
                prompt_hash = task["prompt_hash"]
                cache_file = task["cache_file"]
                row = task["row"]
                
                # Round-robin model/key/provider selection based on global task index
                global_idx = batch_offset + task_idx
                
                # RETRY LOGIC: Try up to 3 times with different models
                MAX_RETRIES = 3
                last_error = None
                
                for attempt in range(MAX_RETRIES):
                    # Rotate to a different model/key on each retry
                    retry_idx = (global_idx + attempt * 7) % len(rotation_pairs)  # Use prime offset
                    task_model, task_key, task_provider = rotation_pairs[retry_idx]
                    
                    try:
                        async with semaphore:
                            # Add exponential backoff for retries
                            if attempt > 0:
                                await asyncio.sleep(0.5 * (2 ** attempt))  # 1s, 2s backoff
                            
                            content = await _call_text_completion_async(
                                session=session,
                                system_instruction="You are a concise JSON-only generator for SPC persona profiles.",
                                user_text=prompt,
                                model=task_model,
                                api_key=task_key,
                                provider=task_provider,
                                expect_json=True,
                            )
                            llm_resp = json.loads(content)
                        
                            payload = {
                                "variant_id": _variant_id_from_row(row),
                                "username": username,
                                "model": task_model,
                                "prompt_hash": prompt_hash,
                                "S": llm_resp.get("S", {}),
                                "P": llm_resp.get("P", {}),
                                "C": llm_resp.get("C", {}),
                                "narratives": llm_resp.get("narratives", {}),
                            }
                            # Overlay style hints from persona card onto P
                            style_over = _style_overrides(row)
                            if style_over:
                                payload.setdefault("P", {})
                                payload["P"]["communication_style"] = style_over
                            _save_cache(cache_file, payload)
                            profile_lookup.setdefault(username, {}).update(payload)
                            completed += 1
                            
                            # Clear the row reference from task to free memory
                            task["row"] = None
                            return  # Success - exit retry loop
                        
                    except aiohttp.ClientResponseError as exc:
                        last_error = exc
                        # 400 errors are usually prompt issues - try different model
                        # 429 errors are rate limits - backoff and retry
                        if exc.status == 429:
                            await asyncio.sleep(1.0 * (2 ** attempt))  # Extra backoff for rate limits
                        elif exc.status >= 500:
                            await asyncio.sleep(0.5 * (2 ** attempt))  # Server errors - retry
                        # 400 errors - try different model immediately
                        continue
                    except json.JSONDecodeError as exc:
                        last_error = exc
                        # Bad JSON response - try different model
                        continue
                    except Exception as exc:
                        last_error = exc
                        continue
                
                # All retries failed
                failed += 1
                if failed <= 10:
                    print(f"[WARN] LLM failed for {username} after {MAX_RETRIES} retries: {last_error}")
                    sys.stdout.flush()
                    profile_lookup.setdefault(username, {}).setdefault("S", {})
                
                # Progress update every 50 within batch
                total_done = completed + failed
                if total_done % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    global_done = batch_offset + total_done
                    global_eta = (len(llm_tasks) - global_done) / rate if rate > 0 else 0
                    print(f"[INFO] Progress: {global_done}/{len(llm_tasks)} ({rate:.1f}/s, ETA: {global_eta:.0f}s)")
                    sys.stdout.flush()
            
            connector = aiohttp.TCPConnector(limit=100)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [_process_one(task, i, session) for i, task in enumerate(batch_tasks)]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return completed, failed
        
        # Process in batches
        total_completed = 0
        total_failed = 0
        overall_start = time.time()
        
        for batch_start in range(0, len(llm_tasks), LLM_BATCH_SIZE):
            batch_end = min(batch_start + LLM_BATCH_SIZE, len(llm_tasks))
            batch = llm_tasks[batch_start:batch_end]
            
            print(f"[INFO] Processing LLM batch {batch_start//LLM_BATCH_SIZE + 1}/{(len(llm_tasks) + LLM_BATCH_SIZE - 1)//LLM_BATCH_SIZE} ({len(batch)} tasks)...")
            sys.stdout.flush()
            
            completed, failed = asyncio.run(_process_llm_batch(batch, batch_start))
            total_completed += completed
            total_failed += failed
            
            # Clear the batch to free memory
            llm_tasks[batch_start:batch_end] = [None] * len(batch)
            
            print(f"[INFO] Batch complete: {completed} succeeded, {failed} failed. Total: {total_completed + total_failed}/{len(llm_tasks)}")
            sys.stdout.flush()
        
        overall_elapsed = time.time() - overall_start
        print(f"[INFO] All LLM calls complete: {total_completed} succeeded, {total_failed} failed in {overall_elapsed:.1f}s ({total_completed/overall_elapsed:.1f}/s)")
        sys.stdout.flush()
    
    # Ensure all entries have at least empty S block
    for row in raw_rows:
        username = row.get("username", "")
        profile_lookup.setdefault(username, {}).setdefault("S", {})

    # STREAMING WRITE: Transform and write rows in batches to reduce memory
    out_path = _determine_output_path(args, personas_cfg)
    _ensure_cache_dir(out_path.parent)
    
    BATCH_SIZE = 100  # Write every 100 personas to disk
    counts: Dict[str, int] = {}
    total_written = 0
    
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        # Process in batches
        for batch_start in range(0, len(raw_rows), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(raw_rows))
            batch_rows = raw_rows[batch_start:batch_end]
            
            # Transform this batch
            batch_final = _transform_rows(batch_rows, seed, single_ratio, profile_lookup)
            
            # Write immediately
            writer.writerows(batch_final)
            fh.flush()  # Ensure data is written to disk
            
            # Track counts
            for row in batch_final:
                counts[row["primary_label"]] = counts.get(row["primary_label"], 0) + 1
            
            total_written += len(batch_final)
            
            # Progress update every batch
            if batch_end % 500 == 0 or batch_end == len(raw_rows):
                print(f"[INFO] Written {total_written}/{len(raw_rows)} personas to CSV")
                sys.stdout.flush()
    
    print(f"Wrote personas to: {out_path}")
    print("Persona counts by primary:")
    for key, val in sorted(counts.items()):
        print(f"  {key:12} {val:5d}")


if __name__ == "__main__":
    main()
