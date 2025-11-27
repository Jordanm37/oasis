#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
from dotenv import load_dotenv
from yaml import safe_load

load_dotenv()

from configs.llm_settings import (
    API_ENDPOINTS,
    API_KEY_ENV_VARS,
    PERSONA_MAX_TOKENS,
    PERSONA_MODEL,
    PERSONA_PROVIDER,
    PERSONA_TEMPERATURE,
    PROVIDER_POOL,
    PROVIDER_POOL_ENABLED,
    PROVIDER_WEIGHTS,
    STRUCTURED_OUTPUT_MODELS,
    get_provider_for_model,
    supports_structured_output,
)
from oasis.persona import (
    PersonaGenerator,
    PersonaSeed,
    PromptSynthesisResult,
    build_requests_from_spec,
    build_llm_prompt,
    load_persona_seeds,
    load_ontology,
    sample_seed,
)
from generation.emission_policy import DEFAULT_LABEL_TO_TOKENS

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
        "recovery_support": args.recovery,
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
    path = Path(entry_path)
    if not path.is_absolute():
        path = base_path.parent / path
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
) -> str:
    """General text/JSON completion helper."""
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
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
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
        "recovery_support": {"LBL:RECOVERY_SUPPORT": 0.02, "LBL:SUPPORTIVE": 0.01},
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
    # Benign and recovery_support are always single-label (non-harmful)
    if primary in ("benign", "recovery_support"):
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
    raw_rows = generator.generate(requests_list)

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
    prompt_rng = random.Random(seed)
    for row in raw_rows:
        username = row.get("username", "")
        entry = profile_lookup.setdefault(username, {})

        variant_spec = _resolve_variant_spec(ontology, row)
        if variant_spec and args.mode != "legacy" and "system_prompt" not in entry:
            prompt_result, prompt_meta, lexicon_sample, style_variation = _synthesize_prompt_for_row(
                row,
                variant_spec,
                mode=args.mode,
                seeds=persona_seeds,
                lexicon_pools=lexicon_pools,
                style_indicator_pools=style_indicator_pools,
                style_sampling_config=style_sampling_config,
                rng=prompt_rng,
                llm_generate=prompt_llm_generate,
            )
            if prompt_result:
                entry["system_prompt"] = prompt_result.system_prompt
                entry["prompt_metadata"] = prompt_meta
                entry["lexicon_samples"] = lexicon_sample
                entry["style_variation"] = style_variation

        if "S" in entry:
            continue

        persona_hint = _persona_hint(username, personality_list)
        prompt, prompt_hash = _build_llm_prompt(row, persona_hint)
        cache_file = cache_dir / f"{username}__{model}__{prompt_hash[:12]}.json"
        cached = _load_cache(cache_file, prompt_hash, model)
        if cached:
            entry.update(cached)
            continue
        if use_llm:
            if not api_key:
                raise RuntimeError("LLM API key not provided. Set --llm-api-key or env XAI_API_KEY.")
            try:
                llm_resp = _call_grok(prompt, model, api_key)
                payload = {
                    "variant_id": _variant_id_from_row(row),
                    "username": username,
                    "model": model,
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
                entry.update(payload)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] LLM failed for {username}: {exc}. Falling back to defaults.")
        entry.setdefault("S", {})

    final_rows = _transform_rows(raw_rows, seed, single_ratio, profile_lookup)
    out_path = _determine_output_path(args, personas_cfg)
    _ensure_cache_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"Wrote personas to: {out_path}")
    print("Persona counts by primary:")
    counts: Dict[str, int] = {}
    for row in final_rows:
        counts[row["primary_label"]] = counts.get(row["primary_label"], 0) + 1
    for key, val in sorted(counts.items()):
        print(f"  {key:12} {val:5d}")


if __name__ == "__main__":
    main()
