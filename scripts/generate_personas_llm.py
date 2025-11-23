#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
from yaml import safe_load

from oasis.persona import (
    PersonaGenerator,
    build_requests_from_spec,
    load_ontology,
)

DEFAULT_ONTOLOGY = Path("configs/personas/ontology_llm.yaml")
FIELDNAMES = [
    "username",
    "description",
    "user_char",
    "primary_label",
    "secondary_label",
    "allowed_labels",
    "label_mode_cap",
    "s_json",
    "p_json",
    "c_json",
    "narratives_json",
    "emission_params_json",
    "pair_probs_json",
]


# -----------------------
# CLI and config helpers
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV with LLM SPC/narratives and six-class taxonomy."
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

    # Counts for six classes
    parser.add_argument("--benign", type=int, default=None)
    parser.add_argument("--recovery", type=int, default=None)
    parser.add_argument("--ed-risk", dest="ed_risk", type=int, default=None)
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--conspiracy", type=int, default=None)

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
        default="grok-2",
        help="LLM model name (xAI / Grok).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="",
        help="LLM API key (falls back to env XAI_API_KEY).",
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
        "incel_misogyny": args.incel,
        "misinfo": args.misinfo,
        "conspiracy": args.conspiracy,
    }


# -----------------------
# LLM helpers with cache
# -----------------------
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _variant_id_from_row(row: Mapping[str, str]) -> str:
    group = str(row.get("persona_group", "")).strip()
    variant = str(row.get("persona_variant", "")).strip()
    return f"{group}.{variant}" if variant else group


def _build_llm_prompt(row: Mapping[str, str]) -> Tuple[str, str]:
    """Return (prompt_text, prompt_hash)."""
    identity = row.get("persona_summary", "") or row.get("description", "")
    topics = row.get("persona_topics", "")
    style = row.get("persona_writing_style", row.get("persona_tone", ""))
    label_primary = row.get("label_primary", "")
    label_secondary = row.get("label_secondary", "")
    variant_id = _variant_id_from_row(row)
    prompt = (
        "You generate structured persona SPC blocks and narratives.\n"
        "Return JSON with keys: S, P, C, narratives.\n"
        "S: {groups:list[str], demographics:{gender_proxy,age_band,region_proxy}, role_in_community:str}\n"
        "P: {traits: dict[str,float], values: dict[str,float], communication_style:{formality,sarcasm_rate,aggression}}\n"
        "C: {stage_in_trajectory:str, offline_stressors:list[str], support_exposure:float}\n"
        "narratives: {c_essay:str, p_intro:str}\n"
        "Keep numbers 0-1 where applicable. No extra text.\n"
        f"Variant: {variant_id}\n"
        f"Identity: {identity}\n"
        f"Topics: {topics}\n"
        f"Style: {style}\n"
        f"Primary label: {label_primary}; Secondary labels: {label_secondary}\n"
    )
    return prompt, _sha256(prompt)


def _call_grok(prompt: str, model: str, api_key: str) -> Dict[str, object]:
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a concise JSON-only generator for SPC persona profiles.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


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


def _allowed_candidates(row: Mapping[str, str]) -> List[str]:
    override = _extract_metadata_list(row, "allowed_labels_override")
    if override:
        return override
    primary = str(row.get("label_primary", "")).strip()
    secondary_raw = str(row.get("label_secondary", "")).strip()
    secondary = [tok.strip() for tok in secondary_raw.split(";") if tok.strip()]
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


def _default_emission_params(primary: str) -> Dict[str, float]:
    defaults: Dict[str, Dict[str, float]] = {
        "incel_misogyny": {"LBL:INCEL_MISOGYNY": 0.03, "LBL:HARASSMENT": 0.02},
        "misinfo": {"LBL:MISINFO_CLAIM": 0.03},
        "conspiracy": {"LBL:CONSPIRACY": 0.02},
        "ed_risk": {"LBL:ED_RISK": 0.03},
        "recovery_support": {"LBL:RECOVERY_SUPPORT": 0.02, "LBL:SUPPORTIVE": 0.01},
        "benign": {"LBL:SUPPORTIVE": 0.02},
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
    if primary == "benign":
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
        variant_id = _variant_id_from_row(row)
        profile = profile_lookup.get(variant_id) or {}
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
        base_prompt = row.get("user_char", "").strip()
        user_char = (base_prompt + "\n\n" + preamble + "\n" + narratives.get("p_intro", "")).strip()

        candidates = _allowed_candidates(row)
        is_double, allowed_labels, secondary_label = _pick_double_mode(rng, primary, candidates, single_ratio)
        label_cap = "double" if is_double else "single"

        final_rows.append(
            {
                "username": row.get("username", f"user_{idx}"),
                "description": row.get("description") or row.get("persona_summary", ""),
                "user_char": user_char,
                "primary_label": primary,
                "secondary_label": secondary_label,
                "allowed_labels": json.dumps(allowed_labels, ensure_ascii=False),
                "label_mode_cap": label_cap,
                "s_json": json.dumps(s_block, ensure_ascii=False),
                "p_json": json.dumps(p_block, ensure_ascii=False),
                "c_json": json.dumps(c_block, ensure_ascii=False),
                "narratives_json": json.dumps(narratives, ensure_ascii=False),
                "emission_params_json": json.dumps(_default_emission_params(primary), ensure_ascii=False),
                "pair_probs_json": json.dumps({}, ensure_ascii=False),
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

    ontology_path = args.ontology or personas_cfg.get("ontology", str(DEFAULT_ONTOLOGY))
    ontology = load_ontology(ontology_path)

    seed = int(personas_cfg.get("seed", args.seed))
    single_ratio = float(personas_cfg.get("single_ratio", args.single_ratio))
    generator = PersonaGenerator(ontology=ontology, seed=seed)

    spec = _coalesce_spec(personas_cfg, plan_cfg, _gather_cli_counts(args))
    requests_list = build_requests_from_spec(generator, spec)
    raw_rows = generator.generate(requests_list)

    # Build per-variant profiles (LLM with cache)
    cache_dir = Path(args.llm_cache_dir)
    use_llm = bool(args.use_llm)
    api_key = args.llm_api_key or os.environ.get("XAI_API_KEY", "")
    model = args.llm_model
    profile_lookup: Dict[str, Dict[str, object]] = {}
    for row in raw_rows:
        variant_id = _variant_id_from_row(row)
        if variant_id in profile_lookup:
            continue
        prompt, prompt_hash = _build_llm_prompt(row)
        cache_file = cache_dir / f"{variant_id}__{model}__{prompt_hash[:12]}.json"
        cached = _load_cache(cache_file, prompt_hash, model)
        if cached:
            profile_lookup[variant_id] = cached
            continue
        if use_llm:
            if not api_key:
                raise RuntimeError("LLM API key not provided. Set --llm-api-key or env XAI_API_KEY.")
            try:
                llm_resp = _call_grok(prompt, model, api_key)
                payload = {
                    "variant_id": variant_id,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "S": llm_resp.get("S", {}),
                    "P": llm_resp.get("P", {}),
                    "C": llm_resp.get("C", {}),
                    "narratives": llm_resp.get("narratives", {}),
                }
                _save_cache(cache_file, payload)
                profile_lookup[variant_id] = payload
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] LLM failed for {variant_id}: {exc}. Falling back to defaults.")
        profile_lookup[variant_id] = {}

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
