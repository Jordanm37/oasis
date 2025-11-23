#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from yaml import safe_load

from oasis.persona import (
    PersonaGenerator,
    build_requests_from_spec,
    load_ontology,
)

DEFAULT_ONTOLOGY = Path("configs/personas/ontology_spc_experiment.yaml")
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
DEFAULT_ALLOWED = {
    "incel": ["incel"],
    "misinfo": ["misinfo", "conspiracy"],
    "benign": ["benign"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV using the ontology with SPC metadata."
    )
    parser.add_argument("--out", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--ontology",
        type=str,
        default=str(DEFAULT_ONTOLOGY),
        help="Path to persona ontology YAML.",
    )
    parser.add_argument("--plan", type=str, default="", help="Optional counts file")
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--benign", type=int, default=None)
    parser.add_argument(
        "--single-ratio",
        type=float,
        default=0.7,
        help="Fraction of personas forced to single-label",
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
    personas = data.get("personas", {})
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
    if not spec:
        raise ValueError("No persona allocation specified.")
    return spec


def _determine_output_path(args: argparse.Namespace, cfg: Mapping[str, object]) -> Path:
    out_path = cfg.get("personas_csv", args.out)
    return Path(os.path.abspath(str(out_path)))


def _gather_cli_counts(args: argparse.Namespace) -> Dict[str, int | None]:
    return {"incel": args.incel, "misinfo": args.misinfo, "benign": args.benign}


def _safe_json(value: Optional[str]) -> Optional[object]:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


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
    for parser in (json.loads, ast.literal_eval):
        try:
            loaded = parser(raw)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except Exception:
            continue
    return [item.strip() for item in raw.split(",") if item.strip()]


def _infer_primary_label(row: Mapping[str, str]) -> str:
    variant = str(row.get("persona_group", "")).lower()
    if "incel" in variant:
        return "incel"
    if "misinfo" in variant or "conspiracy" in variant:
        return "misinfo"
    if "benign" in variant:
        return "benign"
    token = str(row.get("label_primary", "")).lower()
    if "incel" in token:
        return "incel"
    if "misinfo" in token:
        return "misinfo"
    return "benign"


def _default_spc_text(row: Mapping[str, str], key: str) -> str:
    s_pc = _make_spc_blocks(row.get("_primary", ""), row.get("persona_variant", ""))
    s_block, p_block, c_block = s_pc
    if key == "social":
        groups = s_block.get("groups") or []
        role = s_block.get("role_in_community", "participant")
        groups_txt = ", ".join(groups) if groups else "open forums"
        return f"Active in {groups_txt} spaces as a {role}."
    if key == "psych":
        traits = p_block.get("traits") or {}
        top = sorted(traits.items(), key=lambda item: item[1], reverse=True)
        names = [name.replace("_", " ") for name, _ in top[:3]]
        names_txt = ", ".join(names) if names else "balanced poster"
        return f"Inner traits center on being {names_txt}."
    stage = c_block.get("stage_in_trajectory", "benign")
    stressors = c_block.get("offline_stressors", [])
    stress_txt = ", ".join(stressors) if stressors else "day-to-day routines"
    return f"Currently in {stage} stage, navigating {stress_txt}."


def _choose_narratives(row: Mapping[str, str]) -> Dict[str, str]:
    narratives = {}
    base = _safe_json(row.get("narratives_json"))
    if isinstance(base, dict):
        narratives.update({str(k): str(v) for k, v in base.items()})
    intro_override = _extract_metadata(row, "narrative_intro")
    if intro_override:
        narratives["p_intro"] = intro_override
    context_override = _extract_metadata(row, "narrative_context")
    if context_override:
        narratives["c_essay"] = context_override
    if not narratives:
        narratives = {
            "p_intro": "Staying active online to swap stories.",
            "c_essay": "Daily life mixes work, errand runs, and late-night posting.",
        }
    return narratives


def _build_spc_preamble(social: str, psych: str, life: str, intro: str) -> str:
    segments = []
    if social:
        segments.append(f"S | {social}")
    if psych:
        segments.append(f"P | {psych}")
    if life:
        segments.append(f"C | {life}")
    if intro:
        segments.append(f"N | {intro}")
    return "\n".join(segments)


def _double_ratio_config(cfg: Mapping[str, object], default_ratio: float) -> Dict[str, float]:
    ratios = cfg.get("double_ratio", {}) if isinstance(cfg, Mapping) else {}
    out: Dict[str, float] = {"_default": default_ratio}
    if isinstance(ratios, Mapping):
        for k, v in ratios.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    return out


def _select_double_indices(
    group_indices: List[int],
    rows: List[Dict[str, str]],
    group: str,
    default_ratio: float,
    ratio_cfg: Mapping[str, float],
    seed: int,
) -> set[int]:
    if not group_indices:
        return set()
    target_ratio = float(ratio_cfg.get(group, ratio_cfg.get("_default", default_ratio)))
    target_ratio = max(0.0, min(1.0, target_ratio))
    target_total = int(round(len(group_indices) * target_ratio))

    forced_double = {idx for idx in group_indices if _extract_metadata(rows[idx], "double_mode").lower() == "force_double"}
    forced_single = {idx for idx in group_indices if _extract_metadata(rows[idx], "double_mode").lower() == "force_single"}
    remaining = [idx for idx in group_indices if idx not in forced_double and idx not in forced_single]

    residual = max(0, target_total - len(forced_double))
    if residual <= 0:
        return forced_double

    scored = [
        (int(hashlib.sha1(f"{seed}:{group}:{idx}".encode()).hexdigest(), 16), idx)
        for idx in remaining
    ]
    scored.sort()
    selected = {idx for _, idx in scored[: min(residual, len(remaining))]}
    return forced_double | selected


def _determine_allowed_labels(
    row: Mapping[str, str],
    primary: str,
    is_double: bool,
) -> tuple[List[str], str, str]:
    override = _extract_metadata_list(row, "allowed_labels_override")
    base = override if override else DEFAULT_ALLOWED.get(primary, [primary])
    allowed = list(dict.fromkeys(base))
    label_cap = "double" if is_double else "single"
    secondary = allowed[1] if label_cap == "double" and len(allowed) > 1 else ""
    return allowed, secondary, label_cap


def _default_emission_params(primary: str) -> Dict[str, float]:
    default_map = {
        "incel": {"LBL:INCEL_SLANG": 0.04, "LBL:HARASSMENT": 0.02},
        "misinfo": {"LBL:MISINFO_CLAIM": 0.03, "LBL:MISINFO_SOURCE": 0.01},
        "benign": {"LBL:SUPPORTIVE": 0.02},
    }
    return default_map.get(primary, {"LBL:SUPPORTIVE": 0.01})


def _make_spc_blocks(primary: str, variant_slug: str) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    social = {
        "groups": ["mainstream"],
        "demographics": {
            "gender_proxy": "unspecified",
            "age_band": "25-34",
            "region_proxy": "global_north",
        },
        "role_in_community": "conversationalist",
    }
    personal = {
        "traits": {
            "grievance_level": 0.2,
            "institutional_trust": 0.6,
            "empathy": 0.7,
            "sensation_seeking": 0.4,
        },
        "communication_style": {
            "formality": "medium",
            "sarcasm_rate": 0.1,
            "aggression": 0.1,
        },
    }
    context = {
        "stage_in_trajectory": "benign",
        "offline_stressors": [],
        "support_exposure": 0.7,
    }
    slug = (variant_slug or "").lower()
    if primary == "incel":
        social.update({"groups": ["manosphere", "incel_forum"], "role_in_community": "poster"})
        personal["traits"].update({
            "grievance_level": 0.8,
            "institutional_trust": 0.15,
            "empathy": 0.4,
            "sensation_seeking": 0.55,
        })
        context.update({"stage_in_trajectory": "entrenched_incel", "offline_stressors": ["dating_dissatisfaction"], "support_exposure": 0.1})
    elif primary == "misinfo":
        social.update({"groups": ["skeptics_forum", "alt_news"], "role_in_community": "broadcaster"})
        personal["traits"].update({
            "grievance_level": 0.55,
            "institutional_trust": 0.2,
            "empathy": 0.45,
            "sensation_seeking": 0.6,
        })
        context.update({"stage_in_trajectory": "active", "offline_stressors": ["institutional_distrust"], "support_exposure": 0.15})
    return social, personal, context


def _transform_rows(
    raw_rows: List[Dict[str, str]],
    seed: int,
    single_ratio: float,
    ratio_cfg: Mapping[str, float],
) -> List[Dict[str, str]]:
    rows = [dict(row) for row in raw_rows]
    primary_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        primary = _infer_primary_label(row)
        row["_primary"] = primary
        primary_groups[primary].append(idx)

    default_double_ratio = max(0.0, min(1.0, 1.0 - single_ratio))
    double_sets: Dict[str, set[int]] = {}
    for group, indices in primary_groups.items():
        double_sets[group] = _select_double_indices(indices, rows, group, default_double_ratio, ratio_cfg, seed)

    final_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        primary = row["_primary"]
        is_double = idx in double_sets.get(primary, set())
        social = _extract_metadata(row, "social_identity") or _default_spc_text(row, "social")
        psych = _extract_metadata(row, "psych_identity") or _default_spc_text(row, "psych")
        life = _extract_metadata(row, "life_context") or _default_spc_text(row, "life")
        narratives = _choose_narratives(row)
        intro_text = narratives.get("p_intro", "")
        spc_text = _build_spc_preamble(social, psych, life, intro_text)
        base_prompt = row.get("user_char", "").strip()
        user_char = (base_prompt + "\n\n[SPC]\n" + spc_text).strip()

        allowed, secondary, label_cap = _determine_allowed_labels(row, primary, is_double)
        emission_params = row.get("emission_params_json") or json.dumps(_default_emission_params(primary), ensure_ascii=False)
        pair_probs = row.get("pair_probs_json") or json.dumps({}, ensure_ascii=False)

        s_block, p_block, c_block = _make_spc_blocks(primary, row.get("persona_variant", ""))

        final_rows.append(
            {
                "username": row.get("username", f"user_{idx}"),
                "description": row.get("description") or row.get("persona_summary", ""),
                "user_char": user_char,
                "primary_label": primary,
                "secondary_label": secondary,
                "allowed_labels": json.dumps(allowed, ensure_ascii=False),
                "label_mode_cap": label_cap,
                "s_json": json.dumps(s_block, ensure_ascii=False),
                "p_json": json.dumps(p_block, ensure_ascii=False),
                "c_json": json.dumps(c_block, ensure_ascii=False),
                "narratives_json": json.dumps(narratives, ensure_ascii=False),
                "emission_params_json": emission_params,
                "pair_probs_json": pair_probs,
            }
        )
    return final_rows


def _write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("Persona generator produced no rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_list)


def _print_summary(rows: Iterable[Dict[str, str]]) -> None:
    counter = Counter(row.get("primary_label", "unknown") for row in rows)
    print("Persona counts by primary:")
    for label, count in sorted(counter.items()):
        print(f"  {label:<8} {count}")


def main() -> None:
    args = parse_args()
    personas_cfg = _load_personas_cfg(args.config)
    plan_cfg = _load_plan_file(args.plan)

    ontology_path = personas_cfg.get("ontology", args.ontology)
    ontology = load_ontology(ontology_path)

    seed = int(personas_cfg.get("seed", args.seed))
    single_ratio = float(personas_cfg.get("single_ratio", args.single_ratio))
    ratio_cfg = _double_ratio_config(personas_cfg, max(0.0, min(1.0, 1.0 - single_ratio)))
    generator = PersonaGenerator(ontology=ontology, seed=seed)

    spec = _coalesce_spec(personas_cfg, plan_cfg, _gather_cli_counts(args))
    requests = build_requests_from_spec(generator, spec)
    raw_rows = generator.generate(requests)
    final_rows = _transform_rows(raw_rows, seed, single_ratio, ratio_cfg)

    out_path = _determine_output_path(args, personas_cfg)
    _write_csv(out_path, final_rows)
    print(f"Wrote personas to: {out_path}")
    _print_summary(final_rows)


if __name__ == "__main__":
    main()
