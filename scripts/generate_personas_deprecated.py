#!/usr/bin/env python3
"""DEPRECATED: Legacy persona generator with only 3 classes (incel, misinfo, benign).

Use `generate_personas_llm.py` instead for the full 13-class taxonomy with:
- RAG-augmented persona synthesis
- Style indicator sampling
- Lexicon sampling from ontology
- LLM-generated SPC blocks

This file is kept for backwards compatibility and reference only.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "generate_personas_deprecated.py is deprecated. "
    "Use generate_personas_llm.py for the 13-class taxonomy.",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from yaml import safe_load

from oasis.persona import (
    PersonaGenerator,
    build_requests_from_spec,
    load_ontology,
)

DEFAULT_ONTOLOGY = Path("configs/personas/ontology.yaml")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV using the configurable ontology."
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
    parser.add_argument(
        "--plan",
        type=str,
        default="",
        help="Optional YAML/JSON file describing persona counts.",
    )
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--benign", type=int, default=None)
    parser.add_argument(
        "--single-ratio",
        type=float,
        default=0.7,
        help="Fraction of personas constrained to single-label mode.",
    )
    return parser.parse_args()


def _load_personas_cfg(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    personas = data.get("personas", {})
    if not isinstance(personas, Mapping):
        raise TypeError("`personas` section must be a mapping")
    return personas


def _load_plan_file(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plan file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
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
    base_mix = (
        base_cfg.get("mix", base_cfg)
        if isinstance(base_cfg, Mapping)
        else base_cfg
    )
    for source in (base_mix, plan_cfg):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if key in {"seed", "personas_csv", "ontology", "single_ratio"}:
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
    return {
        "incel": args.incel,
        "misinfo": args.misinfo,
        "benign": args.benign,
    }


def _select_indices_for_double(indices: List[int], double_count: int, seed: int) -> set[int]:
    if double_count <= 0 or not indices:
        return set()
    scored = [
        (int(hashlib.sha1(f"{seed}:{idx}".encode()).hexdigest(), 16), idx)
        for idx in indices
    ]
    scored.sort()
    limit = max(0, min(double_count, len(indices)))
    return {idx for _, idx in scored[:limit]}


def _infer_primary_label(row: Mapping[str, object]) -> str:
    group = str(row.get("persona_group", "")).lower()
    if "incel" in group:
        return "incel"
    if "misinfo" in group or "conspiracy" in group:
        return "misinfo"
    if "benign" in group:
        return "benign"
    token = str(row.get("label_primary", "")).lower()
    if "incel" in token:
        return "incel"
    if "misinfo" in token:
        return "misinfo"
    return "benign"


def _determine_allowed_labels(primary: str, is_double: bool) -> tuple[list[str], str, str]:
    if primary == "incel":
        if is_double:
            return ["incel", "misinfo"], "misinfo", "double"
        return ["incel"], "", "single"
    if primary == "misinfo":
        if is_double:
            return ["misinfo", "conspiracy"], "conspiracy", "double"
        return ["misinfo", "conspiracy"], "", "single"
    return ["benign"], "", "single"


def _make_spc_blocks(primary: str, variant_slug: str) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    S = {
        "groups": ["mainstream"],
        "demographics": {
            "gender_proxy": "unspecified",
            "age_band": "25-34",
            "region_proxy": "global_north",
        },
        "role_in_community": "conversationalist",
    }
    P = {
        "traits": {
            "grievance_level": 0.2,
            "institutional_trust": 0.6,
            "empathy": 0.7,
            "sensation_seeking": 0.4,
        },
        "values": {
            "gender_equality": 0.7,
            "individual_responsibility": 0.6,
        },
        "communication_style": {
            "formality": "medium",
            "sarcasm_rate": 0.1,
            "aggression": 0.1,
        },
    }
    C = {
        "stage_in_trajectory": "benign",
        "offline_stressors": [],
        "support_exposure": 0.6,
        "acute_events": [],
    }
    slug = (variant_slug or "").lower()
    if primary == "incel":
        S.update({"groups": ["manosphere", "incel_forum"], "role_in_community": "core_poster"})
        P["traits"].update({
            "grievance_level": 0.85,
            "institutional_trust": 0.15,
            "empathy": 0.35 if "aggressor" in slug else 0.45,
            "sensation_seeking": 0.55,
        })
        P["communication_style"].update({
            "formality": "low",
            "sarcasm_rate": 0.25 if "theorist" in slug else 0.2,
            "aggression": 0.75 if "aggressor" in slug else 0.45,
        })
        C.update({
            "stage_in_trajectory": "entrenched_incel" if "aggressor" in slug else ("pre_incel" if "theorist" in slug else "entrenched_incel"),
            "offline_stressors": ["dating_dissatisfaction"],
            "support_exposure": 0.1,
        })
    elif primary == "misinfo":
        S.update({"groups": ["skeptics_forum", "alt_news"], "role_in_community": "broadcaster"})
        P["traits"].update({
            "grievance_level": 0.55,
            "institutional_trust": 0.2,
            "empathy": 0.45,
            "sensation_seeking": 0.6,
        })
        P["communication_style"].update({
            "formality": "low",
            "sarcasm_rate": 0.3,
            "aggression": 0.4,
        })
        C.update({
            "stage_in_trajectory": "active",
            "offline_stressors": ["institutional_distrust"],
            "support_exposure": 0.15,
        })
    return S, P, C


def _make_narratives(primary: str) -> Dict[str, str]:
    if primary == "incel":
        return {
            "c_essay": "Most days I scroll forums after work, stewing over what feels like a rigged dating system. I post when frustration spikes, looking for others who 'get it'.",
            "p_intro": "It's obvious the game is stacked. Call it blackpill or whatever-you'll see the truth if you stop pretending.",
        }
    if primary == "misinfo":
        return {
            "c_essay": "I don't just accept what institutions say. I dig, compare, and challenge 'official' lines-someone has to.",
            "p_intro": "Do your research. Numbers don't add up, and it's on us to question everything.",
        }
    return {
        "c_essay": "I hang out online to chat hobbies and news. Good conversations, civil tone. I try to keep things supportive.",
        "p_intro": "Hey all-into tech, sports, and helpful threads. Keep it friendly.",
    }


def _format_spc_preamble(S: Dict[str, object], P: Dict[str, object], C: Dict[str, object], intro: str) -> str:
    return (
        "[SPC]\n"
        f"S(groups={S.get('groups')}, role={S.get('role_in_community')}, demo={S.get('demographics', {})})\n"
        f"P(traits={P.get('traits')}, style={P.get('communication_style')})\n"
        f"C(stage={C.get('stage_in_trajectory')}, stressors={C.get('offline_stressors')})\n"
        f"N(c_intro={intro})"
    )


def _default_emission_params(primary: str) -> Dict[str, float]:
    if primary == "incel":
        return {"LBL:INCEL_SLANG": 0.04, "LBL:HARASSMENT": 0.02}
    if primary == "misinfo":
        return {"LBL:MISINFO_CLAIM": 0.03, "LBL:MISINFO_SOURCE": 0.01}
    return {"LBL:SUPPORTIVE": 0.02}


def _transform_rows(
    raw_rows: List[Dict[str, str]],
    seed: int,
    single_ratio: float,
) -> List[Dict[str, str]]:
    primary_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(raw_rows):
        label = _infer_primary_label(row)
        primary_groups[label].append(idx)

    def target_double_count(n: int) -> int:
        return max(0, int(round(n * max(0.0, min(1.0, 1.0 - single_ratio)))))

    double_indices: Dict[str, set[int]] = {}
    for group, indices in primary_groups.items():
        stable = int(hashlib.sha1(group.encode()).hexdigest(), 16)
        double_indices[group] = _select_indices_for_double(
            indices, target_double_count(len(indices)), seed + stable
        )

    final_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        primary = _infer_primary_label(row)
        is_double = idx in double_indices.get(primary, set())
        allowed_labels, secondary_label, label_cap = _determine_allowed_labels(primary, is_double)
        s_block, p_block, c_block = _make_spc_blocks(primary, row.get("persona_variant", ""))
        narratives = _make_narratives(primary)
        preamble = _format_spc_preamble(s_block, p_block, c_block, narratives.get("p_intro", ""))
        user_char = (row.get("user_char", "").strip() + "\n\n" + preamble).strip()
        final_rows.append(
            {
                "username": row.get("username", f"user_{idx}").strip(),
                "description": row.get("description", row.get("persona_summary", "")).strip(),
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
    counter = Counter()
    for row in rows:
        counter[row.get("primary_label", "unknown")] += 1
    print("Persona counts by primary label:")
    for label, count in sorted(counter.items()):
        print(f"  {label:<10} {count}")


def main() -> None:
    args = parse_args()
    personas_cfg = _load_personas_cfg(args.config)
    plan_cfg = _load_plan_file(args.plan)

    ontology_path = personas_cfg.get("ontology", args.ontology)
    ontology = load_ontology(ontology_path)

    seed = int(personas_cfg.get("seed", args.seed))
    single_ratio = float(personas_cfg.get("single_ratio", args.single_ratio))
    generator = PersonaGenerator(ontology=ontology, seed=seed)

    spec = _coalesce_spec(
        personas_cfg,
        plan_cfg,
        _gather_cli_counts(args),
    )
    requests = build_requests_from_spec(generator, spec)
    raw_rows = generator.generate(requests)
    final_rows = _transform_rows(raw_rows, seed, single_ratio)

    out_path = _determine_output_path(args, personas_cfg)
    _write_csv(out_path, final_rows)
    print(f"Wrote personas to: {out_path}")
    _print_summary(final_rows)


if __name__ == "__main__":
    main()
