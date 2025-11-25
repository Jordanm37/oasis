#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from yaml import safe_load


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return safe_load(f) or {}


def generate_personas_csv(manifest_path: Path, output_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    population = manifest.get("population", {})
    personas_list = manifest.get("personas", [])
    
    # Map persona_id to persona config
    persona_map = {p["persona_id"]: p for p in personas_list}

    fieldnames = [
        "username", "description", "user_char", 
        "primary_label", "label_mode_cap", "allowed_labels", 
        "emission_params_json", "style_json", "behavior_json"
    ]

    rows = []
    global_idx = 0

    for p_key, count in population.items():
        # Look up the persona config using the key from population (e.g. "persona_incel_mvp")
        p_cfg = persona_map.get(p_key)
        if not p_cfg:
            print(f"Warning: Population key '{p_key}' not found in 'personas' list. Skipping.")
            continue

        primary = p_cfg.get("primary_label", "benign")
        allowed = p_cfg.get("allowed_labels", [primary])
        style = p_cfg.get("style", {})
        behavior = p_cfg.get("behavior", {})
        emission = p_cfg.get("emission_probs", {})
        
        # Base description/char (could be randomized in a real generator)
        base_char = f"A user interested in {', '.join(p_cfg.get('topics', []))}."
        
        for i in range(count):
            username = f"{primary}_{global_idx:05d}"
            
            row = {
                "username": username,
                "description": f"Bot user {username}",
                "user_char": base_char,
                "primary_label": primary,
                "label_mode_cap": "single", # Default
                "allowed_labels": json.dumps(allowed),
                "emission_params_json": json.dumps(emission),
                "style_json": json.dumps(style),
                "behavior_json": json.dumps(behavior)
            }
            rows.append(row)
            global_idx += 1

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated {len(rows)} personas in {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate personas CSV from manifest.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.yaml")
    ap.add_argument("--out", type=str, required=True, help="Path to output CSV")
    args = ap.parse_args()
    
    manifest_path = Path(os.path.abspath(args.manifest))
    out_path = Path(os.path.abspath(args.out))
    
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
        
    generate_personas_csv(manifest_path, out_path)


if __name__ == "__main__":
    main()

