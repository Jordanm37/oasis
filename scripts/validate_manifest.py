#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

from yaml import safe_load

# Updated required fields for expanded manifest
REQUIRED_MANIFEST_FIELDS = ["rng_seed", "population", "personas", "label_tokens"]
REQUIRED_PERSONA_FIELDS = ["persona_id", "primary_label", "allowed_labels", "style", "behavior", "emission_probs"]


def validate_manifest(path: Path) -> List[str]:
    errs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        data = safe_load(f) or {}
    
    # 1. Check Top-Level Keys
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in data:
            errs.append(f"Manifest missing required field: {key}")
    
    # 2. Validate Population Config
    pop = data.get("population", {})
    if not isinstance(pop, dict) or not pop:
        errs.append("Field 'population' must be a non-empty dictionary.")
    
    # 3. Validate Personas List
    personas = data.get("personas", [])
    if not isinstance(personas, list):
        errs.append("Field 'personas' must be a list.")
    else:
        for idx, p in enumerate(personas):
            if not isinstance(p, dict):
                errs.append(f"Persona at index {idx} is not a dictionary.")
                continue
            
            # Check required persona fields
            for req in REQUIRED_PERSONA_FIELDS:
                if req not in p:
                    errs.append(f"Persona {idx} missing required field: {req}")
            
            # Validate emission_probs are floats
            probs = p.get("emission_probs", {})
            if isinstance(probs, dict):
                for t, v in probs.items():
                    if not isinstance(v, (int, float)):
                        errs.append(f"Persona {idx}: emission_prob for {t} is not a number: {v}")
    
    # 4. Validate Label Tokens
    lt = data.get("label_tokens", {})
    if "inventory" not in lt:
         errs.append("label_tokens missing 'inventory' list")
    if "mapping" not in lt:
         errs.append("label_tokens missing 'mapping' dictionary")

    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate manifest schema.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.yaml")
    args = ap.parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    
    if not manifest_path.exists():
        print(f"Error: File not found {manifest_path}")
        raise SystemExit(1)

    errs = validate_manifest(manifest_path)
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print(f"- {e}")
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
