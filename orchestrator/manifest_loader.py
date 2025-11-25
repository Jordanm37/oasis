from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from yaml import safe_load


@dataclass(frozen=True)
class Manifest:
    data: Dict[str, Any]

    @property
    def run_seed(self) -> int:
        return int(self.data.get("rng_seed", 314159))

    @property
    def post_label_mode_probs(self) -> Dict[str, float]:
        return dict(self.data.get("post_label_mode_probs", {"none": 0.5, "single": 0.4, "double": 0.1}))

    @property
    def multi_label_targets(self) -> Dict[str, float]:
        targets = self.data.get("multi_label_targets", {})
        # Defaults to 20% with 18–22% band
        return {
            "target_rate": float(targets.get("target_rate", 0.20)),
            "min_rate": float(targets.get("min_rate", 0.18)),
            "max_rate": float(targets.get("max_rate", 0.22)),
        }

    @property
    def guidance_config(self) -> Dict[str, Any]:
        r"""Optional guidance configuration for style hints."""
        cfg = self.data.get("guidance", {}) or {}
        enable = bool(cfg.get("enable", False))
        intensity = float(cfg.get("intensity", 1.0))
        use_state = bool(cfg.get("use_state", False))
        token_weighting = bool(cfg.get("token_weighting", False))
        # Clamp intensity
        if intensity < 0.0:
            intensity = 0.0
        if intensity > 1.0:
            intensity = 1.0
        return {
            "enable": enable,
            "intensity": intensity,
            "use_state": use_state,
            "token_weighting": token_weighting,
        }

def load_manifest(path: Path) -> Manifest:
    with path.open("r", encoding="utf-8") as f:
        data = safe_load(f) or {}
    return Manifest(data=data)


def _resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    # Try relative to current working directory first
    cwdp = Path.cwd() / p
    if cwdp.exists():
        return cwdp
    # Then relative to ontology directory or its parent
    rel1 = base.parent / p
    if rel1.exists():
        return rel1
    rel2 = base.parent.parent / p
    return rel2


def load_label_lexicons(ontology_path: Path) -> Dict[str, Dict[str, List[str]]]:
    r"""Build a label→{required, optional} lexicon map from ontology + collections.

    Strategy:
    - Read `lexicon_collections` to map collection-id → file path.
    - For each archetype.variant, read `lexicon_refs` and `label_emission` blocks.
      Any label referenced by that variant inherits the union of terms from its collections.
    """
    with ontology_path.open("r", encoding="utf-8") as f:
        data = safe_load(f) or {}
    collections: Mapping[str, Any] = data.get("lexicon_collections", {}) or {}
    archetypes: Mapping[str, Any] = data.get("archetypes", {}) or {}

    # Load all collections once
    collection_terms: Dict[str, Tuple[List[str], List[str]]] = {}
    for cid, entry in collections.items():
        file_entry = entry.get("file") if isinstance(entry, dict) else None
        if not file_entry:
            continue
        path = _resolve_path(ontology_path, str(file_entry))
        try:
            with path.open("r", encoding="utf-8") as fh:
                lex = safe_load(fh) or {}
        except Exception:
            lex = {}
        req = [str(x) for x in (lex.get("required") or [])]
        opt = [str(x) for x in (lex.get("optional") or [])]
        collection_terms[str(cid)] = (req, opt)

    # Aggregate by label
    label_to_terms: Dict[str, Dict[str, List[str]]] = {}
    for _, arch in (archetypes or {}).items():
        variants = (arch or {}).get("variants", {}) or {}
        for _, var in variants.items():
            meta = var or {}
            refs = [str(x) for x in (meta.get("lexicon_refs") or [])]
            labels = set()
            le = meta.get("label_emission") or {}
            primary = le.get("primary")
            if isinstance(primary, str):
                labels.add(primary)
            sec = le.get("secondary") or []
            if isinstance(sec, list):
                for s in sec:
                    if isinstance(s, str):
                        labels.add(s)
            # Union terms from all referenced collections into each label used by this variant
            req_union: List[str] = []
            opt_union: List[str] = []
            for rid in refs:
                req, opt = collection_terms.get(rid, ([], []))
                req_union.extend(req)
                opt_union.extend(opt)
            if not labels:
                continue
            req_dedup = sorted({w for w in req_union if w})
            opt_dedup = sorted({w for w in opt_union if w})
            for lab in labels:
                bucket = label_to_terms.setdefault(str(lab), {"required": [], "optional": []})
                bucket["required"] = sorted(set(bucket["required"]) | set(req_dedup))
                bucket["optional"] = sorted(set(bucket["optional"]) | set(opt_dedup))

    return label_to_terms


