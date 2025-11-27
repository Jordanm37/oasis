from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from orchestrator.rng import DeterministicRNG

DEFAULT_LABEL_TO_TOKENS: Dict[str, List[str]] = {
    # ==========================================================================
    # 13-class taxonomy token inventory (expanded)
    # ==========================================================================
    # Benign / Recovery cluster
    "benign": [
        "LBL:SUPPORTIVE",
        "LBL:FRIENDLY",
        "LBL:HELPFUL",
        "LBL:POSITIVE",
    ],
    "recovery_support": [
        "LBL:RECOVERY_SUPPORT",
        "LBL:SUPPORTIVE",
        "LBL:SELF_HARM",
        "LBL:PEER_SUPPORT",
        "LBL:COPING_TIP",
        "LBL:MILESTONE",
    ],

    # ED cluster
    "ed_risk": [
        "LBL:ED_RISK",
        "LBL:ED_METHOD",
        "LBL:ED_PROMO",
        "LBL:SELF_HARM",
        "LBL:RESTRICTION_TIP",
        "LBL:BODY_CHECK",
        "LBL:CALORIE_OBSESSION",
        "LBL:FASTING_GLORIFY",
    ],
    "pro_ana": [
        "LBL:MEANSPO",
        "LBL:ED_COACHING",
        "LBL:THINSPO",
        "LBL:PURGE_TIP",
        "LBL:ACCOUNTABILITY_THREAT",
        "LBL:GOAL_WEIGHT",
    ],

    # Incel / Manosphere cluster
    "incel_misogyny": [
        "LBL:INCEL_MISOGYNY",
        "LBL:INCEL_SLANG",
        "LBL:MISOGYNY",
        "LBL:HARASSMENT",
        "LBL:BLACKPILL",
        "LBL:FOID_SLUR",
        "LBL:CHAD_COPE",
        "LBL:DATING_RANT",
        "LBL:LOOKISM",
    ],
    "alpha": [
        "LBL:MISOGYNISTIC_LECTURE",
        "LBL:OBJECTIFICATION",
        "LBL:SMV_THEORY",
        "LBL:REDPILL_WISDOM",
        "LBL:HYPERGAMY_CLAIM",
        "LBL:FRAME_CONTROL",
        "LBL:PLATE_SPINNING",
    ],

    # Misinfo / Conspiracy cluster
    "misinfo": [
        "LBL:MISINFO_CLAIM",
        "LBL:MISINFO_SOURCE",
        "LBL:ANTI_INSTITUTION",
        "LBL:FAKE_STAT",
        "LBL:DEBUNKED_CLAIM",
        "LBL:FEAR_MONGER",
        "LBL:SUPPRESSED_TRUTH",
    ],
    "conspiracy": [
        "LBL:CONSPIRACY",
        "LBL:CONSPIRACY_NARRATIVE",
        "LBL:DEEPSTATE",
        "LBL:ANTI_INSTITUTION",
        "LBL:HIDDEN_AGENDA",
        "LBL:FALSE_FLAG",
        "LBL:COVER_UP",
        "LBL:CONTROLLED_OPP",
        "LBL:WAKE_UP",
    ],
    
    # Culture war cluster
    "trad": [
        "LBL:DOGWHISTLE",
        "LBL:GENDER_ESSENTIALISM",
        "LBL:TRAD_AESTHETIC",
        "LBL:MODERNITY_CRITIQUE",
        "LBL:FAMILY_VALUES",
        "LBL:NATURAL_ORDER",
        "LBL:DECLINE_NARRATIVE",
    ],
    "gamergate": [
        "LBL:CULTURE_WAR",
        "LBL:GATEKEEPING",
        "LBL:WOKE_AGENDA",
        "LBL:FORCED_DIVERSITY",
        "LBL:SJW_ATTACK",
        "LBL:BOYCOTT_CALL",
        "LBL:GAMER_DEFENSE",
    ],

    # Extreme harm cluster
    "extremist": [
        "LBL:VIOLENT_THREAT",
        "LBL:HATE_SLUR",
        "LBL:ACCELERATIONISM",
        "LBL:RACE_WAR",
        "LBL:BOOGALOO",
        "LBL:COLLAPSE_WISH",
        "LBL:ENEMY_DEHUMANIZE",
        "LBL:MARTYR_GLORIFY",
    ],
    "hate_speech": [
        "LBL:HATE_SLUR",
        "LBL:DEHUMANIZATION",
        "LBL:REPLACEMENT_THEORY",
        "LBL:RACIAL_SLUR",
        "LBL:RELIGIOUS_HATE",
        "LBL:ETHNIC_ATTACK",
        "LBL:SUPREMACIST",
        "LBL:VERMIN_RHETORIC",
    ],
    "bullying": [
        "LBL:PERSONAL_ATTACK",
        "LBL:DOXXING_THREAT",
        "LBL:SUICIDE_BAIT",
        "LBL:APPEARANCE_MOCK",
        "LBL:PILE_ON",
        "LBL:SCREENSHOT_SHAME",
        "LBL:GASLIGHT",
        "LBL:SOCIAL_EXCLUSION",
    ],

    # Legacy aliases (for backwards compatibility)
    "incel": [
        "LBL:INCEL_MISOGYNY",
        "LBL:INCEL_SLANG",
        "LBL:MISOGYNY",
        "LBL:HARASSMENT",
        "LBL:BLACKPILL",
    ],
    "recovery": [
        "LBL:RECOVERY_SUPPORT",
        "LBL:SUPPORTIVE",
        "LBL:SELF_HARM",
        "LBL:PEER_SUPPORT",
    ],
}


@dataclass(frozen=True)
class PersonaConfig:
    persona_id: str
    primary_label: str
    allowed_labels: List[str]
    label_mode_cap: str  # "single" | "double"
    benign_on_none_prob: float = 0.6
    max_labels_per_post: int = 2
    emission_probs: Optional[Dict[str, float]] = None  # token -> prob
    pair_probs: Optional[Dict[str, float]] = None      # "L_i↔L_j" -> prob
    allowed_label_tokens: Optional[Dict[str, List[str]]] = None
    prompt_metadata: Optional[Dict[str, object]] = None
    lexicon_samples: Optional[Dict[str, object]] = None
    style_variation: Optional[Dict[str, object]] = None


class EmissionPolicy:
    r"""Compute deterministic per-step emission decisions (none/single/double).

    Decision is driven by:
    - Global post_label_mode_probs (none/single/double)
    - Persona caps and preferences (allowed_labels, pair_probs, emission_probs)
    - A DeterministicRNG keyed by (run_seed, user_id, thread_id, step_idx)
    """

    def __init__(
        self,
        run_seed: int,
        post_label_mode_probs: Dict[str, float],
        label_to_tokens: Optional[Dict[str, List[str]]] = None,
        label_lexicon_terms: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> None:
        self._run_seed = int(run_seed)
        self._post_mode = dict(post_label_mode_probs or {})
        self._label_to_tokens = dict(label_to_tokens or DEFAULT_LABEL_TO_TOKENS)
        # Normalized label→{required, optional} map
        self._label_lexicon_terms: Dict[str, Dict[str, List[str]]] = {}
        for lab, terms in (label_lexicon_terms or {}).items():
            if not isinstance(terms, dict):
                continue
            req = [str(x) for x in (terms.get("required") or [])]
            opt = [str(x) for x in (terms.get("optional") or [])]
            self._label_lexicon_terms[str(lab)] = {"required": req, "optional": opt}

    def decide(
        self,
        user_id: int,
        thread_id: str | int,
        step_idx: int,
        persona: PersonaConfig,
        context: Optional[Dict] = None,
        override_post_mode_probs: Optional[Dict[str, float]] = None,
    ) -> Dict:
        rng_root = DeterministicRNG(self._run_seed).fork(
            f"user:{user_id}", f"thread:{thread_id}", f"step:{step_idx}"
        )
        mode = self._sample_mode(rng_root, persona, override_post_mode_probs)
        if mode == "none":
            return {"mode": "none", "tokens": []}

        # Determine labels for single/double and select tokens.
        labels: List[str] = []
        tokens: List[str] = []
        if mode == "single":
            lab = self._sample_single_label(rng_root, persona)
            tok = self._sample_token_for_label(rng_root, persona, lab, context=context)
            labels = [lab]
            tokens = [tok]
        else:
            l1, l2 = self._sample_label_pair(rng_root, persona)
            t1 = self._sample_token_for_label(rng_root.fork("l1"), persona, l1, context=context)
            t2 = self._sample_token_for_label(rng_root.fork("l2"), persona, l2, context=context)
            tokens = [t1]
            labels = [l1]
            if persona.max_labels_per_post >= 2 and l2 != l1:
                tokens.append(t2)
                labels.append(l2)
        # Label-scoped lexicon sampling derived from ontology (if provided)
        lex_samples: Dict[str, Dict[str, List[str]]] = {}
        if labels:
            for lab in sorted(set(labels)):
                lex = self._sample_lexicon_for_label(rng_root.fork("lex", lab), lab)
                if lex:
                    lex_samples[lab] = lex
        return {
            "mode": "double" if len(tokens) == 2 else ("single" if tokens else "none"),
            "tokens": tokens,
            "label_lexicon_samples": lex_samples,
        }

    def _sample_mode(self, rng: DeterministicRNG, persona: PersonaConfig, override: Optional[Dict[str, float]]) -> str:
        probs = override if override else self._post_mode
        mode = rng.fork("mode").categorical(probs)
        if persona.label_mode_cap == "single" and mode == "double":
            return "single"
        if persona.label_mode_cap == "double":
            return mode
        # default guard
        return "single" if mode not in ("none", "double") else mode

    def _sample_single_label(self, rng: DeterministicRNG, persona: PersonaConfig) -> str:
        # Uniform over allowed_labels as a simple default policy.
        if not persona.allowed_labels:
            return persona.primary_label
        probs = {lab: 1.0 for lab in persona.allowed_labels}
        return rng.fork("single_label").categorical(probs)

    def _sample_label_pair(self, rng: DeterministicRNG, persona: PersonaConfig) -> Tuple[str, str]:
        allowed = persona.allowed_labels or [persona.primary_label]
        if len(allowed) == 1:
            return allowed[0], allowed[0]
        # If pair_probs exists, use it; otherwise sample uniform distinct pair.
        if persona.pair_probs:
            pair = rng.fork("pair").categorical(persona.pair_probs)
            if "↔" in pair:
                a, b = pair.split("↔", 1)
                if a in allowed and b in allowed:
                    return a, b
        # Fallback uniform distinct
        a = rng.fork("pair_a").categorical({lab: 1.0 for lab in allowed})
        remaining = [lab for lab in allowed if lab != a] or [a]
        b = rng.fork("pair_b").categorical({lab: 1.0 for lab in remaining})
        return a, b

    def _sample_token_for_label(self, rng: DeterministicRNG, persona: PersonaConfig, label: str, context: Optional[Dict] = None) -> str:
        # 1. Get candidate tokens for this label (persona override first)
        candidates = None
        if persona.allowed_label_tokens:
            candidates = persona.allowed_label_tokens.get(label)
        if not candidates:
            candidates = self._label_to_tokens.get(label)
        if not candidates:
            # Fallback to canonical token if no variants defined
        return f"LBL:{label.upper()}"

        # 2. Check context for dynamic weighting (e.g., from harm priors)
        dynamic_weights = {}
        if context and "dynamic_token_probs" in context:
            dyn = context["dynamic_token_probs"]
            for tok in candidates:
                if tok in dyn:
                    dynamic_weights[tok] = dyn[tok]

        # 3. If dynamic weights exist and have signal, sample from them
        if dynamic_weights and sum(dynamic_weights.values()) > 0:
            total = sum(dynamic_weights.values())
            probs = {k: v / total for k, v in dynamic_weights.items()}
            return rng.fork("token_dynamic").categorical(probs)

        # 4. Check persona static emission_probs (if defined)
        static_weights = {}
        if persona.emission_probs:
            for tok in candidates:
                if tok in persona.emission_probs:
                    static_weights[tok] = persona.emission_probs[tok]

        # 5. If static weights exist, sample
        if static_weights and sum(static_weights.values()) > 0:
            total = sum(static_weights.values())
            probs = {k: v / total for k, v in static_weights.items()}
            return rng.fork("token_static").categorical(probs)

        # 6. Uniform fallback among candidates
        probs = {tok: 1.0 for tok in candidates}
        return rng.fork("token_uniform").categorical(probs)

    def _sample_lexicon_for_label(self, rng: DeterministicRNG, label: str) -> Dict[str, List[str]]:
        terms = self._label_lexicon_terms.get(label) or {}
        required = [t for t in (terms.get("required") or []) if t]
        optional = [t for t in (terms.get("optional") or []) if t]
        out_req: List[str] = []
        out_opt: List[str] = []
        # At most one required term as a nudge
        if required:
            # deterministic pick
            weights = {w: 1.0 for w in required}
            out_req.append(rng.fork("req").categorical(weights))
        # Up to two optional terms
        pool = list(optional)
        while pool and len(out_opt) < 2:
            weights = {w: 1.0 for w in pool}
            pick = rng.fork(f"opt_{len(out_opt)}").categorical(weights)
            out_opt.append(pick)
            pool.remove(pick)
        if not out_req and not out_opt:
            return {}
        return {"required": out_req, "optional": out_opt}
