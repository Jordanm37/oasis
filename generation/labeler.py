from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

DEFAULT_TOKEN_TO_CATEGORIES: Dict[str, List[str]] = {
    # ==========================================================================
    # Expanded 13-class taxonomy token inventory
    # ==========================================================================

    # -------------------------------------------------------------------------
    # Benign / Recovery cluster
    # -------------------------------------------------------------------------
    "LBL:BENIGN": ["benign"],
    "LBL:SUPPORTIVE": ["recovery_support", "benign"],
    "LBL:FRIENDLY": ["benign"],
    "LBL:HELPFUL": ["benign"],
    "LBL:POSITIVE": ["benign"],
    "LBL:RECOVERY": ["recovery_support"],
    "LBL:RECOVERY_SUPPORT": ["recovery_support"],
    "LBL:PEER_SUPPORT": ["recovery_support"],
    "LBL:COPING_TIP": ["recovery_support"],
    "LBL:MILESTONE": ["recovery_support"],

    # -------------------------------------------------------------------------
    # ED cluster
    # -------------------------------------------------------------------------
    "LBL:ED_RISK": ["ed_risk"],
    "LBL:ED_METHOD": ["ed_risk"],
    "LBL:ED_PROMO": ["ed_risk"],
    "LBL:SELF_HARM": ["ed_risk", "recovery_support"],
    "LBL:RESTRICTION_TIP": ["ed_risk"],
    "LBL:BODY_CHECK": ["ed_risk"],
    "LBL:CALORIE_OBSESSION": ["ed_risk"],
    "LBL:FASTING_GLORIFY": ["ed_risk"],
    "LBL:MEANSPO": ["pro_ana"],
    "LBL:ED_COACHING": ["pro_ana"],
    "LBL:THINSPO": ["pro_ana"],
    "LBL:PURGE_TIP": ["pro_ana"],
    "LBL:ACCOUNTABILITY_THREAT": ["pro_ana"],
    "LBL:GOAL_WEIGHT": ["pro_ana"],

    # -------------------------------------------------------------------------
    # Incel / Manosphere cluster
    # -------------------------------------------------------------------------
    "LBL:INCEL": ["incel_misogyny"],
    "LBL:INCEL_MISOGYNY": ["incel_misogyny"],
    "LBL:INCEL_SLANG": ["incel_misogyny"],
    "LBL:MISOGYNY": ["incel_misogyny"],
    "LBL:BLACKPILL": ["incel_misogyny"],
    "LBL:FOID_SLUR": ["incel_misogyny"],
    "LBL:CHAD_COPE": ["incel_misogyny"],
    "LBL:DATING_RANT": ["incel_misogyny"],
    "LBL:LOOKISM": ["incel_misogyny"],
    "LBL:MISOGYNISTIC_LECTURE": ["alpha"],
    "LBL:OBJECTIFICATION": ["alpha"],
    "LBL:SMV_THEORY": ["alpha"],
    "LBL:REDPILL_WISDOM": ["alpha"],
    "LBL:HYPERGAMY_CLAIM": ["alpha"],
    "LBL:FRAME_CONTROL": ["alpha"],
    "LBL:PLATE_SPINNING": ["alpha"],

    # -------------------------------------------------------------------------
    # Misinfo / Conspiracy cluster
    # -------------------------------------------------------------------------
    "LBL:MISINFO": ["misinfo"],
    "LBL:MISINFO_CLAIM": ["misinfo", "conspiracy"],
    "LBL:MISINFO_SOURCE": ["misinfo"],
    "LBL:FAKE_STAT": ["misinfo"],
    "LBL:DEBUNKED_CLAIM": ["misinfo"],
    "LBL:FEAR_MONGER": ["misinfo"],
    "LBL:SUPPRESSED_TRUTH": ["misinfo"],
    "LBL:CONSPIRACY": ["conspiracy"],
    "LBL:CONSPIRACY_NARRATIVE": ["conspiracy"],
    "LBL:DEEPSTATE": ["conspiracy"],
    "LBL:ANTI_INSTITUTION": ["conspiracy", "misinfo"],
    "LBL:HIDDEN_AGENDA": ["conspiracy"],
    "LBL:FALSE_FLAG": ["conspiracy"],
    "LBL:COVER_UP": ["conspiracy"],
    "LBL:CONTROLLED_OPP": ["conspiracy"],
    "LBL:WAKE_UP": ["conspiracy"],

    # -------------------------------------------------------------------------
    # Culture war cluster
    # -------------------------------------------------------------------------
    "LBL:DOGWHISTLE": ["trad"],
    "LBL:GENDER_ESSENTIALISM": ["trad"],
    "LBL:TRAD_AESTHETIC": ["trad"],
    "LBL:MODERNITY_CRITIQUE": ["trad"],
    "LBL:FAMILY_VALUES": ["trad"],
    "LBL:NATURAL_ORDER": ["trad"],
    "LBL:DECLINE_NARRATIVE": ["trad"],
    "LBL:CULTURE_WAR": ["gamergate"],
    "LBL:GATEKEEPING": ["gamergate"],
    "LBL:WOKE_AGENDA": ["gamergate"],
    "LBL:FORCED_DIVERSITY": ["gamergate"],
    "LBL:SJW_ATTACK": ["gamergate"],
    "LBL:BOYCOTT_CALL": ["gamergate"],
    "LBL:GAMER_DEFENSE": ["gamergate"],

    # -------------------------------------------------------------------------
    # Extreme harm cluster
    # -------------------------------------------------------------------------
    "LBL:VIOLENT_THREAT": ["extremist"],
    "LBL:ACCELERATIONISM": ["extremist"],
    "LBL:RACE_WAR": ["extremist"],
    "LBL:BOOGALOO": ["extremist"],
    "LBL:COLLAPSE_WISH": ["extremist"],
    "LBL:ENEMY_DEHUMANIZE": ["extremist"],
    "LBL:MARTYR_GLORIFY": ["extremist"],
    "LBL:HATE_SLUR": ["extremist", "hate_speech"],
    "LBL:DEHUMANIZATION": ["hate_speech"],
    "LBL:REPLACEMENT_THEORY": ["hate_speech"],
    "LBL:RACIAL_SLUR": ["hate_speech"],
    "LBL:RELIGIOUS_HATE": ["hate_speech"],
    "LBL:ETHNIC_ATTACK": ["hate_speech"],
    "LBL:SUPREMACIST": ["hate_speech"],
    "LBL:VERMIN_RHETORIC": ["hate_speech"],
    "LBL:PERSONAL_ATTACK": ["bullying"],
    "LBL:DOXXING_THREAT": ["bullying"],
    "LBL:SUICIDE_BAIT": ["bullying"],
    "LBL:APPEARANCE_MOCK": ["bullying"],
    "LBL:PILE_ON": ["bullying"],
    "LBL:SCREENSHOT_SHAME": ["bullying"],
    "LBL:GASLIGHT": ["bullying"],
    "LBL:SOCIAL_EXCLUSION": ["bullying"],

    # -------------------------------------------------------------------------
    # Cross-category tokens
    # -------------------------------------------------------------------------
    "LBL:HARASSMENT": ["incel_misogyny", "bullying"],
}

_TOKEN_TO_CATEGORIES: Dict[str, List[str]] = {
    token: categories[:] for token, categories in DEFAULT_TOKEN_TO_CATEGORIES.items()
}


def set_token_category_map(mapping: Mapping[str, Iterable[str]]) -> None:
    """Override the global token→category map used by assign_labels."""

    global _TOKEN_TO_CATEGORIES  # noqa: PLW0603
    new_map: Dict[str, List[str]] = {}
    for token, categories in mapping.items():
        new_map[str(token)] = sorted({str(cat) for cat in categories})
    _TOKEN_TO_CATEGORIES = new_map or {
        token: cats[:] for token, cats in DEFAULT_TOKEN_TO_CATEGORIES.items()
}


def assign_labels(
    emitted_tokens: List[str],
    allowed_labels: Optional[Iterable[str]] = None,
) -> List[str]:
    r"""Map emitted label tokens to final category labels (no confidence).

    Args:
        emitted_tokens: tokens detected in text (e.g., ["LBL:MISINFO_CLAIM"])
        allowed_labels: optional allowed categories for this persona

    Returns:
        sorted unique category_labels
    """
    categories: List[str] = []
    for tok in emitted_tokens:
        categories.extend(_TOKEN_TO_CATEGORIES.get(tok, []))
    if allowed_labels is not None:
        allowed_set = set(allowed_labels)
        categories = [c for c in categories if c in allowed_set]
        if not categories and allowed_set:
            # Fallback to a single allowed category if mapping produced none.
            categories = [sorted(allowed_set)[0]]
    categories = sorted(set(categories))
    return categories


