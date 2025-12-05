"""Label Mapper for Multi-Dataset RAG Imputation System.

This module provides the LabelMapper class that handles mapping between
external dataset labels and OASIS archetype labels and tokens.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import random

logger = logging.getLogger(__name__)


@dataclass
class LabelMapping:
    """Represents a mapping from an external label to OASIS archetypes."""

    source_label: str
    archetypes: List[Dict[str, Any]]  # List of {archetype, weight, tokens}
    primary_archetype: Optional[str] = None
    primary_tokens: Optional[List[str]] = None

    def get_weighted_archetype(self, random_seed: Optional[int] = None) -> str:
        """Get an archetype based on weights.

        Args:
            random_seed: Optional seed for reproducible selection

        Returns:
            Selected archetype name
        """
        if not self.archetypes:
            return "benign"

        if self.primary_archetype:
            return self.primary_archetype

        # If only one archetype, return it
        if len(self.archetypes) == 1:
            return self.archetypes[0]["archetype"]

        # Weighted random selection
        if random_seed is not None:
            random.seed(random_seed)

        weights = [a.get("weight", 1.0) for a in self.archetypes]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        selected = random.choices(self.archetypes, weights=weights, k=1)[0]
        return selected["archetype"]

    def get_tokens_for_archetype(self, archetype: str) -> List[str]:
        """Get tokens for a specific archetype.

        Args:
            archetype: Archetype name

        Returns:
            List of label tokens
        """
        if self.primary_tokens and archetype == self.primary_archetype:
            return self.primary_tokens

        for arch_data in self.archetypes:
            if arch_data["archetype"] == archetype:
                return arch_data.get("tokens", [])

        return []


@dataclass
class CombinationRule:
    """Rule for handling combinations of multiple labels."""

    condition_type: str  # "all_of", "any_of", "not"
    labels: List[str]
    override_archetype: str
    priority_tokens: List[str]

    def matches(self, active_labels: List[str]) -> bool:
        """Check if this rule matches the active labels.

        Args:
            active_labels: List of currently active labels

        Returns:
            True if rule matches
        """
        active_set = set(active_labels)

        if self.condition_type == "all_of":
            return all(label in active_set for label in self.labels)
        elif self.condition_type == "any_of":
            return any(label in active_set for label in self.labels)
        elif self.condition_type == "not":
            return not any(label in active_set for label in self.labels)

        return False


class LabelMapper:
    """Maps external dataset labels to OASIS ontology."""

    def __init__(self, emission_policy_path: Optional[str] = None):
        """Initialize the LabelMapper.

        Args:
            emission_policy_path: Optional path to emission_policy.py for loading token mappings
        """
        self.label_mappings: Dict[str, Dict[str, LabelMapping]] = {}
        self.combination_rules: Dict[str, List[CombinationRule]] = {}
        self.fallback_mappings: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Load token mappings from emission_policy if available
        self.token_mappings: Dict[str, List[str]] = {}
        if emission_policy_path:
            self._load_emission_policy(emission_policy_path)
        else:
            # Use default mappings from emission_policy.py
            self._load_default_token_mappings()

    def _load_default_token_mappings(self):
        """Load default token mappings from emission_policy.py."""
        # Copied from emission_policy.py DEFAULT_LABEL_TO_TOKENS
        self.token_mappings = {
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
        }

    def _load_emission_policy(self, path: str):
        """Load token mappings from emission_policy.py file.

        Args:
            path: Path to emission_policy.py
        """
        # This would parse the Python file to extract DEFAULT_LABEL_TO_TOKENS
        # For now, we use the default mappings
        self._load_default_token_mappings()

    def load_mapping_file(self, dataset_id: str, mapping_file: str) -> None:
        """Load label mapping from YAML file.

        Args:
            dataset_id: Dataset identifier
            mapping_file: Path to mapping YAML file
        """
        if not os.path.exists(mapping_file):
            logger.warning(f"Mapping file not found: {mapping_file}")
            return

        with open(mapping_file, 'r') as f:
            data = yaml.safe_load(f)

        # Parse label mappings
        mappings = {}
        for source_label, mapping_data in data.get("label_mappings", {}).items():
            archetypes = mapping_data.get("archetypes", [])
            mapping = LabelMapping(
                source_label=source_label,
                archetypes=archetypes
            )
            mappings[source_label] = mapping

        self.label_mappings[dataset_id] = mappings

        # Parse combination rules
        rules = []
        for rule_data in data.get("combination_rules", []):
            condition = rule_data.get("condition", {})
            condition_type = "all_of" if "all_of" in condition else \
                            "any_of" if "any_of" in condition else "not"
            labels = condition.get(condition_type, [])

            rule = CombinationRule(
                condition_type=condition_type,
                labels=labels,
                override_archetype=rule_data.get("override_archetype", "benign"),
                priority_tokens=rule_data.get("priority_tokens", [])
            )
            rules.append(rule)

        self.combination_rules[dataset_id] = rules

        # Parse fallback mappings
        self.fallback_mappings[dataset_id] = data.get("fallback_mappings", {})

        # Store metadata
        self.metadata[dataset_id] = data.get("metadata", {})

        logger.info(f"Loaded mapping for {dataset_id} with {len(mappings)} label mappings")

    def map_label(self, dataset_id: str, source_label: str,
                  confidence: float = 1.0) -> Tuple[str, List[str]]:
        """Map a source label to OASIS archetype and tokens.

        Args:
            dataset_id: Dataset identifier
            source_label: Source dataset label
            confidence: Confidence score for the label

        Returns:
            Tuple of (archetype, list of tokens)
        """
        if dataset_id not in self.label_mappings:
            logger.warning(f"No mappings loaded for dataset {dataset_id}")
            return self._get_fallback(dataset_id, confidence)

        mappings = self.label_mappings[dataset_id]

        if source_label not in mappings:
            logger.debug(f"Label {source_label} not found in mappings")
            return self._get_fallback(dataset_id, confidence)

        mapping = mappings[source_label]
        archetype = mapping.get_weighted_archetype()
        tokens = mapping.get_tokens_for_archetype(archetype)

        return archetype, tokens

    def map_multiple_labels(self, dataset_id: str, source_labels: List[str],
                           confidences: Optional[List[float]] = None) -> Tuple[str, List[str]]:
        """Map multiple source labels to OASIS archetype and tokens.

        Args:
            dataset_id: Dataset identifier
            source_labels: List of source dataset labels
            confidences: Optional confidence scores for each label

        Returns:
            Tuple of (archetype, list of tokens)
        """
        if not source_labels:
            return self._get_fallback(dataset_id, 0.5)

        # Check combination rules first
        if dataset_id in self.combination_rules:
            for rule in self.combination_rules[dataset_id]:
                if rule.matches(source_labels):
                    logger.debug(f"Combination rule matched for labels {source_labels}")
                    return rule.override_archetype, rule.priority_tokens

        # If no combination rule matches, use the highest confidence label
        if confidences:
            max_idx = confidences.index(max(confidences))
            primary_label = source_labels[max_idx]
            confidence = confidences[max_idx]
        else:
            primary_label = source_labels[0]
            confidence = 1.0

        return self.map_label(dataset_id, primary_label, confidence)

    def _get_fallback(self, dataset_id: str, confidence: float) -> Tuple[str, List[str]]:
        """Get fallback archetype and tokens.

        Args:
            dataset_id: Dataset identifier
            confidence: Confidence score

        Returns:
            Tuple of (archetype, list of tokens)
        """
        fallback = self.fallback_mappings.get(dataset_id, {})

        # Use low confidence fallback if confidence is low
        if confidence < 0.5:
            archetype = fallback.get("low_confidence_archetype", "benign")
            tokens = fallback.get("low_confidence_tokens", ["LBL:FRIENDLY"])
        else:
            archetype = fallback.get("default_archetype", "benign")
            tokens = fallback.get("default_tokens", ["LBL:FRIENDLY"])

        return archetype, tokens

    def get_all_tokens_for_archetype(self, archetype: str) -> List[str]:
        """Get all possible tokens for an archetype from emission policy.

        Args:
            archetype: Archetype name

        Returns:
            List of all possible tokens
        """
        return self.token_mappings.get(archetype, [])

    def validate_tokens(self, tokens: List[str]) -> List[str]:
        """Validate and clean token list.

        Args:
            tokens: List of tokens to validate

        Returns:
            List of valid tokens
        """
        valid_tokens = []
        all_valid_tokens = set()

        # Collect all valid tokens
        for token_list in self.token_mappings.values():
            all_valid_tokens.update(token_list)

        for token in tokens:
            # Ensure proper format
            if not token.startswith("LBL:"):
                token = f"LBL:{token}"

            if token in all_valid_tokens:
                valid_tokens.append(token)
            else:
                logger.warning(f"Invalid token: {token}")

        return valid_tokens

    def get_mapping_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get statistics about mappings for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'dataset_id': dataset_id,
            'num_label_mappings': 0,
            'num_combination_rules': 0,
            'archetypes_covered': set(),
            'tokens_used': set()
        }

        if dataset_id in self.label_mappings:
            mappings = self.label_mappings[dataset_id]
            stats['num_label_mappings'] = len(mappings)

            for mapping in mappings.values():
                for arch_data in mapping.archetypes:
                    stats['archetypes_covered'].add(arch_data['archetype'])
                    stats['tokens_used'].update(arch_data.get('tokens', []))

        if dataset_id in self.combination_rules:
            stats['num_combination_rules'] = len(self.combination_rules[dataset_id])

        # Convert sets to lists for JSON serialization
        stats['archetypes_covered'] = list(stats['archetypes_covered'])
        stats['tokens_used'] = list(stats['tokens_used'])

        return stats