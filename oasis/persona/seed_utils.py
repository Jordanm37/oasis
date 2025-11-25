from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import csv
import random
import re


_TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True)
class PersonaSeed:
    """Lightweight representation of a PersonaChat-style seed profile."""

    seed_id: int
    persona: str
    chat_lines: List[str]
    keywords: List[str]
    inferred_tone: str = "neutral"

    @property
    def text_blob(self) -> str:
        return " ".join([self.persona, *self.chat_lines]).strip()


def _extract_keywords(text: str, top_n: int = 6) -> List[str]:
    tokens = [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]
    filtered = [tok for tok in tokens if len(tok) > 3]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def load_persona_seeds(
    csv_path: Path,
    *,
    max_rows: Optional[int] = None,
) -> List[PersonaSeed]:
    """Load PersonaChat-style seeds from CSV."""

    seeds: List[PersonaSeed] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            persona = (row.get("Persona") or row.get("persona") or "").strip()
            chat_blob = (row.get("chat") or row.get("Chat") or "").strip()
            if not persona and not chat_blob:
                continue
            chat_lines = [
                line.strip()
                for line in chat_blob.splitlines()
                if line.strip()
            ][:5]
            keywords = _extract_keywords(f"{persona} {' '.join(chat_lines)}")
            seeds.append(
                PersonaSeed(
                    seed_id=idx,
                    persona=persona,
                    chat_lines=chat_lines,
                    keywords=keywords,
                )
            )
            if max_rows and len(seeds) >= max_rows:
                break
    if not seeds:
        raise ValueError(f"No persona seeds loaded from {csv_path}")
    return seeds


def sample_seed(
    seeds: Iterable[PersonaSeed],
    *,
    keyword_hints: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> PersonaSeed:
    """Sample a seed, preferring ones that match the provided keyword hints."""

    rng = rng or random.Random()
    seeds_list = list(seeds)
    if not seeds_list:
        raise ValueError("Seed pool is empty.")
    if keyword_hints:
        hint_set = {hint.lower() for hint in keyword_hints}
        preferred = [
            seed
            for seed in seeds_list
            if hint_set.intersection({k.lower() for k in seed.keywords})
        ]
        if preferred:
            return rng.choice(preferred)
    return rng.choice(seeds_list)

