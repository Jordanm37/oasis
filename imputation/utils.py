"""Shared helpers for label-token replacement and deterministic fallbacks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

LABEL_TOKEN_PATTERN = re.compile(r"<LBL:([A-Z0-9_]+)>")


def extract_label_tokens(text: str | None) -> List[str]:
    """Return the ordered list of label tokens embedded in the text."""
    if not text:
        return []
    return [f"LBL:{match.group(1)}" for match in LABEL_TOKEN_PATTERN.finditer(text)]


@dataclass
class StaticBank:
    """YAML-backed deterministic phrase bank keyed by label token."""

    bank: Dict[str, List[str]]

    @staticmethod
    def load_simple_yaml(path: Path) -> "StaticBank":
        """Load the legacy YAML bank used by build_dataset.py."""
        current_token: str | None = None
        parsed: Dict[str, List[str]] = {}
        if not path.exists():
            return StaticBank(parsed)
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    current_token = line[:-1]
                    parsed[current_token] = []
                    continue
                if current_token and line.lstrip().startswith("- "):
                    item = line.strip()[2:]
                    if (item.startswith('"') and item.endswith('"')) or (
                        item.startswith("'") and item.endswith("'")
                    ):
                        item = item[1:-1]
                    parsed.setdefault(current_token, []).append(item)
        return StaticBank(parsed)

    def deterministic_choice(
        self, token: str, seed: int, record_id: int, occurrence_index: int
    ) -> str:
        """Return a deterministic fallback phrase for a token occurrence."""
        choices = self.bank.get(token, [])
        if not choices:
            return token
        key = f"{seed}:{record_id}:{token}:{occurrence_index}".encode("utf-8")
        idx = int(hashlib.sha256(key).hexdigest(), 16) % max(1, len(choices))
        return choices[idx]


def strip_label_wrappers(token_with_brackets: str) -> str:
    """Convert '<LBL:INCEL>' to 'LBL:INCEL' for convenience."""
    return token_with_brackets.strip("<>").upper()

