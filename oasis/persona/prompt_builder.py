from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .seed_utils import PersonaSeed


@dataclass(frozen=True)
class PromptSynthesisResult:
    system_prompt: str
    persona_goal: str
    style_quirks: str
    lexical_required: str
    lexical_optional: str


def build_llm_prompt_instruction(
    seed: PersonaSeed,
    variant_meta: Dict[str, Any],
) -> str:
    """Render the instruction sent to the LLM."""

    base_lines = [
        "You are designing a unique social-media persona for a simulation.",
        "",
        "Base traits (from a real user seed):",
    ]
    base_lines.append(f"- Persona profile: {seed.persona or 'N/A'}")
    if seed.chat_lines:
        for line in seed.chat_lines[:3]:
            base_lines.append(f"- Chat excerpt: {line}")
    base_lines.append("")

    archetype = variant_meta.get("display_name", "persona")
    worldview = variant_meta.get("worldview", "")
    topics = ", ".join(variant_meta.get("topics", []))
    tone = variant_meta.get("style", {}).get("tone", "neutral")
    instructions = variant_meta.get("instructional_hints", [])

    base_lines.extend(
        [
            f"Transformation target: {archetype}",
            f"- Tone: {tone}",
            f"- Topics: {topics or 'mixed'}",
            f"- Worldview: {worldview}",
        ]
    )
    if instructions:
        base_lines.append("- Additional hints:")
        for hint in instructions:
            base_lines.append(f"  * {hint}")
    indicators = variant_meta.get("style_indicators") or {}
    if isinstance(indicators, dict) and indicators:
        base_lines.append("- Fold this style fingerprint into the [Style] block:")
        for key in sorted(indicators.keys()):
            base_lines.append(f"  * {key}: {indicators[key]}")

    lexicon_refs = variant_meta.get("lexicon_refs", [])
    base_lines.append("")
    base_lines.append(
        f"Lexicon collections to draw from: {', '.join(lexicon_refs) if lexicon_refs else 'none specified'}"
    )

    payload = "\n".join(base_lines)
    payload += textwrap.dedent(
        """

        Task:
          - Write a system prompt describing this specific persona (voice, backstory, motivations).
          - Include a short persona_goal sentence.
          - List 3-5 stylistic quirks, plus required/optional vocabulary terms.
          - Respond in JSON with keys: system_prompt, persona_goal, style_quirks, lexical_required, lexical_optional.
          - Do NOT include label tokens (<LBL:...>). Those are injected elsewhere.
        """
    ).strip()
    return payload


def build_llm_prompt(
    seed: PersonaSeed,
    variant_meta: Dict[str, Any],
    *,
    llm_generate,
    rng_seed: Optional[int] = None,
) -> PromptSynthesisResult:
    """Call the LLM (or stub) to synthesize a persona prompt."""

    instruction = build_llm_prompt_instruction(seed, variant_meta)
    response_text = llm_generate(
        system_instruction="You are an expert persona designer.",
        user_text=instruction,
        config={"seed": rng_seed} if rng_seed is not None else None,
    )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        payload = {}

    def _get(key: str, default: str = "") -> str:
        value = payload.get(key, default)
        if isinstance(value, list):
            return "; ".join(str(v) for v in value)
        return str(value).strip()

    return PromptSynthesisResult(
        system_prompt=_get("system_prompt", instruction),
        persona_goal=_get("persona_goal", "Advance my worldview online."),
        style_quirks=_get("style_quirks", ""),
        lexical_required=_get("lexical_required", ""),
        lexical_optional=_get("lexical_optional", ""),
    )

