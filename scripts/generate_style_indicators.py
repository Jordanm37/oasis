#!/usr/bin/env python3
r"""Generate new style indicators (emoji, tone tags, etc.) for each archetype."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from dotenv import load_dotenv

from oasis.generation.xai_client import XAIConfig
from oasis.generation.xai_client import generate_text as xai_generate
from oasis.persona import PersonaOntology, PersonaVariantSpec, load_ontology

load_dotenv()
LOGGER = logging.getLogger(__name__)

DEFAULT_STYLE_FILE = Path("configs/style_indicators.yaml")
DEFAULT_ONTOLOGY = Path("configs/personas/ontology_unified.yaml")

SYSTEM_PROMPT = (
    "You are a persona style writer for a social-simulation platform.\n"
    "For the requested archetype, you will propose NEW entries for the listed style categories.\n"
    "Each entry must be a concise bullet (~3-10 words), follow the category theme, and feel authentic.\n"
    "DO NOT repeat existing bullet points provided in the context. Keep everything ASCII. Avoid quotes.\n"
    "Return STRICT JSON with the format: {\"emoji\": [\"...\"], \"tone_tag\": [\"...\"], ...}.\n"
)


@dataclass
class StyleContext:
    archetype_id: str
    categories: Dict[str, List[str]]
    variants: List[str] = field(default_factory=list)
    worldviews: List[str] = field(default_factory=list)

    def add_variant(self, variant: PersonaVariantSpec) -> None:
        summary = variant.summary or variant.description
        worldview = (
            str(variant.metadata.get("worldview", "")).strip() or summary or variant.display_name
        )
        self.variants.append(f"{variant.id}: {summary}")
        self.worldviews.append(worldview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand style indicators via Grok.")
    parser.add_argument(
        "--style-file",
        type=str,
        default=str(DEFAULT_STYLE_FILE),
        help="Path to style indicators YAML file.",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=str(DEFAULT_ONTOLOGY),
        help="Path to persona ontology (for context).",
    )
    parser.add_argument(
        "--archetypes",
        nargs="*",
        default=[],
        help="Specific archetype IDs to update (default: all).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Only these categories (e.g., emoji tone_tag). Default: all categories.",
    )
    parser.add_argument(
        "--items-per-category",
        type=int,
        default=5,
        help="Number of new bullets per category to request.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=5,
        help="Max persona variants/worldviews to mention in prompt.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process archetypes concurrently.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum concurrent archetype generations.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.4,
        help="Delay (seconds) between parallel API calls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum tokens Grok may emit per request.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Optional override for XAI_API_KEY.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview changes without writing to disk.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail-fast if any archetype update fails.",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=("standard", "expressive"),
        default="standard",
        help="Switch between default prompt and expressive/creative variant.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", force=True)


def build_xai_config(args: argparse.Namespace) -> XAIConfig:
    import os

    api_key = args.api_key or os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set. Provide via env or --api-key.")
    return XAIConfig(
        api_key=api_key,
        model="grok-4-fast-non-reasoning",
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )


def load_style_file(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Style indicators file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    data: Dict[str, Dict[str, List[str]]] = {}
    for archetype, sections in raw.items():
        cat_map: Dict[str, List[str]] = {}
        for category, entries in (sections or {}).items():
            cat_map[str(category)] = [str(entry) for entry in (entries or [])]
        data[str(archetype)] = cat_map
    return data


def save_style_file(path: Path, data: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
    serializable: Dict[str, Dict[str, List[str]]] = {}
    for archetype, sections in data.items():
        serializable[archetype] = {
            category: list(entries) for category, entries in sections.items()
        }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(serializable, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)


def build_style_contexts(
    style_data: Mapping[str, Mapping[str, List[str]]],
    ontology: PersonaOntology,
    ontology_path: Path,
) -> Dict[str, StyleContext]:
    contexts: Dict[str, StyleContext] = {}
    for archetype, categories in style_data.items():
        contexts[archetype] = StyleContext(archetype_id=archetype, categories=dict(categories))

    for variant in ontology.iter_variants():
        archetype = variant.archetype
        context = contexts.get(archetype)
        if context:
            context.add_variant(variant)

    return contexts


def build_category_prompt(
    context: StyleContext,
    category: str,
    items_per_category: int,
    context_limit: int,
    prompt_variant: str,
) -> str:
    variant_rows = context.variants[:context_limit] or ["(No variant summaries available)"]
    worldview_rows = context.worldviews[:context_limit] or ["(No worldview notes provided)"]
    existing = context.categories.get(category, [])
    preview = existing[:15]
    extra = f" (+{len(existing) - len(preview)} more)" if len(existing) > len(preview) else ""
    formatted = ", ".join(preview) if preview else "(none)"

    lines = [
        f"Archetype: {context.archetype_id}",
        f"Target category: {category}",
        "Persona summaries:",
        "- " + "\n- ".join(variant_rows),
        "\nWorldviews:",
        "- " + "\n- ".join(worldview_rows),
        f"\nExisting entries for {category}: {formatted}{extra}",
        "\nInstruction:",
    ]
    instruction_lines = [
        f"- Propose {items_per_category} NEW entries for this category.",
        "- Keep bullets concise (<=10 words) and category-appropriate.",
        "- Each entry MUST be a JSON string enclosed in double quotes, with NO internal double quotes.",
        '- Example entry: "gentle sunrise emoji with warmth".',
        "- Use ASCII characters only; replace internal quotes with apostrophes.",
        "- Ensure every entry is unique and not already listed above.",
        '- Return JSON EXACTLY as {"items": ["entry1", "entry2", ...]} with no extra keys or text.',
    ]
    if prompt_variant == "expressive":
        instruction_lines.insert(
            1,
            "- Lean into bold, unexpected imagery or metaphors authentic to this archetype.",
        )
        instruction_lines.insert(
            2,
            "- Surface niche slang, regional flavor, or sensory details that feel fresh.",
        )
    lines.extend(instruction_lines)
    return "\n".join(lines)


def parse_style_response(raw: str, categories: Sequence[str]) -> Dict[str, List[str]]:
    import re

    def normalize_json(text: str) -> str:
        # Insert missing commas between adjacent quoted strings
        text = re.sub(r'"\s+"', '", "', text)
        text = re.sub(r'"\s+\[', '", [', text)
        text = re.sub(r'\]\s+"', '], "', text)
        return text

    raw = normalize_json(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Initial JSON parse failed: %s", exc)
        payload = None
        if "```" in raw:
            lines = raw.split("\n")
            in_block = False
            block_lines: List[str] = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    block_lines.append(line)
            if block_lines:
                try:
                    payload = json.loads("\n".join(block_lines))
                    LOGGER.info("Recovered JSON from markdown code block")
                except json.JSONDecodeError:
                    pass
        if payload is None:
            last_brace = raw.rfind("}")
            if last_brace > 0:
                try:
                    payload = json.loads(raw[: last_brace + 1])
                    LOGGER.info("Recovered JSON by truncation")
                except json.JSONDecodeError:
                    pass
        if payload is None:
            fragments = [frag.strip() for frag in raw.splitlines() if frag.strip()]
            combined: Dict[str, List[str]] = {}
            success = True
            for frag in fragments:
                try:
                    data = json.loads(frag)
                except json.JSONDecodeError:
                    success = False
                    break
                for key, value in data.items():
                    combined[key] = value
            if success and combined:
                payload = combined
                LOGGER.info("Recovered JSON by merging per-line fragments")
        if payload is None:
            candidate = raw.strip()
            candidate = re.sub(r'}\s*[\r\n]+\s*{', '},{', candidate)
            candidate = f"[{candidate}]"
            try:
                array_payload = json.loads(candidate)
                merged: Dict[str, List[str]] = {}
                for obj in array_payload:
                    merged.update(obj)
                payload = merged
                LOGGER.info("Recovered JSON by wrapping as array")
            except json.JSONDecodeError:
                pass
        if payload is None:
            pattern = re.compile(r'\{"([^"]+)":\s*\[(.*?)\]\}', re.DOTALL)
            matches = list(pattern.finditer(raw))
            if matches:
                merged: Dict[str, List[str]] = {}
                for match in matches:
                    category, items_blob = match.groups()
                    parts = re.split(r',(?![^\\[]*\\])', items_blob)
                    cleaned_items: List[str] = []
                    for part in parts:
                        text = part.strip().strip('[]"')
                        text = text.replace('"', "").replace("'", "")
                        text = " ".join(text.split())
                        if text:
                            cleaned_items.append(text)
                    merged[category] = cleaned_items
                if merged:
                    payload = merged
                    LOGGER.info("Recovered JSON via regex extraction")
        if payload is None:
            LOGGER.error("Failed to parse JSON response: %s", exc)
            LOGGER.debug("Raw response: %s", raw)
            raise

    results: Dict[str, List[str]] = {}
    for category in categories:
        values = payload.get(category, [])
        if not isinstance(values, list):
            raise ValueError(f"Category '{category}' missing or not a list in response.")
        cleaned = [" ".join(str(item).strip().split()) for item in values if str(item).strip()]
        results[category] = cleaned
    return results


def append_unique(existing: List[str], candidates: Sequence[str]) -> List[str]:
    existing_lower = {item.lower() for item in existing}
    appended = 0
    for item in candidates:
        key = item.lower()
        if key not in existing_lower:
            existing.append(item)
            existing_lower.add(key)
            appended += 1
    return existing


async def process_archetype(
    archetype_id: str,
    context: StyleContext,
    target_categories: Sequence[str],
    args: argparse.Namespace,
    xai_config: XAIConfig,
    style_data: Dict[str, Dict[str, List[str]]],
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[Dict[str, List[str]]], Optional[Exception]]:
    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            aggregated: Dict[str, List[str]] = {}
            for category in target_categories:
                prompt = build_category_prompt(
                    context=context,
                    category=category,
                    items_per_category=args.items_per_category,
                    context_limit=args.context_limit,
                    prompt_variant=args.prompt_variant,
                )
                await asyncio.sleep(args.rate_limit_delay)
                response = await loop.run_in_executor(
                    None,
                    xai_generate,
                    SYSTEM_PROMPT,
                    prompt,
                    xai_config,
                )
                parsed = parse_style_response(response, [category])
                items = parsed.get(category, [])
                aggregated[category] = items
                if not args.dry_run:
                    existing = style_data[archetype_id].setdefault(category, [])
                    append_unique(existing, items)

            return (archetype_id, aggregated, None)
        except Exception as exc:
            return (archetype_id, None, exc)


async def run_async(
    args: argparse.Namespace,
    contexts: Dict[str, StyleContext],
    target_archetypes: Sequence[str],
    target_categories: Sequence[str],
    style_data: Dict[str, Dict[str, List[str]]],
) -> List[Tuple[str, Dict[str, List[str]]]]:
    xai_config = build_xai_config(args)
    semaphore = asyncio.Semaphore(args.max_parallel)
    tasks = []

    for archetype_id in target_archetypes:
        context = contexts.get(archetype_id)
        if not context:
            LOGGER.warning("Archetype '%s' not found in style file.", archetype_id)
            continue
        tasks.append(
            process_archetype(
                archetype_id=archetype_id,
                context=context,
                target_categories=target_categories,
                args=args,
                xai_config=xai_config,
                style_data=style_data,
                semaphore=semaphore,
            )
        )

    results: List[Tuple[str, Dict[str, List[str]]]] = []
    for archetype_id, new_entries, exc in await asyncio.gather(*tasks):
        if exc:
            if args.strict:
                raise exc
            LOGGER.exception("Archetype '%s' failed: %s", archetype_id, exc)
            continue
        if new_entries:
            results.append((archetype_id, new_entries))
            LOGGER.info(
                "Archetype '%s': added %s",
                archetype_id,
                ", ".join(f"{len(v)} {k}" for k, v in new_entries.items()),
            )
    return results


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    style_path = Path(args.style_file)
    ontology_path = Path(args.ontology)

    style_data = load_style_file(style_path)
    ontology = load_ontology(ontology_path)
    contexts = build_style_contexts(style_data, ontology, ontology_path)

    target_archetypes = args.archetypes or sorted(style_data.keys())
    categories = args.categories
    if not categories:
        # Use union of categories for first archetype (assuming consistent structure).
        categories = sorted({cat for sections in style_data.values() for cat in sections.keys()})

    if args.parallel:
        LOGGER.info(
            "Running style generation in parallel (max %d concurrent archetypes).",
            args.max_parallel,
        )
        results = asyncio.run(
            run_async(args, contexts, target_archetypes, categories, style_data)
        )
    else:
        LOGGER.info("Running style generation sequentially.")
        xai_config = build_xai_config(args)
        results = []
        for archetype_id in target_archetypes:
            context = contexts.get(archetype_id)
            if not context:
                LOGGER.warning("Archetype '%s' not found in style file.", archetype_id)
                continue
            aggregated: Dict[str, List[str]] = {}
            for category in categories:
                prompt = build_category_prompt(
                    context=context,
                    category=category,
                    items_per_category=args.items_per_category,
                    context_limit=args.context_limit,
                    prompt_variant=args.prompt_variant,
                )
                time.sleep(args.rate_limit_delay)
                response = xai_generate(
                    SYSTEM_PROMPT,
                    prompt,
                    xai_config,
                )
                parsed = parse_style_response(response, [category])
                items = parsed.get(category, [])
                aggregated[category] = items
                if not args.dry_run:
                    existing = style_data[archetype_id].setdefault(category, [])
                    append_unique(existing, items)
            LOGGER.info(
                "Archetype '%s': added %s",
                archetype_id,
                ", ".join(f"{len(v)} {k}" for k, v in aggregated.items()),
            )
            results.append((archetype_id, aggregated))

    if not args.dry_run:
        save_style_file(style_path, style_data)
        LOGGER.info("Updated style indicators saved to %s", style_path)
    else:
        LOGGER.info("Dry-run complete; no files were modified.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Aborted by user.")
        sys.exit(130)

