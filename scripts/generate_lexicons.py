#!/usr/bin/env python3
r"""Procedurally expand lexicon YAML files with Grok suggestions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableSet, Optional, Sequence, Tuple

import yaml
from dotenv import load_dotenv

from oasis.generation.xai_client import XAIConfig
from oasis.generation.xai_client import generate_text as xai_generate
from oasis.persona import PersonaOntology, PersonaVariantSpec, load_ontology

# Load environment variables early
load_dotenv()

LOGGER = logging.getLogger(__name__)


DEFAULT_ONTOLOGY = Path("configs/personas/ontology_unified.yaml")
SYSTEM_PROMPT = (
    "You are a safety-critical lexicon curator for an agent-based social media simulator. "
    "Your task is to propose new vocabulary items for the supplied persona class so that "
    "LLM agents will sound authentic without repeating prior entries. "
    "ALWAYS reply with strict JSON using this schema: "
    '{"required": ["token1", "..."], "optional": ["tokenA", "..."]}. '
    "Tokens must be short ASCII phrases (1-3 words), no punctuation except apostrophes or hyphens, "
    "and must align with the described worldview. Do not repeat supplied tokens."
)


@dataclass
class LexiconContext:
    r"""Metadata needed to craft prompts for one lexicon collection."""

    collection_id: str
    file_path: Path
    variant_rows: List[str] = field(default_factory=list)
    worldview_rows: List[str] = field(default_factory=list)

    def add_variant(self, variant: PersonaVariantSpec) -> None:
        summary = variant.summary or variant.description
        worldview = (
            str(variant.metadata.get("worldview", "")).strip() or summary or variant.display_name
        )
        self.variant_rows.append(f"{variant.id}: {summary}")
        self.worldview_rows.append(worldview)


@dataclass
class LexiconUpdate:
    r"""Holds final update summary for a single file."""

    collection_id: str
    file_path: Path
    required_added: List[str]
    optional_added: List[str]


def parse_args() -> argparse.Namespace:
    r"""Build CLI parser for lexicon generation."""

    parser = argparse.ArgumentParser(
        description="Call Grok via xai_client to expand ontology-linked lexicons."
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=str(DEFAULT_ONTOLOGY),
        help="Path to ontology YAML with lexicon_collections entries.",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=[],
        help="Explicit collection IDs to update (default: all collections in ontology).",
    )
    parser.add_argument(
        "--required-target",
        type=int,
        default=6,
        help="Desired number of NEW required tokens per iteration.",
    )
    parser.add_argument(
        "--optional-target",
        type=int,
        default=12,
        help="Desired number of NEW optional tokens per iteration.",
    )
    parser.add_argument(
        "--required-total",
        type=int,
        default=0,
        help="Target total count for required tokens (0 = no limit, just run once).",
    )
    parser.add_argument(
        "--optional-total",
        type=int,
        default=0,
        help="Target total count for optional tokens (0 = no limit, just run once).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of generation iterations per collection.",
    )
    parser.add_argument(
        "--retry-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry failed iterations up to 3 times before giving up.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process collections in parallel (faster but may hit rate limits).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of collections to process simultaneously.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between parallel API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=5,
        help="How many persona variants to mention in each LLM prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, only print the proposed changes without writing files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grok-4-fast-non-reasoning",
        help="xAI model name to use when calling Grok.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for Grok requests.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-p sampling value for Grok requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=600,
        help="Maximum number of tokens Grok may emit per request.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Optional override for XAI_API_KEY environment variable.",
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
        help="Fail-fast if any collection update encounters an error.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    r"""Configure root logger once."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,  # Override any existing configuration
    )


def build_xai_config(args: argparse.Namespace) -> XAIConfig:
    r"""Create an XAIConfig honoring CLI overrides."""

    import os
    api_key = args.api_key or os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY not found. Set it via environment variable or use --api-key flag."
        )
    return XAIConfig(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )


def load_lexicon(file_path: Path) -> Dict[str, List[str]]:
    r"""Load a lexicon YAML (required/optional lists)."""

    if not file_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    required = [str(entry) for entry in raw.get("required", [])]
    optional = [str(entry) for entry in raw.get("optional", [])]
    return {"required": required, "optional": optional}


def save_lexicon(file_path: Path, payload: Mapping[str, Sequence[str]]) -> None:
    r"""Persist lexicon data to disk (ASCII-safe)."""

    content = {
        "required": [entry for entry in payload.get("required", [])],
        "optional": [entry for entry in payload.get("optional", [])],
    }
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            content,
            handle,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )


def resolve_collections(ontology: PersonaOntology, ontology_path: Path) -> Dict[str, LexiconContext]:
    r"""Map lexicon collection IDs to prompt contexts."""

    contexts: Dict[str, LexiconContext] = {}
    for collection_id, entry in (ontology.lexicon_collections or {}).items():
        file_entry = entry.get("file")
        if not file_entry:
            LOGGER.warning("Collection '%s' missing file entry; skipping", collection_id)
            continue
        file_path = Path(file_entry)
        if not file_path.is_absolute():
            # Resolve relative to the workspace root, not ontology file location
            file_path = Path.cwd() / file_path
        if not file_path.exists():
            # Try relative to ontology parent as fallback
            alt_path = ontology_path.parent / file_entry
            if alt_path.exists():
                file_path = alt_path
        contexts[collection_id] = LexiconContext(
            collection_id=collection_id,
            file_path=file_path,
        )

    for variant in ontology.iter_variants():
        refs = variant.metadata.get("lexicon_refs") if variant.metadata else None
        if not refs:
            continue
        for ref in refs:
            context = contexts.get(ref)
            if context:
                context.add_variant(variant)
    return contexts


def build_user_prompt(
    context: LexiconContext,
    lexicon_data: Mapping[str, Sequence[str]],
    required_target: int,
    optional_target: int,
    context_limit: int,
) -> str:
    r"""Craft the Grok user prompt for one collection."""

    variant_rows = context.variant_rows[:context_limit] or ["(No variant summaries available)"]
    worldview_rows = context.worldview_rows[:context_limit] or ["(No worldview notes provided)"]
    
    # Show full existing lists for better deduplication context
    required_existing = lexicon_data.get("required", [])
    optional_existing = lexicon_data.get("optional", [])
    
    req_count = len(required_existing)
    opt_count = len(optional_existing)
    
    # Sample a subset if lists are very long to keep prompt manageable
    if req_count > 50:
        req_sample = ", ".join(required_existing[:25]) + f" ... ({req_count - 25} more)"
    else:
        req_sample = ", ".join(required_existing) or "(none)"
    
    if opt_count > 100:
        opt_sample = ", ".join(optional_existing[:50]) + f" ... ({opt_count - 50} more)"
    else:
        opt_sample = ", ".join(optional_existing) or "(none)"
    
    return (
        f"Collection ID: {context.collection_id}\n"
        f"Referenced persona variants:\n- " + "\n- ".join(variant_rows) + "\n\n"
        f"Worldview notes:\n- " + "\n- ".join(worldview_rows) + "\n\n"
        f"Current token counts: {req_count} required, {opt_count} optional\n"
        f"Existing required tokens: {req_sample}\n"
        f"Existing optional tokens (sample): {opt_sample}\n\n"
        f"Generate {required_target} NEW required tokens and {optional_target} NEW optional tokens "
        "that reflect the worldview above. Tokens MUST be unique and NOT repeat any existing tokens. "
        "Tokens must be concrete, feel like authentic slang or shorthand used by the described community. "
        "Keep everything in lowercase except for proper nouns/acronyms that truly require capitalization. "
        "Focus on creating diverse vocabulary that captures different aspects of the community's discourse."
    )


def extract_terms(raw_response: str) -> Tuple[List[str], List[str]]:
    r"""Parse Grok JSON output with fallback for malformed responses."""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        # Try to salvage partial JSON by finding the last complete object
        LOGGER.warning("Initial JSON parse failed: %s", exc)
        # Attempt to extract JSON from markdown code blocks
        if "```json" in raw_response or "```" in raw_response:
            lines = raw_response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            if json_lines:
                try:
                    payload = json.loads("\n".join(json_lines))
                    LOGGER.info("Recovered JSON from markdown code block")
                except json.JSONDecodeError:
                    pass
        
        # If still failing, try truncating to last complete brace
        if "payload" not in locals():
            last_brace = raw_response.rfind("}")
            if last_brace > 0:
                truncated = raw_response[:last_brace + 1]
                try:
                    payload = json.loads(truncated)
                    LOGGER.info("Recovered JSON by truncating to last complete brace")
                except json.JSONDecodeError:
                    pass
        
        # Final fallback: return empty lists
        if "payload" not in locals():
            LOGGER.error("Could not recover valid JSON from response (length=%d)", len(raw_response))
            LOGGER.debug("Raw response snippet: %s...", raw_response[:500])
            raise ValueError(f"LLM response was not valid JSON and could not be recovered: {exc}") from exc
    
    required_raw = payload.get("required") or []
    optional_raw = payload.get("optional") or []
    if not isinstance(required_raw, list) or not isinstance(optional_raw, list):
        raise ValueError("LLM response missing 'required'/'optional' lists")
    required = [sanitize_token(entry) for entry in required_raw]
    optional = [sanitize_token(entry) for entry in optional_raw]
    return required, optional


def sanitize_token(entry: str) -> str:
    r"""Normalize whitespace and ensure ASCII output for tokens."""

    token = " ".join(str(entry).strip().split())
    token = token.encode("ascii", "ignore").decode("ascii")
    return token


def filter_new_terms(
    candidates: Sequence[str],
    existing: Sequence[str],
    already_added: MutableSet[str],
    limit: int,
) -> List[str]:
    r"""Keep up to `limit` new tokens not present in existing/added sets (case-insensitive)."""

    existing_lower = {token.lower() for token in existing}
    results: List[str] = []
    for token in candidates:
        clean = token.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in existing_lower or key in already_added:
            continue
        results.append(clean)
        already_added.add(key)
        if len(results) >= limit:
            break
    return results


def update_collection(
    context: LexiconContext,
    lexicon_data: Dict[str, List[str]],
    args: argparse.Namespace,
    xai_config: XAIConfig,
) -> LexiconUpdate | None:
    r"""Call Grok iteratively until reaching target totals for a collection."""

    required_total_target = args.required_total
    optional_total_target = args.optional_total
    
    # If no totals specified, run once
    if required_total_target <= 0 and optional_total_target <= 0:
        return _single_update_iteration(context, lexicon_data, args, xai_config)
    
    # Iterative mode: keep generating until we hit targets
    iteration = 0
    total_req_added = 0
    total_opt_added = 0
    
    while iteration < args.max_iterations:
        current_req_count = len(lexicon_data["required"])
        current_opt_count = len(lexicon_data["optional"])
        
        # Check if we've reached targets
        req_remaining = max(0, required_total_target - current_req_count)
        opt_remaining = max(0, optional_total_target - current_opt_count)
        
        if req_remaining <= 0 and opt_remaining <= 0:
            LOGGER.info(
                "Collection '%s' reached targets: %d required, %d optional",
                context.collection_id,
                current_req_count,
                current_opt_count,
            )
            break
        
        # Adjust per-iteration targets based on remaining
        iter_req_target = min(args.required_target, req_remaining) if req_remaining > 0 else 0
        iter_opt_target = min(args.optional_target, opt_remaining) if opt_remaining > 0 else 0
        
        if iter_req_target <= 0 and iter_opt_target <= 0:
            break
        
        LOGGER.info(
            "Collection '%s' iteration %d/%d: current=%d req + %d opt, requesting +%d req + %d opt",
            context.collection_id,
            iteration + 1,
            args.max_iterations,
            current_req_count,
            current_opt_count,
            iter_req_target,
            iter_opt_target,
        )
        
        # Temporarily adjust args for this iteration
        saved_req = args.required_target
        saved_opt = args.optional_target
        args.required_target = iter_req_target
        args.optional_target = iter_opt_target
        
        try:
            update = _single_update_iteration(context, lexicon_data, args, xai_config)
        finally:
            args.required_target = saved_req
            args.optional_target = saved_opt
        
        if update is None:
            LOGGER.warning(
                "Collection '%s' iteration %d produced no new tokens; stopping early",
                context.collection_id,
                iteration + 1,
            )
            break
        
        total_req_added += len(update.required_added)
        total_opt_added += len(update.optional_added)
        iteration += 1
    
    if iteration >= args.max_iterations:
        LOGGER.warning(
            "Collection '%s' hit max iterations (%d) before reaching targets",
            context.collection_id,
            args.max_iterations,
        )
    
    if total_req_added == 0 and total_opt_added == 0:
        return None
    
    return LexiconUpdate(
        collection_id=context.collection_id,
        file_path=context.file_path,
        required_added=[f"(total: {total_req_added} across {iteration} iterations)"],
        optional_added=[f"(total: {total_opt_added} across {iteration} iterations)"],
    )


def _single_update_iteration(
    context: LexiconContext,
    lexicon_data: Dict[str, List[str]],
    args: argparse.Namespace,
    xai_config: XAIConfig,
    rate_limit_delay: Optional[float] = None,
) -> LexiconUpdate | None:
    r"""Single iteration: call Grok once and update lexicon lists."""

    if args.required_target <= 0 and args.optional_target <= 0:
        LOGGER.info("No targets requested for collection '%s'; skipping", context.collection_id)
        return None

    user_prompt = build_user_prompt(
        context=context,
        lexicon_data=lexicon_data,
        required_target=args.required_target,
        optional_target=args.optional_target,
        context_limit=args.context_limit,
    )
    LOGGER.debug("Prompt for %s:\n%s", context.collection_id, user_prompt)
    
    # Rate limiting for parallel execution
    if rate_limit_delay and rate_limit_delay > 0:
        time.sleep(rate_limit_delay)
    
    # Retry logic
    max_retries = 3 if args.retry_on_error else 1
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = xai_generate(
                system_instruction=SYSTEM_PROMPT,
                user_text=user_prompt,
                config=xai_config,
            )
            if not response:
                raise RuntimeError(f"Grok returned an empty response for collection '{context.collection_id}'")
            
            proposed_required, proposed_optional = extract_terms(response)
            LOGGER.debug("Raw Grok response for %s: %s", context.collection_id, response[:500])
            
            # Success - break retry loop
            break
            
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                LOGGER.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    context.collection_id,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                LOGGER.error(
                    "All %d attempts failed for %s: %s",
                    max_retries,
                    context.collection_id,
                    exc,
                )
                raise

    dedup_tracker: MutableSet[str] = set()
    new_required = filter_new_terms(
        candidates=proposed_required,
        existing=lexicon_data["required"],
        already_added=dedup_tracker,
        limit=args.required_target,
    )
    new_optional = filter_new_terms(
        candidates=proposed_optional,
        existing=lexicon_data["optional"],
        already_added=dedup_tracker,
        limit=args.optional_target,
    )

    if not new_required and not new_optional:
        LOGGER.info("No novel tokens produced for collection '%s'", context.collection_id)
        return None

    lexicon_data["required"].extend(new_required)
    lexicon_data["optional"].extend(new_optional)
    return LexiconUpdate(
        collection_id=context.collection_id,
        file_path=context.file_path,
        required_added=new_required,
        optional_added=new_optional,
    )


async def process_collection_async(
    collection_id: str,
    context: LexiconContext,
    args: argparse.Namespace,
    xai_config: XAIConfig,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[LexiconUpdate], Optional[Exception]]:
    r"""Process a single collection asynchronously with semaphore-based rate limiting."""
    
    async with semaphore:
        try:
            # Load lexicon data
            lexicon_data = load_lexicon(context.file_path)
            
            # Run update in thread pool since it's blocking I/O
            loop = asyncio.get_event_loop()
            update = await loop.run_in_executor(
                None,
                update_collection,
                context,
                lexicon_data,
                args,
                xai_config,
            )
            
            # Save if not dry-run
            if update and not args.dry_run:
                await loop.run_in_executor(None, save_lexicon, context.file_path, lexicon_data)
            
            return (collection_id, update, None)
            
        except Exception as exc:
            return (collection_id, None, exc)


async def main_async(args: argparse.Namespace) -> List[LexiconUpdate]:
    r"""Async main function for parallel processing."""
    
    ontology_path = Path(args.ontology)
    ontology = load_ontology(ontology_path)
    contexts = resolve_collections(ontology, ontology_path)
    
    if not contexts:
        LOGGER.error("Ontology has no lexicon collections; nothing to do.")
        return []
    
    target_collections = args.collections or sorted(contexts.keys())
    xai_config = build_xai_config(args)
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.max_parallel)
    
    # Launch all collections in parallel
    tasks = []
    for collection_id in target_collections:
        context = contexts.get(collection_id)
        if not context:
            LOGGER.warning(f"Collection '{collection_id}' not found in ontology.")
            continue
        
        task = process_collection_async(collection_id, context, args, xai_config, semaphore)
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    updates: List[LexiconUpdate] = []
    for result in results:
        if isinstance(result, Exception):
            if args.strict:
                raise result
            LOGGER.exception("Collection processing failed: %s", result)
            continue
        
        collection_id, update, exc = result
        
        if exc:
            if args.strict:
                raise exc
            LOGGER.exception("Collection '%s' failed: %s", collection_id, exc)
            continue
        
        if update is None:
            continue
        
        updates.append(update)
        
        # Log update
        if args.dry_run:
            if "total:" in ", ".join(update.required_added):
                LOGGER.info(
                    "[DRY-RUN] %s completed: %s",
                    collection_id,
                    ", ".join(update.required_added + update.optional_added),
                )
            else:
                LOGGER.info(
                    "[DRY-RUN] %s -> required %+d, optional %+d",
                    collection_id,
                    len(update.required_added),
                    len(update.optional_added),
                )
                LOGGER.info("  Proposed required: %s", ", ".join(update.required_added) or "-")
                LOGGER.info("  Proposed optional: %s", ", ".join(update.optional_added) or "-")
        else:
            if "total:" in ", ".join(update.required_added):
                LOGGER.info("Updated %s: %s", collection_id, ", ".join(update.required_added + update.optional_added))
            else:
                LOGGER.info(
                    "Updated %s with %d required and %d optional tokens.",
                    collection_id,
                    len(update.required_added),
                    len(update.optional_added),
                )
    
    return updates


def main() -> None:
    r"""Entry point."""

    args = parse_args()
    configure_logging(verbose=args.verbose)
    
    if args.parallel:
        LOGGER.info("Running in PARALLEL mode with max %d concurrent collections", args.max_parallel)
        updates = asyncio.run(main_async(args))
    else:
        LOGGER.info("Running in SEQUENTIAL mode")
        ontology_path = Path(args.ontology)
        ontology = load_ontology(ontology_path)
        contexts = resolve_collections(ontology, ontology_path)
        if not contexts:
            LOGGER.error("Ontology has no lexicon collections; nothing to do.")
            sys.exit(1)

        target_collections = args.collections or sorted(contexts.keys())
        xai_config = build_xai_config(args)
        updates: List[LexiconUpdate] = []

        for collection_id in target_collections:
            context = contexts.get(collection_id)
            if not context:
                message = f"Collection '{collection_id}' not found in ontology."
                if args.strict:
                    raise KeyError(message)
                LOGGER.warning(message)
                continue
            try:
                lexicon_data = load_lexicon(context.file_path)
            except FileNotFoundError as exc:
                if args.strict:
                    raise
                LOGGER.error(str(exc))
                continue

            try:
                update = update_collection(context, lexicon_data, args, xai_config)
            except Exception as exc:  # pragma: no cover - defensive
                if args.strict:
                    raise
                LOGGER.exception("Collection '%s' failed: %s", collection_id, exc)
                continue

            if update is None:
                continue

            updates.append(update)
            if args.dry_run:
                if "total:" in ", ".join(update.required_added):
                    # Summary update from iterative mode
                    LOGGER.info(
                        "[DRY-RUN] %s completed: %s",
                        collection_id,
                        ", ".join(update.required_added + update.optional_added),
                    )
                else:
                    # Single iteration
                    LOGGER.info(
                        "[DRY-RUN] %s -> required %+d, optional %+d",
                        collection_id,
                        len(update.required_added),
                        len(update.optional_added),
                    )
                    LOGGER.info("  Proposed required: %s", ", ".join(update.required_added) or "-")
                    LOGGER.info("  Proposed optional: %s", ", ".join(update.optional_added) or "-")
            else:
                save_lexicon(context.file_path, lexicon_data)
                if "total:" in ", ".join(update.required_added):
                    LOGGER.info("Updated %s (%s): %s", collection_id, context.file_path, ", ".join(update.required_added + update.optional_added))
                else:
                    LOGGER.info(
                        "Updated %s (%s) with %d required and %d optional tokens.",
                        collection_id,
                        context.file_path,
                        len(update.required_added),
                        len(update.optional_added),
                    )

    if not updates:
        LOGGER.info("No lexicon files were changed.")
    elif args.dry_run:
        LOGGER.info("Dry-run complete. Re-run without --dry-run to persist the changes.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Aborted by user.")
        sys.exit(130)

