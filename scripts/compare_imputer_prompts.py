#!/usr/bin/env python3
"""
Compare imputer prompt variants on a small dataset.

Runs two simulations with different prompts and compares outputs.

Usage:
    poetry run python scripts/compare_imputer_prompts.py --personas 20 --steps 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Three prompt variants to compare
PROMPT_VARIANTS = {
    "baseline": {
        "system": (
            "You replace placeholder label tokens (e.g., <LBL:INCEL_MISOGYNY>) "
            "inside social-media posts. Only replace the placeholder spans, keep the author's "
            "voice, and keep the rest of the text identical. Output JSON."
        ),
        "suffix": (
            'Return JSON of the form {"replacement": "..."} where the replacement is a short span '
            "that captures the label's intent in the author's voice."
        ),
    },
    "improved": {
        "system": (
            "You replace placeholder label tokens inside social-media posts with realistic content. "
            "Rules: (1) Output 1-15 words only, (2) Match the author's tone exactly, "
            "(3) Never include the original token text, (4) Output JSON {\"replacement\": \"...\"}."
        ),
        "suffix": (
            'Return JSON {"replacement": "..."} with a SHORT phrase (1-15 words) that:\n'
            "- Sounds natural in context\n"
            "- Matches the post's tone and style\n"
            "- Captures the label's meaning without being generic"
        ),
    },
    "hybrid": {
        "system": (
            "You replace placeholder label tokens inside social-media posts. "
            "Rules: (1) Output 2-10 words, (2) STAY IN CHARACTER as the post author - never sound educational, "
            "(3) Never include the token text itself, (4) Output JSON {\"replacement\": \"...\"}."
        ),
        "suffix": (
            'Return JSON {"replacement": "..."} with a phrase (2-10 words) that:\n'
            "- Sounds like something THIS USER would actually say\n"
            "- Fits naturally in the sentence\n"
            "- Expresses the label's sentiment authentically"
        ),
    },
}


@dataclass
class ComparisonResult:
    """Result from comparing prompt outputs."""

    record_id: int
    record_type: str  # "post" or "comment"
    original_text: str
    token: str
    baseline_output: str
    improved_output: str
    hybrid_output: str = ""


def extract_tokens_from_text(text: str) -> List[str]:
    """Extract label tokens from text."""
    import re
    pattern = r'<?(LBL:[A-Z_]+)>?'
    return re.findall(pattern, text)


async def run_imputation_with_prompt(
    db_path: Path,
    variant_name: str,
    prompts: Dict[str, str],
) -> Dict[int, str]:
    """Run imputation with a specific prompt variant.

    Returns dict of record_id -> imputed_text
    """
    from configs.llm_settings import (
        IMPUTATION_MAX_TOKENS,
        IMPUTATION_MODEL,
        IMPUTATION_PROVIDER,
        IMPUTATION_TEMPERATURE,
    )
    from orchestrator.model_provider import LLMProviderSettings, create_model_backend

    print(f"\n  Running with '{variant_name}' prompt...")

    # Get records with tokens
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT post_id AS id, content, 'post' AS type
        FROM post
        WHERE content LIKE '%LBL:%'
        UNION ALL
        SELECT comment_id AS id, content, 'comment' AS type
        FROM comment
        WHERE content LIKE '%LBL:%'
        LIMIT 50
    """)
    records = cur.fetchall()
    conn.close()

    if not records:
        print("  No records with tokens found")
        return {}

    print(f"  Found {len(records)} records with tokens")

    # Create LLM backend
    llm_settings = LLMProviderSettings(
        provider=IMPUTATION_PROVIDER,
        model_name=IMPUTATION_MODEL,
    )
    backend = create_model_backend(llm_settings)
    backend.model_config_dict["temperature"] = float(IMPUTATION_TEMPERATURE)
    backend.model_config_dict["max_tokens"] = int(IMPUTATION_MAX_TOKENS)

    results = {}

    for record in records:
        record_id = record["id"]
        content = record["content"]
        tokens = extract_tokens_from_text(content)

        if not tokens:
            continue

        # Build prompt for first token
        token = tokens[0]
        user_prompt = (
            f"Original post (do not rewrite unrelated text):\n"
            f"{content}\n\n"
            f"Placeholder token #1: {token}\n\n"
            f"{prompts['suffix']}"
        )

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await backend.arun(messages)
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if isinstance(message, dict):
                raw_content = str(message.get("content", "") or "")
            else:
                raw_content = str(getattr(message, "content", "") or "")

            # Parse JSON response
            try:
                data = json.loads(raw_content.strip())
                replacement = data.get("replacement", raw_content)
            except json.JSONDecodeError:
                replacement = raw_content.split("\n")[0][:100]

            results[record_id] = replacement

        except Exception as e:
            print(f"    Error on record {record_id}: {e}")
            results[record_id] = f"[ERROR: {e}]"

    print(f"  Completed {len(results)} imputations")
    return results


def compare_outputs(
    baseline_results: Dict[int, str],
    improved_results: Dict[int, str],
    hybrid_results: Dict[int, str],
    db_path: Path,
) -> List[ComparisonResult]:
    """Compare outputs from all prompt variants."""

    # Get original records
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT post_id AS id, content, 'post' AS type
        FROM post
        WHERE content LIKE '%LBL:%'
        UNION ALL
        SELECT comment_id AS id, content, 'comment' AS type
        FROM comment
        WHERE content LIKE '%LBL:%'
    """)
    records = {row["id"]: dict(row) for row in cur.fetchall()}
    conn.close()

    comparisons = []

    for record_id in baseline_results:
        if record_id not in improved_results or record_id not in hybrid_results:
            continue

        record = records.get(record_id, {})
        tokens = extract_tokens_from_text(record.get("content", ""))

        comparisons.append(ComparisonResult(
            record_id=record_id,
            record_type=record.get("type", "unknown"),
            original_text=record.get("content", "")[:200],
            token=tokens[0] if tokens else "unknown",
            baseline_output=baseline_results[record_id],
            improved_output=improved_results[record_id],
            hybrid_output=hybrid_results[record_id],
        ))

    return comparisons


def print_comparison_report(comparisons: List[ComparisonResult]) -> None:
    """Print a formatted comparison report."""

    print("\n" + "=" * 80)
    print("PROMPT COMPARISON REPORT (3 variants)")
    print("=" * 80)

    for i, comp in enumerate(comparisons, 1):  # Show all examples
        print(f"\n--- Example {i} ({comp.record_type} #{comp.record_id}) ---")
        print(f"Token: {comp.token}")
        print(f"Original: {comp.original_text[:80]}...")
        print(f"\n  Baseline:  {comp.baseline_output}")
        print(f"  Improved:  {comp.improved_output}")
        print(f"  Hybrid:    {comp.hybrid_output}")

        # Simple quality indicators
        baseline_len = len(comp.baseline_output.split())
        improved_len = len(comp.improved_output.split())
        hybrid_len = len(comp.hybrid_output.split())
        print(f"\n  Words: baseline={baseline_len}, improved={improved_len}, hybrid={hybrid_len}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate stats
    baseline_lengths = [len(c.baseline_output.split()) for c in comparisons]
    improved_lengths = [len(c.improved_output.split()) for c in comparisons]
    hybrid_lengths = [len(c.hybrid_output.split()) for c in comparisons]

    avg_baseline = sum(baseline_lengths) / len(baseline_lengths) if baseline_lengths else 0
    avg_improved = sum(improved_lengths) / len(improved_lengths) if improved_lengths else 0
    avg_hybrid = sum(hybrid_lengths) / len(hybrid_lengths) if hybrid_lengths else 0

    print(f"Total comparisons: {len(comparisons)}")
    print(f"\nAverage word count:")
    print(f"  Baseline: {avg_baseline:.1f}")
    print(f"  Improved: {avg_improved:.1f}")
    print(f"  Hybrid:   {avg_hybrid:.1f}")

    # Count how many are in the ideal 2-10 word range
    def in_range(length):
        return 2 <= length <= 10

    baseline_in_range = sum(1 for l in baseline_lengths if in_range(l))
    improved_in_range = sum(1 for l in improved_lengths if in_range(l))
    hybrid_in_range = sum(1 for l in hybrid_lengths if in_range(l))

    n = len(comparisons)
    print(f"\nIn ideal range (2-10 words):")
    print(f"  Baseline: {baseline_in_range}/{n} ({100*baseline_in_range/n:.0f}%)")
    print(f"  Improved: {improved_in_range}/{n} ({100*improved_in_range/n:.0f}%)")
    print(f"  Hybrid:   {hybrid_in_range}/{n} ({100*hybrid_in_range/n:.0f}%)")


async def run_comparison(personas: int = 20, steps: int = 1) -> None:
    """Run the full comparison pipeline."""

    print("\n" + "=" * 60)
    print("IMPUTER PROMPT A/B COMPARISON")
    print("=" * 60)

    # Step 1: Generate a small test dataset
    print("\n[Step 1] Generating test dataset...")

    import subprocess
    from scripts.run_full_pipeline import PipelineConfig, get_python_cmd

    run_id = f"prompt_comparison_{datetime.now().strftime('%H%M%S')}"
    output_dir = Path("./data/prompt_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        run_id=run_id,
        output_dir=output_dir,
        total_personas=personas,
        warmup_steps=0,
        simulation_steps=steps,
        benign_ratio=0.3,  # More toxic for more tokens
    )

    # Run simulation without imputation
    cmd = get_python_cmd() + [
        "scripts/run_full_pipeline.py",
        "--run-id", run_id,
        "--output-dir", str(output_dir),
        "--total-personas", str(personas),
        "--steps", str(steps),
        "--warmup-steps", "0",
        "--benign-ratio", "0.3",
        "--no-report",
    ]

    # We need to modify run_full_pipeline to skip imputation for this test
    # For now, just run the pipeline and then re-impute with both prompts

    print(f"Running: {' '.join(cmd[:8])}...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Pipeline failed: {result.stderr[:500]}")
        return

    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.1f}s")

    db_path = config.db_path
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    # Step 2: Run imputation with baseline prompt
    print("\n[Step 2] Testing baseline prompt...")
    baseline_results = await run_imputation_with_prompt(
        db_path, "baseline", PROMPT_VARIANTS["baseline"]
    )

    # Step 3: Run imputation with improved prompt
    print("\n[Step 3] Testing improved prompt...")
    improved_results = await run_imputation_with_prompt(
        db_path, "improved", PROMPT_VARIANTS["improved"]
    )

    # Step 4: Run imputation with hybrid prompt
    print("\n[Step 4] Testing hybrid prompt...")
    hybrid_results = await run_imputation_with_prompt(
        db_path, "hybrid", PROMPT_VARIANTS["hybrid"]
    )

    # Step 5: Compare results
    print("\n[Step 5] Comparing outputs...")
    comparisons = compare_outputs(baseline_results, improved_results, hybrid_results, db_path)

    if comparisons:
        print_comparison_report(comparisons)

        # Save to file
        report_path = output_dir / f"{run_id}_comparison.json"
        with open(report_path, "w") as f:
            json.dump([
                {
                    "record_id": c.record_id,
                    "type": c.record_type,
                    "token": c.token,
                    "original": c.original_text,
                    "baseline": c.baseline_output,
                    "improved": c.improved_output,
                    "hybrid": c.hybrid_output,
                }
                for c in comparisons
            ], f, indent=2)
        print(f"\nFull comparison saved to: {report_path}")
    else:
        print("No comparisons generated")


def main():
    parser = argparse.ArgumentParser(description="Compare imputer prompt variants")
    parser.add_argument("--personas", type=int, default=20, help="Number of personas")
    parser.add_argument("--steps", type=int, default=1, help="Simulation steps")
    args = parser.parse_args()

    asyncio.run(run_comparison(personas=args.personas, steps=args.steps))


if __name__ == "__main__":
    main()
