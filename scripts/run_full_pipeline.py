#!/usr/bin/env python3
"""
Full Pipeline Script for OASIS Social Simulation

This script runs the complete pipeline from persona generation to training dataset export:
1. Persona Generation (via LLM)
2. Graph Building (social network edges)
3. Simulation Run (posts/comments generation)
4. RAG LLM Imputation (replace label tokens with natural language)
5. Report Generation (HTML viewer + stats)
6. Training Dataset Export (JSONL with username, content, label)

Usage:
    poetry run python scripts/run_full_pipeline.py --run-id my_experiment --total-personas 100

Configuration is pulled from configs/llm_settings.py for LLM provider/model settings.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Helper: Detect if running on Modal (no poetry) vs local (with poetry)
# =============================================================================

def get_python_cmd() -> List[str]:
    """Return the command prefix for running Python scripts.
    
    On Modal/cloud: use python3 directly
    Locally: use poetry run python3
    """
    import shutil
    if shutil.which("poetry") is not None:
        return ["poetry", "run", "python3"]
    else:
        return ["python3"]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full pipeline run."""
    
    # Run identification
    run_id: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    seed: int = 314159
    
    # Output paths (derived from run_id)
    output_dir: Path = field(default_factory=lambda: Path("./data/runs"))
    
    # Persona generation
    total_personas: int = 100
    benign_ratio: float = 0.60  # 60% benign, 40% toxic by default
    
    @property
    def persona_distribution(self) -> Dict[str, float]:
        """Compute persona distribution based on benign_ratio.
        
        Benign class (benign + recovery_ed) gets benign_ratio.
        Toxic classes (11 total) split (1 - benign_ratio) uniformly.
        """
        # Note: recovery_ed is now included in benign ratio
        benign_classes = ["benign", "recovery_ed"]
        toxic_classes = [
            "ed_risk", "pro_ana", "incel_misogyny", "alpha",
            "misinfo", "conspiracy", "trad", "gamergate",
            "extremist", "hate_speech", "bullying"
        ]
        
        benign_per_class = self.benign_ratio / len(benign_classes)
        toxic_per_class = (1.0 - self.benign_ratio) / len(toxic_classes)
        
        dist = {}
        for c in benign_classes:
            dist[c] = benign_per_class
        for c in toxic_classes:
            dist[c] = toxic_per_class
        return dist
    ontology_path: Path = Path("configs/personas/ontology_unified.yaml")
    
    # Simulation
    warmup_steps: int = 1
    simulation_steps: int = 2
    
    # RAG Imputation
    rag_workers: int = 4
    rag_batch_size: int = 8
    
    # Enhanced features
    enable_thread_dynamics: bool = True   # Pile-ons, echo chambers, debates
    enable_obfuscation: bool = True       # Leetspeak, asterisks on toxic terms
    
    # Post-processing
    generate_report: bool = True
    static_bank_path: Path = Path("./data/label_tokens_static_bank.yaml")
    
    @property
    def personas_csv(self) -> Path:
        return self.output_dir / f"personas_{self.run_id}.csv"
    
    @property
    def edges_csv(self) -> Path:
        return self.output_dir / f"edges_{self.run_id}.csv"
    
    @property
    def db_path(self) -> Path:
        return self.output_dir / f"{self.run_id}.db"
    
    @property
    def dataset_jsonl(self) -> Path:
        return self.output_dir / f"{self.run_id}_dataset.jsonl"
    
    @property
    def training_jsonl(self) -> Path:
        return self.output_dir / f"{self.run_id}_training.jsonl"
    
    @property
    def report_html(self) -> Path:
        return self.output_dir / f"{self.run_id}_report.html"
    
    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"manifest_{self.run_id}.yaml"
    
    def compute_persona_counts(self) -> Dict[str, int]:
        """Compute integer counts from distribution percentages.
        
        Handles cases where total_personas < number of classes by:
        1. First ensuring benign class gets its share
        2. Then distributing remaining among toxic classes
        
        Note: recovery_ed is included in benign count
        """
        # Note: "benign" and "recovery_ed" are non-toxic
        benign_classes = ["benign", "recovery_ed"]
        toxic_classes = [
            "ed_risk", "pro_ana", "incel_misogyny", "alpha",
            "misinfo", "conspiracy", "trad", "gamergate",
            "extremist", "hate_speech", "bullying"
        ]
        
        # Calculate target counts
        num_benign_total = max(1, round(self.total_personas * self.benign_ratio))
        num_toxic = self.total_personas - num_benign_total
        
        counts = {}
        
        # Distribute benign personas
        if num_benign_total >= len(benign_classes):
            base = num_benign_total // len(benign_classes)
            remainder = num_benign_total % len(benign_classes)
            for i, c in enumerate(benign_classes):
                counts[c] = base + (1 if i < remainder else 0)
        else:
            # Fallback if very few benign slots
            counts["benign"] = num_benign_total
            counts["recovery_ed"] = 0
        
        # Distribute toxic personas
        if num_toxic >= len(toxic_classes):
            base = num_toxic // len(toxic_classes)
            remainder = num_toxic % len(toxic_classes)
            for i, c in enumerate(toxic_classes):
                counts[c] = base + (1 if i < remainder else 0)
        elif num_toxic > 0:
            # Not enough for all toxic classes - distribute what we have
            for i, c in enumerate(toxic_classes):
                counts[c] = 1 if i < num_toxic else 0
        else:
            # No toxic personas
            for c in toxic_classes:
                counts[c] = 0
        
        # Filter out zero counts
        counts = {k: v for k, v in counts.items() if v > 0}
        
        return counts


# =============================================================================
# Pipeline Steps
# =============================================================================

def log_step(step_num: int, total_steps: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"[Step {step_num}/{total_steps}] {message}")
    print(f"{'='*60}")
    sys.stdout.flush()


def step_1_generate_personas(config: PipelineConfig) -> bool:
    """Generate personas using LLM."""
    log_step(1, 6, "Generating Personas")
    
    counts = config.compute_persona_counts()
    print(f"Persona distribution ({config.total_personas} total):")
    for persona, count in counts.items():
        print(f"  {persona}: {count}")
    
    # Build command for generate_personas_llm.py
    cmd = get_python_cmd() + [
        "scripts/generate_personas_llm.py",
        "--out", str(config.personas_csv),
        "--seed", str(config.seed),
        "--ontology", str(config.ontology_path),
        "--mode", "rag",  # RAG mode with parallel prompt synthesis
    ]
    
    # Add persona counts (map to generate_personas_llm.py argument names)
    persona_arg_map = {
        "benign": "--benign",
        "recovery_ed": "--recovery",
        "ed_risk": "--ed-risk",
        "pro_ana": "--pro-ana",
        "incel_misogyny": "--incel",
        "alpha": "--alpha",
        "misinfo": "--misinfo",
        "conspiracy": "--conspiracy",
        "trad": "--trad",
        "gamergate": "--gamergate",
        "extremist": "--extremist",
        "hate_speech": "--hate",      # Note: --hate not --hate-speech
        "bullying": "--bully",        # Note: --bully not --bullying
    }
    
    for persona, count in counts.items():
        if persona in persona_arg_map:
            cmd.extend([persona_arg_map[persona], str(count)])
    
    print(f"\nRunning: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Persona generation failed with code {result.returncode}")
        return False
    
    if not config.personas_csv.exists():
        print(f"ERROR: Personas CSV not created at {config.personas_csv}")
        return False
    
    # Count generated personas
    with open(config.personas_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        actual_count = sum(1 for _ in reader)
    
    print(f"\n✓ Generated {actual_count} personas to {config.personas_csv}")
    return True


def step_2_build_graph(config: PipelineConfig) -> bool:
    """Build social network graph (edges)."""
    log_step(2, 6, "Building Social Graph")
    
    # Read personas to get user_ids
    user_ids = []
    with open(config.personas_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "user_id" in row:
                user_ids.append(int(row["user_id"]))
            else:
                user_ids.append(len(user_ids))
    
    if not user_ids:
        print("ERROR: No user_ids found in personas CSV")
        return False
    
    # Generate edges (simple random graph)
    random.seed(config.seed)
    edges = []
    num_edges = len(user_ids) * 3  # Average 3 follows per user
    
    for _ in range(num_edges):
        follower = random.choice(user_ids)
        followee = random.choice(user_ids)
        if follower != followee:
            edges.append((follower, followee))
    
    # Remove duplicates
    edges = list(set(edges))
    
    # Write edges CSV
    with open(config.edges_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["follower_id", "followee_id"])
        for follower, followee in edges:
            writer.writerow([follower, followee])
    
    print(f"✓ Generated {len(edges)} edges to {config.edges_csv}")
    return True


def step_3_create_manifest(config: PipelineConfig) -> bool:
    """Create a manifest file for the simulation."""
    log_step(3, 6, "Creating Manifest")
    
    counts = config.compute_persona_counts()
    
    # Map persona names to manifest population keys
    population_map = {
        "benign": "persona_benign_mvp",
        "recovery_ed": "persona_recovery_mvp",
        "ed_risk": "persona_ed_risk_mvp",
        "pro_ana": "persona_pro_ana_mvp",
        "incel_misogyny": "persona_incel_mvp",
        "alpha": "persona_alpha_mvp",
        "misinfo": "persona_misinfo_mvp",
        "conspiracy": "persona_conspiracy_mvp",
        "trad": "persona_trad_mvp",
        "gamergate": "persona_gamergate_mvp",
        "extremist": "persona_extremist_mvp",
        "hate_speech": "persona_hate_speech_mvp",
        "bullying": "persona_bullying_mvp",
    }
    
    population = {population_map.get(k, f"persona_{k}_mvp"): v for k, v in counts.items()}
    
    manifest_content = f"""# Auto-generated manifest for {config.run_id}
run_id: "{config.run_id}"

ontology: "{config.ontology_path}"

population:
"""
    for key, count in population.items():
        manifest_content += f"  {key}: {count}\n"
    
    manifest_content += f"""
personas_csv: "{config.personas_csv}"
edges_csv: "{config.edges_csv}"

graph:
  type: "seeded"
  seed: {config.seed}

post_label_mode_probs:
  none: 0.4
  single: 0.4
  double: 0.2

label_tokens:
  inventory:
    - "LBL:INCEL_SLANG"
    - "LBL:INCEL_MISOGYNY"
    - "LBL:MISOGYNISTIC_LECTURE"
    - "LBL:ED_PROMO"
    - "LBL:MEANSPO"
    - "LBL:MISINFO_CLAIM"
    - "LBL:DEEPSTATE"
    - "LBL:GENDER_ESSENTIALISM"
    - "LBL:GATEKEEPING"
    - "LBL:DOXXING_THREAT"
    - "LBL:SLUR"
    - "LBL:SUPPORTIVE"
"""
    
    with open(config.manifest_path, "w") as f:
        f.write(manifest_content)
    
    print(f"✓ Created manifest at {config.manifest_path}")
    return True


def step_4_run_simulation(config: PipelineConfig) -> bool:
    """Run the social simulation."""
    log_step(4, 6, "Running Simulation")
    
    # Remove old database if exists
    if config.db_path.exists():
        config.db_path.unlink()
        print(f"Removed existing database: {config.db_path}")
    
    cmd = get_python_cmd() + [
        "scripts/run_production_sim.py",
        "--manifest", str(config.manifest_path),
        "--personas-csv", str(config.personas_csv),
        "--db", str(config.db_path),
        "--steps", str(config.simulation_steps),
        "--warmup-steps", str(config.warmup_steps),
        "--rag-imputer", "off",  # We'll run imputation separately
        "--edges-csv", str(config.edges_csv),
        "--fresh-db",
    ]
    
    # Add enhanced feature flags
    if config.enable_thread_dynamics:
        cmd.append("--enable-thread-dynamics")
    if config.enable_obfuscation:
        cmd.append("--enable-obfuscation")
    
    print(f"\nRunning simulation with {config.warmup_steps} warmup + {config.simulation_steps} steps...")
    print(f"Thread dynamics: {config.enable_thread_dynamics}, Obfuscation: {config.enable_obfuscation}")
    print(f"Command: {' '.join(cmd[:10])}...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Simulation failed with code {result.returncode}")
        return False
    
    if not config.db_path.exists():
        print(f"ERROR: Database not created at {config.db_path}")
        return False
    
    # Count results
    conn = sqlite3.connect(str(config.db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post")
    post_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment")
    comment_count = cur.fetchone()[0]
    conn.close()
    
    print(f"\n✓ Simulation completed in {elapsed:.1f}s")
    print(f"  Posts: {post_count}")
    print(f"  Comments: {comment_count}")
    return True


async def step_5_run_imputation(config: PipelineConfig) -> bool:
    """Run RAG LLM imputation to replace label tokens."""
    log_step(5, 6, "Running RAG LLM Imputation")
    
    from configs.llm_settings import (IMPUTATION_MAX_TOKENS, IMPUTATION_MODEL,
                                      IMPUTATION_PROVIDER,
                                      IMPUTATION_TEMPERATURE)
    from oasis.imputation.rag_llm_imputer import RagImputer, RagImputerConfig
    from orchestrator.model_provider import LLMProviderSettings

    # Count tokens before
    conn = sqlite3.connect(str(config.db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post WHERE content LIKE '%LBL:%'")
    post_tokens = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE content LIKE '%LBL:%'")
    comment_tokens = cur.fetchone()[0]
    conn.close()
    
    total_tokens = post_tokens + comment_tokens
    print(f"Found {post_tokens} posts and {comment_tokens} comments with label tokens")
    
    if total_tokens == 0:
        print("No tokens to impute, skipping...")
        return True
    
    print(f"Provider: {IMPUTATION_PROVIDER}")
    print(f"Model: {IMPUTATION_MODEL}")
    print(f"Workers: {config.rag_workers}")
    
    # Build LLM settings
    llm_settings = LLMProviderSettings(
        provider=IMPUTATION_PROVIDER,
        model_name=IMPUTATION_MODEL,
    )
    
    # Create imputer config
    imputer_config = RagImputerConfig(
        db_path=config.db_path,
        llm_settings=llm_settings,
        mode="sync",
        batch_size=config.rag_batch_size,
        max_workers=config.rag_workers,
        temperature=IMPUTATION_TEMPERATURE,
        max_tokens=IMPUTATION_MAX_TOKENS,
        static_bank_path=config.static_bank_path,
        run_seed=config.seed,
        personas_csv_path=config.personas_csv,
    )
    
    # Run imputer
    imputer = RagImputer(imputer_config)
    
    start_time = time.time()
    await imputer.start()
    
    # Reset tracking to scan from beginning
    imputer._last_post_id = 0
    imputer._last_comment_id = 0
    
    await imputer.enqueue_new_rows()
    await imputer.flush()
    await imputer.shutdown()
    elapsed = time.time() - start_time
    
    # Verify imputation
    conn = sqlite3.connect(str(config.db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    imputed_posts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    imputed_comments = cur.fetchone()[0]
    conn.close()
    
    print(f"\n✓ Imputation completed in {elapsed:.1f}s")
    print(f"  Imputed posts: {imputed_posts}")
    print(f"  Imputed comments: {imputed_comments}")
    return True


def step_6_export_datasets(config: PipelineConfig) -> bool:
    """Export datasets: full JSONL and training-friendly format."""
    log_step(6, 6, "Exporting Datasets")
    
    # Run build_dataset.py for full export
    cmd = get_python_cmd() + [
        "scripts/build_dataset.py",
        "--db", str(config.db_path),
        "--out", str(config.dataset_jsonl),
        "--personas-csv", str(config.personas_csv),
        "--static-bank", str(config.static_bank_path),
    ]
    
    print("Exporting full dataset...")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: Dataset export failed with code {result.returncode}")
    
    # Create training-friendly format
    print("\nCreating training-friendly dataset...")
    
    # Load personas for username lookup
    # First, load persona data from CSV (indexed by row order, 0-based)
    persona_rows: List[Dict[str, str]] = []
    with open(config.personas_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            persona_rows.append(row)
    
    # Get username -> user_id mapping from database
    # The 'name' column contains the username (user_name is often empty)
    conn_temp = sqlite3.connect(str(config.db_path))
    cur_temp = conn_temp.cursor()
    cur_temp.execute("SELECT user_id, name FROM user")
    username_to_uid = {name: uid for uid, name in cur_temp.fetchall() if name}
    conn_temp.close()
    
    # Build uid -> persona mapping using username as key
    uid_to_persona: Dict[int, Dict[str, str]] = {}
    for row in persona_rows:
        username = row.get("username", "")
        if username in username_to_uid:
            uid_to_persona[username_to_uid[username]] = row
    
    # Read from database and create training format
    conn = sqlite3.connect(str(config.db_path))
    cur = conn.cursor()
    
    # Check for text_rag_imputed column
    cur.execute("PRAGMA table_info(post)")
    post_cols = [col[1] for col in cur.fetchall()]
    has_rag_col = "text_rag_imputed" in post_cols
    
    training_records = []
    
    # Process posts
    if has_rag_col:
        cur.execute("SELECT post_id, user_id, content, text_rag_imputed FROM post")
    else:
        cur.execute("SELECT post_id, user_id, content, NULL FROM post")
    
    for post_id, user_id, content, imputed in cur.fetchall():
        persona_data = uid_to_persona.get(user_id, {})
        username = persona_data.get("username", f"user_{user_id}")
        label = persona_data.get("primary_label", "unknown")
        
        # Use imputed text if available, otherwise original
        text = imputed if imputed and imputed.strip() else content
        
        training_records.append({
            "username": username,
            "content": text or "",
            "label": label,
            "type": "post",
            "id": f"p_{post_id}",
        })
    
    # Check for text_rag_imputed column in comments
    cur.execute("PRAGMA table_info(comment)")
    comment_cols = [col[1] for col in cur.fetchall()]
    has_comment_rag_col = "text_rag_imputed" in comment_cols
    
    # Process comments
    if has_comment_rag_col:
        cur.execute("SELECT comment_id, user_id, content, text_rag_imputed FROM comment")
    else:
        cur.execute("SELECT comment_id, user_id, content, NULL FROM comment")
    
    for comment_id, user_id, content, imputed in cur.fetchall():
        persona_data = uid_to_persona.get(user_id, {})
        username = persona_data.get("username", f"user_{user_id}")
        label = persona_data.get("primary_label", "unknown")
        
        text = imputed if imputed and imputed.strip() else content
        
        training_records.append({
            "username": username,
            "content": text or "",
            "label": label,
            "type": "comment",
            "id": f"c_{comment_id}",
        })
    
    conn.close()
    
    # Write training dataset
    with open(config.training_jsonl, "w") as f:
        for record in training_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✓ Exported {len(training_records)} records to {config.training_jsonl}")
    
    # Print label distribution
    from collections import Counter
    label_counts = Counter(r["label"] for r in training_records)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(training_records)
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Print sample record
    if training_records:
        print("\nSample training record:")
        sample = training_records[0]
        print(f"  username: {sample['username']}")
        print(f"  label: {sample['label']}")
        print(f"  content: {sample['content'][:80]}...")
    
    # Generate report if requested
    if config.generate_report:
        print("\nGenerating HTML report...")
        report_cmd = get_python_cmd() + [
            "scripts/report_production.py",
            "--db", str(config.db_path),
            "--out", str(config.report_html),
            "--personas-csv", str(config.personas_csv),
        ]
        result = subprocess.run(report_cmd, capture_output=True)
        if result.returncode == 0 and config.report_html.exists():
            print(f"✓ Generated report at {config.report_html}")
        else:
            print("WARNING: Report generation failed (non-critical)")
    
    return True


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline(config: PipelineConfig) -> bool:
    """Run the full pipeline."""
    print("\n" + "="*60)
    print("OASIS Full Pipeline")
    print("="*60)
    print(f"Run ID: {config.run_id}")
    print(f"Total Personas: {config.total_personas}")
    print(f"Benign/Toxic Ratio: {config.benign_ratio*100:.0f}% / {(1-config.benign_ratio)*100:.0f}%")
    print(f"Simulation Steps: {config.warmup_steps} warmup + {config.simulation_steps} main")
    print(f"Output Directory: {config.output_dir}")
    print("="*60)
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Step 1: Generate Personas
    if not step_1_generate_personas(config):
        return False
    
    # Step 2: Build Graph
    if not step_2_build_graph(config):
        return False
    
    # Step 3: Create Manifest
    if not step_3_create_manifest(config):
        return False
    
    # Step 4: Run Simulation
    if not step_4_run_simulation(config):
        return False
    
    # Step 5: Run Imputation
    if not await step_5_run_imputation(config):
        return False
    
    # Step 6: Export Datasets
    if not step_6_export_datasets(config):
        return False
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutput files:")
    print(f"  Personas:        {config.personas_csv}")
    print(f"  Edges:           {config.edges_csv}")
    print(f"  Database:        {config.db_path}")
    print(f"  Full Dataset:    {config.dataset_jsonl}")
    print(f"  Training Data:   {config.training_jsonl}")
    if config.generate_report:
        print(f"  HTML Report:     {config.report_html}")
    print("="*60)
    
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full OASIS simulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run with 50 personas
  poetry run python scripts/run_full_pipeline.py --run-id test --total-personas 50 --steps 1

  # Full run with 200 personas
  poetry run python scripts/run_full_pipeline.py --run-id prod_v1 --total-personas 200 --steps 3

  # Custom output directory
  poetry run python scripts/run_full_pipeline.py --run-id exp1 --output-dir ./experiments/exp1
        """
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique identifier for this run (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--total-personas",
        type=int,
        default=100,
        help="Total number of personas to generate (default: 100)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of simulation steps (default: 2)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of warmup steps (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help="Random seed for reproducibility (default: 314159)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/runs",
        help="Output directory for all files (default: ./data/runs)"
    )
    parser.add_argument(
        "--rag-workers",
        type=int,
        default=4,
        help="Number of RAG imputation workers (default: 4)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default="configs/personas/ontology_unified.yaml",
        help="Path to persona ontology YAML"
    )
    parser.add_argument(
        "--benign-ratio",
        type=float,
        default=0.60,
        help="Ratio of benign personas (default: 0.60 = 60%% benign, 40%% toxic)"
    )
    parser.add_argument(
        "--enable-thread-dynamics",
        action="store_true",
        default=True,
        help="Enable coordinated behaviors: pile-ons, echo chambers, debates (default: enabled)"
    )
    parser.add_argument(
        "--no-thread-dynamics",
        action="store_true",
        help="Disable thread dynamics"
    )
    parser.add_argument(
        "--enable-obfuscation",
        action="store_true",
        default=True,
        help="Enable post-imputation obfuscation: leetspeak, asterisks (default: enabled)"
    )
    parser.add_argument(
        "--no-obfuscation",
        action="store_true",
        help="Disable obfuscation"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Handle flag overrides (--no-* takes precedence)
    enable_thread_dynamics = not args.no_thread_dynamics
    enable_obfuscation = not args.no_obfuscation
    
    # Build config
    config = PipelineConfig(
        run_id=args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=args.seed,
        output_dir=Path(args.output_dir),
        total_personas=args.total_personas,
        benign_ratio=args.benign_ratio,
        warmup_steps=args.warmup_steps,
        simulation_steps=args.steps,
        rag_workers=args.rag_workers,
        generate_report=not args.no_report,
        ontology_path=Path(args.ontology),
        enable_thread_dynamics=enable_thread_dynamics,
        enable_obfuscation=enable_obfuscation,
    )
    
    # Run pipeline
    success = asyncio.run(run_pipeline(config))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

