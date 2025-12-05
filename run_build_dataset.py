"""
Build dataset from OASIS simulation database on Modal.

Usage:
    modal run run_build_dataset.py
"""
import modal

app = modal.App("build-dataset")
volume = modal.Volume.from_name("oasis-data", create_if_missing=False)

# Use same image as modal_sim.py to ensure all dependencies are available
oasis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "sqlite3", "curl")
    .pip_install(
        "camel-ai==0.2.70",
        "tiktoken",
        "pyyaml",
        "python-dotenv",
        "aiohttp",
        "openai>=1.0.0",
        "pydantic>=2.0",
        "numpy",
        "pandas",
        "networkx",
        "scipy",
        "pettingzoo",
        "igraph",
        "scikit-learn",
        "neo4j",
        "tenacity",
    )
    .run_commands("mkdir -p /app/data/runs")
    # Include all required directories (matching modal_sim.py exactly)
    .add_local_dir("oasis", "/app/oasis", copy=True, ignore=["__pycache__", "*.pyc", "datasets"])
    .add_local_dir("scripts", "/app/scripts", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("configs", "/app/configs", copy=True, ignore=["__pycache__", "*.pyc", "rag", "*.jsonl"])
    .add_local_dir("orchestrator", "/app/orchestrator", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("generation", "/app/generation", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_file("data/label_tokens_static_bank.yaml", "/app/data/label_tokens_static_bank.yaml")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
)


@app.function(
    image=oasis_image,
    volumes={"/app/data/runs": volume},
    timeout=1800,
    cpu=4,
)
def build_dataset(
    run_id: str = "prod_5k_v3_comments",
    train_ratio: float = 0.70,
    test_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
):
    """Build training dataset from simulation database."""
    import os
    import subprocess
    import sys
    
    # Setup paths (matching modal_sim.py pattern)
    sys.path.insert(0, "/app")
    os.chdir("/app")
    os.environ["PYTHONPATH"] = "/app"
    
    runs_dir = "/app/data/runs"
    db_path = f"{runs_dir}/{run_id}.db"
    out_path = f"{runs_dir}/{run_id}_dataset.jsonl"
    static_bank = "/app/data/label_tokens_static_bank.yaml"
    
    # Check for personas CSV (try without _comments suffix first)
    base_run_id = run_id.replace("_comments", "")
    personas_csv = f"{runs_dir}/personas_{base_run_id}.csv"
    if not os.path.exists(personas_csv):
        personas_csv = None
        print(f"No personas CSV found at {runs_dir}/personas_{base_run_id}.csv")
    
    print(f"{'='*60}")
    print(f"BUILD DATASET")
    print(f"{'='*60}")
    print(f"Database: {db_path} (exists: {os.path.exists(db_path)})")
    print(f"Output: {out_path}")
    print(f"Static bank: {static_bank} (exists: {os.path.exists(static_bank)})")
    print(f"Personas CSV: {personas_csv} (exists: {os.path.exists(personas_csv) if personas_csv else 'N/A'})")
    print(f"Split ratios: train={train_ratio}, test={test_ratio}, holdout={holdout_ratio}")
    
    # Verify database exists
    if not os.path.exists(db_path):
        available = [f for f in os.listdir(runs_dir) if f.endswith(".db")]
        raise FileNotFoundError(f"Database not found: {db_path}\nAvailable: {available}")
    
    # Build command (correct path: scripts/build_dataset.py)
    cmd = [
        "python3", "scripts/build_dataset.py",
        "--db", db_path,
        "--out", out_path,
        "--static-bank", static_bank,
        "--train-ratio", str(train_ratio),
        "--test-ratio", str(test_ratio),
        "--holdout-ratio", str(holdout_ratio),
    ]
    
    if personas_csv and os.path.exists(personas_csv):
        cmd.extend(["--personas-csv", personas_csv])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run without capture_output to see real-time progress
    result = subprocess.run(cmd, cwd="/app")
    
    if result.returncode != 0:
        raise RuntimeError(f"Build failed with exit code {result.returncode}")
    
    # Commit volume to save output
    volume.commit()
    
    # Show output stats
    if os.path.exists(out_path):
        import json
        from collections import Counter
        
        splits = Counter()
        labels = Counter()
        total_records = 0
        
        with open(out_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                total_records += 1
                splits[rec.get("split", "unknown")] += 1
                for lbl in rec.get("category_labels", []):
                    labels[lbl] += 1
        
        file_size = os.path.getsize(out_path)
        
        print(f"\n{'='*60}")
        print(f"DATASET BUILT SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Output: {out_path}")
        print(f"Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"Total records: {total_records:,}")
        print(f"\nSplit distribution:")
        for split, count in sorted(splits.items()):
            pct = count / total_records * 100
            print(f"  {split}: {count:,} ({pct:.1f}%)")
        print(f"\nLabel distribution (top 20):")
        for lbl, count in labels.most_common(20):
            print(f"  {lbl}: {count:,}")
        
        return {
            "status": "success",
            "output_path": out_path,
            "total_records": total_records,
            "splits": dict(splits),
            "labels": dict(labels.most_common(20)),
        }
    
    return {"status": "error", "message": "Output file not created"}


@app.local_entrypoint()
def main():
    """Run the dataset builder."""
    result = build_dataset.remote()
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Status: {result.get('status')}")
    if result.get('total_records'):
        print(f"Records: {result['total_records']:,}")
