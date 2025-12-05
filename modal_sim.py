"""
Modal deployment for OASIS simulation.

Setup:
    1. pip install modal
    2. modal setup  (authenticate)
    3. modal secret create groq-keys GROQ_API_KEY=gsk_xxx GROQ_API_KEY_2=gsk_yyy
    4. modal run modal_sim.py

To run with custom parameters:
    modal run modal_sim.py --steps 120 --total-records 6500
"""

import modal

# Create the Modal app
app = modal.App("oasis-simulation")

# Define the image with all dependencies
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
        # For RAG/vector retrieval
        "chromadb",
        "sentence-transformers",
    )
    .run_commands("mkdir -p /app/data/runs")
    # Only upload essential directories (must be last)
    .add_local_dir("oasis", "/app/oasis", copy=True, ignore=["__pycache__", "*.pyc", "datasets"])
    .add_local_dir("scripts", "/app/scripts", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("configs", "/app/configs", copy=True, ignore=["__pycache__", "*.pyc", "rag", "*.jsonl"])
    .add_local_dir("orchestrator", "/app/orchestrator", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("generation", "/app/generation", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("imputation", "/app/imputation", copy=True, ignore=["__pycache__", "*.pyc"])
    # Add essential data files (not the large runs directory)
    .add_local_file("data/label_tokens_static_bank.yaml", "/app/data/label_tokens_static_bank.yaml")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
)

# Persistent volume for storing results AND ChromaDB indices
# ChromaDB was uploaded via: modal volume put oasis-data ./data/imputation/chromadb /imputation/chromadb
volume = modal.Volume.from_name("oasis-data", create_if_missing=True)


@app.function(
    image=oasis_image,
    timeout=28800,  # 8 hours max for large runs
    secrets=[
        modal.Secret.from_name("groq-keys"),
        modal.Secret.from_name("xai-keys", required_keys=[]),  # Optional
        modal.Secret.from_name("gemini-keys", required_keys=[]),  # Optional
        modal.Secret.from_name("openai-keys", required_keys=[]),  # Optional
    ],
    volumes={"/app/data/runs": volume},
    cpu=24,
    memory=8192,
)
def run_simulation(
    run_id: str = "modal_run",
    total_personas: int = 100,
    steps: int = 120,
    warmup_steps: int = 2,
    benign_ratio: float = 0.65,
    rag_workers: int = 4,
    enable_thread_dynamics: bool = True,
    enable_obfuscation: bool = True,
):
    """Run the full OASIS pipeline on Modal.
    
    Enhanced parameters:
    - max_step_iterations=1: Single action per step (faster generation)
    - refresh_rec_post_count=8: More posts shown per refresh
    - max_rec_post_len=10: Recommendation pool size
    - enable_thread_dynamics: Coordinated behaviors (pile-ons, echo chambers)
    - enable_obfuscation: Apply leetspeak/asterisks to harmful terms
    """
    import os
    import subprocess
    import sys

    # Add app to path
    sys.path.insert(0, "/app")
    
    # Setup ChromaDB symlink (volume has /imputation/chromadb, code expects /app/data/imputation/chromadb)
    chromadb_volume_path = "/app/data/runs/imputation/chromadb"
    chromadb_expected_path = "/app/data/imputation/chromadb"
    if os.path.exists(chromadb_volume_path) and not os.path.exists(chromadb_expected_path):
        os.makedirs("/app/data/imputation", exist_ok=True)
        os.symlink(chromadb_volume_path, chromadb_expected_path)
        print(f"Linked ChromaDB: {chromadb_expected_path} -> {chromadb_volume_path}")
    
    print(f"=== Starting OASIS Simulation on Modal ===")
    print(f"Run ID: {run_id}")
    print(f"Personas: {total_personas}")
    print(f"Steps: {steps}")
    print(f"Benign ratio: {benign_ratio}")
    print(f"Thread dynamics: {enable_thread_dynamics}")
    print(f"Obfuscation: {enable_obfuscation}")
    print(f"max_step_iterations: 1 (configured in llm_config.py)")
    print(f"RecSys: refresh=8, max_pool=10, following=5")
    
    # Check API keys (supports multiple providers for parallel execution)
    print("\n=== API Keys ===")
    # Groq keys (up to 40)
    groq_count = 0
    for i in range(1, 41):
        key_name = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
        key_val = os.environ.get(key_name, "")
        if key_val:
            groq_count += 1
            print(f"  {key_name}: set ({key_val[:10]}...)")
    print(f"  Total Groq keys: {groq_count}")
    # xAI key
    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        print(f"  XAI_API_KEY: set ({xai_key[:10]}...)")
    # Gemini key
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        print(f"  GEMINI_API_KEY: set ({gemini_key[:10]}...)")
    # OpenAI key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        print(f"  OPENAI_API_KEY: set ({openai_key[:10]}...)")
    
    # List files to verify upload
    print("\n=== Files in /app ===")
    for f in os.listdir("/app"):
        print(f"  {f}")
    
    # Set PYTHONPATH so imports work
    os.environ["PYTHONPATH"] = "/app"
    
    # Run the full pipeline with all enhanced parameters
    cmd = [
        "python3", "scripts/run_full_pipeline.py",
        "--run-id", run_id,
        "--total-personas", str(total_personas),
        "--steps", str(steps),
        "--warmup-steps", str(warmup_steps),
        "--benign-ratio", str(benign_ratio),
        "--rag-workers", str(rag_workers),
        "--output-dir", "./data/runs",
    ]
    
    # Add optional flags
    if enable_thread_dynamics:
        cmd.append("--enable-thread-dynamics")
    if enable_obfuscation:
        cmd.append("--enable-obfuscation")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/app")
    
    if result.returncode != 0:
        print(f"Pipeline failed with exit code {result.returncode}")
        return {"status": "failed", "exit_code": result.returncode}
    
    # Commit volume changes
    volume.commit()
    
    print("\n=== Simulation Complete ===")
    
    # List output files
    print("\n=== Output Files ===")
    for f in os.listdir("/app/data/runs"):
        if run_id in f:
            size = os.path.getsize(f"/app/data/runs/{f}")
            print(f"  {f}: {size/1024:.1f} KB")
    
    return {"status": "success", "run_id": run_id}


@app.function(
    image=oasis_image,
    timeout=28800,  # 8 hours for long runs
    secrets=[
        modal.Secret.from_name("groq-keys"),
        modal.Secret.from_name("xai-keys", required_keys=[]),
        modal.Secret.from_name("gemini-keys", required_keys=[]),
        modal.Secret.from_name("openai-keys", required_keys=[]),
    ],
    volumes={"/app/data/runs": volume},
    cpu=24,
    memory=8192,
)
def continue_simulation(
    steps: int = 120,
    run_id: str = "dataset_6500",
    enable_thread_dynamics: bool = True,
    enable_obfuscation: bool = True,
):
    """Continue an existing simulation with more steps using existing personas/graph."""
    import os
    import subprocess
    import sys
    
    sys.path.insert(0, "/app")
    os.chdir("/app")
    
    # Setup ChromaDB symlink (volume has /imputation/chromadb, code expects /app/data/imputation/chromadb)
    chromadb_volume_path = "/app/data/runs/imputation/chromadb"
    chromadb_expected_path = "/app/data/imputation/chromadb"
    if os.path.exists(chromadb_volume_path) and not os.path.exists(chromadb_expected_path):
        os.makedirs("/app/data/imputation", exist_ok=True)
        os.symlink(chromadb_volume_path, chromadb_expected_path)
        print(f"Linked ChromaDB: {chromadb_expected_path} -> {chromadb_volume_path}")
    
    # Set environment for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["MODAL_ENVIRONMENT"] = "true"
    
    # Print API keys summary
    print("\n=== API Keys ===")
    groq_count = sum(1 for i in range(1, 25) if os.environ.get("GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"))
    print(f"  Total Groq keys: {groq_count}")
    if os.environ.get("XAI_API_KEY"):
        print(f"  XAI_API_KEY: set")
    if os.environ.get("GEMINI_API_KEY"):
        print(f"  GEMINI_API_KEY: set")
    if os.environ.get("OPENAI_API_KEY"):
        print(f"  OPENAI_API_KEY: set")
    
    # Find the files for this run_id
    runs_dir = "/app/data/runs"
    files = os.listdir(runs_dir) if os.path.exists(runs_dir) else []
    
    # Find matching files
    manifest_path = None
    personas_csv = None
    db_path = None
    edges_csv = None
    
    for f in files:
        if run_id in f:
            full_path = f"{runs_dir}/{f}"
            if f.endswith(".yaml") and "manifest" in f:
                manifest_path = full_path
            elif f.endswith(".csv") and "personas" in f:
                personas_csv = full_path
            elif f.endswith(".db"):
                db_path = full_path
            elif f.endswith(".csv") and "edges" in f:
                edges_csv = full_path
    
    if not all([manifest_path, personas_csv, db_path, edges_csv]):
        print(f"Missing files for run_id: {run_id}")
        print(f"Found: manifest={manifest_path}, personas={personas_csv}, db={db_path}, edges={edges_csv}")
        print(f"Available files: {files}")
        return {"status": "failed", "error": "Missing required files"}
    
    print(f"\n=== Continuing Simulation ===")
    print(f"Run ID: {run_id}")
    print(f"Steps: {steps}")
    print(f"Thread dynamics: {enable_thread_dynamics}")
    print(f"Obfuscation: {enable_obfuscation}")
    print(f"DB: {db_path}")
    
    cmd = [
        "python3", "scripts/run_production_sim.py",
        "--manifest", manifest_path,
        "--personas-csv", personas_csv,
        "--db", db_path,
        "--steps", str(steps),
        "--warmup-steps", "0",
        "--rag-imputer", "background",
        "--rag-workers", "4",
        "--edges-csv", edges_csv,
    ]
    
    if enable_thread_dynamics:
        cmd.append("--enable-thread-dynamics")
    if enable_obfuscation:
        cmd.append("--enable-obfuscation")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/app", env=env)
    
    volume.commit()
    
    return {"status": "success" if result.returncode == 0 else "failed"}


@app.function(
    image=oasis_image,
    volumes={"/app/data/runs": volume},
)
def check_progress(run_id: str = "dataset_6500"):
    """Check the current progress of a simulation."""
    import os
    import sqlite3

    # Try to find the database file
    runs_dir = "/app/data/runs"
    db_path = None
    
    # List all available DBs
    available_dbs = []
    if os.path.exists(runs_dir):
        available_dbs = [f for f in os.listdir(runs_dir) if f.endswith(".db")]
        # Find matching DB
        for db in available_dbs:
            if run_id in db:
                db_path = f"{runs_dir}/{db}"
                break
    
    if not db_path or not os.path.exists(db_path):
        return {"error": f"No database found for run_id: {run_id}", "available_dbs": available_dbs}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM post")
    posts = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM comment")
    comments = cur.fetchone()[0]
    
    # Check if text_rag_imputed column exists
    cur.execute("PRAGMA table_info(post)")
    post_cols = [col[1] for col in cur.fetchall()]
    has_imputed_col = "text_rag_imputed" in post_cols
    
    posts_imputed = 0
    comments_imputed = 0
    if has_imputed_col:
        cur.execute("SELECT COUNT(*) FROM post WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
        posts_imputed = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM comment WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
        comments_imputed = cur.fetchone()[0]
    
    conn.close()
    
    return {
        "db_path": db_path,
        "posts": posts,
        "comments": comments,
        "total": posts + comments,
        "posts_imputed": posts_imputed,
        "comments_imputed": comments_imputed,
        "imputation_ready": has_imputed_col,
    }


@app.function(
    image=oasis_image,
    volumes={"/app/data/runs": volume},
)
def list_files(run_id: str = ""):
    """List all files in the data/runs directory."""
    import os
    
    files = []
    runs_dir = "/app/data/runs"
    
    if os.path.exists(runs_dir):
        for f in sorted(os.listdir(runs_dir)):
            if run_id == "" or run_id in f:
                path = f"{runs_dir}/{f}"
                size = os.path.getsize(path)
                files.append({"name": f, "size_kb": round(size/1024, 1)})
    
    return {"files": files, "count": len(files)}


@app.function(
    image=oasis_image,
    timeout=14400,  # 4 hours for imputation
    secrets=[
        modal.Secret.from_name("groq-keys"),
        modal.Secret.from_name("xai-keys", required_keys=[]),
        modal.Secret.from_name("gemini-keys", required_keys=[]),
        modal.Secret.from_name("openai-keys", required_keys=[]),
    ],
    volumes={"/app/data/runs": volume},
    cpu=8,
    memory=4096,
)
def run_imputation(
    run_id: str = "prod_100u_50s_20b",
    workers: int = 8,
    batch_size: int = 32,
):
    """Run RAG LLM imputation on an existing database."""
    import os
    import subprocess
    import sys
    import sqlite3
    
    sys.path.insert(0, "/app")
    os.chdir("/app")
    
    # Setup ChromaDB symlink
    chromadb_volume_path = "/app/data/runs/imputation/chromadb"
    chromadb_expected_path = "/app/data/imputation/chromadb"
    if os.path.exists(chromadb_volume_path) and not os.path.exists(chromadb_expected_path):
        os.makedirs("/app/data/imputation", exist_ok=True)
        os.symlink(chromadb_volume_path, chromadb_expected_path)
        print(f"Linked ChromaDB: {chromadb_expected_path} -> {chromadb_volume_path}")
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    
    # Find the database
    runs_dir = "/app/data/runs"
    db_path = f"{runs_dir}/{run_id}.db"
    
    if not os.path.exists(db_path):
        available = [f for f in os.listdir(runs_dir) if f.endswith(".db")]
        return {"status": "failed", "error": f"No database found: {db_path}", "available_dbs": available}
    
    # Find personas CSV
    personas_csv = f"{runs_dir}/personas_{run_id}.csv"
    
    # Print summary
    print(f"\n=== RAG Imputation ===")
    print(f"Run ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"Personas: {personas_csv if os.path.exists(personas_csv) else 'not found'}")
    print(f"Workers: {workers}")
    print(f"Batch size: {batch_size}")
    
    groq_count = sum(1 for i in range(1, 41) if os.environ.get("GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"))
    print(f"Groq API keys: {groq_count}")
    
    # Build command
    cmd = [
        "python3", "scripts/run_rag_imputer.py",
        "--db", db_path,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/app", env=env)
    
    # Commit volume changes
    volume.commit()
    
    # Get final stats
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post")
    total_posts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment")
    total_comments = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM post WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    posts_imputed = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE text_rag_imputed IS NOT NULL AND text_rag_imputed != ''")
    comments_imputed = cur.fetchone()[0]
    conn.close()
    
    print(f"\n=== Imputation Complete ===")
    print(f"Posts: {posts_imputed}/{total_posts} imputed")
    print(f"Comments: {comments_imputed}/{total_comments} imputed")
    
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "posts_imputed": posts_imputed,
        "comments_imputed": comments_imputed,
        "total_posts": total_posts,
        "total_comments": total_comments,
    }


@app.local_entrypoint()
def main(
    action: str = "run",
    steps: int = 200,
    total_personas: int = 100,
    run_id: str = "dataset_6500",
    benign_ratio: float = 0.65,
    enable_thread_dynamics: bool = True,
    enable_obfuscation: bool = True,
):
    """
    Local entrypoint for Modal commands.
    
    Usage:
        modal run modal_sim.py                           # Full pipeline (default)
        modal run modal_sim.py --action continue         # Continue existing sim
        modal run modal_sim.py --action check            # Check progress
        modal run modal_sim.py --action list             # List result files
        
    Enhanced parameters (all enabled by default):
        --enable-thread-dynamics    # Pile-ons, echo chambers, debates
        --enable-obfuscation        # Leetspeak, asterisks on toxic terms
        
    Pre-configured improvements:
        - max_step_iterations=1     # Single action per step (faster)
        - refresh_rec_post_count=8  # More posts shown per refresh
        - max_rec_post_len=10       # Recommendation pool size
        - following_post_count=5    # More posts from followed users
    """
    if action == "run":
        print(f"Starting full pipeline: {total_personas} personas, {steps} steps")
        print(f"Run ID: {run_id}")
        print(f"Benign ratio: {benign_ratio}")
        print(f"Thread dynamics: {enable_thread_dynamics}")
        print(f"Obfuscation: {enable_obfuscation}")
        print("-" * 50)
        result = run_simulation.remote(
            run_id=run_id,
            total_personas=total_personas,
            steps=steps,
            benign_ratio=benign_ratio,
            enable_thread_dynamics=enable_thread_dynamics,
            enable_obfuscation=enable_obfuscation,
        )
        print(f"Result: {result}")
        
    elif action == "continue":
        print(f"Continuing simulation '{run_id}' with {steps} more steps")
        result = continue_simulation.remote(steps=steps, run_id=run_id)
        print(f"Result: {result}")
        
    elif action == "check":
        result = check_progress.remote(run_id=run_id)
        print(f"Progress: {result}")
        
    elif action == "list":
        result = list_files.remote(run_id=run_id)
        print(f"Files ({result['count']} total):")
        for f in result["files"]:
            print(f"  {f['name']}: {f['size_kb']} KB")
    
    elif action == "impute":
        print(f"Running RAG imputation for '{run_id}'")
        print("-" * 50)
        result = run_imputation.remote(run_id=run_id)
        print(f"Result: {result}")
