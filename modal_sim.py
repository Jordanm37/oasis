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
        # For TwHIN-BERT recommendation system
        "transformers",
        "torch",
        # For Gemini LLM provider
        "google-genai",
    )
    .run_commands(
        "mkdir -p /app/data/runs",
        # Pre-download TwHIN-BERT model during image build (avoids slow runtime download)
        "python3 -c \"from transformers import AutoTokenizer, AutoModel; "
        "print('Downloading TwHIN-BERT...'); "
        "AutoTokenizer.from_pretrained('Twitter/twhin-bert-base'); "
        "AutoModel.from_pretrained('Twitter/twhin-bert-base'); "
        "print('TwHIN-BERT cached!')\"",
    )
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
    timeout=14400,  # 4 hours for comment pass
    secrets=[
        modal.Secret.from_name("groq-keys"),
        modal.Secret.from_name("xai-keys", required_keys=[]),
        modal.Secret.from_name("gemini-keys", required_keys=[]),
        modal.Secret.from_name("openai-keys", required_keys=[]),
    ],
    volumes={"/app/data/runs": volume},
    cpu=32,
    memory=8192,
)
def run_comment_pass(
    run_id: str = "dataset_6500",
    steps: int = 10,
    enable_obfuscation: bool = True,
    enable_thread_dynamics: bool = True,
    copy_db: bool = False,
    recsys: str = "random",
):
    """Run a comment-only pass on an existing simulation.
    
    This adds more comments to existing posts without creating new posts.
    Useful for increasing comment density in a dataset.
    
    Args:
        run_id: The run ID to find the database/personas for.
        steps: Number of comment-only steps to run.
        enable_obfuscation: Enable post-imputation obfuscation.
        enable_thread_dynamics: Enable pile-ons, echo chambers, debates.
        copy_db: If True, copy DB to <run_id>_comments.db before modifying.
        recsys: RecSys type: "random", "twhin-bert", or "reddit".
    """
    import os
    import shutil
    import sqlite3
    import subprocess
    import sys
    
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
    env["MODAL_ENVIRONMENT"] = "true"
    
    # Find the files for this run_id (exact match to avoid picking up derivatives like _comments)
    runs_dir = "/app/data/runs"
    files = os.listdir(runs_dir) if os.path.exists(runs_dir) else []
    
    # Expected exact file names for this run_id
    expected_db = f"{run_id}.db"
    expected_manifest = f"manifest_{run_id}.yaml"
    expected_personas = f"personas_{run_id}.csv"
    expected_edges = f"edges_{run_id}.csv"
    
    manifest_path = None
    personas_csv = None
    db_path = None
    edges_csv = None
    
    for f in files:
        full_path = f"{runs_dir}/{f}"
        if f == expected_db:
            db_path = full_path
        elif f == expected_manifest:
            manifest_path = full_path
        elif f == expected_personas:
            personas_csv = full_path
        elif f == expected_edges:
            edges_csv = full_path
    
    if not all([manifest_path, personas_csv, db_path, edges_csv]):
        return {"status": "failed", "error": f"Missing files for run_id: {run_id}. Expected: {expected_db}, {expected_manifest}, {expected_personas}, {expected_edges}"}
    
    # If copy_db is True, copy all files to new names with _comments suffix
    new_run_id = run_id
    if copy_db:
        new_run_id = f"{run_id}_comments"
        print(f"\n=== Copying database to {new_run_id} ===")
        
        # Copy database using SQLite backup API (not shutil.copy2 - corrupts on volumes!)
        new_db_path = f"{runs_dir}/{new_run_id}.db"
        print(f"  Using SQLite backup API for reliable copy...")
        src_conn = sqlite3.connect(db_path)
        dst_conn = sqlite3.connect(new_db_path)
        src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        print(f"  Database backed up: {db_path} -> {new_db_path}")
        db_path = new_db_path
        
        # Copy manifest
        new_manifest = f"{runs_dir}/manifest_{new_run_id}.yaml"
        shutil.copy2(manifest_path, new_manifest)
        print(f"  Copied: {manifest_path} -> {new_manifest}")
        manifest_path = new_manifest
        
        # Copy personas
        new_personas = f"{runs_dir}/personas_{new_run_id}.csv"
        shutil.copy2(personas_csv, new_personas)
        print(f"  Copied: {personas_csv} -> {new_personas}")
        personas_csv = new_personas
        
        # Copy edges
        new_edges = f"{runs_dir}/edges_{new_run_id}.csv"
        shutil.copy2(edges_csv, new_edges)
        print(f"  Copied: {edges_csv} -> {new_edges}")
        edges_csv = new_edges
        
        # Commit volume to sync
        print("  Syncing volume...")
        volume.commit()
        import time
        time.sleep(1)
        
        # Quick integrity check
        test_conn = sqlite3.connect(db_path)
        test_cur = test_conn.cursor()
        test_cur.execute("PRAGMA quick_check")
        result = test_cur.fetchone()[0]
        test_cur.execute("SELECT COUNT(*) FROM post")
        post_count = test_cur.fetchone()[0]
        test_cur.execute("SELECT COUNT(*) FROM comment") 
        comment_count = test_cur.fetchone()[0]
        test_conn.close()
        if result == "ok":
            print(f"  ✓ Database verified: {post_count} posts, {comment_count} comments")
        else:
            print(f"  WARNING: Quick check returned: {result}")
    
    # Get current counts
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post")
    posts_before = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment")
    comments_before = cur.fetchone()[0]
    conn.close()
    
    print(f"\n=== Comment-Only Pass ===")
    print(f"Run ID: {new_run_id}")
    print(f"Steps: {steps}")
    print(f"RecSys: {recsys}")
    print(f"DB: {db_path}")
    print(f"Before: {posts_before} posts, {comments_before} comments")
    print(f"Mode: COMMENT-ONLY (no new posts will be created)")
    if copy_db:
        print(f"Copy mode: Original {run_id} preserved, working on {new_run_id}")
    
    # Print API keys summary
    groq_count = sum(1 for i in range(1, 41) if os.environ.get("GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"))
    print(f"Groq API keys: {groq_count}")
    
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
        "--comment-only",  # Key flag: restricts to CREATE_COMMENT only
        "--recsys", recsys,  # RecSys type
    ]
    
    if enable_thread_dynamics:
        cmd.append("--enable-thread-dynamics")
    if enable_obfuscation:
        cmd.append("--enable-obfuscation")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/app", env=env)
    
    # Commit volume changes
    volume.commit()
    
    # Get final counts
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post")
    posts_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment")
    comments_after = cur.fetchone()[0]
    conn.close()
    
    print(f"\n=== Comment Pass Complete ===")
    print(f"Run ID: {new_run_id}")
    print(f"RecSys: {recsys}")
    print(f"Posts: {posts_before} -> {posts_after} (change: {posts_after - posts_before})")
    print(f"Comments: {comments_before} -> {comments_after} (change: +{comments_after - comments_before})")
    if copy_db:
        print(f"New database: {db_path}")
    
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "run_id": new_run_id,
        "db_path": db_path,
        "recsys": recsys,
        "posts_before": posts_before,
        "posts_after": posts_after,
        "comments_before": comments_before,
        "comments_after": comments_after,
        "new_comments": comments_after - comments_before,
    }


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
    run_id: str = "dataset_6500",
    workers: int = 4,
    batch_size: int = 32,
    reset: bool = False,
    retry_failed: bool = False,
):
    """Run RAG LLM imputation on an existing database.
    
    Uses llama-3.3-70b-versatile via Groq (configured in llm_settings.py).
    
    Args:
        run_id: The run ID to find the database for.
        workers: Number of concurrent imputation workers.
        batch_size: Batch size for processing records.
        reset: If True, clear existing imputed text and re-run from scratch.
        retry_failed: If True, re-process items where imputed text still contains LBL: tokens.
    """
    import os
    import subprocess
    import sys
    
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
    db_path = None
    
    if os.path.exists(runs_dir):
        for f in os.listdir(runs_dir):
            if run_id in f and f.endswith(".db") and not f.endswith("-wal") and not f.endswith("-shm"):
                db_path = f"{runs_dir}/{f}"
                break
    
    if not db_path or not os.path.exists(db_path):
        available = [f for f in os.listdir(runs_dir) if f.endswith(".db")] if os.path.exists(runs_dir) else []
        return {"status": "failed", "error": f"No database found for run_id: {run_id}", "available_dbs": available}

    # Auto-detect personas CSV for persona-aware imputation
    personas_csv = None
    expected_personas = f"personas_{run_id}.csv"
    if os.path.exists(f"{runs_dir}/{expected_personas}"):
        personas_csv = f"{runs_dir}/{expected_personas}"

    # Print API keys summary
    print("\n=== RAG Imputation ===")
    print(f"Run ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"Personas CSV: {personas_csv or 'not found (persona context disabled)'}")
    print(f"Workers: {workers}")
    print(f"Batch size: {batch_size}")
    print(f"Reset: {reset}")
    print(f"Retry failed: {retry_failed}")
    print(f"Model: llama-3.3-70b-versatile (Groq)")

    groq_count = sum(1 for i in range(1, 41) if os.environ.get("GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"))
    print(f"Groq API keys available: {groq_count}")

    # Build command
    cmd = [
        "python3", "scripts/run_rag_imputer.py",
        "--db", db_path,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]

    # Add personas CSV if found
    if personas_csv:
        cmd.extend(["--personas-csv", personas_csv])

    if reset:
        cmd.append("--reset")
    elif retry_failed:
        cmd.append("--retry-failed")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/app", env=env)
    
    # Commit volume changes
    volume.commit()
    
    # Get final stats
    import sqlite3
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
    workers: int = 4,
    batch_size: int = 32,
    reset: bool = False,
    retry_failed: bool = False,
    copy_db: bool = False,
    recsys: str = "random",
):
    """
    Local entrypoint for Modal commands.
    
    Usage:
        modal run modal_sim.py                           # Full pipeline (default)
        modal run modal_sim.py --action continue         # Continue existing sim
        modal run modal_sim.py --action comment-pass     # Add comments to existing posts
        modal run modal_sim.py --action check            # Check progress
        modal run modal_sim.py --action list             # List result files
        modal run modal_sim.py --action impute           # Run RAG imputation
        
    Comment-pass options (for --action comment-pass):
        --steps 10                  # Number of comment-only steps to run
        --run-id <id>               # Run ID to add comments to
        --copy-db                   # Copy DB before modifying (preserves original)
        --recsys twhin-bert         # RecSys: random, twhin-bert, reddit
        
    Imputation options (for --action impute):
        --workers 4                 # Concurrent imputation workers
        --batch-size 32             # Batch size for processing
        --reset                     # Clear existing imputed text first
        --retry-failed              # Re-process items where imputed text still has LBL tokens
        
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
        print(f"Model: llama-3.3-70b-versatile (Groq)")
        print(f"Workers: {workers}, Batch size: {batch_size}, Reset: {reset}, Retry failed: {retry_failed}")
        print("-" * 50)
        result = run_imputation.remote(
            run_id=run_id,
            workers=workers,
            batch_size=batch_size,
            reset=reset,
            retry_failed=retry_failed,
        )
        print(f"Result: {result}")
        
    elif action == "comment-pass":
        print(f"Running comment-only pass for '{run_id}'")
        print(f"Steps: {steps}")
        print(f"RecSys: {recsys}")
        print(f"Copy DB: {copy_db}")
        print(f"Mode: COMMENT-ONLY (no new posts, only comments)")
        print("-" * 50)
        result = run_comment_pass.remote(
            run_id=run_id,
            steps=steps,
            enable_obfuscation=enable_obfuscation,
            copy_db=copy_db,
            recsys=recsys,
        )
        print(f"Result: {result}")
