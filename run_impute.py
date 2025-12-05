"""Quick imputation runner."""
import modal

app = modal.App("run-impute")

oasis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "sqlite3", "curl")
    .pip_install(
        "camel-ai==0.2.70", "tiktoken", "pyyaml", "python-dotenv", "aiohttp",
        "openai>=1.0.0", "pydantic>=2.0", "numpy", "pandas", "networkx",
        "scipy", "pettingzoo", "igraph", "scikit-learn", "neo4j", "tenacity",
        "chromadb", "sentence-transformers",
    )
    .run_commands("mkdir -p /app/data/runs")
    .add_local_dir("oasis", "/app/oasis", copy=True, ignore=["__pycache__", "*.pyc", "datasets"])
    .add_local_dir("scripts", "/app/scripts", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("configs", "/app/configs", copy=True, ignore=["__pycache__", "*.pyc", "rag", "*.jsonl"])
    .add_local_dir("orchestrator", "/app/orchestrator", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("imputation", "/app/imputation", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir("generation", "/app/generation", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_file("data/label_tokens_static_bank.yaml", "/app/data/label_tokens_static_bank.yaml")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
)

volume = modal.Volume.from_name("oasis-data", create_if_missing=False)

@app.function(
    image=oasis_image,
    timeout=14400,
    secrets=[modal.Secret.from_name("groq-keys")],
    volumes={"/app/data/runs": volume},
    cpu=8,
    memory=4096,
)
def run_imputation(run_id: str, workers: int = 8, batch_size: int = 64):
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
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    
    db_path = f"/app/data/runs/{run_id}.db"
    personas_csv = f"/app/data/runs/personas_{run_id}.csv"
    
    cmd = [
        "python3", "scripts/run_rag_imputer.py",
        "--db", db_path,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/app", env=env)
    volume.commit()
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post WHERE text_rag_imputed IS NOT NULL")
    posts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM comment WHERE text_rag_imputed IS NOT NULL")
    comments = cur.fetchone()[0]
    conn.close()
    
    return {"posts_imputed": posts, "comments_imputed": comments}

@app.local_entrypoint()
def main(run_id: str = "prod_5k_v3_comments", workers: int = 8, batch_size: int = 64):
    print(f"Running imputation for {run_id} with {workers} workers, batch {batch_size}")
    result = run_imputation.remote(run_id, workers, batch_size)
    print(f"Result: {result}")
