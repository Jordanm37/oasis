# Running OASIS on Modal

Modal is a serverless platform that makes it easy to run Python workloads in the cloud.

## Quick Start (5 minutes)

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Authenticate

```bash
modal setup
```

This opens a browser to authenticate with your Modal account (free tier available).

### 3. Create Secrets

Store your Groq API keys as Modal secrets:

```bash
modal secret create groq-keys \
  GROQ_API_KEY=gsk_your_first_key_here \
  GROQ_API_KEY_2=gsk_your_second_key_here
```

### 4. Run the Simulation

**Full pipeline (new run):**
```bash
modal run modal_sim.py --steps 120 --total-personas 100 --run-id my_run
```

**Continue existing simulation:**
```bash
modal run modal_sim.py --action continue --steps 120
```

**Check progress:**
```bash
modal run modal_sim.py --action check
```

**List result files:**
```bash
modal run modal_sim.py --action download --run-id my_run
```

## Reaching 6,500 Records

To generate ~6,500 records from scratch:

```bash
# Option A: 100 personas × 200 steps ≈ 6,000-7,000 records
modal run modal_sim.py --total-personas 100 --steps 200 --run-id dataset_6500

# Option B: 150 personas × 130 steps ≈ 6,500 records  
modal run modal_sim.py --total-personas 150 --steps 130 --run-id dataset_6500
```

## Downloading Results

After the simulation completes, download the files:

```bash
# List available files
modal run modal_sim.py --action download --run-id dataset_6500

# Download via Modal volume
modal volume get oasis-data ./local_data/
```

Or use the Modal dashboard at https://modal.com to browse and download files.

## Cost Estimate

- **CPU:** ~$0.10/hour (2 vCPU)
- **Memory:** ~$0.05/hour (4GB)
- **Total for 6,500 records:** ~$0.50-1.00 (1-2 hours runtime)

Modal's free tier includes $30/month of credits, which is plenty for this workload.

## Troubleshooting

### "Secret not found"
```bash
modal secret create groq-keys GROQ_API_KEY=gsk_xxx
```

### "Volume not found"
The volume is created automatically on first run.

### Timeout errors
Increase the timeout in `modal_sim.py`:
```python
@app.function(timeout=28800)  # 8 hours
```

### View logs
```bash
modal app logs oasis-simulation
```

