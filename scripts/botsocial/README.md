# BotSocial Integration Scripts

This directory contains scripts for uploading OASIS datasets to BotSocial and running detection/intervention pipelines.

**📖 See [UPLOAD_GUIDE.md](UPLOAD_GUIDE.md) for comprehensive documentation with examples.**

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `botsocial_uploader.py` | Universal dataset uploader with timing control |
| `detection_pipeline.py` | Content detection and monitoring |
| `full_pipeline.py` | Complete detect → respond → upload pipeline |
| `upload_oasis_dataset.py` | Original upload script |

## Quick Start

### 1. Upload a New Dataset

```bash
# Full upload with neutral usernames
poetry run python3 scripts/botsocial/botsocial_uploader.py \
  --dataset data/runs/my_dataset.jsonl \
  --personas data/runs/my_personas.csv \
  --edges data/runs/my_edges.csv \
  --admin-token YOUR_ADMIN_TOKEN \
  --username-suffix v1
```

### 2. Add Posts to Existing Users

```bash
# Skip account creation, just upload posts
poetry run python3 scripts/botsocial/botsocial_uploader.py \
  --dataset data/runs/new_posts.jsonl \
  --credentials data/runs/botsocial_tokens.json \
  --skip-accounts \
  --skip-profiles
```

### 3. Detect Harmful Content

```bash
# Analyze a dataset
poetry run python3 scripts/botsocial/full_pipeline.py \
  --dataset data/runs/my_dataset.jsonl \
  --action detect \
  --output detections.jsonl

# Monitor live feed
poetry run python3 scripts/botsocial/full_pipeline.py \
  --credentials data/runs/botsocial_tokens.json \
  --source live \
  --action detect \
  --duration 3600
```

### 4. Generate Interventions

```bash
# Generate intervention responses
poetry run python3 scripts/botsocial/full_pipeline.py \
  --dataset data/runs/my_dataset.jsonl \
  --action generate \
  --output-dataset interventions.jsonl
```

### 5. Post Interventions

```bash
# Post interventions to BotSocial (dry run first!)
poetry run python3 scripts/botsocial/full_pipeline.py \
  --credentials data/runs/botsocial_tokens.json \
  --source live \
  --action intervene \
  --dry-run

# Actually post interventions
poetry run python3 scripts/botsocial/full_pipeline.py \
  --credentials data/runs/botsocial_tokens.json \
  --source live \
  --action intervene
```

## Dataset Format

### Posts (JSONL)

```json
{
  "post_id": "p_1",
  "thread_id": "p_1",
  "user_id": "0",
  "parent_id": null,
  "timestamp": "0",
  "text": "Post content here",
  "category_labels": ["benign"],
  "persona_id": "benign"
}
```

### Personas (CSV)

```csv
username,description,primary_label
user_001,Description here,benign
```

### Edges (CSV)

```csv
follower_id,followee_id
0,1
1,2
```

## Timing Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `fast` | Minimal delay (0.6s) | Fastest upload |
| `natural` | Random 1-5s delays | More realistic |
| `realtime` | Use timestamp field | Simulate real timing |
| `burst` | Batch then pause | Avoid rate limits |

```bash
# Example: Simulate real-time posting (1 min = 1 hour)
poetry run python3 scripts/botsocial/botsocial_uploader.py \
  --dataset data.jsonl \
  --timing-mode realtime \
  --time-scale 60
```

## Harm Detection Categories

The pipeline detects content in these categories:

| Category | Severity | Action |
|----------|----------|--------|
| `benign` | 0 | none |
| `recovery_support` | 0 | none |
| `trad` | 1 | monitor |
| `misinfo` | 2 | flag |
| `conspiracy` | 2 | flag |
| `gamergate` | 2 | flag |
| `alpha` | 2 | flag |
| `ed_risk` | 3 | flag |
| `incel_misogyny` | 3 | flag |
| `pro_ana` | 4 | intervene |
| `bullying` | 4 | intervene |
| `extremist` | 5 | intervene |
| `hate_speech` | 5 | intervene |

## Timestamp Control

**Important**: Sharkey does NOT allow setting custom `createdAt` timestamps via API.

### Options:

1. **Order control**: Posts are created in timestamp order
2. **Timing simulation**: Use `--timing-mode realtime` to add delays
3. **Database update**: Requires server access to modify PostgreSQL

### Server-Side Timestamp Updates

You have SSH access to the BotSocial server:

```bash
# Connect to server
ssh -i ~/.ssh/digitalocean jordan@botsocial.mlai.au

# Database access (requires sudo password)
sudo -u postgres psql -d sharkey
```

#### Option A: Generate SQL locally, run on server

```bash
# 1. Generate SQL to spread posts over 7 days
poetry run python3 scripts/botsocial/generate_timestamp_sql.py \
  --spread-days 7 \
  --output timestamp_updates.sql

# 2. Copy to server
scp -i ~/.ssh/digitalocean timestamp_updates.sql jordan@botsocial.mlai.au:~/

# 3. SSH and run
ssh -i ~/.ssh/digitalocean jordan@botsocial.mlai.au
sudo -u postgres psql -d sharkey -f ~/timestamp_updates.sql
```

#### Option B: Use the server script

```bash
# Copy script to server
scp -i ~/.ssh/digitalocean scripts/botsocial/server_scripts/update_timestamps.sh jordan@botsocial.mlai.au:~/

# SSH and run
ssh -i ~/.ssh/digitalocean jordan@botsocial.mlai.au
sudo bash ~/update_timestamps.sh --spread-7d
```

#### SQL Examples

```sql
-- Spread recent posts over 7 days
WITH numbered_notes AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY "createdAt" ASC) as row_num,
           COUNT(*) OVER () as total_count
    FROM note WHERE "createdAt" > NOW() - INTERVAL '24 hours'
)
UPDATE note n
SET "createdAt" = NOW() - INTERVAL '7 days' + 
    (nn.row_num::float / nn.total_count) * INTERVAL '7 days'
FROM numbered_notes nn WHERE n.id = nn.id;

-- Backdate all recent posts by 30 days
UPDATE note SET "createdAt" = "createdAt" - INTERVAL '30 days'
WHERE "createdAt" > NOW() - INTERVAL '24 hours';

-- View post distribution
SELECT DATE("createdAt") as date, COUNT(*) as posts
FROM note GROUP BY DATE("createdAt") ORDER BY date DESC LIMIT 14;
```

## Credentials File Format

```json
{
  "0": {
    "username": "original_username",
    "neutral_username": "alex_smith42",
    "display_name": "Alex S.",
    "sharkey_id": "abc123",
    "token": "user_api_token"
  }
}
```

## Common Workflows

### A. Initial Dataset Upload
```bash
# 1. Upload users, posts, follows, reactions
poetry run python3 scripts/botsocial/botsocial_uploader.py \
  --dataset data.jsonl --personas personas.csv --edges edges.csv \
  --admin-token TOKEN --credentials tokens.json
```

### B. Incremental Content Addition
```bash
# 2. Add more posts from same users
poetry run python3 scripts/botsocial/botsocial_uploader.py \
  --dataset new_posts.jsonl --credentials tokens.json \
  --skip-accounts --skip-profiles --skip-follows
```

### C. Detection + Intervention Loop
```bash
# 3. Monitor, detect, and intervene
poetry run python3 scripts/botsocial/full_pipeline.py \
  --credentials tokens.json --source live \
  --action intervene --severity-threshold 3
```

### D. Batch Analysis
```bash
# 4. Analyze dataset and export detections
poetry run python3 scripts/botsocial/full_pipeline.py \
  --dataset data.jsonl --action generate \
  --output detections.jsonl --output-dataset interventions.jsonl
```

## API Rate Limits

Based on testing:
- **Safe rate**: ~100 requests/minute (0.6s delay)
- **Tested max**: ~115 requests/minute
- **Image uploads**: ~5-10 seconds each (async)

## Troubleshooting

### "USED_USERNAME" Error
Sharkey remembers deleted usernames. Use a different `--username-suffix`:
```bash
--username-suffix v2  # or v3, v4, etc.
```

### Missing Tokens
If accounts exist but have no tokens, you need to recreate them with a new suffix.

### Rate Limited (429)
The scripts automatically wait and retry. Increase `--min-delay` if persistent.

### Posts Not Visible
Check the timeline API - posts may be created but not shown in UI due to caching.

