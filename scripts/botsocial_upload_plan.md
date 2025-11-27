# BotSocial Dataset Upload Plan

## ✅ Executive Summary

**Dataset**: `prod_100p_100s_65_35_dataset.jsonl`
- **Posts**: 1,363 total (481 root posts, 882 replies)
- **Users**: 100 unique users (user_id: 0-99)
- **Personas**: 13 types (benign, recovery_support, bullying, extremist, etc.)
- **Timestamps**: Relative (0-16), need realistic timing

**Additional Data Available**:
- `personas_prod_100p_100s_65_35.csv` - Full persona details with usernames, descriptions
- `edges_prod_100p_100s_65_35.csv` - **288 follow relationships** already defined!
- `manifest_prod_100p_100s_65_35.yaml` - Population breakdown

**Target**: BotSocial Sharkey instance at `https://botsocial.mlai.au`

---

## 🎯 ACTION CHECKLIST

### Prerequisites (Get from Tom)
- [ ] **Tom's admin API token** - Required for account creation
  - Tom: Go to https://botsocial.mlai.au → Settings → API → Generate Token
  - Select permissions: `read:account, write:notes, admin:*`
  - Send token securely

### Once You Have Admin Token
```bash
# 1. Test connection
poetry run python3 scripts/botsocial/test_connection.py --admin-token "YOUR_TOKEN"

# 2. Run full upload (~5-6 hours)
poetry run python3 scripts/botsocial/upload_oasis_dataset.py \
    --admin-token "YOUR_TOKEN" \
    --dataset data/runs/prod_100p_100s_65_35_dataset.jsonl \
    --personas data/runs/personas_prod_100p_100s_65_35.csv \
    --edges data/runs/edges_prod_100p_100s_65_35.csv
```

---

---

## Phase 1: Account Creation & Profile Setup

### 1.1 Account Creation Strategy

**Challenge**: Need to create 100 user accounts with API tokens.

**Solution**: Use admin API endpoint `admin/accounts/create`

```python
# Endpoint: POST /api/admin/accounts/create
{
    "i": "ADMIN_TOKEN",
    "username": "user_0",
    "password": "SecurePass123!"
}
```

**Naming Convention**:
| User ID | Username Format | Example |
|---------|-----------------|---------|
| 0-99    | `sim_user_{id}` | `sim_user_0`, `sim_user_42` |

### 1.2 Display Name Generation by Persona

Generate realistic display names based on `persona_id`:

```python
PERSONA_NAME_TEMPLATES = {
    "benign": ["Alex", "Jordan", "Taylor", "Casey", "Morgan"],  # Neutral names
    "recovery_support": ["Hope Coach", "Recovery Guide", "Wellness Advocate"],
    "bullying": ["EdgeLord", "TruthBomb", "RealTalk"],  # Generic handles
    "extremist": ["FreedomFighter", "PatriotX", "Awakened"],
    "incel_misogyny": ["RedPillTruth", "BlackPillReality"],
    "ed_risk": ["FitGoals", "DisciplineDaily"],
    "pro_ana": ["ThinspoBabe", "GoalGetter"],
    "misinfo": ["TruthSeeker", "WakeUpNow"],
    "conspiracy": ["DeepTruth", "QuestionAll"],
    "hate_speech": ["NationFirst", "TruePatriot"],
    "gamergate": ["GamerGate99", "AntiWoke"],
    "trad": ["TraditionalValues", "HomesteadMom"],
    "alpha": ["AlphaMindset", "HighValue"],
}
```

**Implementation**:
```python
def generate_display_name(user_id: str, persona_id: str) -> str:
    """Generate a display name based on persona."""
    base_names = PERSONA_NAME_TEMPLATES.get(persona_id, ["User"])
    name = random.choice(base_names)
    suffix = hashlib.md5(user_id.encode()).hexdigest()[:4]
    return f"{name}_{suffix}"
```

### 1.3 Profile Image Strategy

**Avatar Generation**: Use Picsum Photos with deterministic seeds

```python
def get_avatar_url(user_id: str, persona_id: str) -> str:
    """Generate deterministic avatar URL based on user/persona."""
    seed = f"avatar-{user_id}-{persona_id}"
    return f"https://picsum.photos/seed/{seed}/400/400"

def get_banner_url(user_id: str, persona_id: str) -> str:
    """Generate deterministic banner URL."""
    seed = f"banner-{user_id}-{persona_id}"
    return f"https://picsum.photos/seed/{seed}/1500/500"
```

**Alternative - Persona-Themed Images**:
For more realistic personas, use themed image categories:

| Persona | Avatar Theme | Banner Theme |
|---------|--------------|--------------|
| benign | Generic portraits | Nature/lifestyle |
| recovery_support | Warm, supportive imagery | Sunrise/hope imagery |
| extremist | Abstract/patriotic | Dark/intense |
| trad | Traditional/classic | Farmhouse/pastoral |

**Upload Workflow** (async - 204 response):
```python
# 1. Trigger upload
requests.post(f"{API_BASE}/drive/files/upload-from-url", json={
    "i": token, "url": avatar_url, "isSensitive": False
})

# 2. Wait for processing (3-5 seconds)
time.sleep(3)

# 3. Query for new file ID
files = requests.post(f"{API_BASE}/drive/files", json={"i": token, "limit": 1}).json()
avatar_id = files[0]['id']

# 4. Set avatar
requests.post(f"{API_BASE}/i/update", json={"i": token, "avatarId": avatar_id})
```

---

## Phase 2: Realistic Post Timing

### 2.1 The Timestamp Problem

**Current State**: Timestamps are relative integers (0-16)
**Goal**: Create realistic-looking temporal patterns

### 2.2 Timing Strategy Options

#### Option A: Compressed Simulation (Recommended)
Create posts over a short real-world period (hours) with realistic-looking gaps:

```python
def calculate_post_delay(current_timestamp: str, prev_timestamp: str) -> float:
    """Calculate delay between posts based on relative timestamps."""
    curr = int(current_timestamp)
    prev = int(prev_timestamp) if prev_timestamp else 0
    
    timestamp_diff = curr - prev
    
    if timestamp_diff == 0:
        # Same timestamp = near-simultaneous (within minutes)
        return random.uniform(5, 60)  # 5-60 seconds
    else:
        # Different timestamp = larger gap
        return random.uniform(60, 180) * timestamp_diff  # 1-3 min per timestamp unit
```

**Total Time Estimate**: ~45-90 minutes for full dataset

#### Option B: Batch Upload with Post-Dating
Upload all posts quickly, then use Sharkey's internal timestamp ordering:

- Sharkey orders by `createdAt` timestamp
- Posts within same "timestamp" group uploaded in batches
- Result: Natural feed ordering matches intended timeline

#### Option C: Real-Time Simulation
Spread posts over actual time periods (days/weeks):
- Most realistic appearance
- Impractical for testing (weeks to complete)
- Only use if simulating real user behavior

### 2.3 Recommended Approach: Smart Batching

```python
def group_posts_by_timestamp(posts: List[dict]) -> dict:
    """Group posts by timestamp for batch processing."""
    groups = defaultdict(list)
    for post in posts:
        groups[post['timestamp']].append(post)
    return dict(sorted(groups.items(), key=lambda x: int(x[0])))

# Process each timestamp group with realistic delays
for timestamp, posts in sorted_groups.items():
    for post in posts:
        create_post(post)
        time.sleep(random.uniform(2, 5))  # Small delay within group
    
    # Larger delay between timestamp groups
    time.sleep(random.uniform(30, 90))
```

---

## Phase 3: Social Interactions

### 3.1 Follow Relationships

**Current Dataset**: No explicit follow data.

**Strategy**: Create plausible follow networks based on interactions:

```python
def build_follow_graph(posts: List[dict]) -> Dict[str, Set[str]]:
    """Build follow relationships from reply patterns."""
    follows = defaultdict(set)
    
    for post in posts:
        if post['parent_id']:
            # Users who reply likely follow the original poster
            replier = post['user_id']
            # Find original post's user
            original_post = find_post(post['parent_id'])
            if original_post:
                follows[replier].add(original_post['user_id'])
    
    return follows
```

**API Endpoint**:
```python
# POST /api/following/create
requests.post(f"{API_BASE}/following/create", json={
    "i": follower_token,
    "userId": followee_sharkey_id
})
```

### 3.2 Reactions (Likes)

**Current Dataset**: No explicit reaction data.

**Strategy**: Generate synthetic reactions based on persona alignment:

```python
PERSONA_REACTION_AFFINITIES = {
    "benign": {"benign": 0.7, "recovery_support": 0.8, "bullying": 0.1},
    "recovery_support": {"recovery_support": 0.9, "benign": 0.6, "pro_ana": 0.05},
    "bullying": {"bullying": 0.4, "extremist": 0.3, "benign": 0.2},
    # ... etc
}

def should_react(viewer_persona: str, post_persona: str) -> bool:
    """Determine if viewer should react to post based on persona alignment."""
    affinity = PERSONA_REACTION_AFFINITIES.get(viewer_persona, {}).get(post_persona, 0.1)
    return random.random() < affinity
```

**API Endpoint**:
```python
# POST /api/notes/reactions/create
requests.post(f"{API_BASE}/notes/reactions/create", json={
    "i": user_token,
    "noteId": note_id,
    "reaction": "❤️"  # or "👍", "🎉", etc.
})
```

### 3.3 Reposts (Renotes)

**Strategy**: High-engagement posts get reposted by aligned users:

```python
def select_posts_for_repost(posts: List[dict], count: int = 50) -> List[dict]:
    """Select posts likely to be reposted."""
    # Prioritize extremist/viral content patterns
    viral_personas = ["extremist", "misinfo", "conspiracy"]
    candidates = [p for p in posts if p['persona_id'] in viral_personas and not p['parent_id']]
    return random.sample(candidates, min(count, len(candidates)))
```

**API Endpoint**:
```python
# POST /api/notes/create (with renoteId for quote, or just renoteId for simple repost)
requests.post(f"{API_BASE}/notes/create", json={
    "i": user_token,
    "renoteId": original_note_id,
    # "text": "Quote comment"  # Optional for quote-repost
})
```

---

## Phase 4: Reply Threading

### 4.1 Thread Structure

The dataset uses `thread_id` and `parent_id` for threading:
- `parent_id = null` → Root post
- `parent_id = "p_23"` → Reply to post p_23

**Implementation**:
```python
def upload_with_threading(posts: List[dict]) -> Dict[str, str]:
    """Upload posts maintaining thread structure."""
    post_id_mapping = {}  # Maps dataset post_id to Sharkey note_id
    
    # Sort: root posts first, then replies ordered by timestamp
    root_posts = [p for p in posts if p['parent_id'] is None]
    replies = [p for p in posts if p['parent_id'] is not None]
    
    # Upload root posts first
    for post in sorted(root_posts, key=lambda x: int(x['timestamp'])):
        note_id = create_note(post)
        post_id_mapping[post['post_id']] = note_id
        time.sleep(RATE_LIMIT_DELAY)
    
    # Upload replies (sorted by timestamp)
    for reply in sorted(replies, key=lambda x: int(x['timestamp'])):
        parent_note_id = post_id_mapping.get(reply['parent_id'])
        if parent_note_id:
            note_id = create_note_reply(reply, parent_note_id)
            post_id_mapping[reply['post_id']] = note_id
            time.sleep(RATE_LIMIT_DELAY)
    
    return post_id_mapping
```

---

## Phase 5: Rate Limiting & Error Handling

### 5.1 Sharkey Rate Limits

From instance metadata:
- **Default**: 10 requests/minute
- **Note creation**: 5 requests/minute
- **Timeline**: 15 requests/minute
- **Admin**: 30 requests/minute

### 5.2 Rate Limit Strategy

```python
class RateLimitedClient:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 12  # 5 req/min = 12 sec between requests
    
    def wait_for_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def create_note(self, token: str, text: str, reply_id: str = None) -> str:
        self.wait_for_rate_limit()
        
        data = {"i": token, "text": text, "visibility": "public"}
        if reply_id:
            data["replyId"] = reply_id
        
        response = requests.post(f"{API_BASE}/notes/create", json=data)
        
        if response.status_code == 429:
            # Rate limited - exponential backoff
            time.sleep(60)
            return self.create_note(token, text, reply_id)
        
        return response.json()['createdNote']['id']
```

### 5.3 Estimated Upload Time

| Phase | Items | Time per Item | Total Time |
|-------|-------|---------------|------------|
| Account Creation | 100 | 2 sec | ~3-4 min |
| Profile Setup (avatars) | 100 | 8 sec | ~13-15 min |
| Post Upload | 1,363 | 12 sec | ~4.5 hours |
| Reactions | ~500 | 3 sec | ~25 min |
| Follows | ~300 | 3 sec | ~15 min |

**Total Estimated Time**: ~5-6 hours

### 5.4 Parallelization Options

To speed up, we can:
1. **Parallel account creation** (different tokens don't share rate limits)
2. **Batch uploads** with multiple admin tokens
3. **Skip profile images** initially (add later)

---

## Implementation Roadmap

### Step 1: Prerequisites (Tom Required)
- [ ] Get Tom's admin API token
- [ ] Verify admin token works with `admin/accounts/create`

### Step 2: Account Creation Script
- [ ] Create 100 accounts via admin API
- [ ] Store account info and passwords securely
- [ ] Generate API tokens for each account

### Step 3: Profile Setup Script
- [ ] Generate display names per persona
- [ ] Upload avatars (Picsum Photos)
- [ ] Upload banners
- [ ] Update user profiles

### Step 4: Post Upload Script
- [ ] Parse JSONL dataset
- [ ] Build thread dependency graph
- [ ] Upload root posts first (timestamp order)
- [ ] Upload replies with proper `replyId`
- [ ] Track post_id → note_id mapping

### Step 5: Interaction Generation Script
- [ ] Generate follow relationships from interactions
- [ ] Create follows via API
- [ ] Generate reactions based on persona affinity
- [ ] Optionally add reposts for viral content

### Step 6: Verification
- [ ] Verify post counts match
- [ ] Verify threading works correctly
- [ ] Spot-check reactions and follows
- [ ] Test timeline rendering

---

## Files to Create

```
scripts/
├── botsocial/
│   ├── __init__.py
│   ├── client.py          # API client with rate limiting
│   ├── accounts.py        # Account creation & token management
│   ├── profiles.py        # Profile setup & image upload
│   ├── posts.py           # Post/reply creation
│   ├── interactions.py    # Follows, reactions, reposts
│   ├── timing.py          # Realistic timing utilities
│   └── upload_dataset.py  # Main orchestration script
├── config/
│   └── botsocial_config.yaml  # Credentials & settings
└── data/
    └── account_tokens.json    # Generated tokens (gitignored)
```

---

## Quick Start Commands

```bash
# After implementation, run:

# 1. Create accounts (requires admin token)
poetry run python3 scripts/botsocial/accounts.py --admin-token "XXX" --count 100

# 2. Setup profiles with images
poetry run python3 scripts/botsocial/profiles.py --config config/botsocial_config.yaml

# 3. Upload dataset
poetry run python3 scripts/botsocial/upload_dataset.py \
    data/runs/prod_100p_100s_65_35_dataset.jsonl \
    --config config/botsocial_config.yaml

# 4. Generate interactions (optional)
poetry run python3 scripts/botsocial/interactions.py \
    --follows --reactions --config config/botsocial_config.yaml
```

---

## Next Steps

1. **Get Admin Token from Tom** - Essential first step
2. **Decide on timing strategy** - Compressed (hours) vs real-time (days)
3. **Decide on interaction depth** - Just posts vs full social graph
4. **Create implementation scripts** - Start with accounts.py

---

## Questions to Resolve

1. **Token Management**: How will Tom generate 100 user tokens? 
   - Option A: Server-side script (requires SSH access)
   - Option B: Admin API auto-generation (if supported)
   - Option C: Manual generation (impractical for 100 users)

2. **Image Hosting**: Use Picsum (external) or upload local images?

3. **Persistence**: Keep account credentials in version control or external secrets?

4. **Post Images**: Should posts have images? Dataset doesn't specify, but could add for realism.

---

## 🔍 Key Data Insights

### Persona Usernames (from CSV)
The personas CSV already has generated usernames like:
- `benign_0016_a2924e` - Friendly on-topic poster
- `gamer_0000_c40a70` - Culture war gatekeeper
- `recovery_0005_738893` - Recovery support mentor
- `extrem_0000_4bdd90` - Extremist agitator

These are the actual usernames to create on BotSocial!

### Follow Relationships (288 edges)
The edges CSV contains real follow relationships:
```csv
follower_id,followee_id
92,58
38,87
84,20
...
```
These map to user IDs (0-99) from the dataset.

### Population Breakdown (from manifest)
```yaml
persona_benign_mvp: 32      # 32% benign
persona_recovery_mvp: 32    # 32% recovery support  
persona_bullying_mvp: 6     # 6% bullying
# ... other harmful personas: 3 each (36%)
```

---

## 📂 Files Created

| File | Purpose |
|------|---------|
| `scripts/botsocial/__init__.py` | Module initialization |
| `scripts/botsocial/test_connection.py` | Test admin access |
| `scripts/botsocial/upload_oasis_dataset.py` | Main upload script |
| `scripts/botsocial_upload_plan.md` | This plan |

---

*Plan created: November 26, 2025*
*Dataset: prod_100p_100s_65_35_dataset.jsonl*
*Target: botsocial.mlai.au*
*Status: Ready - awaiting admin token from Tom*

