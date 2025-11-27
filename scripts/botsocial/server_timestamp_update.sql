-- =============================================================================
-- Sharkey Timestamp Update Script
-- =============================================================================
-- 
-- This script updates post timestamps in the Sharkey database.
-- Run this on the server: sudo -u postgres psql -d sharkey -f server_timestamp_update.sql
--
-- WARNING: This modifies the database directly. Backup first!
-- sudo -u postgres pg_dump sharkey > sharkey-backup-$(date +%Y%m%d).sql
--
-- =============================================================================

-- View current note structure
-- \d note

-- =============================================================================
-- OPTION 1: Spread posts over time (e.g., last 7 days)
-- =============================================================================

-- This spreads all posts from the last hour evenly over the past 7 days
-- Adjust the interval as needed

/*
UPDATE note
SET "createdAt" = "createdAt" - (
    (ROW_NUMBER() OVER (ORDER BY "createdAt" DESC) - 1) * INTERVAL '10 minutes'
)
WHERE "createdAt" > NOW() - INTERVAL '1 hour';
*/

-- =============================================================================
-- OPTION 2: Set specific timestamps based on a mapping
-- =============================================================================

-- Example: Update specific notes by ID
/*
UPDATE note SET "createdAt" = '2024-01-15 10:30:00'::timestamp WHERE id = 'note_id_here';
UPDATE note SET "createdAt" = '2024-01-15 11:00:00'::timestamp WHERE id = 'another_note_id';
*/

-- =============================================================================
-- OPTION 3: Spread posts by user (simulate realistic posting patterns)
-- =============================================================================

-- This creates a more realistic timeline where each user's posts are spread out
/*
WITH numbered_notes AS (
    SELECT 
        id,
        "userId",
        ROW_NUMBER() OVER (PARTITION BY "userId" ORDER BY "createdAt") as user_post_num,
        COUNT(*) OVER (PARTITION BY "userId") as user_total_posts
    FROM note
    WHERE "createdAt" > NOW() - INTERVAL '2 hours'
)
UPDATE note n
SET "createdAt" = NOW() - INTERVAL '7 days' + (
    (nn.user_post_num::float / GREATEST(nn.user_total_posts, 1)) * INTERVAL '7 days'
)
FROM numbered_notes nn
WHERE n.id = nn.id;
*/

-- =============================================================================
-- OPTION 4: Backdate all recent posts by X days
-- =============================================================================

-- Simple: Move all posts from last hour back by 30 days
/*
UPDATE note
SET "createdAt" = "createdAt" - INTERVAL '30 days'
WHERE "createdAt" > NOW() - INTERVAL '1 hour';
*/

-- =============================================================================
-- USEFUL QUERIES
-- =============================================================================

-- Count posts by date
SELECT 
    DATE("createdAt") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE("createdAt")
ORDER BY post_date DESC
LIMIT 20;

-- View recent posts with timestamps
SELECT 
    id,
    LEFT(text, 50) as text_preview,
    "userId",
    "createdAt"
FROM note
ORDER BY "createdAt" DESC
LIMIT 20;

-- Count posts by user
SELECT 
    "userId",
    COUNT(*) as post_count,
    MIN("createdAt") as first_post,
    MAX("createdAt") as last_post
FROM note
GROUP BY "userId"
ORDER BY post_count DESC
LIMIT 20;

