
-- =============================================================================
-- Spread posts over 7 days
-- Generated: 2025-11-26T16:47:56.468867
-- =============================================================================

-- First, let's see what we're working with
SELECT 
    COUNT(*) as total_posts,
    MIN("createdAt") as earliest,
    MAX("createdAt") as latest
FROM note
WHERE "createdAt" > NOW() - INTERVAL '48 hours';

-- Spread posts evenly over 7 days
-- Posts will be ordered by their original creation time
WITH numbered_notes AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY "createdAt" ASC) as row_num,
        COUNT(*) OVER () as total_count
    FROM note
    WHERE "createdAt" > NOW() - INTERVAL '48 hours'
)
UPDATE note n
SET "createdAt" = '2025-11-19 16:47:56'::timestamp + (
    (nn.row_num::float / GREATEST(nn.total_count, 1)) * INTERVAL '7 days'
)
FROM numbered_notes nn
WHERE n.id = nn.id;

-- Verify the change
SELECT 
    DATE("createdAt") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE("createdAt")
ORDER BY post_date DESC
LIMIT 14;
