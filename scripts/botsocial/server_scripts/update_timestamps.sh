#!/bin/bash
# =============================================================================
# Sharkey Timestamp Update Script
# =============================================================================
#
# This script updates post timestamps in the Sharkey database to create
# a more realistic timeline spread.
#
# Usage (on server):
#   sudo bash update_timestamps.sh [OPTION]
#
# Options:
#   --spread-7d    Spread recent posts over last 7 days
#   --spread-30d   Spread recent posts over last 30 days
#   --backdate N   Backdate recent posts by N days
#   --dry-run      Show what would be changed without changing
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SPREAD_DAYS=7
DRY_RUN=false
ACTION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spread-7d)
            ACTION="spread"
            SPREAD_DAYS=7
            shift
            ;;
        --spread-30d)
            ACTION="spread"
            SPREAD_DAYS=30
            shift
            ;;
        --backdate)
            ACTION="backdate"
            SPREAD_DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ACTION" ]; then
    echo "Usage: sudo bash update_timestamps.sh [--spread-7d|--spread-30d|--backdate N] [--dry-run]"
    exit 1
fi

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Sharkey Timestamp Update${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Backup reminder
echo -e "${YELLOW}WARNING: This will modify the database!${NC}"
echo -e "${YELLOW}Make sure you have a backup:${NC}"
echo "  sudo -u postgres pg_dump sharkey > sharkey-backup-\$(date +%Y%m%d).sql"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Show current state
echo "Current post distribution:"
sudo -u postgres psql -d sharkey -c "
SELECT 
    DATE(\"createdAt\") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE(\"createdAt\")
ORDER BY post_date DESC
LIMIT 10;
"

# Count recent posts
RECENT_COUNT=$(sudo -u postgres psql -d sharkey -t -c "
SELECT COUNT(*) FROM note WHERE \"createdAt\" > NOW() - INTERVAL '2 hours';
")
echo ""
echo "Posts in last 2 hours: $RECENT_COUNT"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Would apply: $ACTION with $SPREAD_DAYS days"
    exit 0
fi

# Confirm
read -p "Continue with $ACTION ($SPREAD_DAYS days)? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Apply changes
if [ "$ACTION" = "spread" ]; then
    echo "Spreading posts over $SPREAD_DAYS days..."
    
    sudo -u postgres psql -d sharkey -c "
    WITH numbered_notes AS (
        SELECT 
            id,
            ROW_NUMBER() OVER (ORDER BY \"createdAt\" ASC) as row_num,
            COUNT(*) OVER () as total_count
        FROM note
        WHERE \"createdAt\" > NOW() - INTERVAL '2 hours'
    )
    UPDATE note n
    SET \"createdAt\" = NOW() - INTERVAL '$SPREAD_DAYS days' + (
        (nn.row_num::float / GREATEST(nn.total_count, 1)) * INTERVAL '$SPREAD_DAYS days'
    )
    FROM numbered_notes nn
    WHERE n.id = nn.id;
    "
    
elif [ "$ACTION" = "backdate" ]; then
    echo "Backdating posts by $SPREAD_DAYS days..."
    
    sudo -u postgres psql -d sharkey -c "
    UPDATE note
    SET \"createdAt\" = \"createdAt\" - INTERVAL '$SPREAD_DAYS days'
    WHERE \"createdAt\" > NOW() - INTERVAL '2 hours';
    "
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""

# Show new state
echo "New post distribution:"
sudo -u postgres psql -d sharkey -c "
SELECT 
    DATE(\"createdAt\") as post_date,
    COUNT(*) as post_count
FROM note
GROUP BY DATE(\"createdAt\")
ORDER BY post_date DESC
LIMIT 10;
"

