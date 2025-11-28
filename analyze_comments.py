"""Analyze comment distribution in the database."""
import modal

app = modal.App("analyze-db")
volume = modal.Volume.from_name("oasis-data", create_if_missing=False)

@app.function(volumes={"/app/data/runs": volume}, timeout=120)
def analyze_comments(run_id: str = "prod_5k_v3"):
    import sqlite3
    import os
    
    db_path = f"/app/data/runs/{run_id}.db"
    if not os.path.exists(db_path):
        return {"error": f"Database not found: {db_path}"}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get comment count distribution per post
    cur.execute("""
        SELECT 
            p.post_id,
            COUNT(c.comment_id) as comment_count
        FROM post p
        LEFT JOIN comment c ON p.post_id = c.post_id
        GROUP BY p.post_id
        ORDER BY comment_count DESC
    """)
    results = cur.fetchall()
    
    # Calculate statistics
    comment_counts = [r[1] for r in results]
    total_posts = len(comment_counts)
    total_comments = sum(comment_counts)
    
    # Distribution buckets
    buckets = {
        "0 comments": 0,
        "1 comment": 0,
        "2-5 comments": 0,
        "6-10 comments": 0,
        "11-20 comments": 0,
        "21-50 comments": 0,
        "50+ comments": 0
    }
    
    for count in comment_counts:
        if count == 0:
            buckets["0 comments"] += 1
        elif count == 1:
            buckets["1 comment"] += 1
        elif count <= 5:
            buckets["2-5 comments"] += 1
        elif count <= 10:
            buckets["6-10 comments"] += 1
        elif count <= 20:
            buckets["11-20 comments"] += 1
        elif count <= 50:
            buckets["21-50 comments"] += 1
        else:
            buckets["50+ comments"] += 1
    
    # Top 10 most commented posts
    top_10 = results[:10]
    
    # Stats
    avg_comments = total_comments / total_posts if total_posts > 0 else 0
    max_comments = max(comment_counts) if comment_counts else 0
    posts_with_comments = sum(1 for c in comment_counts if c > 0)
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"COMMENT DISTRIBUTION ANALYSIS: {run_id}")
    print(f"{'='*60}")
    print(f"\nTotal posts: {total_posts:,}")
    print(f"Total comments: {total_comments:,}")
    print(f"Posts with comments: {posts_with_comments:,} ({100*posts_with_comments/total_posts:.1f}%)")
    print(f"Posts without comments: {total_posts - posts_with_comments:,} ({100*(total_posts-posts_with_comments)/total_posts:.1f}%)")
    print(f"Average comments/post: {avg_comments:.2f}")
    print(f"Max comments on single post: {max_comments}")
    
    print(f"\n{'='*40}")
    print("DISTRIBUTION BUCKETS:")
    print(f"{'='*40}")
    for bucket, count in buckets.items():
        pct = 100 * count / total_posts if total_posts > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{bucket:20s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    print(f"\n{'='*40}")
    print("TOP 10 MOST COMMENTED POSTS:")
    print(f"{'='*40}")
    for post_id, count in top_10:
        print(f"Post {post_id}: {count} comments")
    
    return {
        "total_posts": total_posts,
        "total_comments": total_comments,
        "posts_with_comments": posts_with_comments,
        "avg_comments": avg_comments,
        "max_comments": max_comments,
        "distribution": buckets
    }


@app.local_entrypoint()
def main(run_id: str = "prod_5k_v3"):
    result = analyze_comments.remote(run_id)
    print(f"\nResult: {result}")

