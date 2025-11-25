#!/usr/bin/env python3
"""Monitor lexicon generation progress."""

import time
from pathlib import Path
import yaml

def get_counts():
    """Get current token counts from all lexicon files."""
    counts = {}
    for file in sorted(Path("configs/lexicons").glob("*.yaml")):
        try:
            with open(file) as f:
                data = yaml.safe_load(f) or {}
            req = len(data.get("required", []))
            opt = len(data.get("optional", []))
            counts[file.stem] = (req, opt)
        except Exception as e:
            counts[file.stem] = (0, 0)
    return counts

def main():
    """Monitor and display progress."""
    print("Lexicon Generation Progress Monitor")
    print("=" * 80)
    print("Target: 200 required, 1000 optional per collection")
    print("=" * 80)
    
    last_counts = {}
    iteration = 0
    
    while True:
        iteration += 1
        counts = get_counts()
        
        print(f"\n[Iteration {iteration}] {time.strftime('%H:%M:%S')}")
        print(f"{'Collection':<20} {'Required':>10} {'Optional':>10} {'Total':>10} {'Progress':>12}")
        print("-" * 80)
        
        for name in sorted(counts.keys()):
            req, opt = counts[name]
            total = req + opt
            req_pct = min(100, (req / 200) * 100)
            opt_pct = min(100, (opt / 1000) * 100)
            avg_pct = (req_pct + opt_pct) / 2
            
            # Show delta from last check
            delta_str = ""
            if name in last_counts:
                last_req, last_opt = last_counts[name]
                delta_req = req - last_req
                delta_opt = opt - last_opt
                if delta_req > 0 or delta_opt > 0:
                    delta_str = f" (+{delta_req}/{delta_opt})"
            
            print(f"{name:<20} {req:>10} {opt:>10} {total:>10} {avg_pct:>10.1f}%{delta_str}")
        
        last_counts = counts
        
        # Check if all done
        all_done = all(req >= 200 and opt >= 1000 for req, opt in counts.values())
        if all_done:
            print("\n✅ All collections reached target!")
            break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

