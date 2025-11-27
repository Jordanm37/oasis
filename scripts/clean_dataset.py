#!/usr/bin/env python3
"""
Dataset cleaning and re-imputation script.

Detects unimputed label tokens in JSONL datasets and either:
1. Reports them for re-imputation (--detect-only)
2. Re-imputes them using the static bank fallback (--impute)
3. Strips them as a last resort (--strip)

Usage:
    # Detect only (recommended first step)
    poetry run python3 scripts/clean_dataset.py <input.jsonl> --detect-only

    # Re-impute using static bank
    poetry run python3 scripts/clean_dataset.py <input.jsonl> --impute --static-bank <bank.yaml>

    # Strip as last resort
    poetry run python3 scripts/clean_dataset.py <input.jsonl> --strip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.imputation.utils import (LABEL_TOKEN_PATTERN, StaticBank,
                                    extract_label_tokens)

# Patterns that indicate unimputed content
ISSUE_PATTERNS = {
    "unimputed_label_token": re.compile(r'#?LBL:\s*[A-Z_]+'),
    "bracketed_label_token": re.compile(r'<LBL:'),
    "json_markdown_block": re.compile(r'```'),
    "replacement_placeholder": re.compile(r'\{"replacement":'),
}

# Cleaning patterns for strip mode
STRIP_PATTERNS = [
    # Label tokens (various formats)
    (r'\s*#?LBL:\s*[A-Z_]+[^.!?\s]*', ''),         # LBL:TOKEN or #LBL: TOKEN
    (r'\s*LBL:[A-Z_]+\s*', ''),                    # LBL:TOKEN
    (r'\s*<LBL:[A-Z_]+>\s*', ''),                  # <LBL:TOKEN>
    (r'\s*\[LBL:[A-Z_]+\]\s*', ''),                # [LBL:TOKEN]
    # JSON placeholders from RAG imputer
    (r'\s*```json\s*', ''),                        # ```json marker
    (r'\s*```\s*', ''),                            # closing ``` marker
    (r'\s*\{"replacement":\s*"[^"]*"\}\s*', ''),   # {"replacement": "..."}
    (r'\s*\{"replacement":[^}]+\}\s*', ''),        # {"replacement": ...}
    # Trailing whitespace cleanup
    (r'\s+$', ''),
    (r'^\s+', ''),
    (r'\s{2,}', ' '),
]


def check_for_issues(content: str) -> Dict[str, bool]:
    """Return dict of issue types found in content."""
    issues = {}
    for issue_name, pattern in ISSUE_PATTERNS.items():
        if pattern.search(content):
            issues[issue_name] = True
    return issues


def get_content_field(record: Dict[str, Any]) -> Tuple[str, str]:
    """Return (content, field_name) from record."""
    if 'content' in record:
        return record.get('content', ''), 'content'
    if 'text' in record:
        return record.get('text', ''), 'text'
    return '', ''


def strip_content(content: str) -> str:
    """Apply all strip patterns to content string."""
    cleaned = content
    for pattern, replacement in STRIP_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def impute_content(
    content: str,
    static_bank: StaticBank,
    seed: int,
    record_id: int,
) -> str:
    """Replace label tokens using static bank deterministic selection."""
    occurrence: Dict[str, int] = defaultdict(int)

    def replace_match(m: re.Match[str]) -> str:
        token = m.group(0)
        # Normalize to LBL:TOKEN format
        if token.startswith('#'):
            token = token[1:].strip()
        if token.startswith('<') and token.endswith('>'):
            token = token[1:-1]
        # Get deterministic replacement
        occ = occurrence[token]
        occurrence[token] += 1
        replacement = static_bank.deterministic_choice(token, seed, record_id, occ)
        return replacement

    # Replace both formats: LBL:TOKEN and <LBL:TOKEN>
    new_content = LABEL_TOKEN_PATTERN.sub(replace_match, content)
    # Also handle #LBL: TOKEN format
    new_content = re.sub(r'#?LBL:\s*[A-Z_]+', replace_match, new_content)
    return new_content


def process_dataset(
    input_path: Path,
    output_path: Optional[Path],
    mode: str,
    static_bank: Optional[StaticBank] = None,
    seed: int = 314159,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process dataset with specified mode."""
    
    records: List[Dict[str, Any]] = []
    issues_by_type: Dict[str, int] = defaultdict(int)
    issues_by_record: List[Dict[str, Any]] = []
    total_records = 0
    records_with_issues = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue
            
            total_records += 1
            content, field_name = get_content_field(record)
            issues = check_for_issues(content)
            
            if issues:
                records_with_issues += 1
                for issue_type in issues:
                    issues_by_type[issue_type] += 1
                
                # Store issue details for reporting
                record_id = record.get('post_id') or record.get('id') or f"line_{line_num}"
                issues_by_record.append({
                    "line": line_num,
                    "record_id": record_id,
                    "issues": list(issues.keys()),
                    "tokens": extract_label_tokens(content),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                })
                
                # Process based on mode
                if mode == "impute" and static_bank:
                    # Extract numeric ID for deterministic imputation
                    numeric_id = line_num
                    if isinstance(record_id, str):
                        match = re.search(r'\d+', record_id)
                        if match:
                            numeric_id = int(match.group())
                    record[field_name] = impute_content(content, static_bank, seed, numeric_id)
                elif mode == "strip":
                    record[field_name] = strip_content(content)
                # else mode == "detect" - no changes
            
            records.append(record)

    # Summary
    summary = {
        "total_records": total_records,
        "records_with_issues": records_with_issues,
        "issues_by_type": dict(issues_by_type),
        "issue_details": issues_by_record if mode == "detect" else [],
    }

    # Write output if not detect-only and not dry-run
    if mode != "detect" and not dry_run and output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Verify cleaning
        remaining_issues = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                content, _ = get_content_field(record)
                if check_for_issues(content):
                    remaining_issues += 1
        summary["remaining_issues"] = remaining_issues

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Detect and fix unimputed label tokens in datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect issues only (recommended first step)
  poetry run python3 scripts/clean_dataset.py data/runs/dataset.jsonl --detect-only

  # Re-impute using static bank
  poetry run python3 scripts/clean_dataset.py data/runs/dataset.jsonl --impute \\
      --static-bank data/label_tokens_static_bank.yaml

  # Strip as last resort (loses semantic content)
  poetry run python3 scripts/clean_dataset.py data/runs/dataset.jsonl --strip
        """
    )
    parser.add_argument('input', type=Path, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=Path, help='Output file (default: overwrite input)')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--detect-only', action='store_true', 
                           help='Only detect issues, do not modify (recommended first)')
    mode_group.add_argument('--impute', action='store_true',
                           help='Re-impute using static bank (requires --static-bank)')
    mode_group.add_argument('--strip', action='store_true',
                           help='Strip unimputed tokens (last resort, loses semantic content)')
    
    parser.add_argument('--static-bank', type=Path, 
                       default=Path('data/label_tokens_static_bank.yaml'),
                       help='Path to static phrase bank YAML (for --impute mode)')
    parser.add_argument('--seed', type=int, default=314159,
                       help='Seed for deterministic imputation')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without writing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed issue information')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine mode
    if args.detect_only:
        mode = "detect"
    elif args.impute:
        mode = "impute"
    else:
        mode = "strip"

    # Load static bank if imputing
    static_bank = None
    if mode == "impute":
        if not args.static_bank.exists():
            print(f"Error: Static bank not found: {args.static_bank}", file=sys.stderr)
            print("Use --static-bank to specify the path.", file=sys.stderr)
            return 1
        static_bank = StaticBank.load_simple_yaml(args.static_bank)
        print(f"[impute] Loaded static bank from {args.static_bank}")

    output_path = args.output or args.input

    # Process
    print(f"\n=== Processing: {args.input} ===")
    print(f"Mode: {mode}")
    
    summary = process_dataset(
        input_path=args.input,
        output_path=output_path,
        mode=mode,
        static_bank=static_bank,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Records with issues: {summary['records_with_issues']}")
    
    if summary['issues_by_type']:
        print("\nIssues by type:")
        for issue_type, count in sorted(summary['issues_by_type'].items(), key=lambda x: -x[1]):
            print(f"  - {issue_type}: {count}")
    
    if mode == "detect" and args.verbose and summary['issue_details']:
        print("\n=== Issue Details ===")
        for issue in summary['issue_details'][:20]:  # Limit to first 20
            print(f"\nLine {issue['line']} ({issue['record_id']}):")
            print(f"  Issues: {', '.join(issue['issues'])}")
            if issue['tokens']:
                print(f"  Tokens: {', '.join(issue['tokens'])}")
            print(f"  Preview: {issue['content_preview']}")
        if len(summary['issue_details']) > 20:
            print(f"\n... and {len(summary['issue_details']) - 20} more records with issues")

    if mode == "detect":
        if summary['records_with_issues'] > 0:
            print(f"\n⚠ Found {summary['records_with_issues']} records with unimputed tokens.")
            print("\nRecommended next steps:")
            print("  1. Re-run RAG imputation on the source database:")
            print("     poetry run python3 scripts/run_rag_imputer.py --db <db_path> --reset")
            print("  2. Re-export the dataset:")
            print("     poetry run python3 scripts/build_dataset.py --db <db_path> --out <output.jsonl>")
            print("  3. Or use static bank imputation as fallback:")
            print(f"     poetry run python3 scripts/clean_dataset.py {args.input} --impute")
        else:
            print("\n✓ No issues found - dataset is clean!")
        return 0

    if args.dry_run:
        print(f"\nDry run complete. No files modified.")
        return 0

    # Check result
    remaining = summary.get('remaining_issues', 0)
    if remaining == 0:
        print(f"\n✓ Cleaned dataset written to: {output_path}")
        print("✓ Verification passed: No issues remaining")
    else:
        print(f"\n⚠ Warning: {remaining} records still have issues after {mode}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
