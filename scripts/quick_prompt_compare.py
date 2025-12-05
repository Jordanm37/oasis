#!/usr/bin/env python3
"""Quick baseline vs hybrid prompt comparison on existing test data."""

import asyncio
import json
import sqlite3
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from configs.llm_settings import (
    IMPUTATION_MAX_TOKENS,
    IMPUTATION_MODEL,
    IMPUTATION_PROVIDER,
    IMPUTATION_TEMPERATURE,
)
from orchestrator.model_provider import LLMProviderSettings, create_model_backend

PROMPTS = {
    "baseline": {
        "system": (
            "You replace placeholder label tokens (e.g., <LBL:INCEL_MISOGYNY>) "
            "inside social-media posts. Only replace the placeholder spans, keep the author's "
            "voice, and keep the rest of the text identical. Output JSON."
        ),
        "suffix": (
            'Return JSON of the form {"replacement": "..."} where the replacement is a short span '
            "that captures the label's intent in the author's voice."
        ),
        "include_persona": False,
    },
    "hybrid": {
        "system": (
            "You replace placeholder label tokens inside social-media posts. "
            "Rules: (1) Output 2-10 words, (2) STAY IN CHARACTER as the post author - never sound educational, "
            "(3) Never include the token text itself, (4) MATCH the toxicity/intensity level of the surrounding text, "
            "(5) Output JSON {\"replacement\": \"...\"}."
        ),
        "suffix": (
            'Return JSON {"replacement": "..."} with a phrase (2-10 words) that:\n'
            "- Sounds like something THIS USER would actually say\n"
            "- Fits naturally in the sentence\n"
            "- Matches the INTENSITY of the post (aggressive posts get aggressive replacements, subtle posts get subtle ones)\n"
            "- Expresses the label's sentiment authentically"
        ),
        "include_persona": False,
    },
    "hybrid_persona": {
        "system": (
            "You replace placeholder label tokens inside social-media posts. "
            "You will be given the author's persona/profile to help you stay in character. "
            "Rules: (1) Output 2-10 words, (2) STAY IN CHARACTER as the post author - never sound educational, "
            "(3) Never include the token text itself, (4) MATCH the toxicity/intensity level of the surrounding text, "
            "(5) Output JSON {\"replacement\": \"...\"}."
        ),
        "suffix": (
            'Return JSON {"replacement": "..."} with a phrase (2-10 words) that:\n'
            "- Sounds like something THIS USER would actually say based on their persona\n"
            "- Fits naturally in the sentence\n"
            "- Matches the INTENSITY of the post\n"
            "- Expresses the label's sentiment authentically"
        ),
        "include_persona": True,
    },
}


def extract_tokens(text):
    return re.findall(r'<?(LBL:[A-Z_]+)>?', text)


async def run_imputation(db_path, variant, prompts, personas_csv=None):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get user info if we need persona context
    user_info = {}
    if prompts.get("include_persona") and personas_csv:
        import csv
        # Load personas
        personas_by_username = {}
        with open(personas_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                personas_by_username[row.get("username", "")] = row

        # Map user_id -> persona via username
        cur.execute("SELECT user_id, name FROM user")
        for uid, name in cur.fetchall():
            if name in personas_by_username:
                user_info[uid] = personas_by_username[name]

    # Fetch posts with user_id
    cur.execute("""
        SELECT post_id AS id, user_id, content FROM post WHERE content LIKE '%LBL:%'
        UNION ALL
        SELECT comment_id AS id, user_id, content FROM comment WHERE content LIKE '%LBL:%'
    """)
    records = cur.fetchall()
    conn.close()

    print(f"  Found {len(records)} records with tokens")

    llm_settings = LLMProviderSettings(provider=IMPUTATION_PROVIDER, model_name=IMPUTATION_MODEL)
    backend = create_model_backend(llm_settings)
    backend.model_config_dict["temperature"] = float(IMPUTATION_TEMPERATURE)
    backend.model_config_dict["max_tokens"] = int(IMPUTATION_MAX_TOKENS)

    results = {}
    for rec in records:
        tokens = extract_tokens(rec["content"])
        if not tokens:
            continue
        token = tokens[0]

        # Build persona context if available
        persona_context = ""
        if prompts.get("include_persona") and rec["user_id"] in user_info:
            persona = user_info[rec["user_id"]]
            persona_context = (
                f"\n\nAuthor persona:\n"
                f"- Username: {persona.get('username', 'unknown')}\n"
                f"- Type: {persona.get('primary_label', 'unknown')}\n"
                f"- Bio: {persona.get('bio', '')[:150]}\n"
            )

        user_prompt = f"Original post:\n{rec['content']}{persona_context}\n\nPlaceholder token #1: {token}\n\n{prompts['suffix']}"
        messages = [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user_prompt}]
        try:
            response = await backend.arun(messages)
            content = str(getattr(response.choices[0].message, "content", ""))
            try:
                data = json.loads(content.strip())
                replacement = data.get("replacement", content)
            except json.JSONDecodeError:
                replacement = content.split("\n")[0][:100]
            results[rec["id"]] = {"text": rec["content"][:100], "token": token, "output": replacement}
        except Exception as e:
            results[rec["id"]] = {"text": rec["content"][:100], "token": token, "output": f"[ERROR: {e}]"}
    return results


async def main():
    db_path = Path("data/prompt_tests/prompt_comparison_000502.db")
    personas_csv = Path("data/prompt_tests/personas_prompt_comparison_000502.csv")

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print("Testing BASELINE prompt...")
    baseline = await run_imputation(db_path, "baseline", PROMPTS["baseline"])
    print(f"  Completed {len(baseline)} imputations")

    print("\nTesting HYBRID prompt...")
    hybrid = await run_imputation(db_path, "hybrid", PROMPTS["hybrid"])
    print(f"  Completed {len(hybrid)} imputations")

    print("\nTesting HYBRID + PERSONA prompt...")
    if personas_csv.exists():
        hybrid_persona = await run_imputation(db_path, "hybrid_persona", PROMPTS["hybrid_persona"], personas_csv)
        print(f"  Completed {len(hybrid_persona)} imputations")
    else:
        print(f"  Skipping - personas CSV not found: {personas_csv}")
        hybrid_persona = {}

    print("\n" + "="*80)
    print("BASELINE vs HYBRID vs HYBRID+PERSONA COMPARISON")
    print("="*80)

    for i, (rec_id, b_data) in enumerate(baseline.items(), 1):
        h_data = hybrid.get(rec_id, {})
        hp_data = hybrid_persona.get(rec_id, {})
        print(f"\n--- Example {i} ---")
        print(f"Token: {b_data['token']}")
        print(f"Original: {b_data['text']}...")
        print(f"  Baseline:       {b_data['output']}")
        print(f"  Hybrid:         {h_data.get('output', 'N/A')}")
        if hp_data:
            print(f"  Hybrid+Persona: {hp_data.get('output', 'N/A')}")
        b_len = len(b_data['output'].split())
        h_len = len(h_data.get('output', '').split())
        hp_len = len(hp_data.get('output', '').split()) if hp_data else 0
        words_str = f"baseline={b_len}, hybrid={h_len}"
        if hp_data:
            words_str += f", hybrid+persona={hp_len}"
        print(f"  Words: {words_str}")

    # Summary
    b_lengths = [len(d['output'].split()) for d in baseline.values()]
    h_lengths = [len(hybrid.get(k, {}).get('output', '').split()) for k in baseline.keys()]
    hp_lengths = [len(hybrid_persona.get(k, {}).get('output', '').split()) for k in baseline.keys()] if hybrid_persona else []

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples: {len(baseline)}")
    print(f"\nAverage word count:")
    print(f"  Baseline:       {sum(b_lengths)/len(b_lengths):.1f}")
    print(f"  Hybrid:         {sum(h_lengths)/len(h_lengths):.1f}")
    if hp_lengths:
        print(f"  Hybrid+Persona: {sum(hp_lengths)/len(hp_lengths):.1f}")

    in_range = lambda l: 2 <= l <= 10
    b_ok = sum(1 for l in b_lengths if in_range(l))
    h_ok = sum(1 for l in h_lengths if in_range(l))
    hp_ok = sum(1 for l in hp_lengths if in_range(l)) if hp_lengths else 0
    n = len(baseline)
    print(f"\nIn ideal range (2-10 words):")
    print(f"  Baseline:       {b_ok}/{n} ({100*b_ok/n:.0f}%)")
    print(f"  Hybrid:         {h_ok}/{n} ({100*h_ok/n:.0f}%)")
    if hp_lengths:
        print(f"  Hybrid+Persona: {hp_ok}/{n} ({100*hp_ok/n:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
