import os
import re
import json
import numpy as np
from dotenv import load_dotenv

MODIFIER_TERMS = [
    "stable", "nonspecific", "minimal", "mild", "moderate", "severe", 
    "borderline", "prominent", "subcentimeter", "tiny", "small", "large", 
    "focal", "patchy", "scattered", "trace"
]

HEDGE_TERMS = [
    "suggestive of", "consistent with", "likely", "concerning for", 
    "suspicious for", "possible", "probable", "versus", "may represent"
]

def analyze_prompts(entry_list):
    prompts = []
    for entry in entry_list:
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            texts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            texts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
        prompts.extend(texts)

    word_counts = [len(p.split()) for p in prompts]
    char_counts = [len(p) for p in prompts]
    
    # Adjective modifiers
    has_modifier = [any(re.search(r'\b' + re.escape(m) + r'\b', p, re.IGNORECASE) for m in MODIFIER_TERMS) for p in prompts]
    
    # Measurements
    has_measurement = [bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b|measuring', p, re.IGNORECASE)) for p in prompts]
    
    # Hedge phrases
    has_hedge = [any(re.search(r'\b' + re.escape(h) + r'\b', p, re.IGNORECASE) for h in HEDGE_TERMS) for p in prompts]
    
    # Punctuation
    has_comma = [',' in p for p in prompts]
    has_hyphen = ['-' in p for p in prompts]
    
    # Vocabulary uniqueness
    all_words = [w.lower().strip(',.-') for p in prompts for w in p.split()]
    unique_words = set(all_words)

    stats = {
        "n_prompts": len(prompts),
        "mean_words": np.mean(word_counts),
        "median_words": np.median(word_counts),
        "max_words": np.max(word_counts),
        "min_words": np.min(word_counts),
        "mean_chars": np.mean(char_counts),
        "modifier_pct": np.mean(has_modifier) * 100,
        "measurement_pct": np.mean(has_measurement) * 100,
        "hedge_pct": np.mean(has_hedge) * 100,
        "comma_pct": np.mean(has_comma) * 100,
        "hyphen_pct": np.mean(has_hyphen) * 100,
        "total_words": len(all_words),
        "vocab_size": len(unique_words),
        "ttr": len(unique_words) / len(all_words) if all_words else 0
    }
    return stats, prompts

def main():
    load_dotenv(override=True)
    dataset_json = os.environ["DATASET_JSON"]
    with open(dataset_json) as f:
        meta = json.load(f)

    val_cases = meta.get("val", [])
    cases_50 = val_cases[:50]
    cases_150 = val_cases[50:]

    stats_50, prompts_50 = analyze_prompts(cases_50)
    stats_150, prompts_150 = analyze_prompts(cases_150)

    print("=========================================================================")
    print("=== QUANTITATIVE TEXT SHIFT ANALYSIS: CASES 1-50 vs CASES 51-200 ===")
    print("=========================================================================\n")

    print(f"{'Metric':<35} | {'Cases 1-50 (Paper Val)':<22} | {'Cases 51-200 (New Val)':<22} | {'Shift / Ratio':<15}")
    print("-" * 102)
    print(f"{'Total Prompts':<35} | {stats_50['n_prompts']:<22} | {stats_150['n_prompts']:<22} | -")
    print(f"{'Mean Word Count per Prompt':<35} | {stats_50['mean_words']:<22.2f} | {stats_150['mean_words']:<22.2f} | {stats_150['mean_words']/stats_50['mean_words']:.2f}x")
    print(f"{'Median Word Count per Prompt':<35} | {stats_50['median_words']:<22.1f} | {stats_150['median_words']:<22.1f} | {stats_150['median_words']/stats_50['median_words']:.2f}x")
    print(f"{'Max Word Count in Prompt':<35} | {stats_50['max_words']:<22} | {stats_150['max_words']:<22} | {stats_150['max_words']/stats_50['max_words']:.2f}x")
    print(f"{'Mean Character Length':<35} | {stats_50['mean_chars']:<22.1f} | {stats_150['mean_chars']:<22.1f} | {stats_150['mean_chars']/stats_50['mean_chars']:.2f}x")
    print(f"{'Prompts with Clinical Modifiers':<35} | {stats_50['modifier_pct']:<21.1f}% | {stats_150['modifier_pct']:<21.1f}% | +{stats_150['modifier_pct']-stats_50['modifier_pct']:.1f}%")
    print(f"{'Prompts with Measurements (mm/cm)':<35} | {stats_50['measurement_pct']:<21.1f}% | {stats_150['measurement_pct']:<21.1f}% | +{stats_150['measurement_pct']-stats_50['measurement_pct']:.1f}%")
    print(f"{'Prompts with Hedge/Uncertainty':<35} | {stats_50['hedge_pct']:<21.1f}% | {stats_150['hedge_pct']:<21.1f}% | +{stats_150['hedge_pct']-stats_50['hedge_pct']:.1f}%")
    print(f"{'Prompts with Comma Punctuation':<35} | {stats_50['comma_pct']:<21.1f}% | {stats_150['comma_pct']:<21.1f}% | +{stats_150['comma_pct']-stats_50['comma_pct']:.1f}%")
    print(f"{'Total Word Tokens':<35} | {stats_50['total_words']:<22} | {stats_150['total_words']:<22} | -")
    print(f"{'Unique Vocabulary Size':<35} | {stats_50['vocab_size']:<22} | {stats_150['vocab_size']:<22} | -")
    print(f"{'Type-Token Ratio (Vocabulary Diversity)':<35} | {stats_50['ttr']:<22.4f} | {stats_150['ttr']:<22.4f} | -")

    print("\n--- SAMPLE PROMPTS FROM CASES 1-50 ---")
    for p in prompts_50[:5]:
        print(f"  * '{p}'")

    print("\n--- SAMPLE PROMPTS FROM CASES 51-200 ---")
    for p in prompts_150[:5]:
        print(f"  * '{p}'")

if __name__ == "__main__":
    main()
