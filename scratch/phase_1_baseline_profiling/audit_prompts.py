import json
import os
import re
from collections import Counter
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    dataset_json = os.environ["DATASET_JSON"]
    
    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    def extract_prompts(cases):
        prompts = []
        for entry in cases:
            findings = entry.get("findings", {})
            if isinstance(findings, dict):
                sorted_keys = sorted(findings.keys(), key=int)
                for k in sorted_keys:
                    val = findings[k]
                    text = val.get("text", "") if isinstance(val, dict) else str(val)
                    prompts.append(text)
            else:
                for f in findings:
                    text = f["text"] if isinstance(f, dict) else str(f)
                    prompts.append(text)
        return prompts

    prompts_50 = extract_prompts(val_cases[:50])
    prompts_150 = extract_prompts(val_cases[50:])
    
    print(f"=== PROMPT AUDIT SUMMARY ===")
    print(f"First 50 cases total prompts: {len(prompts_50)}")
    print(f"Next 150 cases total prompts: {len(prompts_150)}")
    
    def analyze_texts(texts, name):
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]
        has_digits = sum(1 for t in texts if re.search(r'\d', t))
        has_mm_cm = sum(1 for t in texts if re.search(r'\b(mm|cm)\b', t, re.IGNORECASE))
        has_punctuation = sum(1 for t in texts if re.search(r'[^\w\s]', t))
        
        words = [w.lower().strip(".,;:()") for t in texts for w in t.split()]
        vocab = set(words)
        most_common = Counter(words).most_common(15)
        
        print(f"\n--- {name} ---")
        print(f"Mean word length: {sum(word_counts)/len(word_counts):.2f} (min: {min(word_counts)}, max: {max(word_counts)})")
        print(f"Mean char length: {sum(char_counts)/len(char_counts):.2f}")
        print(f"Prompts with numbers: {has_digits} / {len(texts)} ({has_digits/len(texts)*100:.1f}%)")
        print(f"Prompts with measurements (mm/cm): {has_mm_cm} / {len(texts)} ({has_mm_cm/len(texts)*100:.1f}%)")
        print(f"Prompts with punctuation: {has_punctuation} / {len(texts)} ({has_punctuation/len(texts)*100:.1f}%)")
        print(f"Total vocabulary size (unique words): {len(vocab)}")
        print(f"Top 15 words: {most_common}")

    analyze_texts(prompts_50, "FIRST 50 (PAPER VAL)")
    analyze_texts(prompts_150, "NEXT 150 (NEW MICCAI VAL)")

if __name__ == "__main__":
    main()
