import os
import json
import torch
import numpy as np
from dotenv import load_dotenv
from voxtell.inference.predictor import VoxTellPredictor

def main():
    load_dotenv(override=True)
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = "/home/jdeferrari/rex_project/models/voxtell_v1.1"

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    
    # Select prompts from cases 1..50 (Paper val) vs cases 51..200 (New val)
    prompts_50 = []
    for entry in val_cases[:50]:
        findings = entry.get("findings", {})
        texts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted(findings.keys(), key=int)] if isinstance(findings, dict) else [f["text"] if isinstance(f, dict) else str(f) for f in findings]
        prompts_50.extend(texts)
        
    prompts_150 = []
    for entry in val_cases[50:100]:
        findings = entry.get("findings", {})
        texts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted(findings.keys(), key=int)] if isinstance(findings, dict) else [f["text"] if isinstance(f, dict) else str(f) for f in findings]
        prompts_150.extend(texts)

    print(f"Loaded {len(prompts_50)} prompts from paper val, {len(prompts_150)} prompts from new val.")
    
    with torch.no_grad():
        emb_50 = predictor.embed_text_prompts(prompts_50[:20])  # (N, D)
        emb_150 = predictor.embed_text_prompts(prompts_150[:20]) # (N, D)
        
    print(f"Embedding shape 50: {emb_50.shape}")
    print(f"Embedding shape 150: {emb_150.shape}")
    
    print("\n--- SAMPLE PROMPTS & EMBEDDING NORMS ---")
    print("Paper Val Prompts:")
    for p, e in zip(prompts_50[:5], emb_50[:5]):
        norm = torch.norm(e).item()
        print(f"  [{norm:.4f}] '{p}'")
        
    print("\nNew Val Prompts (Before Cleaning):")
    for p, e in zip(prompts_150[:5], emb_150[:5]):
        norm = torch.norm(e).item()
        print(f"  [{norm:.4f}] '{p}'")

    # Test specific problematic prompt cleaning
    test_orig = "Stable, nonspecific 6 mm subpleural nodule in the medial segment of the middle lobe"
    test_clean = "6 mm subpleural nodule in the middle lobe"
    
    with torch.no_grad():
        e_orig = predictor.embed_text_prompts([test_orig])[0]
        e_clean = predictor.embed_text_prompts([test_clean])[0]
        
    cos_sim = torch.nn.functional.cosine_similarity(e_orig, e_clean, dim=0).item()
    print(f"\n--- PROMPT CLEANING EMBEDDING SIMILARITY ---")
    print(f"Original: '{test_orig}'")
    print(f"Cleaned : '{test_clean}'")
    print(f"Cosine Similarity: {cos_sim:.4f}")

if __name__ == "__main__":
    main()
