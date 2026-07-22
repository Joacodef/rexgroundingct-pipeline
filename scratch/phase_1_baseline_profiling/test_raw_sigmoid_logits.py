import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

def main():
    load_dotenv(override=True)
    img_dir = os.environ["IMG_RAW_DIR"]
    gt_dir = os.environ["SEG_RAW_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = "/home/jdeferrari/rex_project/models/voxtell_v1.1"

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    # Pick 3 cases from the second split (cases 50..55)
    test_cases = val_cases[50:53]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    reader = NibabelIOWithReorient()
    
    for entry in test_cases:
        scan_id = entry["name"].replace(".nii.gz", "")
        nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path): continue
        
        img, img_props = reader.read_images([nifti_path])
        
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
            
        print(f"\n==========================================")
        print(f"=== TESTING RAW LOGITS / SIGMOID: {scan_id} ===")
        print(f"==========================================")
        
        data_prep, bbox, orig_shape = predictor.preprocess(img)
        embeddings = predictor.embed_text_prompts(prompts)
        
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data_prep, embeddings).to("cpu")
            probs = torch.sigmoid(logits.float()).numpy()
            
        print(f"Raw Sigmoid Probs Shape: {probs.shape}")
        
        for f_idx, p_text in enumerate(prompts):
            p_map = probs[f_idx]
            max_prob = p_map.max()
            mean_prob = p_map.mean()
            p99_prob = np.percentile(p_map, 99.9)
            
            count_05 = np.sum(p_map > 0.5)
            count_03 = np.sum(p_map > 0.3)
            count_02 = np.sum(p_map > 0.2)
            count_01 = np.sum(p_map > 0.1)
            
            print(f"\nFinding #{f_idx} ('{p_text[:40]}...'):")
            print(f"  Max prob: {max_prob:.4f} | 99.9th percentile: {p99_prob:.4f} | Mean prob: {mean_prob:.6f}")
            print(f"  Voxels > 0.5: {count_05}")
            print(f"  Voxels > 0.3: {count_03}")
            print(f"  Voxels > 0.2: {count_02}")
            print(f"  Voxels > 0.1: {count_01}")

if __name__ == "__main__":
    main()
