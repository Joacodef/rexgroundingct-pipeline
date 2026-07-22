import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)
gt_dir = os.environ["SEG_RAW_DIR"]
pred_dir = os.environ["DATA_PRED_DIR"]
dataset_json = os.environ["DATASET_JSON"]

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def evaluate_subset(entries, label):
    dices = []
    hits = 0
    for entry in tqdm(entries, desc=f"Evaluating {label}"):
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
        pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)
        
        if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0)
            
        if pred_img.ndim == 3: pred_img = np.expand_dims(pred_img, axis=-1)
        if pred_img.ndim == 4 and pred_img.shape[-1] < np.min(pred_img.shape[:-1]):
            pred_img = np.moveaxis(pred_img, -1, 0)
            
        for f_idx in range(gt_img.shape[0]):
            d = compute_dice(pred_img[f_idx], gt_img[f_idx])
            dices.append(d)
            if d >= 0.1: hits += 1
            
    avg_d = np.mean(dices) if dices else 0.0
    hr = hits / len(dices) if dices else 0.0
    print(f"\n=== {label} ===")
    print(f"Evaluated cases: {len(entries)}, findings: {len(dices)}")
    print(f"Average Dice: {avg_d:.4f}")
    print(f"Hit Rate (>=0.1): {hr:.4f}\n")
    return dices, hits

def main():
    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    # 1. First 50 cases
    dices_50, hits_50 = evaluate_subset(val_cases[:50], "First 50 Cases (Paper Val Split)")
    
    # 2. Next 150 cases
    dices_150, hits_150 = evaluate_subset(val_cases[50:], "Next 150 Cases (New MICCAI Val Split)")
    
    # 3. Combined 200 cases
    all_d = dices_50 + dices_150
    all_h = hits_50 + hits_150
    print(f"=== COMBINED 200 CASES ===")
    print(f"Total findings: {len(all_d)}")
    print(f"Average Dice: {np.mean(all_d):.4f}")
    print(f"Hit Rate (>=0.1): {all_h / len(all_d):.4f}")

if __name__ == "__main__":
    main()
