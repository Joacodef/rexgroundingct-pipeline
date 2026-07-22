import os
import json
import numpy as np
import nibabel as nib
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

def main():
    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    print(f"Total val cases in dataset.json: {len(val_cases)}")
    
    results = []
    for idx, entry in enumerate(val_cases):
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        gt_nii = nib.load(gt_path, mmap=True)
        pred_nii = nib.load(pred_path, mmap=True)
        
        gt_img = np.asanyarray(gt_nii.dataobj)
        pred_img = np.asanyarray(pred_nii.dataobj)
        
        if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0)
            
        if pred_img.ndim == 3: pred_img = np.expand_dims(pred_img, axis=-1)
        if pred_img.ndim == 4 and pred_img.shape[-1] < np.min(pred_img.shape[:-1]):
            pred_img = np.moveaxis(pred_img, -1, 0)
            
        case_dices = []
        case_hits = 0
        for f_idx in range(gt_img.shape[0]):
            d = compute_dice(pred_img[f_idx], gt_img[f_idx])
            case_dices.append(d)
            if d >= 0.1: case_hits += 1
            
        results.append({
            "idx": idx,
            "scan_id": scan_id,
            "dices": case_dices,
            "hits": case_hits
        })

    # 1. First 50 cases (Paper split)
    res_50 = results[:50]
    dices_50 = [d for r in res_50 for d in r["dices"]]
    hits_50 = sum(r["hits"] for r in res_50)
    print(f"\n--- FIRST 50 CASES (PAPER VAL SPLIT) ---")
    print(f"Evaluated cases: {len(res_50)}, findings: {len(dices_50)}")
    print(f"Average Dice: {np.mean(dices_50):.4f}")
    print(f"Hit Rate (>=0.1): {hits_50 / len(dices_50):.4f}")
    
    # 2. Next 150 cases (Cases 50..200)
    res_150 = results[50:]
    dices_150 = [d for r in res_150 for d in r["dices"]]
    hits_150 = sum(r["hits"] for r in res_150)
    print(f"\n--- NEXT 150 CASES (NEW MICCAI VAL SPLIT) ---")
    print(f"Evaluated cases: {len(res_150)}, findings: {len(dices_150)}")
    print(f"Average Dice: {np.mean(dices_150):.4f}")
    print(f"Hit Rate (>=0.1): {hits_150 / len(dices_150):.4f}")
    
    # 3. All 200 cases
    all_dices = [d for r in results for d in r["dices"]]
    all_hits = sum(r["hits"] for r in results)
    print(f"\n--- ALL 200 CASES ---")
    print(f"Evaluated cases: {len(results)}, findings: {len(all_dices)}")
    print(f"Average Dice: {np.mean(all_dices):.4f}")
    print(f"Hit Rate (>=0.1): {all_hits / len(all_dices):.4f}")

if __name__ == "__main__":
    main()
