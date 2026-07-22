import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    load_dotenv(override=True)
    gt_dir = os.environ["SEG_RAW_DIR"]
    pred_dir = "/tmp/jdeferrari/rexgroundingct_preprocessed/voxtell_val_normalized_preds"
    dataset_json = os.environ["DATASET_JSON"]

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    pred_files = set(os.listdir(pred_dir))
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")
    
    dices = []
    hits = 0
    total_findings = 0
    empty_preds = 0
    
    cases_1_50_dices = []
    cases_51_200_dices = []

    for idx, entry in enumerate(val_cases):
        scan_id = entry["name"].replace(".nii.gz", "")
        pred_fname = f"{scan_id}.nii.gz"
        if pred_fname not in pred_files:
            continue
            
        pred_path = os.path.join(pred_dir, pred_fname)
        gt_path = os.path.join(gt_dir, pred_fname)
        if not os.path.exists(gt_path): continue
            
        pred_nii = nib.load(pred_path)
        pred_data = np.asanyarray(pred_nii.dataobj)
        if pred_data.ndim == 3: pred_data = np.expand_dims(pred_data, axis=-1)
        if pred_data.ndim == 4 and pred_data.shape[-1] < np.min(pred_data.shape[:-1]):
            pred_data_f = np.moveaxis(pred_data, -1, 0)
        else:
            pred_data_f = pred_data
            
        gt_nii = nib.load(gt_path)
        gt_data = np.asanyarray(gt_nii.dataobj)
        if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data_f = np.moveaxis(gt_data, -1, 0)
        else:
            gt_data_f = gt_data
            
        n_channels = min(pred_data_f.shape[0], gt_data_f.shape[0])
        for c in range(n_channels):
            pm = pred_data_f[c]
            gm = gt_data_f[c]
            
            d = compute_dice(pm, gm)
            dices.append(d)
            total_findings += 1
            if d >= 0.1: hits += 1
            if np.sum(pm > 0) == 0: empty_preds += 1
            
            if idx < 50:
                cases_1_50_dices.append(d)
            else:
                cases_51_200_dices.append(d)

    print("\n==========================================")
    print(f"=== INTERMEDIATE EVALUATION ON {len(dices)} FINDINGS ===")
    print("==========================================")
    print(f"Total Findings Evaluated: {total_findings}")
    print(f"Overall Average Dice    : {np.mean(dices):.4f}")
    print(f"Hit Rate (Dice >= 0.1)  : {hits/total_findings*100:.2f}% ({hits}/{total_findings})")
    print(f"Empty Mask Rate         : {empty_preds/total_findings*100:.2f}% ({empty_preds}/{total_findings})")
    
    if cases_1_50_dices:
        print(f"\nPaper Val Split (Cases 1-50) Avg Dice: {np.mean(cases_1_50_dices):.4f} (N={len(cases_1_50_dices)} findings)")
    if cases_51_200_dices:
        print(f"New Val Split (Cases 51-200) Avg Dice: {np.mean(cases_51_200_dices):.4f} (N={len(cases_51_200_dices)} findings)")

if __name__ == "__main__":
    main()
