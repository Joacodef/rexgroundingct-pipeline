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
    
    # Try TMP_PREP_DIR first, then DATA_PREP_DIR
    gt_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    pred_dir = os.getenv("TMP_PRED_DIR") or os.getenv("DATA_PRED_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    split = "val"

    # Set up logs folder and output to a log file
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/diagnose_dice.log"
    
    import sys
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)

    print("=" * 70)
    print("          DIAGNOSTIC SCRIPT: DICE & ORIENTATION INVESTIGATION")
    print("=" * 70)
    print(f"GT DIR      : {gt_dir}")
    print(f"PRED DIR    : {pred_dir}")
    print(f"DATASET_JSON: {dataset_json}")
    print("=" * 70)

    if not all([gt_dir, pred_dir, dataset_json]):
        print("[ERROR] Missing environment variables in .env.")
        return

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{split}'")
        return

    checked_cases = 0
    for entry in entries:
        if checked_cases >= 5:
            break
            
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}_seg.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        
        print(f"\n--> Inspecting Case: {scan_id}")
        if not os.path.exists(gt_path):
            print(f"    [WARNING] GT not found: {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"    [WARNING] Prediction not found: {pred_path}")
            continue
            
        checked_cases += 1
        
        # Load raw nibabel headers and data
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_img = gt_nii.get_fdata(dtype=np.float32)
        pred_img = pred_nii.get_fdata(dtype=np.float32)
        
        print(f"    [GT] Loaded Shape: {gt_img.shape} | Unique values: {np.unique(gt_img).tolist()[:10]} | Non-zero voxels: {int((gt_img > 0).sum())}")
        print(f"    [Pred] Loaded Shape: {pred_img.shape} | Unique values: {np.unique(pred_img).tolist()[:10]} | Non-zero voxels: {int((pred_img > 0).sum())}")
        
        # Shape adjustment logic matching evaluate.py
        # 1. Expand GT dims if 3D
        gt_adjusted = gt_img.copy()
        if gt_adjusted.ndim == 3:
            gt_adjusted = np.expand_dims(gt_adjusted, axis=-1)
            
        # 2. Transpose findings to front if at the end
        if gt_adjusted.ndim == 4:
            if gt_adjusted.shape[-1] < np.min(gt_adjusted.shape[:-1]):
                gt_adjusted = np.moveaxis(gt_adjusted, -1, 0)
                
        print(f"    [GT Adjusted Shape] : {gt_adjusted.shape}")
        print(f"    [Pred Loaded Shape] : {pred_img.shape}")
        
        if gt_adjusted.shape != pred_img.shape:
            print(f"    [ERROR] Shape mismatch! GT Adjusted: {gt_adjusted.shape}, Pred: {pred_img.shape}")
            continue
            
        num_findings = gt_adjusted.shape[0]
        for f_idx in range(num_findings):
            dice = compute_dice(pred_img[f_idx], gt_adjusted[f_idx])
            gt_sum = (gt_adjusted[f_idx] > 0).sum()
            pred_sum = (pred_img[f_idx] > 0).sum()
            overlap = np.logical_and(pred_img[f_idx] > 0, gt_adjusted[f_idx] > 0).sum()
            print(f"      - Finding {f_idx}: Dice = {dice:.6f} | GT Vol = {int(gt_sum)} vox | Pred Vol = {int(pred_sum)} vox | Overlap = {int(overlap)} vox")
            
            # If Dice is low and both have volumes, check alternative permutations and flips
            if dice < 0.1 and gt_sum > 0 and pred_sum > 0:
                print("        [DEBUG] Checking alternative spatial layouts for Dice improvements...")
                spatial_pred = pred_img[f_idx]
                spatial_gt = gt_adjusted[f_idx]
                
                # Check spatial transpositions (permutations of axes)
                permutations = [
                    ((0, 1, 2), "No Permutation"),
                    ((0, 2, 1), "Swap Y-Z"),
                    ((1, 0, 2), "Swap X-Y"),
                    ((1, 2, 0), "Shift left"),
                    ((2, 0, 1), "Shift right"),
                    ((2, 1, 0), "Swap X-Z")
                ]
                
                for axes, label in permutations:
                    # Permute prediction
                    perm_pred = np.transpose(spatial_pred, axes)
                    if perm_pred.shape != spatial_gt.shape:
                        continue
                        
                    # Calculate Dice for normal and flipped variations
                    # (flips can happen due to orientation changes like RAS vs LPS, etc.)
                    variations = [
                        (perm_pred, "Normal"),
                        (np.flip(perm_pred, 0), "Flip X"),
                        (np.flip(perm_pred, 1), "Flip Y"),
                        (np.flip(perm_pred, 2), "Flip Z"),
                        (np.flip(np.flip(perm_pred, 0), 1), "Flip X-Y"),
                        (np.flip(np.flip(perm_pred, 1), 2), "Flip Y-Z"),
                        (np.flip(np.flip(perm_pred, 0), 2), "Flip X-Z"),
                        (np.flip(np.flip(np.flip(perm_pred, 0), 1), 2), "Flip X-Y-Z"),
                    ]
                    
                    for v_pred, v_label in variations:
                        perm_overlap = np.logical_and(v_pred > 0, spatial_gt > 0).sum()
                        if perm_overlap > 0:
                            perm_dice = 2. * perm_overlap / ((spatial_gt > 0).sum() + (v_pred > 0).sum())
                            if perm_dice > 0.05:
                                print(f"          * Layout [{label}] with [{v_label}] gives Dice = {perm_dice:.6f} (Overlap: {int(perm_overlap)} vox)")

    print(f"\nDiagnostic finished. Log saved to: {log_path}")

if __name__ == "__main__":
    main()
