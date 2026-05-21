"""
Consolidated Spatial Alignment Diagnostic Tool.

This script merges the functionality of `diagnose_zero_dice.py`, `test_orientation.py`,
and `test_perfect_alignment.py`. It exhaustively checks various axis permutations
and flips to identify the correct spatial mapping between VoxTell predictions
and ground truth NIfTI segmentation masks.
"""

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

def get_bbox_and_centroid(mask):
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return None, None
    bbox = [[int(indices[i].min()), int(indices[i].max())] for i in range(len(indices))]
    centroid = [float(indices[i].mean()) for i in range(len(indices))]
    return bbox, centroid

def main():
    load_dotenv(override=True)
    
    gt_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    pred_dir = os.getenv("TMP_PRED_DIR") or os.getenv("DATA_PRED_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    
    print("=" * 70)
    print("      CONSOLIDATED DIAGNOSTIC: SPATIAL ALIGNMENT & AXIS INVESTIGATION")
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
        
    entries = metadata.get("val", [])
    if not entries:
        print("[ERROR] No entries found in validation split.")
        return

    # Check up to 5 cases by default
    max_cases = 5
    checked_cases = 0
    
    for entry in entries:
        if checked_cases >= max_cases:
            break
            
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}_seg.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        checked_cases += 1
        print(f"\n--> Analyzing Case: {scan_id}")
        
        # Load NIfTIs
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_img = gt_nii.get_fdata(dtype=np.float32)
        pred_img = pred_nii.get_fdata(dtype=np.float32)
        
        print(f"    [GT] Loaded Shape   : {gt_img.shape} | Non-zero voxels: {int((gt_img > 0).sum())}")
        print(f"    [Pred] Loaded Shape : {pred_img.shape} | Non-zero voxels: {int((pred_img > 0).sum())}")
        
        # Standardize GT dimensions (F, X, Y, Z)
        gt_adjusted = gt_img.copy()
        if gt_adjusted.ndim == 3:
            gt_adjusted = np.expand_dims(gt_adjusted, axis=-1)
        if gt_adjusted.ndim == 4:
            if gt_adjusted.shape[-1] < np.min(gt_adjusted.shape[:-1]):
                gt_adjusted = np.moveaxis(gt_adjusted, -1, 0)
                
        print(f"    [GT Standardized]   : {gt_adjusted.shape}")
        
        if gt_adjusted.shape != pred_img.shape:
            print(f"    [ERROR] Shape mismatch! GT Standardized: {gt_adjusted.shape}, Pred: {pred_img.shape}")
            continue
            
        num_findings = gt_adjusted.shape[0]
        for f_idx in range(num_findings):
            gt_f = gt_adjusted[f_idx]
            pred_f = pred_img[f_idx]
            
            dice = compute_dice(pred_f, gt_f)
            gt_vol = (gt_f > 0).sum()
            pred_vol = (pred_f > 0).sum()
            overlap = np.logical_and(pred_f > 0, gt_f > 0).sum()
            
            print(f"      - Finding {f_idx}: Dice = {dice:.6f} | GT Vol = {int(gt_vol)} vox | Pred Vol = {int(pred_vol)} vox | Overlap = {int(overlap)} vox")
            
            gt_bbox, gt_cent = get_bbox_and_centroid(gt_f)
            if gt_bbox:
                print(f"        GT Bounding Box : {gt_bbox}")
                print(f"        GT Centroid     : {[f'{c:.1f}' for c in gt_cent]}")
                
            pred_bbox, pred_cent = get_bbox_and_centroid(pred_f)
            if pred_bbox:
                print(f"        Pred Bounding Box: {pred_bbox}")
                print(f"        Pred Centroid    : {[f'{c:.1f}' for c in pred_cent]}")
            
            # Exhaustively check permutations if overlap is zero or very low
            if dice < 0.1 and gt_vol > 0 and pred_vol > 0:
                print("        [DEBUG] Checking alternative permutations/flips for alignment...")
                
                permutations = [
                    ((0, 1, 2), "No Permutation"),
                    ((0, 2, 1), "Swap Y-Z"),
                    ((1, 0, 2), "Swap X-Y"),
                    ((1, 2, 0), "Shift left"),
                    ((2, 0, 1), "Shift right"),
                    ((2, 1, 0), "Swap X-Z")
                ]
                
                for axes, perm_label in permutations:
                    perm_pred = np.transpose(pred_f, axes)
                    if perm_pred.shape != gt_f.shape:
                        continue
                        
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
                        perm_overlap = np.logical_and(v_pred > 0, gt_f > 0).sum()
                        if perm_overlap > 0:
                            perm_dice = 2. * perm_overlap / ((gt_f > 0).sum() + (v_pred > 0).sum())
                            if perm_dice > 0.02:
                                print(f"          * Layout [{perm_label}] with [{v_label}] -> Dice = {perm_dice:.6f} (Overlap: {int(perm_overlap)} vox)")

if __name__ == "__main__":
    main()
