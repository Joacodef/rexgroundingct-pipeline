"""
Verification script for ReXGroundingCT test split predictions.

This script ensures that the generated 4D NIfTI predictions meet the structural,
spatial orientation (affine), and dimensionality requirements of the challenge.
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib

def verify_predictions(dataset_json, raw_img_dir, pred_dir, split="test"):
    print("=" * 80)
    print(f"      VERIFYING PREDICTIONS FOR SPLIT: {split.upper()}")
    print("=" * 80)
    print(f"Dataset JSON : {dataset_json}")
    print(f"Raw Image Dir: {raw_img_dir}")
    print(f"Pred Dir     : {pred_dir}")
    print("=" * 80)

    if not os.path.exists(dataset_json):
        print(f"[ERROR] Dataset JSON not found: {dataset_json}")
        return False

    with open(dataset_json, "r") as f:
        metadata = json.load(f)

    entries = metadata.get(split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{split}' in {dataset_json}")
        return False

    all_passed = True
    verified_count = 0

    for entry in entries:
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue

        raw_path = os.path.join(raw_img_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")

        print(f"\n--> Checking Case: {scan_id}")

        # 1. Existence check
        if not os.path.exists(raw_path):
            print(f"   [ERROR] Raw image file not found: {raw_path}")
            all_passed = False
            continue

        if not os.path.exists(pred_path):
            print(f"   [ERROR] Prediction NIfTI file not found: {pred_path}")
            all_passed = False
            continue

        # 2. Load NIfTI headers
        try:
            raw_nib = nib.load(raw_path)
            pred_nib = nib.load(pred_path)
        except Exception as e:
            print(f"   [ERROR] Failed to load NIfTIs: {str(e)}")
            all_passed = False
            continue

        # 3. Check findings structure in dataset.json
        findings = entry.get("findings", {})
        expected_num_findings = len(findings)
        print(f"   Expected findings count: {expected_num_findings}")

        # 4. Shape comparison
        raw_shape = raw_nib.shape
        pred_shape = pred_nib.shape
        expected_pred_shape = (expected_num_findings,) + raw_shape

        print(f"   Raw Shape: {raw_shape}")
        print(f"   Pred Shape: {pred_shape} (Expected: {expected_pred_shape})")

        if pred_shape != expected_pred_shape:
            print(f"   [ERROR] Shape Mismatch! Got {pred_shape}, expected {expected_pred_shape}")
            all_passed = False
        else:
            print("   [PASS] Shape validation successful.")

        # 5. Spatial Affine alignment comparison
        raw_affine = raw_nib.affine
        pred_affine = pred_nib.affine
        
        # We compare only the upper-left 3x3 scaling/rotation part and the translation vector
        affine_diff = np.abs(raw_affine - pred_affine)
        max_diff = np.max(affine_diff)

        print(f"   Affine match max difference: {max_diff:.8f}")
        if max_diff > 1e-5:
            print(f"   [ERROR] Spatial Alignment Mismatch! The prediction affine deviates from the raw image affine.")
            print(f"   Raw Affine:\n{raw_affine}")
            print(f"   Pred Affine:\n{pred_affine}")
            all_passed = False
        else:
            print("   [PASS] Spatial Affine Alignment verified.")

        # 6. Data properties check
        try:
            pred_data = pred_nib.get_fdata(dtype=np.float32)
            
            # Check for NaN / Inf
            nan_count = np.isnan(pred_data).sum()
            inf_count = np.isinf(pred_data).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"   [ERROR] Prediction data contains invalid values: NaNs={nan_count}, Infs={inf_count}")
                all_passed = False
            
            # Check range of values (should be 0 or 1)
            unique_vals = np.unique(pred_data)
            print(f"   Unique values in prediction: {unique_vals.tolist()}")
            
            if not np.all(np.isin(unique_vals, [0, 1])):
                print(f"   [WARNING] Prediction contains values other than 0 and 1! Got: {unique_vals}")
                # Don't fail the verification but issue a strong warning
                
            # Check non-zero predictions
            active_voxels = np.sum(pred_data > 0)
            total_voxels = pred_data.size
            percent_active = (active_voxels / total_voxels) * 100
            print(f"   Active voxels (predicted mask): {active_voxels} / {total_voxels} ({percent_active:.4f}%)")
            
        except Exception as e:
            print(f"   [ERROR] Failed reading prediction voxel data: {str(e)}")
            all_passed = False

        verified_count += 1

    print("\n" + "=" * 80)
    if all_passed and verified_count > 0:
        print(f"SUCCESS: All {verified_count} cases verified and passed spatial/structural alignment.")
        print("=" * 80)
        return True
    else:
        print(f"FAILURE: Spatial or structural discrepancies found across {verified_count} cases.")
        print("=" * 80)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify spatial and structural correctness of predictions")
    parser.add_argument("--dataset_json", type=str, default="data/dataset.json", help="Path to dataset.json")
    parser.add_argument("--raw_img_dir", type=str, default="data/raw/images", help="Directory containing raw images")
    parser.add_argument("--pred_dir", type=str, default="data/predictions", help="Directory containing prediction NIfTIs")
    parser.add_argument("--split", type=str, default="test", help="Split to verify")
    args = parser.parse_args()

    success = verify_predictions(
        dataset_json=args.dataset_json,
        raw_img_dir=args.raw_img_dir,
        pred_dir=args.pred_dir,
        split=args.split
    )
    import sys
    sys.exit(0 if success else 1)
