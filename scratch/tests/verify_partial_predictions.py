import os
import json
import numpy as np
import nibabel as nib

def verify_partial_predictions(dataset_json="data/dataset.json", raw_img_dir="data/raw/images", pred_dir="data/predictions_test"):
    print("=" * 80)
    print("    AUDITING INTEGRITY OF CURRENTLY GENERATED TEST PREDICTIONS")
    print("=" * 80)
    
    if not os.path.exists(dataset_json):
        print(f"[ERROR] dataset.json not found: {dataset_json}")
        return
        
    with open(dataset_json, "r") as f:
        metadata = json.load(f)
        
    test_entries = metadata.get("test", [])
    if not test_entries:
        print("[ERROR] No test split entries found in dataset.json")
        return
        
    generated_files = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")]
    print(f"Total test cases in dataset.json: {len(test_entries)}")
    print(f"Total test predictions completed: {len(generated_files)}")
    print("=" * 80)
    
    if len(generated_files) == 0:
        print("[INFO] No predictions generated yet. Nothing to verify.")
        return
        
    passed_count = 0
    failed_count = 0
    
    for entry in test_entries:
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        if not os.path.exists(pred_path):
            # Skip cases that have not finished generating yet
            continue
            
        raw_path = os.path.join(raw_img_dir, f"{scan_id}.nii.gz")
        print(f"--> Auditing {scan_id}... ", end="")
        
        # Check raw image exists
        if not os.path.exists(raw_path):
            print(f"[FAILED] Raw image missing: {raw_path}")
            failed_count += 1
            continue
            
        try:
            raw_nib = nib.load(raw_path)
            pred_nib = nib.load(pred_path)
        except Exception as e:
            print(f"[FAILED] Error loading NIfTI headers: {str(e)}")
            failed_count += 1
            continue
            
        # Check dimension shape matching original CT + findings channels
        findings = entry.get("findings", {})
        expected_num_findings = len(findings)
        raw_shape = raw_nib.shape
        pred_shape = pred_nib.shape
        expected_pred_shape = (expected_num_findings,) + raw_shape
        
        if pred_shape != expected_pred_shape:
            print(f"[FAILED] Shape Mismatch! Got {pred_shape}, expected {expected_pred_shape}")
            failed_count += 1
            continue
            
        # Check affine spatial alignment matching original CT scan
        raw_affine = raw_nib.affine
        pred_affine = pred_nib.affine
        max_affine_diff = np.max(np.abs(raw_affine - pred_affine))
        
        if max_affine_diff > 1e-5:
            print(f"[FAILED] Affine Mismatch! Diff: {max_affine_diff:.8f}")
            failed_count += 1
            continue
            
        # Load arrays and verify data properties
        try:
            pred_data = pred_nib.get_fdata(dtype=np.float32)
            
            # Check NaNs or Infs
            if np.isnan(pred_data).any() or np.isinf(pred_data).any():
                print("[FAILED] Contains NaNs or Infs")
                failed_count += 1
                continue
                
            # Check binary value ranges [0, 1]
            unique_vals = np.unique(pred_data)
            if not np.all(np.isin(unique_vals, [0.0, 1.0])):
                print(f"[WARNING] Non-binary values found: {unique_vals.tolist()}")
                
            # Check active volume
            active_voxels = np.sum(pred_data > 0)
            percent_active = (active_voxels / pred_data.size) * 100
            
            print(f"[OK] Shape={pred_shape}, AffineDiff={max_affine_diff:.8f}, Values={unique_vals.tolist()}, Active={percent_active:.4f}%")
            passed_count += 1
            
        except Exception as e:
            print(f"[FAILED] Error reading voxel array: {str(e)}")
            failed_count += 1
            
    print("=" * 80)
    print(f"AUDIT SUMMARY:")
    print(f"  Total Checked   : {passed_count + failed_count}")
    print(f"  PASSED Integrity: {passed_count}")
    print(f"  FAILED Integrity: {failed_count}")
    print("=" * 80)

if __name__ == "__main__":
    verify_partial_predictions()
