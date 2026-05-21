"""
Consolidated Metadata Inspection Tool.

This script merges the functionality of `inspect_dataset_json.py`, `inspect_locations.py`,
and `inspect_voxtell.py`. It inspects the dataset JSON structure, computes
bounding boxes and centroids of annotations, and prints the VoxTell model methods
and text embeddings.
"""

import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

def get_bbox_and_centroid(mask):
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return None, None
    bbox = [[int(indices[i].min()), int(indices[i].max())] for i in range(len(indices))]
    centroid = [float(indices[i].mean()) for i in range(len(indices))]
    return bbox, centroid

def main():
    load_dotenv(override=True)
    dataset_json = os.getenv("DATASET_JSON")
    gt_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    
    print("=" * 70)
    print("            CONSOLIDATED METADATA & DATASET INSPECTOR")
    print("=" * 70)
    
    if not dataset_json or not os.path.exists(dataset_json):
        print(f"[ERROR] Dataset JSON not found at: {dataset_json}")
        return
        
    # 1. Dataset JSON Structure
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    print("\n[Dataset JSON Structure]")
    print(f"  Keys in dataset.json: {list(metadata.keys())}")
    for split in ["train", "val", "test"]:
        entries = metadata.get(split, [])
        print(f"  Split '{split}': count = {len(entries)}")
        if entries:
            print(f"    Example '{split}' entry keys: {list(entries[0].keys())}")
            if "findings" in entries[0]:
                print(f"    Example findings: {entries[0]['findings']}")
                
    # 2. GT Annotation Locations (Bounding Boxes and Centroids)
    print("\n[Spatial Bounding Boxes and Centroids (Validation)]")
    val_cases = metadata.get("val", [])[:3] # Analyze up to 3 validation cases
    for entry in val_cases:
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}_seg.nii.gz")
        
        print(f"\n  Case: {scan_id}")
        if not os.path.exists(gt_path):
            print(f"    [WARNING] Preprocessed GT mask not found at: {gt_path}")
            continue
            
        gt_nii = nib.load(gt_path)
        gt_img = gt_nii.get_fdata(dtype=np.float32)
        
        # Adjust dimensions to (F, X, Y, Z)
        if gt_img.ndim == 3:
            gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4:
            if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
                gt_img = np.moveaxis(gt_img, -1, 0)
                
        print(f"    GT Dimensions (F, X, Y, Z): {gt_img.shape}")
        findings = entry.get("findings", {})
        
        for f_idx in range(gt_img.shape[0]):
            finding_text = findings.get(str(f_idx)) or findings.get(f_idx) or "Unknown Finding"
            gt_mask = gt_img[f_idx] > 0
            vol = gt_mask.sum()
            
            print(f"    - Finding {f_idx}: {finding_text[:50]}...")
            print(f"      GT Volume: {int(vol)} voxels")
            if vol > 0:
                bbox, centroid = get_bbox_and_centroid(gt_mask)
                print(f"      Bounding Box: X{bbox[0]}, Y{bbox[1]}, Z{bbox[2]}")
                print(f"      Centroid    : {[f'{c:.1f}' for c in centroid]}")
                
    # 3. Optional Predictor Inspection
    print("\n[Optional VoxTell Predictor Inspection]")
    try:
        from voxtell.inference.predictor import VoxTellPredictor
        import inspect
        print("  VoxTellPredictor successfully imported!")
        print("  Methods available in VoxTellPredictor:")
        methods = [m[0] for m in inspect.getmembers(VoxTellPredictor, predicate=inspect.isfunction)]
        print(f"    {methods}")
    except ImportError as e:
        print(f"  [INFO] VoxTell not installed in current environment, skipping class inspection. (Details: {e})")

if __name__ == "__main__":
    main()
