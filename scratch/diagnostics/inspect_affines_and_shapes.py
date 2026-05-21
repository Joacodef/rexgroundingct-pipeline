"""
Consolidated Affine and Shape Inspection Tool.

This script merges the functionality of `inspect_affines.py`, `diagnose_monai_shapes.py`,
and `test_nib_load.py`. It inspects file dimensions, zoom spacing, affine matrices,
and axcodes using both raw Nibabel loaders and MONAI pipelines.
"""

import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    
    data_prep_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    img_raw_dir = os.getenv("IMG_RAW_DIR")
    seg_raw_dir = os.getenv("SEG_RAW_DIR")
    
    print("=" * 70)
    print("      CONSOLIDATED DIAGNOSTIC: AFFINES, SHAPES, & READERS")
    print("=" * 70)
    print(f"PREPROCESSED DIR: {data_prep_dir}")
    print(f"RAW IMAGE DIR   : {img_raw_dir}")
    print(f"RAW SEG DIR     : {seg_raw_dir}")
    print("=" * 70)

    if not all([data_prep_dir, dataset_json]):
        print("[ERROR] Missing base configuration variables in .env.")
        return

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entry = metadata["val"][0] # Use first validation case
    scan_id = entry["name"].replace(".nii.gz", "")
    
    prep_ct_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
    prep_seg_path = os.path.join(data_prep_dir, f"{scan_id}_seg.nii.gz")
    raw_ct_path = os.path.join(img_raw_dir, entry["name"]) if img_raw_dir else None
    raw_seg_path = os.path.join(seg_raw_dir, entry["name"]) if seg_raw_dir else None

    # 1. Inspect using raw Nibabel (Preprocessed and Raw side-by-side)
    print("\n--- [1. Raw Nibabel Properties] ---")
    for name, path in [
        ("Preprocessed CT", prep_ct_path),
        ("Preprocessed Seg", prep_seg_path),
        ("Raw CT", raw_ct_path),
        ("Raw Seg", raw_seg_path)
    ]:
        if not path or not os.path.exists(path):
            print(f"  {name}: File not found at {path}")
            continue
            
        nii = nib.load(path)
        print(f"\n  {name}:")
        print(f"    Shape        : {nii.shape}")
        print(f"    Zooms        : {nii.header.get_zooms()}")
        print(f"    Axcode       : {''.join(nib.aff2axcodes(nii.affine))}")
        print(f"    Affine Matrix:\n{nii.affine}")

    # 2. Inspect using MONAI LoadImaged reader
    print("\n--- [2. MONAI LoadImaged Pipeline] ---")
    if raw_ct_path and raw_seg_path and os.path.exists(raw_ct_path) and os.path.exists(raw_seg_path):
        try:
            from monai.transforms import LoadImaged
            loader = LoadImaged(keys=["image", "label"], reader="NibabelReader")
            loaded_data = loader({"image": raw_ct_path, "label": raw_seg_path})
            
            print(f"  [Image] Class: {type(loaded_data['image'])}")
            print(f"  [Image] Shape: {loaded_data['image'].shape}")
            if hasattr(loaded_data['image'], 'meta'):
                print(f"  [Image] Spacing: {loaded_data['image'].meta.get('pixdim')}")
                
            print(f"  [Label] Class: {type(loaded_data['label'])}")
            print(f"  [Label] Shape: {loaded_data['label'].shape}")
        except ImportError:
            print("  [INFO] MONAI not installed or error importing. Skipping MONAI load tests.")
    else:
        print("  [WARNING] Raw files not available for MONAI load test.")

    # 3. Compare standard loaders and nnU-Net Reader
    print("\n--- [3. nnUNet Reader with Reorient] ---")
    if os.path.exists(prep_ct_path):
        try:
            from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
            reader = NibabelIOWithReorient()
            img, img_props = reader.read_images([prep_ct_path])
            print(f"  Loaded Shape           : {img.shape}")
            print(f"  Spacings for nnUNet    : {img_props.get('spacing')}")
            if 'nibabel_stuff' in img_props:
                print(f"  Original Affine        :\n{img_props['nibabel_stuff'].get('original_affine')}")
                print(f"  Reoriented Affine      :\n{img_props['nibabel_stuff'].get('reoriented_affine')}")
        except ImportError:
            print("  [INFO] nnUNet Reader not imported, skipping reader tests.")

if __name__ == "__main__":
    main()
