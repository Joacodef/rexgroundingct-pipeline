import json
import os
import nibabel as nib
from dotenv import load_dotenv

load_dotenv(override=True)
dataset_json = os.environ.get("DATASET_JSON")
img_raw_dir = os.environ.get("IMG_RAW_DIR")

with open(dataset_json, 'r') as f:
    metadata = json.load(f)

val_entries = metadata.get("val", [])
if len(val_entries) > 53:
    entry = val_entries[53]
    scan_id = entry.get("name").replace(".nii.gz", "")
    nifti_path = os.path.join(img_raw_dir, f"{scan_id}.nii.gz")
    
    findings = entry.get('findings', {})
    num_findings = len(findings)
    
    print(f"Scan ID: {scan_id}")
    print(f"Number of findings (F dimension): {num_findings}")
    
    if os.path.exists(nifti_path):
        img = nib.load(nifti_path)
        shape = img.shape
        print(f"Volume Shape: {shape}")
    else:
        print("Raw NIfTI file not found.")
else:
    print("Less than 54 entries in val split.")
