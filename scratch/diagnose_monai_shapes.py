import os
import json
import torch
import nibabel as nib
from dotenv import load_dotenv
from monai.transforms import LoadImaged

load_dotenv(override=True)

DATASET_JSON = os.getenv("DATASET_JSON")
IMG_DIR = os.getenv("IMG_RAW_DIR")
SEG_DIR = os.getenv("SEG_RAW_DIR")

print("--- DIAGNOSTIC SCRIPT ---")
print(f"DATASET_JSON: {DATASET_JSON}")
print(f"IMG_RAW_DIR: {IMG_DIR}")
print(f"SEG_RAW_DIR: {SEG_DIR}")

with open(DATASET_JSON, 'r') as f:
    metadata = json.load(f)

first_case = metadata['val'][0]
print(f"\nFirst case: {first_case['name']}")

img_path = os.path.join(IMG_DIR, first_case["name"])
seg_path = os.path.join(SEG_DIR, first_case["name"])

# 1. Inspect using raw Nibabel
raw_img = nib.load(img_path)
raw_seg = nib.load(seg_path)

print(f"\n[Raw Nibabel] Image shape: {raw_img.shape}, affine diagonal: {raw_img.affine.diagonal()}")
print(f"[Raw Nibabel] Segmentation shape: {raw_seg.shape}, affine diagonal: {raw_seg.affine.diagonal()}")

# 2. Inspect using LoadImaged
loader = LoadImaged(keys=["image", "label"], reader="NibabelReader")
loaded_data = loader({"image": img_path, "label": seg_path})

print(f"\n[LoadImaged] Image class: {type(loaded_data['image'])}")
print(f"[LoadImaged] Image shape: {loaded_data['image'].shape}")
print(f"[LoadImaged] Image meta keys: {list(loaded_data['image'].meta.keys()) if hasattr(loaded_data['image'], 'meta') else 'No meta'}")

print(f"\n[LoadImaged] Label class: {type(loaded_data['label'])}")
print(f"[LoadImaged] Label shape: {loaded_data['label'].shape}")
print(f"[LoadImaged] Label meta keys: {list(loaded_data['label'].meta.keys()) if hasattr(loaded_data['label'], 'meta') else 'No meta'}")
