import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import torch
import numpy as np
import nibabel as nib
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    raw_img_dir = os.getenv("IMG_RAW_DIR")
    raw_seg_dir = os.getenv("SEG_RAW_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
    
    entries = metadata["val"][:5] # Test first 5 cases
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_dir = os.getenv("MODEL_DIR")
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    voxtell_weights_dir = os.path.join(models_root, "voxtell_v1.1")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    print("==========================================================")
    print("Hypothesis Test: Raw Images with Exact Authors' Pipeline")
    print("==========================================================")
    
    reader = NibabelIOWithReorient()
    
    for entry in entries:
        scan_id = entry["name"].replace(".nii.gz", "")
        raw_img_path = os.path.join(raw_img_dir, entry["name"])
        raw_seg_path = os.path.join(raw_seg_dir, entry["name"])
        
        if not os.path.exists(raw_img_path) or not os.path.exists(raw_seg_path):
            print(f"Skipping {scan_id} (missing raw files)")
            continue
            
        print(f"\n--> CASE: {scan_id}")
        
        # 1. Load raw image using NibabelIOWithReorient (reorients to RAS and returns (1, Z, Y, X))
        img_raw, img_properties = reader.read_images([raw_img_path])
        print(f"  Raw image loaded shape (Z, Y, X): {img_raw.shape[1:]} | Spacing: {img_properties['spacing']}")
        
        # 2. Load GT mask, reorient to standard canonical RAS, and move findings axis to front (F, X, Y, Z)
        gt_nii = nib.load(raw_seg_path)
        gt_ras = nib.as_closest_canonical(gt_nii)
        gt_img = gt_ras.get_fdata(dtype=np.float32)
        if gt_img.ndim == 3:
            gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4:
            if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
                gt_img = np.moveaxis(gt_img, -1, 0) # Shape: (F, X, Y, Z)
                
        # 3. Extract prompts
        findings = entry.get('findings', {})
        sorted_keys = sorted(findings.keys(), key=int)
        text_prompts = []
        for k in sorted_keys:
            val = findings[k]
            if isinstance(val, dict):
                text_prompts.append(val.get('text', ''))
            else:
                text_prompts.append(str(val))
        
        # 4. Predict
        with torch.no_grad():
            voxtell_seg = predictor.predict_single_image(img_raw, text_prompts)
            
        # Revert axis order from (F, Z, Y, X) to (F, X, Y, Z) using transpose to match gt_img
        pred_4d = np.transpose(voxtell_seg, (0, 3, 2, 1))
        
        # Print shape sanity checks
        print(f"  Pred shape: {pred_4d.shape} | GT shape: {gt_img.shape}")
        
        for f_idx in range(gt_img.shape[0]):
            dice = compute_dice(pred_4d[f_idx], gt_img[f_idx])
            gt_vol = (gt_img[f_idx] > 0).sum()
            pred_vol = (pred_4d[f_idx] > 0).sum()
            overlap = np.logical_and(pred_4d[f_idx] > 0, gt_img[f_idx] > 0).sum()
            print(f"    Finding {f_idx}: Dice = {dice:.6f} | GT Vol = {gt_vol} | Pred Vol = {pred_vol} | Overlap = {overlap}")

if __name__ == "__main__":
    main()
