import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
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
    
    entry = metadata["val"][0] # train_13082_a_1
    scan_id = entry["name"].replace(".nii.gz", "")
    raw_img_path = os.path.join(raw_img_dir, entry["name"])
    raw_seg_path = os.path.join(raw_seg_dir, entry["name"])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_dir = os.getenv("MODEL_DIR")
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    voxtell_weights_dir = os.path.join(models_root, "voxtell_v1.1")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    reader = NibabelIOWithReorient()
    img_raw, img_properties = reader.read_images([raw_img_path])
    
    findings = entry.get('findings', {})
    sorted_keys = sorted(findings.keys(), key=int)
    text_prompts = [str(findings[k]) for k in sorted_keys]
    
    print("==================================================")
    print(f"Testing Back-Reorientation for Case 1: {scan_id}")
    print("==================================================")
    
    with torch.no_grad():
        pred_seg = predictor.predict_single_image(img_raw, text_prompts) # shape: (F, Z, Y, X)
        
    print(f"Original RAS prediction shape: {pred_seg.shape}")
    
    # Reorient the prediction back from RAS to original image space
    # 1. Transpose from (F, Z, Y, X) to (X, Y, Z, F) in RAS orientation space
    pred_xyzf = np.transpose(pred_seg, (3, 2, 1, 0))
    
    # 2. Create NIfTI image in RAS space using reoriented_affine
    reoriented_affine = img_properties['nibabel_stuff']['reoriented_affine']
    pred_nib = nib.Nifti1Image(pred_xyzf, reoriented_affine)
    
    # 3. Get the transformation to go from RAS back to original orientation
    original_affine = img_properties['nibabel_stuff']['original_affine']
    img_ornt = io_orientation(original_affine)
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    
    # 4. Apply back-reorientation
    pred_nib_back = pred_nib.as_reoriented(from_canonical)
    
    # 5. Extract data and transpose to (F, X, Y, Z)
    pred_back_data = pred_nib_back.get_fdata() # shape: (X, Y, Z, F)
    pred_back_fxyz = np.transpose(pred_back_data, (3, 0, 1, 2))
    
    # Load raw GT directly
    gt_nii = nib.load(raw_seg_path)
    gt_data = gt_nii.get_fdata(dtype=np.float32) # shape: (F, X, Y, Z) or similar
    
    # Sanity expand if 3D
    if gt_data.ndim == 3:
        gt_data = np.expand_dims(gt_data, axis=-1)
    if gt_data.ndim == 4:
        if gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data = np.moveaxis(gt_data, -1, 0) # Shape: (F, X, Y, Z)
            
    print(f"Back-reoriented shape: {pred_back_fxyz.shape} | Raw GT shape: {gt_data.shape}")
    
    for f_idx in range(gt_data.shape[0]):
        dice = compute_dice(pred_back_fxyz[f_idx], gt_data[f_idx])
        print(f"  Finding {f_idx} ({text_prompts[f_idx]}): Dice = {dice:.6f}")

if __name__ == "__main__":
    main()
