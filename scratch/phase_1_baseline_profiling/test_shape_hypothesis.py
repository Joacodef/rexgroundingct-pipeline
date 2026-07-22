import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

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
    img_dir = os.environ["IMG_RAW_DIR"]
    gt_dir = os.environ["SEG_RAW_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = "/home/jdeferrari/rex_project/models/voxtell_v1.1"

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    val_entries = metadata.get("val", [])
    
    # Target problematic scans from HANDSHAKE.md
    target_ids = ["train_13013_a_1", "train_13102_a_1"]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    reader = NibabelIOWithReorient()
    
    for scan_id in target_ids:
        entry = next((e for e in val_entries if e["name"].replace(".nii.gz", "") == scan_id), None)
        if not entry:
            print(f"Entry {scan_id} not found in val split.")
            continue
            
        nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        
        print(f"\n==========================================")
        print(f"=== TESTING SCAN: {scan_id} ===")
        print(f"==========================================")
        
        img, img_properties = reader.read_images([nifti_path])
        print(f"Read img shape (RAS space): {img.shape}")
        
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata(dtype=np.float32)
        print(f"Raw GT shape: {gt_data.shape}")
        print(f"Original image affine:\n{img_properties['nibabel_stuff']['original_affine']}")
        print(f"Original image orientation: {nib.orientations.aff2axcodes(img_properties['nibabel_stuff']['original_affine'])}")
        print(f"Reoriented image affine:\n{img_properties['nibabel_stuff']['reoriented_affine']}")
        
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
            
        print(f"Prompts ({len(prompts)}): {prompts}")
        
        with torch.no_grad():
            voxtell_seg = predictor.predict_single_image(img, prompts)
            
        print(f"VoxTell predicted raw shape: {voxtell_seg.shape}")
        print(f"Total non-zero predicted voxels across findings: {np.sum(voxtell_seg > 0)}")
        
        # -------------------------------------------------------------
        # Method 1: Current voxtell_inference.py pipeline
        # -------------------------------------------------------------
        pred_xyzf_1 = np.transpose(voxtell_seg, (3, 2, 1, 0))
        reoriented_affine = img_properties['nibabel_stuff']['reoriented_affine']
        pred_nib_1 = nib.Nifti1Image(pred_xyzf_1, reoriented_affine)
        
        original_affine = img_properties['nibabel_stuff']['original_affine']
        img_ornt = io_orientation(original_affine)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        
        pred_nib_back_1 = pred_nib_1.as_reoriented(from_canonical)
        pred_back_data_1 = np.asanyarray(pred_nib_back_1.dataobj).astype(np.uint8)
        pred_back_fxyz_1 = np.transpose(pred_back_data_1, (3, 0, 1, 2))
        
        if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data_f = np.moveaxis(gt_data, -1, 0)
        else:
            gt_data_f = gt_data
            
        for f_idx in range(gt_data_f.shape[0]):
            gt_m = gt_data_f[f_idx]
            pr_m = pred_back_fxyz_1[f_idx]
            d = compute_dice(pr_m, gt_m)
            gt_vox = np.sum(gt_m > 0)
            pr_vox = np.sum(pr_m > 0)
            
            # Check bounding box centers of GT vs Pred if both > 0
            bbox_info = ""
            if gt_vox > 0 and pr_vox > 0:
                gt_coords = np.argwhere(gt_m > 0)
                pr_coords = np.argwhere(pr_m > 0)
                gt_center = gt_coords.mean(axis=0)
                pr_center = pr_coords.mean(axis=0)
                bbox_info = f" | GT center: {gt_center.round(1)} | Pred center: {pr_center.round(1)} | Diff: {(pr_center-gt_center).round(1)}"
                
            print(f"Finding #{f_idx} ('{prompts[f_idx][:30]}...'): Dice = {d:.4f} | GT vox = {gt_vox} | Pred vox = {pr_vox}{bbox_info}")

if __name__ == "__main__":
    main()
