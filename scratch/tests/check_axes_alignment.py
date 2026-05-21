import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation
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

def get_bbox_and_centroid(mask):
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return None, None
    bbox = [[int(indices[i].min()), int(indices[i].max())] for i in range(len(indices))]
    centroid = [float(indices[i].mean()) for i in range(len(indices))]
    return bbox, centroid

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
    print(f"Diagnosing Case 1 with Corrected Affine: {scan_id}")
    print("==================================================")
    
    with torch.no_grad():
        pred_seg = predictor.predict_single_image(img_raw, text_prompts) # shape: (F, Z, Y, X)
        
    # Correct alignment of GT by copying the image affine
    img_nii = nib.load(raw_img_path)
    gt_nii = nib.load(raw_seg_path)
    
    gt_data = gt_nii.get_fdata(dtype=np.float32) # shape: (F, X, Y, Z)
    # Move F to the end to get (X, Y, Z, F)
    gt_data_xyzf = np.moveaxis(gt_data, 0, -1)
    
    # Wrap in NIfTI using the image's affine
    gt_seg_nii = nib.Nifti1Image(gt_data_xyzf, img_nii.affine)
    
    # Reorient to RAS using the image's affine orientation
    img_ornt = io_orientation(img_nii.affine)
    gt_reoriented = gt_seg_nii.as_reoriented(img_ornt)
    
    # Transpose back to (F, Z, Y, X) to match prediction shape
    gt_seg = gt_reoriented.get_fdata().transpose((3, 2, 1, 0))
    
    print(f"Pred shape: {pred_seg.shape} | GT shape: {gt_seg.shape}")
    
    for f_idx in range(gt_seg.shape[0]):
        print(f"\nFinding {f_idx}: {text_prompts[f_idx]}")
        gt_f = gt_seg[f_idx]
        pred_f = pred_seg[f_idx]
        
        gt_bbox, gt_cent = get_bbox_and_centroid(gt_f)
        pred_bbox, pred_cent = get_bbox_and_centroid(pred_f)
        
        print(f"  GT Vol: {int((gt_f > 0).sum())} vox | Centroid: {gt_cent} | Bbox: {gt_bbox}")
        print(f"  Pred Vol: {int((pred_f > 0).sum())} vox | Centroid: {pred_cent} | Bbox: {pred_bbox}")
        
        if gt_bbox and pred_bbox:
            # Check all permutations of spatial axes (0, 1, 2)
            permutations = [
                ((0, 1, 2), "No Permutation (Z, Y, X)"),
                ((0, 2, 1), "Swap Y-X"),
                ((1, 0, 2), "Swap Z-Y"),
                ((1, 2, 0), "Shift left"),
                ((2, 0, 1), "Shift right"),
                ((2, 1, 0), "Swap Z-X")
            ]
            for axes, perm_label in permutations:
                perm_pred = np.transpose(pred_f, axes)
                if perm_pred.shape != gt_f.shape:
                    continue
                variations = [
                    (perm_pred, "Normal"),
                    (np.flip(perm_pred, 0), "Flip Z"),
                    (np.flip(perm_pred, 1), "Flip Y"),
                    (np.flip(perm_pred, 2), "Flip X"),
                    (np.flip(np.flip(perm_pred, 0), 1), "Flip Z-Y"),
                    (np.flip(np.flip(perm_pred, 1), 2), "Flip Y-X"),
                    (np.flip(np.flip(perm_pred, 0), 2), "Flip Z-X"),
                    (np.flip(np.flip(np.flip(perm_pred, 0), 1), 2), "Flip Z-Y-X"),
                ]
                for v_pred, v_label in variations:
                    overlap = np.logical_and(v_pred > 0, gt_f > 0).sum()
                    if overlap > 0:
                        dice = 2. * overlap / ((gt_f > 0).sum() + (v_pred > 0).sum())
                        print(f"    * [{perm_label}] with [{v_label}] -> Dice = {dice:.6f} (Overlap: {int(overlap)} vox)")

if __name__ == "__main__":
    main()
