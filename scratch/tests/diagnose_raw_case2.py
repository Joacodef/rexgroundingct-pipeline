import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import torch
import numpy as np
import nibabel as nib
from voxtell.inference.predictor import VoxTellPredictor
from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd, MapTransform

class CopyAffined(MapTransform):
    def __init__(self, keys, ref_key="image", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        ref_tensor = d[self.ref_key]
        for key in self.key_iterator(d):
            if key == self.ref_key:
                continue
            target_tensor = d[key]
            if hasattr(target_tensor, "affine") and hasattr(ref_tensor, "affine"):
                target_tensor.affine = ref_tensor.affine.clone()
            if hasattr(target_tensor, "meta") and hasattr(ref_tensor, "meta"):
                for meta_k in ["affine", "original_affine", "pixdim", "space", "spatial_shape"]:
                    if meta_k in ref_tensor.meta:
                        val = ref_tensor.meta[meta_k]
                        if hasattr(val, "clone"):
                            target_tensor.meta[meta_k] = val.clone()
                        elif hasattr(val, "copy"):
                            target_tensor.meta[meta_k] = val.copy()
                        else:
                            target_tensor.meta[meta_k] = val
        return d

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
    
    entry = metadata["val"][1] # train_13591_a_1
    scan_id = entry["name"].replace(".nii.gz", "")
    raw_img_path = os.path.join(raw_img_dir, entry["name"])
    raw_seg_path = os.path.join(raw_seg_dir, entry["name"])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_dir = os.getenv("MODEL_DIR")
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    voxtell_weights_dir = os.path.join(models_root, "voxtell_v1.1")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    pipeline_raw = Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureChannelFirstd(keys=["label"], channel_dim=0),
        CopyAffined(keys=["label"], ref_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])
    
    data_dict = pipeline_raw({"image": raw_img_path, "label": raw_seg_path})
    img_raw_tensor = data_dict["image"]
    seg_raw_tensor = data_dict["label"]
    
    img_raw_np = img_raw_tensor.numpy()[0]
    seg_raw_np = seg_raw_tensor.numpy()
    
    findings = entry.get('findings', {})
    sorted_keys = sorted(findings.keys(), key=int)
    text_prompts = [str(findings[k]) for k in sorted_keys]
    
    print("==================================================")
    print(f"Diagnosing Case 2 raw prediction and alignment: {scan_id}")
    print("==================================================")
    
    with torch.no_grad():
        pred_seg = predictor.predict_single_image(img_raw_np, text_prompts)
        
    for f_idx in range(seg_raw_np.shape[0]):
        print(f"\nFinding {f_idx}: {text_prompts[f_idx]}")
        gt_f = seg_raw_np[f_idx]
        pred_f = pred_seg[f_idx]
        
        gt_bbox, gt_cent = get_bbox_and_centroid(gt_f)
        pred_bbox, pred_cent = get_bbox_and_centroid(pred_f)
        
        print(f"  GT Vol: {int((gt_f > 0).sum())} vox | Centroid: {gt_cent} | Bbox: {gt_bbox}")
        print(f"  Pred Vol: {int((pred_f > 0).sum())} vox | Centroid: {pred_cent} | Bbox: {pred_bbox}")
        
        if gt_bbox and pred_bbox:
            # Let's check permutations/flips of prediction to align with GT
            permutations = [
                ((0, 1, 2), "No Permutation"),
                ((0, 2, 1), "Swap Y-Z"),
                ((1, 0, 2), "Swap X-Y"),
                ((1, 2, 0), "Shift left"),
                ((2, 0, 1), "Shift right"),
                ((2, 1, 0), "Swap X-Z")
            ]
            for axes, perm_label in permutations:
                perm_pred = np.transpose(pred_f, axes)
                if perm_pred.shape != gt_f.shape:
                    continue
                variations = [
                    (perm_pred, "Normal"),
                    (np.flip(perm_pred, 0), "Flip X"),
                    (np.flip(perm_pred, 1), "Flip Y"),
                    (np.flip(perm_pred, 2), "Flip Z"),
                    (np.flip(np.flip(perm_pred, 0), 1), "Flip X-Y"),
                    (np.flip(np.flip(perm_pred, 1), 2), "Flip Y-Z"),
                    (np.flip(np.flip(perm_pred, 0), 2), "Flip X-Z"),
                    (np.flip(np.flip(np.flip(perm_pred, 0), 1), 2), "Flip X-Y-Z"),
                ]
                for v_pred, v_label in variations:
                    overlap = np.logical_and(v_pred > 0, gt_f > 0).sum()
                    if overlap > 0:
                        dice = 2. * overlap / ((gt_f > 0).sum() + (v_pred > 0).sum())
                        print(f"    * [{perm_label}] with [{v_label}] -> Dice = {dice:.6f} (Overlap: {int(overlap)} vox)")

if __name__ == "__main__":
    main()
