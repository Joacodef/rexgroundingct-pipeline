import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

def compute_dice(pred_bool, gt_bool):
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

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    # Pick 3 cases from the second split (cases 50..200)
    test_cases = val_cases[50:53]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    reader = NibabelIOWithReorient()
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for entry in test_cases:
        scan_id = entry["name"].replace(".nii.gz", "")
        nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path): continue
        
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata(dtype=np.float32)
        if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data_f = np.moveaxis(gt_data, -1, 0)
        else:
            gt_data_f = gt_data
            
        img, img_props = reader.read_images([nifti_path])
        
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
            
        print(f"\n==========================================")
        print(f"=== TESTING PROBABILITY MAPS: {scan_id} ===")
        print(f"==========================================")
        
        # Override predictor.predict_single_image to get raw continuous probabilities
        # In VoxTellPredictor, predict_single_image calls predictor.predict_from_files or sliding_window_inference
        # VoxTell predictor returns binary array (or continuous logits/probs).
        # Let's inspect predictor.predict_single_image code in voxtell package.
        with torch.no_grad():
            raw_output = predictor.predict_single_image(img, prompts)
            
        print(f"Raw output shape: {raw_output.shape}, dtype: {raw_output.dtype}, min: {raw_output.min()}, max: {raw_output.max()}")
        print(f"Voxel value distribution:")
        print(f"  > 0.0: {np.sum(raw_output > 0)}")
        print(f"  > 0.1: {np.sum(raw_output > 0.1)}")
        print(f"  > 0.3: {np.sum(raw_output > 0.3)}")
        print(f"  > 0.5: {np.sum(raw_output > 0.5)}")
        
        for f_idx in range(min(len(prompts), gt_data_f.shape[0])):
            gt_m = gt_data_f[f_idx] > 0
            pr_m = raw_output[f_idx]
            
            print(f"\nFinding #{f_idx} ('{prompts[f_idx][:35]}...'): GT Voxels = {np.sum(gt_m)}")
            for t in thresholds:
                pred_t = pr_m > t
                d = compute_dice(pred_t, gt_m)
                print(f"  Threshold {t:.1f} -> Pred Voxels: {np.sum(pred_t)} | Dice: {d:.4f}")

if __name__ == "__main__":
    main()
