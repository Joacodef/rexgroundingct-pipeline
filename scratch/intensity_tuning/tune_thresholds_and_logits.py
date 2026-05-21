import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Restrict visibility to compatible GPU 0/1 to bypass Blackwell sm_120 driver mismatch warnings
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import torch
import numpy as np
import nibabel as nib
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def compute_dice(pred_bool, gt_bool):
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    data_prep_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    val_cases = metadata.get("val", [])
    if not val_cases:
        print("Error: No validation cases found in dataset.json")
        return
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    download_dir = os.getenv("MODEL_DIR")
    voxtell_weights_dir = os.path.join(download_dir, "voxtell_v1.0")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    reader = NibabelIOWithReorient()
    
    # We can evaluate on a range of sigmoid thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    # Let's test on the first 5 validation cases
    cases_to_test = val_cases[:5]
    
    # Select which window to use for testing thresholds (defaulting to soft-tissue window)
    window_names = {
        "Raw": lambda x: x,
        "Soft-tissue [-125, 275]": lambda x: np.clip(x, -125, 275)
    }
    
    print("==========================================================================")
    print("CONSOLIDATED LOGIT & THRESHOLD ANALYSIS SYSTEM")
    print("==========================================================================")
    
    for w_name, w_func in window_names.items():
        print(f"\n==========================================")
        print(f"EVALUATING WINDOW TYPE: {w_name}")
        print(f"==========================================")
        
        for case_idx, entry in enumerate(cases_to_test):
            scan_id = entry["name"].replace(".nii.gz", "")
            nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
            gt_path = os.path.join(data_prep_dir, f"{scan_id}_seg.nii.gz")
            
            if not os.path.exists(nifti_path) or not os.path.exists(gt_path):
                print(f"Skipping {scan_id} (missing files)")
                continue
                
            print(f"\n  Case {case_idx}: {scan_id}")
            
            # Load GT mask
            gt_nii = nib.load(gt_path)
            gt_img = gt_nii.get_fdata(dtype=np.float32)
            if gt_img.ndim == 3:
                gt_img = np.expand_dims(gt_img, axis=-1)
            if gt_img.ndim == 4:
                if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
                    gt_img = np.moveaxis(gt_img, -1, 0)
            
            # Load input and apply window function
            img_raw, img_properties = reader.read_images([nifti_path])
            img_processed = w_func(img_raw)
            
            # Preprocess image to model grid
            data, bbox, orig_shape = predictor.preprocess(img_processed)
            
            # Get text prompts
            findings = entry.get('findings', {})
            sorted_keys = sorted(findings.keys(), key=int)
            text_prompts = [str(findings[k]) for k in sorted_keys]
            
            # Embed text prompts
            embeddings = predictor.embed_text_prompts(text_prompts)
            
            # Predict sliding window logits
            predictor.network = predictor.network.to(device)
            with torch.no_grad():
                logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
                
            # Logit & Sigmoid statistical analysis
            sig = torch.sigmoid(logits.float()).numpy()
            
            print(f"    [Stats] Logits min: {logits.min().item():.3f} | max: {logits.max().item():.3f} | mean: {logits.mean().item():.3f}")
            print(f"    [Stats] Sigmoid min: {sig.min().item():.6f} | max: {sig.max().item():.6f} | mean: {sig.mean().item():.6f}")
            
            # Revert cropping to get full size in Z, Y, X
            from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
            
            sig_full = np.zeros([sig.shape[0], *orig_shape], dtype=np.float32)
            for i in range(sig.shape[0]):
                sig_full[i] = insert_crop_into_image(sig_full[i], sig[i], bbox)
                
            # Transpose back to (F, X, Y, Z) to align with gt_img
            sig_full_transposed = np.transpose(sig_full, (0, 3, 2, 1))
            
            # Search thresholds
            for th in thresholds:
                dices = []
                print(f"    --- Threshold: {th} ---")
                for f_idx in range(gt_img.shape[0]):
                    pred_bool = sig_full_transposed[f_idx] > th
                    gt_bool = gt_img[f_idx] > 0
                    dice = compute_dice(pred_bool, gt_bool)
                    dices.append(dice)
                    
                    gt_vol = gt_bool.sum()
                    pred_vol = pred_bool.sum()
                    overlap = np.logical_and(pred_bool, gt_bool).sum()
                    
                    # Print if any activation found or dice is non-zero
                    if dice > 0.0 or pred_vol > 0:
                        print(f"      Finding {f_idx} (\"{text_prompts[f_idx][:25]}...\"): Dice = {dice:.6f} | GT Vol = {gt_vol} | Pred Vol = {pred_vol} | Overlap = {overlap}")
                print(f"      Mean Dice across findings: {np.mean(dices):.6f}")

if __name__ == "__main__":
    main()
