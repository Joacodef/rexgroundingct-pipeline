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

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
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
    
    # Allow testing on case index 1 (train_13591_a_1) or other validation cases
    val_cases = metadata.get("val", [])
    if not val_cases:
        print("Error: No validation cases found in dataset.json")
        return
        
    case_idx = 1
    if case_idx >= len(val_cases):
        case_idx = 0
    entry = val_cases[case_idx]
    
    scan_id = entry["name"].replace(".nii.gz", "")
    nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
    gt_path = os.path.join(data_prep_dir, f"{scan_id}_seg.nii.gz")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    download_dir = os.getenv("MODEL_DIR")
    voxtell_weights_dir = os.path.join(download_dir, "voxtell_v1.0")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    # Load using NibabelIOWithReorient (gives shape 1, Z, Y, X)
    reader = NibabelIOWithReorient()
    img_raw, img_properties = reader.read_images([nifti_path])
    
    # Load GT mask
    gt_nii = nib.load(gt_path)
    gt_img = gt_nii.get_fdata(dtype=np.float32)
    if gt_img.ndim == 3:
        gt_img = np.expand_dims(gt_img, axis=-1)
    if gt_img.ndim == 4:
        if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0) # Shape: (F, X, Y, Z)
            
    findings = entry.get('findings', {})
    sorted_keys = sorted(findings.keys(), key=int)
    text_prompts = [str(findings[k]) for k in sorted_keys]
    
    print("==========================================================================")
    print(f"CONSOLIDATED INTENSITY & WINDOW TUNING SEARCH ON {scan_id}")
    print(f"Prompts: {text_prompts}")
    print("==========================================================================")
    
    # Pre-embed prompts (shared across experiments to save GPU/CPU compute)
    embeddings = predictor.embed_text_prompts(text_prompts)
    predictor.network = predictor.network.to(device)
    
    # Define windowing experiments (includes raw, cropped, clipped, thresholded, and shifted versions)
    windows = {
        "1. Raw image (no window)": img_raw.copy(),
        "2. Soft-tissue window [-125, 275]": np.clip(img_raw, -125, 275),
        "3. Pulmonary window [-1000, 200]": np.clip(img_raw, -1000, 200),
        "4. Standard Lung/Soft Window [-1000, 400]": np.clip(img_raw, -1000, 400),
        "5. Broad Range [-1000, 1000]": np.clip(img_raw, -1000, 1000),
        "6. Soft-tissue window [-125, 275] Shifted (+125)": np.clip(img_raw, -125, 275) + 125,
        "7. Pulmonary window [-1000, 200] Shifted (+1000)": np.clip(img_raw, -1000, 200) + 1000,
        "8. Standard Lung/Soft Window [-1000, 400] Shifted (+1000)": np.clip(img_raw, -1000, 400) + 1000,
        "9. Standard Broad Window [-1024, 300] Shifted (+1024)": np.clip(img_raw, -1024, 300) + 1024,
        "10. Threshold < -500 set to 0": np.where(img_raw < -500, 0, img_raw),
        "11. Threshold < -800 set to 0": np.where(img_raw < -800, 0, img_raw),
        "12. Threshold < -1000 set to 0": np.where(img_raw < -1000, 0, img_raw),
        "13. Clip to [-1000, 1000] and then threshold < -900 to 0": np.where(np.clip(img_raw, -1000, 1000) < -900, 0, np.clip(img_raw, -1000, 1000)),
    }
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_results = {}
    for f_idx in range(gt_img.shape[0]):
        best_results[f_idx] = {"dice": -1.0, "desc": ""}
        
    for name, img_exp in windows.items():
        print(f"\n--- Running window: {name} ---")
        print(f"  Input min: {img_exp.min():.2f} | max: {img_exp.max():.2f} | mean: {img_exp.mean():.2f} | std: {img_exp.std():.2f}")
        
        # Preprocess using predictor's crop and sizing
        data, bbox, orig_shape = predictor.preprocess(img_exp)
        
        # Predict sliding window logits using embeddings
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
            
        probs = torch.sigmoid(logits.float()).numpy() # Shape: (F, Z_cropped, Y_cropped, X_cropped)
        probs_4d = np.transpose(probs, (0, 3, 2, 1)) # Shape: (F, X_cropped, Y_cropped, Z_cropped)
        
        # Insert back to original shape so that dice is computed exactly in correct dimensions
        probs_full_transposed = np.zeros([gt_img.shape[0], *gt_img.shape[1:]], dtype=np.float32)
        for f_idx in range(gt_img.shape[0]):
            # Revert cropping via target shape
            from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
            reverted_probs_3d = np.zeros(orig_shape, dtype=np.float32)
            reverted_probs_3d = insert_crop_into_image(reverted_probs_3d, probs[f_idx], bbox)
            # Transpose to match gt_img: Shape (X, Y, Z)
            probs_full_transposed[f_idx] = np.transpose(reverted_probs_3d, (2, 1, 0))
            
        for th in thresholds:
            for f_idx in range(gt_img.shape[0]):
                pred_binary = probs_full_transposed[f_idx] > th
                dice = compute_dice(pred_binary, gt_img[f_idx])
                gt_vol = (gt_img[f_idx] > 0).sum()
                pred_vol = pred_binary.sum()
                overlap = np.logical_and(pred_binary, gt_img[f_idx] > 0).sum()
                
                if dice > best_results[f_idx]["dice"]:
                    best_results[f_idx] = {
                        "dice": dice,
                        "desc": f"Window '{name}' + Th {th:.2f} (Pred Vol: {pred_vol}, Overlap: {overlap}, GT Vol: {gt_vol})"
                    }
                
                # Report if there's any performance or interesting overlap
                if dice > 0.01 or pred_vol > 0:
                    print(f"    Finding {f_idx}: Th {th:.2f} | Dice = {dice:.6f} | GT Vol = {gt_vol} | Pred Vol = {pred_vol} | Overlap = {overlap}")
                    
    print("\n" + "="*80)
    print("               CONSOLIDATED BEST INTENSITY & WINDOW SEARCH RESULTS")
    print("="*80)
    for f_idx in range(gt_img.shape[0]):
        print(f"Finding {f_idx} (\"{text_prompts[f_idx][:50]}...\"):")
        print(f"  Best Dice: {best_results[f_idx]['dice']:.6f}")
        print(f"  Config:    {best_results[f_idx]['desc']}")
        print("-"*80)

if __name__ == "__main__":
    main()
