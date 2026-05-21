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

def main():
    data_prep_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
    
    entry = metadata["val"][1] # train_13591_a_1
    scan_id = entry["name"].replace(".nii.gz", "")
    nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
    gt_path = os.path.join(data_prep_dir, f"{scan_id}_seg.nii.gz")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_dir = os.getenv("MODEL_DIR")
    voxtell_weights_dir = os.path.join(download_dir, "voxtell_v1.0")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    
    # Load using NibabelIOWithReorient
    reader = NibabelIOWithReorient()
    img_raw, img_properties = reader.read_images([nifti_path])
    
    # Run diagnostics for two key windows: soft-tissue and threshold < -500
    windows_to_test = {
        "Soft-Tissue Window [-125, 275]": np.clip(img_raw, -125, 275),
        "Threshold < -500 set to 0": np.where(img_raw < -500, 0, img_raw)
    }
    
    # Load GT mask
    gt_nii = nib.load(gt_path)
    gt_img = gt_nii.get_fdata(dtype=np.float32)
    if gt_img.ndim == 3:
        gt_img = np.expand_dims(gt_img, axis=-1)
    if gt_img.ndim == 4:
        if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0) # Shape (F, X, Y, Z)
            
    findings = entry.get('findings', {})
    sorted_keys = sorted(findings.keys(), key=int)
    text_prompts = [str(findings[k]) for k in sorted_keys]
    
    print("==================================================")
    print(f"DIAGNOSING FINDING 0 SPATIAL PROPERTIES ON {scan_id}")
    print("==================================================")
    
    # Get GT coordinates for Finding 0 and Finding 1
    for f_idx in [0, 1]:
        gt_f = gt_img[f_idx] > 0
        gt_vol = gt_f.sum()
        if gt_vol > 0:
            coords = np.argwhere(gt_f)
            c_min = coords.min(axis=0)
            c_max = coords.max(axis=0)
            c_mean = coords.mean(axis=0)
            print(f"GT Finding {f_idx} ({text_prompts[f_idx]}):")
            print(f"  Volume: {gt_vol} voxels")
            print(f"  Bounding Box: X[{c_min[0]} to {c_max[0]}], Y[{c_min[1]} to {c_max[1]}], Z[{c_min[2]} to {c_max[2]}]")
            print(f"  Center of mass: {c_mean}")
        else:
            print(f"GT Finding {f_idx} is empty!")
            
    # Embed text prompts
    embeddings = predictor.embed_text_prompts(text_prompts)
    predictor.network = predictor.network.to(device)
    
    for w_name, img_windowed in windows_to_test.items():
        print(f"\n==========================================")
        print(f"WINDOW: {w_name}")
        print(f"==========================================")
        
        # Get raw logits from the predictor
        data, bbox, orig_shape = predictor.preprocess(img_windowed)
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
            
        probs = torch.sigmoid(logits.float()).numpy() # (F, Z, Y, X)
        
        # Revert cropping to match original size and shape
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        probs_full_transposed = np.zeros([gt_img.shape[0], *gt_img.shape[1:]], dtype=np.float32)
        for f_idx in range(gt_img.shape[0]):
            reverted_probs_3d = np.zeros(orig_shape, dtype=np.float32)
            reverted_probs_3d = insert_crop_into_image(reverted_probs_3d, probs[f_idx], bbox)
            probs_full_transposed[f_idx] = np.transpose(reverted_probs_3d, (2, 1, 0)) # Shape (X, Y, Z)
    
        for f_idx in [0, 1]:
            p_f = probs_full_transposed[f_idx]
            print(f"\nPredictions for Finding {f_idx}:")
            print(f"  Prob min: {p_f.min():.6f} | max: {p_f.max():.6f} | mean: {p_f.mean():.6f} | std: {p_f.std():.6f}")
            
            # Test different thresholds
            for th in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                pred_f = p_f > th
                pred_vol = pred_f.sum()
                overlap = np.logical_and(pred_f, gt_img[f_idx] > 0).sum()
                dice = 2. * overlap / (pred_vol + (gt_img[f_idx] > 0).sum()) if (pred_vol + (gt_img[f_idx] > 0).sum()) > 0 else 0
                print(f"    Th > {th:.2f}: Vol = {pred_vol:6d} vox | Overlap = {overlap:4d} vox | Dice = {dice:.6f}")
                if pred_vol > 0:
                    coords = np.argwhere(pred_f)
                    c_min = coords.min(axis=0)
                    c_max = coords.max(axis=0)
                    print(f"      Pred Bbox: X[{c_min[0]} to {c_max[0]}], Y[{c_min[1]} to {c_max[1]}], Z[{c_min[2]} to {c_max[2]}]")

if __name__ == "__main__":
    main()
