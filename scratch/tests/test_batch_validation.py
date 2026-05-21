import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Restrict visibility to compatible GPU 0/1 to bypass Blackwell sm_120 driver mismatch warnings
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
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
    
    entries = metadata["val"][:5] # Test first 5 cases
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_dir = os.getenv("MODEL_DIR")
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    voxtell_weights_dir = os.path.join(models_root, "voxtell_v1.1")
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    reader = NibabelIOWithReorient()
    
    print("==================================================")
    print("Evaluating official VoxTell preprocessing (raw HU + global nnUNet normalisation) on 5 cases")
    print("==================================================")
    
    all_dices = []
    
    for entry in entries:
        scan_id = entry["name"].replace(".nii.gz", "")
        nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
        gt_path = os.path.join(data_prep_dir, f"{scan_id}_seg.nii.gz")
        
        if not os.path.exists(nifti_path) or not os.path.exists(gt_path):
            print(f"Skipping {scan_id} (missing files)")
            continue
            
        print(f"\n--> CASE: {scan_id}")
        
        # Load using NibabelIOWithReorient
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
        text_prompts = []
        for k in sorted_keys:
            val = findings[k]
            if isinstance(val, dict):
                text_prompts.append(val.get('text', ''))
            else:
                text_prompts.append(str(val))
                
        # Run predictor to get raw probabilities (bypass binary casting in predict_single_image)
        # Preprocess image
        data, bbox, orig_shape = predictor.preprocess(img_raw)
        # Embed prompts
        embeddings = predictor.embed_text_prompts(text_prompts)
        
        predictor.network = predictor.network.to(device)
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
            
        # Apply sigmoid and threshold = 0.7
        probs = torch.sigmoid(logits.float()).numpy()
        pred_binary = probs > 0.7
        
        # Insert back to original shape
        segmentation_reverted_cropping = np.zeros(
            [pred_binary.shape[0], *orig_shape],
            dtype=np.uint8
        )
        for i in range(pred_binary.shape[0]):
            from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
            segmentation_reverted_cropping[i] = insert_crop_into_image(
                segmentation_reverted_cropping[i], pred_binary[i], bbox
            )
            
        # Transpose back to (F, X, Y, Z)
        pred_4d = np.transpose(segmentation_reverted_cropping, (0, 3, 2, 1))
        
        for f_idx in range(gt_img.shape[0]):
            dice = compute_dice(pred_4d[f_idx], gt_img[f_idx])
            all_dices.append(dice)
            gt_vol = (gt_img[f_idx] > 0).sum()
            pred_vol = (pred_4d[f_idx] > 0).sum()
            overlap = np.logical_and(pred_4d[f_idx] > 0, gt_img[f_idx] > 0).sum()
            print(f"  Finding {f_idx}: Dice = {dice:.6f} | GT Vol = {gt_vol} | Pred Vol = {pred_vol} | Overlap = {overlap}")
            
    print(f"\nAverage Dice on these 5 cases: {np.mean(all_dices):.4f}")

if __name__ == "__main__":
    main()
