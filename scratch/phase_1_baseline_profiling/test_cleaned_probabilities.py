import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor
from scripts.voxtell.prompt_normalizer import clean_finding_prompt

def main():
    load_dotenv(override=True)
    img_dir = os.environ["IMG_RAW_DIR"]
    gt_dir = os.environ["SEG_RAW_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = "/home/jdeferrari/rex_project/models/voxtell_v1.1"

    scan_id = "train_18382_b_2"
    nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
    
    reader = NibabelIOWithReorient()
    img, img_props = reader.read_images([nifti_path])
    
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata(dtype=np.float32)
    if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
    if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
        gt_data_f = np.moveaxis(gt_data, -1, 0)
    else:
        gt_data_f = gt_data
        
    original_prompts = [
        "Stable, nonspecific 6 mm subpleural nodule in the lateral basal segment of the left lower lobe",
        "Subcentimeter, minimal, nonspecific focal ground-glass opacities in the posterobasal segment of the right lower lobe and in the right middle lobe"
    ]
    cleaned_prompts = [clean_finding_prompt(p) for p in original_prompts]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    
    data_prep, bbox, orig_shape = predictor.preprocess(img)
    
    print("\n--- 1. ORIGINAL PROMPTS LOGITS ---")
    emb_orig = predictor.embed_text_prompts(original_prompts)
    with torch.no_grad():
        logits_orig = predictor.predict_sliding_window_return_logits(data_prep, emb_orig).to("cpu")
        probs_orig = torch.sigmoid(logits_orig.float()).numpy()
        
    print("\n--- 2. CLEANED PROMPTS LOGITS ---")
    emb_clean = predictor.embed_text_prompts(cleaned_prompts)
    with torch.no_grad():
        logits_clean = predictor.predict_sliding_window_return_logits(data_prep, emb_clean).to("cpu")
        probs_clean = torch.sigmoid(logits_clean.float()).numpy()
        
    for f_idx in range(len(original_prompts)):
        p_orig = probs_orig[f_idx]
        p_clean = probs_clean[f_idx]
        gt_vox = int(np.sum(gt_data_f[f_idx] > 0))
        
        print(f"\n==========================================")
        print(f"Finding #{f_idx} (GT Voxels: {gt_vox}):")
        print(f"  ORIGINAL: '{original_prompts[f_idx]}'")
        print(f"    Max prob: {p_orig.max():.4f} | 99.9th percentile: {np.percentile(p_orig, 99.9):.4f}")
        print(f"    Voxels > 0.5: {np.sum(p_orig > 0.5)} | Voxels > 0.2: {np.sum(p_orig > 0.2)} | Voxels > 0.1: {np.sum(p_orig > 0.1)}")
        
        print(f"  CLEANED : '{cleaned_prompts[f_idx]}'")
        print(f"    Max prob: {p_clean.max():.4f} | 99.9th percentile: {np.percentile(p_clean, 99.9):.4f}")
        print(f"    Voxels > 0.5: {np.sum(p_clean > 0.5)} | Voxels > 0.2: {np.sum(p_clean > 0.2)} | Voxels > 0.1: {np.sum(p_clean > 0.1)}")

if __name__ == "__main__":
    main()
